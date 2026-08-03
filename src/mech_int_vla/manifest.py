"""Deterministic episode manifest construction for all preregistered splits."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import (
    ConditionSpec,
    ProtocolConfig,
    ProtocolConfigError,
    SplitName,
    TaskSpec,
)
from .determinism import hash_order, hash_seed
from .guard import assert_calibration_ready, assert_locked_test_ready


@dataclass(frozen=True)
class EpisodeSpec:
    suite: str
    task_id: int
    task_rank: int
    split: SplitName
    base_init_state_id: int
    condition_index: int
    condition_name: str
    condition_family: str
    condition_parameters: Mapping[str, Any]
    reset_seed: int
    inference_seed: int
    policy_revision: str
    code_commit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "condition_parameters",
            _freeze_parameters(self.condition_parameters),
        )

    @property
    def episode_id(self) -> str:
        return (
            f"{self.suite}-task{self.task_id}-{self.split.value}-"
            f"init{self.base_init_state_id:02d}-cell{self.condition_index}"
        )

    def noise_seed(self, stream: str, draw_index: int) -> int:
        """Derive a common-random-number seed independent of predictor/model name."""

        if not stream or draw_index < 0:
            raise ValueError("stream must be nonempty and draw_index nonnegative")
        return hash_seed("policy-noise-v1", self.inference_seed, stream, draw_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "task_rank": self.task_rank,
            "split": self.split.value,
            "base_init_state_id": self.base_init_state_id,
            "condition_index": self.condition_index,
            "condition_name": self.condition_name,
            "condition_family": self.condition_family,
            "condition_parameters": dict(self.condition_parameters),
            "reset_seed": self.reset_seed,
            "inference_seed": self.inference_seed,
            "policy_revision": self.policy_revision,
            "code_commit": self.code_commit,
        }


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    split: SplitName
    task: TaskSpec
    episodes: tuple[EpisodeSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split": self.split.value,
            "task": {
                "rank": self.task.rank,
                "suite": self.task.suite,
                "task_id": self.task.task_id,
                "language": self.task.language,
                "primary_object": self.task.primary_object,
                "planar_symmetry_order": self.task.planar_symmetry_order,
            },
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _freeze_parameter_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_parameters(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter_value(nested) for nested in value)
    return value


def _freeze_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_parameter_value(value) for key, value in parameters.items()}
    )


def _realized_translation(condition: ConditionSpec, direction: str) -> dict[str, Any]:
    parameters = dict(condition.parameters)
    magnitude = float(parameters["magnitude_m"])
    if direction == "x_positive":
        delta = (magnitude, 0.0)
    elif direction == "x_negative":
        delta = (-magnitude, 0.0)
    elif direction == "y_positive":
        delta = (0.0, magnitude)
    elif direction == "y_negative":
        delta = (0.0, -magnitude)
    else:
        diagonal_signs = {
            "x_positive_y_positive": (1.0, 1.0),
            "x_positive_y_negative": (1.0, -1.0),
            "x_negative_y_positive": (-1.0, 1.0),
            "x_negative_y_negative": (-1.0, -1.0),
        }
        if direction not in diagonal_signs:
            raise ProtocolConfigError(f"unknown translation direction: {direction}")
        signs = diagonal_signs[direction]
        component = magnitude / math.sqrt(2.0)
        delta = (signs[0] * component, signs[1] * component)
    parameters["assigned_direction"] = direction
    parameters["delta_xy_m"] = [delta[0], delta[1]]
    return parameters


def _balanced_directions(
    init_ids: tuple[int, ...], *, task: TaskSpec, split: SplitName, diagonal: bool
) -> dict[int, str]:
    directions = (
        (
            "x_positive_y_positive",
            "x_positive_y_negative",
            "x_negative_y_positive",
            "x_negative_y_negative",
        )
        if diagonal
        else ("x_positive", "x_negative", "y_positive", "y_negative")
    )
    if len(init_ids) % len(directions):
        raise ProtocolConfigError(
            "balanced direction assignment requires equal cell counts"
        )
    ordered_ids = hash_order(
        init_ids, "translation-init-order-v1", task.suite, task.task_id, split.value
    )
    ordered_directions = hash_order(
        directions,
        "translation-direction-order-v1",
        task.suite,
        task.task_id,
        split.value,
    )
    return {
        init_id: ordered_directions[position % len(ordered_directions)]
        for position, init_id in enumerate(ordered_ids)
    }


def _discovery_assignments(
    init_ids: tuple[int, ...], conditions: tuple[ConditionSpec, ...], task: TaskSpec
) -> dict[int, tuple[ConditionSpec, ...]]:
    if len(init_ids) % 2 or len(conditions) != 6:
        raise ProtocolConfigError(
            "version-1 Reality Gate assignment requires an even init count and six cells"
        )
    by_magnitude: dict[float, list[ConditionSpec]] = {}
    for condition in conditions:
        try:
            magnitude = abs(float(condition.parameters["value"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolConfigError(
                "Reality Gate cells must contain numeric yaw values"
            ) from exc
        by_magnitude.setdefault(magnitude, []).append(condition)
    if len(by_magnitude) != 3 or any(
        len(magnitude_cells) != 2 for magnitude_cells in by_magnitude.values()
    ):
        raise ProtocolConfigError(
            "Reality Gate must define two signed cells at each of three magnitudes"
        )

    assignments: dict[int, list[ConditionSpec]] = {init_id: [] for init_id in init_ids}
    for magnitude in sorted(by_magnitude):
        ordered_ids = hash_order(
            init_ids,
            "reality-gate-sign-assignment-v1",
            task.suite,
            task.task_id,
            magnitude,
        )
        signed_cells = hash_order(
            by_magnitude[magnitude],
            "reality-gate-sign-order-v1",
            task.suite,
            task.task_id,
            magnitude,
        )
        midpoint = len(ordered_ids) // 2
        for position, init_id in enumerate(ordered_ids):
            assignments[init_id].append(signed_cells[position >= midpoint])
    return {init_id: tuple(cells) for init_id, cells in assignments.items()}


def _episode(
    *,
    task: TaskSpec,
    split: SplitName,
    init_id: int,
    condition_index: int,
    condition: ConditionSpec,
    parameters: Mapping[str, Any],
    config: ProtocolConfig,
    policy_revision: str,
    code_commit: str,
) -> EpisodeSpec:
    return EpisodeSpec(
        suite=task.suite,
        task_id=task.task_id,
        task_rank=task.rank,
        split=split,
        base_init_state_id=init_id,
        condition_index=condition_index,
        condition_name=condition.name,
        condition_family=condition.family,
        condition_parameters=dict(parameters),
        reset_seed=config.split.seed_for(
            split,
            task_rank=task.rank,
            base_init_state_id=init_id,
            condition_index=condition_index,
        ),
        inference_seed=hash_seed(
            "episode-inference-v1",
            task.suite,
            task.task_id,
            split.value,
            init_id,
            condition_index,
            condition.name,
        ),
        policy_revision=policy_revision,
        code_commit=code_commit,
    )


def generate_episode_manifest(
    split: SplitName | str,
    task: TaskSpec,
    config: ProtocolConfig,
    *,
    policy_revision: str,
    code_commit: str,
    repo_root: str | Path | None = None,
) -> Manifest:
    """Instantiate a deterministic manifest, guarding protected splits fail-closed."""

    split = SplitName(split)
    if task not in config.task_order.tasks:
        raise ProtocolConfigError("task is not present in the configured task order")
    if not policy_revision or not code_commit:
        raise ProtocolConfigError("policy_revision and code_commit are required")

    if split in {SplitName.CALIBRATION, SplitName.LOCKED_TEST}:
        if repo_root is None:
            raise ProtocolConfigError(
                f"repo_root is required to instantiate {split.value}"
            )
        receipt = (
            assert_calibration_ready(
                repo_root,
                config.split.calibration_guard,
                protocol=config,
                task=task,
                policy_revision=policy_revision,
            )
            if split is SplitName.CALIBRATION
            else assert_locked_test_ready(
                repo_root,
                config.split.locked_test_guard,
                task=task,
                policy_revision=policy_revision,
                selection=config.split.calibration_selection,
            )
        )
        if code_commit != receipt.head_commit:
            raise ProtocolConfigError(
                f"{split.value} manifest code_commit must equal the guarded HEAD commit"
            )

    return reconstruct_episode_manifest(
        split,
        task,
        config,
        policy_revision=policy_revision,
        code_commit=code_commit,
    )


def reconstruct_episode_manifest(
    split: SplitName | str,
    task: TaskSpec,
    config: ProtocolConfig,
    *,
    policy_revision: str,
    code_commit: str,
) -> Manifest:
    """Purely reconstruct frozen topology for historical receipt validation.

    Unlike :func:`generate_episode_manifest`, this function does not authorize
    collection of a protected split.  It exists so a manifest collected at the
    exact Reality-Gate or Calibration lock commit remains independently
    revalidatable after HEAD advances to a later lock commit.
    """

    split = SplitName(split)
    if task not in config.task_order.tasks:
        raise ProtocolConfigError("task is not present in the configured task order")
    if not policy_revision or not code_commit:
        raise ProtocolConfigError("policy_revision and code_commit are required")

    init_ids = config.split.splits[split].init_state_ids.ids()
    episodes: list[EpisodeSpec] = []
    if split is SplitName.DISCOVERY:
        iid = ConditionSpec("iid", "iid", 0)
        assignments = _discovery_assignments(
            init_ids, config.perturbations.reality_gate_cells, task
        )
        for init_id in init_ids:
            episodes.append(
                _episode(
                    task=task,
                    split=split,
                    init_id=init_id,
                    condition_index=0,
                    condition=iid,
                    parameters={},
                    config=config,
                    policy_revision=policy_revision,
                    code_commit=code_commit,
                )
            )
            for condition_index, condition in enumerate(assignments[init_id], start=1):
                episodes.append(
                    _episode(
                        task=task,
                        split=split,
                        init_id=init_id,
                        condition_index=condition_index,
                        condition=condition,
                        parameters=condition.parameters,
                        config=config,
                        policy_revision=policy_revision,
                        code_commit=code_commit,
                    )
                )
    else:
        conditions = (
            config.perturbations.calibration_rollouts
            if split is SplitName.CALIBRATION
            else config.perturbations.locked_test_rollouts
        )
        translation = next(
            condition
            for condition in conditions
            if "direction_assignment" in condition.parameters
        )
        assignment_name = str(translation.parameters["direction_assignment"])
        if assignment_name not in {"balanced_hash_cardinal", "balanced_hash_diagonal"}:
            raise ProtocolConfigError(
                f"unsupported direction assignment: {assignment_name}"
            )
        directions = _balanced_directions(
            init_ids,
            task=task,
            split=split,
            diagonal=assignment_name == "balanced_hash_diagonal",
        )
        for init_id in init_ids:
            for condition in conditions:
                assert condition.index is not None
                parameters = (
                    _realized_translation(condition, directions[init_id])
                    if condition is translation
                    else condition.parameters
                )
                episodes.append(
                    _episode(
                        task=task,
                        split=split,
                        init_id=init_id,
                        condition_index=condition.index,
                        condition=condition,
                        parameters=parameters,
                        config=config,
                        policy_revision=policy_revision,
                        code_commit=code_commit,
                    )
                )

    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ProtocolConfigError("manifest contains duplicate episode IDs")
    if len({episode.reset_seed for episode in episodes}) != len(episodes):
        raise ProtocolConfigError("manifest contains duplicate reset seeds")
    return Manifest(schema_version=1, split=split, task=task, episodes=tuple(episodes))
