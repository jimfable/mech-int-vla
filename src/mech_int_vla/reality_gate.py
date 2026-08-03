"""Pure, fail-closed decisions for the preregistered Discovery Reality Gate.

This module never discovers or loads rollout artifacts.  Callers must supply a
validated Discovery :class:`~mech_int_vla.manifest.Manifest` and the exact raw
artifact identities for cells that were executed.  The decision layer then
proves coverage and applies only the point-estimate rules frozen in
``PREREG.md``; Wilson intervals are retained as descriptive metadata.

The orientation helper fixes two details that would otherwise be ambiguous:
every recorded control state (including the terminal frame) has equal weight,
and every episode must have a complete integer control-step cadence beginning at
zero.  For planar symmetry order ``s``, its physical circular standard deviation
is

``sqrt(-2 log |mean(exp(i s theta_rel))|) / s``.

Dividing by ``s`` returns radians in physical-angle units rather than in the
symmetry-multiplied representation space.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from numbers import Integral, Real
from typing import Any, Final

import numpy as np

from .artifacts import RolloutArtifact
from .config import ProtocolConfig, SplitName, TaskSpec
from .evaluation import (
    DynamicRangeGateResult,
    EvaluationError,
    RateInterval,
    ReproductionGateResult,
    dynamic_range_gate_decision,
    reproduction_gate_decision,
)
from .failure_events import ArtifactIdentity, FailureEventFreezeManifest
from .manifest import EpisodeSpec, Manifest, generate_episode_manifest

REALITY_GATE_SCHEMA_VERSION: Final = 1
REALITY_GATE_PROTOCOL: Final = "preregistered-reality-gate-v1"
REALITY_GATE_LOCK_SCHEMA_VERSION: Final = 1
REALITY_GATE_LOCK_PROTOCOL: Final = "preregistered-reality-gate-lock-v1"
VARIABLE_FALLBACK_ORDER: Final[tuple[str, ...]] = (
    "orientation",
    "planar_position",
    "contact",
)
ORIENTATION_WEIGHTING: Final = "equal_weight_per_recorded_control_state"
ORIENTATION_CADENCE: Final = "every_control_step_including_terminal_frame"
ORIENTATION_MINIMUM_FINITE_FRACTION: Final = 0.90
ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES: Final = 15.0

_SELECTION_RULE: Final = "first_task_passing_reproduction_and_dynamic_range_gates"
_DISCOVERY_INIT_IDS: Final = tuple(range(10))
_EXPECTED_YAWS: Final = (-45.0, -30.0, -15.0, 15.0, 30.0, 45.0)
_EXPECTED_TASKS: Final = (
    (
        1,
        "libero_10",
        5,
        "pick up the book and place it in the back compartment of the caddy",
        "black_book",
        2,
    ),
    (
        2,
        "libero_10",
        2,
        "turn on the stove and put the moka pot on it",
        "moka_pot",
        1,
    ),
    (
        3,
        "libero_10",
        9,
        "put the yellow and white mug in the microwave and close it",
        "white_yellow_mug",
        1,
    ),
)
_TASK_GATE_DECISION_FACTORY_TOKEN: Final = object()
_REALITY_GATE_RECEIPT_FACTORY_TOKEN: Final = object()
_DISCOVERY_CELL_FACTORY_TOKEN: Final = object()
_DISCOVERY_ATTEMPT_FACTORY_TOKEN: Final = object()
_ORIENTATION_ROLLOUT_FACTORY_TOKEN: Final = object()
_ORIENTATION_ARITHMETIC_FACTORY_TOKEN: Final = object()


class RealityGateError(ValueError):
    """Raised when supplied evidence cannot support the frozen gate decision."""


@dataclass(frozen=True)
class ValidityRetryEvidence:
    """Minimal receipt evidence copied from a raw artifact's retry audit.

    The primary raw artifact's metadata hash binds the complete initial and retry
    validity payloads.  This object retains only the fields needed to prove that
    the mandatory identical reset was performed after an invalid initial reset.
    Disagreement is evidence to retain, not a reason to discard the cell.
    """

    performed: bool
    same_reset_seed_and_condition: bool | None = None
    agrees_on_invalidity: bool | None = None
    agrees_on_reasons: bool | None = None

    def __post_init__(self) -> None:
        if type(self.performed) is not bool:
            raise RealityGateError("validity retry performed must be boolean")
        details = (
            self.same_reset_seed_and_condition,
            self.agrees_on_invalidity,
            self.agrees_on_reasons,
        )
        if not self.performed:
            if any(value is not None for value in details):
                raise RealityGateError(
                    "an unperformed validity retry cannot carry retry fields"
                )
            return
        if self.same_reset_seed_and_condition is not True:
            raise RealityGateError(
                "a performed validity retry must use the same reset seed and condition"
            )
        if (
            type(self.agrees_on_invalidity) is not bool
            or type(self.agrees_on_reasons) is not bool
        ):
            raise RealityGateError(
                "a performed validity retry must retain both agreement booleans"
            )

    def to_dict(self) -> dict[str, Any]:
        if not self.performed:
            return {"performed": False}
        return {
            "performed": True,
            "same_reset_seed_and_condition": True,
            "agrees_on_invalidity": self.agrees_on_invalidity,
            "agrees_on_reasons": self.agrees_on_reasons,
        }


@dataclass(frozen=True)
class DiscoveryCellResult:
    """Derived outcome for one manifested Discovery cell.

    Unexecuted perturbation cells remain represented with ``executed=False`` so
    the decision can prove exact 40-cell manifest coverage while also proving
    that perturbations were not attempted after a reproduction failure.

    Instances are created only by :func:`task_discovery_attempt_from_rollouts`;
    callers cannot assert validity, success, retry, or artifact-identity fields.
    """

    episode_id: str
    executed: bool
    artifact: ArtifactIdentity | None
    valid: bool | None
    success: bool | None
    validity_retry: ValidityRetryEvidence | None
    _factory_token: InitVar[object] = None
    _validation_marker: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _DISCOVERY_CELL_FACTORY_TOKEN:
            raise RealityGateError(
                "DiscoveryCellResult is produced only from RolloutArtifact evidence"
            )
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise RealityGateError("episode_id must be a non-empty string")
        if type(self.executed) is not bool:
            raise RealityGateError("executed must be boolean")
        if self.executed:
            if not isinstance(self.artifact, ArtifactIdentity):
                raise RealityGateError(
                    "an executed cell requires a validated ArtifactIdentity"
                )
            if self.artifact.episode_id != self.episode_id:
                raise RealityGateError("cell and artifact episode IDs differ")
            if type(self.valid) is not bool or type(self.success) is not bool:
                raise RealityGateError(
                    "an executed cell requires boolean valid and success values"
                )
            if not self.valid and self.success:
                raise RealityGateError("an invalid episode cannot be successful")
            if not isinstance(self.validity_retry, ValidityRetryEvidence):
                raise RealityGateError(
                    "an executed cell requires validated validity-retry evidence"
                )
            if self.valid and self.validity_retry.performed:
                raise RealityGateError(
                    "a valid initial reset cannot have a performed validity retry"
                )
            if not self.valid and not self.validity_retry.performed:
                raise RealityGateError(
                    "an invalid initial reset requires one performed validity retry"
                )
        elif any(
            value is not None
            for value in (
                self.artifact,
                self.valid,
                self.success,
                self.validity_retry,
            )
        ):
            raise RealityGateError(
                "an unexecuted cell cannot carry an artifact, validity, or outcome"
            )
        object.__setattr__(
            self,
            "_validation_marker",
            _DISCOVERY_CELL_FACTORY_TOKEN,
        )


@dataclass(frozen=True)
class TaskDiscoveryAttempt:
    """Exact trusted representation of one ordered-task rollout attempt."""

    manifest: Manifest
    cells: tuple[DiscoveryCellResult, ...]
    executed_rollouts: tuple[RolloutArtifact, ...]
    _factory_token: InitVar[object] = None
    _validation_marker: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _DISCOVERY_ATTEMPT_FACTORY_TOKEN:
            raise RealityGateError(
                "TaskDiscoveryAttempt is produced only from RolloutArtifact evidence"
            )
        if not isinstance(self.manifest, Manifest):
            raise RealityGateError("manifest must be a validated Manifest")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(cell, DiscoveryCellResult) for cell in self.cells
        ):
            raise RealityGateError(
                "cells must be a tuple of validated DiscoveryCellResult values"
            )
        if any(
            cell._validation_marker is not _DISCOVERY_CELL_FACTORY_TOKEN
            for cell in self.cells
        ):
            raise RealityGateError("attempt cells must be trusted derived results")
        if not isinstance(self.executed_rollouts, tuple) or any(
            not isinstance(raw, RolloutArtifact) for raw in self.executed_rollouts
        ):
            raise RealityGateError(
                "executed_rollouts must be a RolloutArtifact tuple"
            )
        executed_cell_ids = tuple(
            cell.episode_id for cell in self.cells if cell.executed
        )
        if tuple(raw.episode_id for raw in self.executed_rollouts) != executed_cell_ids:
            raise RealityGateError(
                "executed rollouts must exactly match executed cells in manifest order"
            )
        object.__setattr__(
            self,
            "_validation_marker",
            _DISCOVERY_ATTEMPT_FACTORY_TOKEN,
        )


def task_discovery_attempt_from_rollouts(
    protocol: ProtocolConfig,
    manifest: Manifest,
    executed_rollouts: tuple[RolloutArtifact, ...],
) -> TaskDiscoveryAttempt:
    """Build one trusted Discovery attempt from exact raw rollout evidence.

    Only two execution shapes are legal: all ten IID cells and no perturbations
    after reproduction fails, or all forty cells after reproduction passes.
    Unexecuted perturbation cells in the first form are inserted here rather
    than accepted from a caller.
    """

    _validate_protocol(protocol)
    episodes = _validated_discovery_manifest(protocol, manifest)
    if not isinstance(executed_rollouts, tuple) or any(
        not isinstance(raw, RolloutArtifact) for raw in executed_rollouts
    ):
        raise RealityGateError(
            "executed_rollouts must be a tuple of validated RolloutArtifact values"
        )
    observed_ids = tuple(raw.episode_id for raw in executed_rollouts)
    if len(set(observed_ids)) != len(observed_ids):
        raise RealityGateError("executed rollout episode IDs must be unique")
    expected_by_id = {episode.episode_id: episode for episode in episodes}
    extra = sorted(set(observed_ids) - set(expected_by_id))
    if extra:
        raise RealityGateError(f"executed rollouts are absent from manifest: {extra}")
    iid_ids = {
        episode.episode_id
        for episode in episodes
        if episode.condition_family == "iid"
    }
    all_ids = set(expected_by_id)
    observed_set = set(observed_ids)
    if observed_set not in (iid_ids, all_ids):
        missing = sorted(all_ids - observed_set)
        raise RealityGateError(
            "executed rollouts must be exactly the 10 IID cells or all 40 cells: "
            f"missing={missing}"
        )

    by_id = {raw.episode_id: raw for raw in executed_rollouts}
    derived_cells: dict[str, DiscoveryCellResult] = {}
    for episode_id, raw in by_id.items():
        derived_cells[episode_id] = _discovery_cell_from_rollout(
            manifest.task,
            expected_by_id[episode_id],
            raw,
        )
    base_vlm_revisions = {
        raw.metadata["model"]["base_vlm_revision"] for raw in executed_rollouts
    }
    if len(base_vlm_revisions) != 1:
        raise RealityGateError(
            "all executed rollouts must use one base-VLM revision"
        )
    iid_successes = sum(
        bool(derived_cells[episode_id].success) for episode_id in iid_ids
    )
    reproduction_passes = (
        iid_successes >= protocol.task_order.gates.baseline_min_successes
    )
    if observed_set == iid_ids and reproduction_passes:
        raise RealityGateError(
            "reproduction passed; all thirty perturbation rollouts are required"
        )
    if observed_set == all_ids and not reproduction_passes:
        raise RealityGateError(
            "perturbations were executed for a reproduction-failing task"
        )

    cells: list[DiscoveryCellResult] = []
    ordered_rollouts: list[RolloutArtifact] = []
    for episode in episodes:
        raw = by_id.get(episode.episode_id)
        if raw is None:
            cells.append(
                DiscoveryCellResult(
                    episode_id=episode.episode_id,
                    executed=False,
                    artifact=None,
                    valid=None,
                    success=None,
                    validity_retry=None,
                    _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
                )
            )
        else:
            cells.append(derived_cells[episode.episode_id])
            ordered_rollouts.append(raw)
    return TaskDiscoveryAttempt(
        manifest=manifest,
        cells=tuple(cells),
        executed_rollouts=tuple(ordered_rollouts),
        _factory_token=_DISCOVERY_ATTEMPT_FACTORY_TOKEN,
    )


def _discovery_cell_from_rollout(
    task: TaskSpec,
    episode: EpisodeSpec,
    raw: RolloutArtifact,
) -> DiscoveryCellResult:
    _validate_rollout_against_episode(raw, task, episode)
    try:
        valid = raw.metadata["validity"]["valid"]
        success = raw.metadata["outcome"]["success"]
        retry = raw.metadata["validity_retry"]
    except (KeyError, TypeError) as exc:
        raise RealityGateError(
            f"{episode.episode_id}: rollout lacks validity/outcome/retry metadata"
        ) from exc
    if type(valid) is not bool or type(success) is not bool:
        raise RealityGateError(
            f"{episode.episode_id}: rollout validity and success must be booleans"
        )
    if not isinstance(retry, Mapping):
        raise RealityGateError(
            f"{episode.episode_id}: validity_retry must be a metadata mapping"
        )
    if valid:
        if set(retry) != {"performed"} or retry.get("performed") is not False:
            raise RealityGateError(
                f"{episode.episode_id}: valid reset must record no validity retry"
            )
        retry_evidence = ValidityRetryEvidence(False)
    else:
        required = {
            "performed",
            "same_reset_seed_and_condition",
            "agrees_on_invalidity",
            "agrees_on_reasons",
        }
        if not required.issubset(retry):
            raise RealityGateError(
                f"{episode.episode_id}: invalid reset lacks mandatory retry evidence"
            )
        retry_evidence = ValidityRetryEvidence(
            performed=retry["performed"],
            same_reset_seed_and_condition=retry[
                "same_reset_seed_and_condition"
            ],
            agrees_on_invalidity=retry["agrees_on_invalidity"],
            agrees_on_reasons=retry["agrees_on_reasons"],
        )
    try:
        identity = ArtifactIdentity(
            episode_id=episode.episode_id,
            metadata_sha256=raw.hashes.metadata_sha256,
            trajectory_sha256=raw.hashes.trajectory_sha256,
        )
    except (AttributeError, ValueError) as exc:
        raise RealityGateError(
            f"{episode.episode_id}: rollout content hashes are invalid"
        ) from exc
    return DiscoveryCellResult(
        episode_id=episode.episode_id,
        executed=True,
        artifact=identity,
        valid=valid,
        success=success,
        validity_retry=retry_evidence,
        _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class TaskGateDecision:
    """Immutable result and provenance for one evaluated task attempt.

    Instances are produced only by :func:`evaluate_task_attempt`.  Preventing
    direct construction keeps caller-supplied gate-result objects from being
    repackaged as protocol decisions without evaluating the manifested cells.
    """

    task: TaskSpec
    manifest_sha256: str
    policy_revision: str
    code_commit: str
    cells: tuple[DiscoveryCellResult, ...]
    reproduction: ReproductionGateResult
    dynamic_range: DynamicRangeGateResult | None
    passes: bool
    _factory_token: InitVar[object] = None
    _validation_marker: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _TASK_GATE_DECISION_FACTORY_TOKEN:
            raise RealityGateError(
                "TaskGateDecision is produced only by evaluate_task_attempt"
            )
        if not isinstance(self.task, TaskSpec):
            raise RealityGateError("task must be a validated TaskSpec")
        if not _is_sha256(self.manifest_sha256):
            raise RealityGateError("manifest_sha256 must be a lowercase SHA-256")
        if not isinstance(self.policy_revision, str) or not self.policy_revision:
            raise RealityGateError("policy_revision must be a non-empty string")
        if not isinstance(self.code_commit, str) or not self.code_commit:
            raise RealityGateError("code_commit must be a non-empty string")
        if not isinstance(self.cells, tuple) or len(self.cells) != 40:
            raise RealityGateError("a task decision must retain exactly 40 cells")
        if not isinstance(self.reproduction, ReproductionGateResult):
            raise RealityGateError("reproduction result is invalid")
        if self.dynamic_range is not None and not isinstance(
            self.dynamic_range, DynamicRangeGateResult
        ):
            raise RealityGateError("dynamic-range result is invalid")
        if type(self.passes) is not bool:
            raise RealityGateError("passes must be boolean")
        if not self.reproduction.passes and self.dynamic_range is not None:
            raise RealityGateError(
                "dynamic range cannot be evaluated after reproduction failure"
            )
        if self.reproduction.passes and self.dynamic_range is None:
            raise RealityGateError(
                "dynamic range is required after reproduction passes"
            )
        expected_pass = bool(
            self.reproduction.passes
            and self.dynamic_range is not None
            and self.dynamic_range.passes
        )
        if self.passes != expected_pass:
            raise RealityGateError("task pass flag disagrees with gate results")
        object.__setattr__(
            self,
            "_validation_marker",
            _TASK_GATE_DECISION_FACTORY_TOKEN,
        )

    @property
    def raw_artifacts(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            cell.artifact
            for cell in self.cells
            if cell.executed and cell.artifact is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": _task_metadata(self.task),
            "manifest_sha256": self.manifest_sha256,
            "policy_revision": self.policy_revision,
            "code_commit": self.code_commit,
            "cells": [_cell_metadata(cell) for cell in self.cells],
            "reproduction": _reproduction_metadata(self.reproduction),
            "dynamic_range": (
                None
                if self.dynamic_range is None
                else _dynamic_metadata(self.dynamic_range)
            ),
            "passes": self.passes,
        }


@dataclass(frozen=True)
class RealityGateReceipt:
    """Canonical, hash-ready selection receipt built by the trusted decider."""

    task_order: tuple[TaskSpec, ...]
    attempts: tuple[TaskGateDecision, ...]
    selected_task: TaskSpec | None
    schema_version: int = REALITY_GATE_SCHEMA_VERSION
    protocol: str = REALITY_GATE_PROTOCOL
    _factory_token: InitVar[object] = None
    _validation_marker: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _REALITY_GATE_RECEIPT_FACTORY_TOKEN:
            raise RealityGateError(
                "RealityGateReceipt is produced only by decide_reality_gate"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RealityGateError("unsupported Reality Gate receipt schema")
        if self.protocol != REALITY_GATE_PROTOCOL:
            raise RealityGateError("unsupported Reality Gate receipt protocol")
        if (
            not isinstance(self.task_order, tuple)
            or not self.task_order
            or any(not isinstance(task, TaskSpec) for task in self.task_order)
        ):
            raise RealityGateError("task_order must be a non-empty TaskSpec tuple")
        if _task_identities(self.task_order) != _EXPECTED_TASKS:
            raise RealityGateError(
                "receipt task order differs from the exact amended three-task protocol"
            )
        if (
            not isinstance(self.attempts, tuple)
            or not self.attempts
            or any(
                not isinstance(attempt, TaskGateDecision) for attempt in self.attempts
            )
        ):
            raise RealityGateError("attempts must retain a non-empty decision tuple")
        if any(
            attempt._validation_marker is not _TASK_GATE_DECISION_FACTORY_TOKEN
            for attempt in self.attempts
        ):
            raise RealityGateError(
                "receipt attempts must be trusted evaluated task decisions"
            )
        attempted_tasks = tuple(attempt.task for attempt in self.attempts)
        if attempted_tasks != self.task_order[: len(attempted_tasks)]:
            raise RealityGateError(
                "receipt attempts must be the exact ordered task prefix"
            )
        provenance_pairs = {
            (attempt.policy_revision, attempt.code_commit) for attempt in self.attempts
        }
        if len(provenance_pairs) != 1:
            raise RealityGateError(
                "all task attempts must use one policy revision and code commit"
            )
        passing = tuple(attempt for attempt in self.attempts if attempt.passes)
        if self.selected_task is None:
            if passing or len(self.attempts) != len(self.task_order):
                raise RealityGateError("no-task receipt is not an exhaustive failure")
        elif (
            len(passing) != 1
            or passing[0].task != self.selected_task
            or self.attempts[-1] != passing[0]
        ):
            raise RealityGateError(
                "selected task must be the sole pass and final attempted task"
            )
        object.__setattr__(
            self,
            "_validation_marker",
            _REALITY_GATE_RECEIPT_FACTORY_TOKEN,
        )

    @property
    def variable_fallback_order(self) -> tuple[str, ...]:
        return VARIABLE_FALLBACK_ORDER

    @property
    def policy_revision(self) -> str:
        return self.attempts[0].policy_revision

    @property
    def code_commit(self) -> str:
        return self.attempts[0].code_commit

    def to_dict(self) -> dict[str, Any]:
        """Return canonical-JSON-compatible receipt metadata.

        Each attempt embeds the Discovery manifest SHA-256 and every supplied
        raw metadata/trajectory identity.  No filesystem path is accepted or
        inferred at this boundary.
        """

        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "decision_basis": "point_estimates",
            "wilson_intervals_role": "descriptive_only",
            "policy_revision": self.policy_revision,
            "code_commit": self.code_commit,
            "task_order": [_task_metadata(task) for task in self.task_order],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_task": (
                None
                if self.selected_task is None
                else _task_metadata(self.selected_task)
            ),
            "all_attempts_retained": True,
            "variable_fallback_order": list(VARIABLE_FALLBACK_ORDER),
        }

    def canonical_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class OrientationState:
    """One exact theta extraction at an identified control state.

    ``None`` and IEEE non-finite real values both represent an unavailable
    extraction and count against the finite-fraction criterion.
    """

    episode_id: str
    control_step: int
    theta_rel_rad: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise RealityGateError("orientation episode_id must be non-empty")
        if (
            not isinstance(self.control_step, Integral)
            or isinstance(self.control_step, bool)
            or self.control_step < 0
        ):
            raise RealityGateError("orientation control_step must be nonnegative")
        object.__setattr__(self, "control_step", int(self.control_step))
        if self.theta_rel_rad is not None:
            if not isinstance(self.theta_rel_rad, Real) or isinstance(
                self.theta_rel_rad, bool
            ):
                raise RealityGateError("theta_rel_rad must be a real number or None")
            object.__setattr__(self, "theta_rel_rad", float(self.theta_rel_rad))


@dataclass(frozen=True)
class OrientationEligibility:
    """Immutable eligibility derived from exact raw rollout trajectories."""

    source_artifacts: tuple[ArtifactIdentity, ...]
    terminal_control_steps: tuple[tuple[ArtifactIdentity, int], ...]
    state_sha256: str
    symmetry_order: int
    extraction_unit_tests_passed: bool
    total_states: int
    finite_states: int
    finite_fraction: float
    resultant_length: float | None
    physical_circular_sd_rad: float | None
    physical_circular_sd_is_infinite: bool
    minimum_finite_fraction: float
    minimum_physical_circular_sd_degrees: float
    weighting: str
    cadence: str
    eligible: bool
    _factory_token: InitVar[object] = None
    _validation_marker: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token not in (
            _ORIENTATION_ROLLOUT_FACTORY_TOKEN,
            _ORIENTATION_ARITHMETIC_FACTORY_TOKEN,
        ):
            raise RealityGateError(
                "OrientationEligibility is produced only from RolloutArtifact evidence"
            )
        if not isinstance(self.source_artifacts, tuple) or not self.source_artifacts:
            raise RealityGateError("orientation source_artifacts cannot be empty")
        if any(
            not isinstance(item, ArtifactIdentity) for item in self.source_artifacts
        ):
            raise RealityGateError(
                "orientation sources must be ArtifactIdentity values"
            )
        source_ids = tuple(item.episode_id for item in self.source_artifacts)
        if len(set(source_ids)) != len(source_ids) or source_ids != tuple(
            sorted(source_ids)
        ):
            raise RealityGateError(
                "orientation source artifacts must have unique, sorted episode IDs"
            )
        if (
            not isinstance(self.terminal_control_steps, tuple)
            or not self.terminal_control_steps
        ):
            raise RealityGateError(
                "orientation terminal_control_steps must be a non-empty tuple"
            )
        cadence_artifacts: list[ArtifactIdentity] = []
        expected_total_states = 0
        for item in self.terminal_control_steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise RealityGateError(
                    "each terminal-step entry must pair an ArtifactIdentity with an int"
                )
            artifact, terminal_step = item
            if not isinstance(artifact, ArtifactIdentity):
                raise RealityGateError(
                    "terminal-step entries must contain ArtifactIdentity values"
                )
            if type(terminal_step) is not int or terminal_step < 0:
                raise RealityGateError(
                    "terminal_control_step values must be nonnegative integers"
                )
            cadence_artifacts.append(artifact)
            expected_total_states += terminal_step + 1
        if tuple(cadence_artifacts) != self.source_artifacts:
            raise RealityGateError(
                "terminal-step evidence must exactly map the sorted source artifacts"
            )
        if not _is_sha256(self.state_sha256):
            raise RealityGateError("state_sha256 must be a lowercase SHA-256")
        if (
            type(self.symmetry_order) is not int
            or self.symmetry_order < 1
            or type(self.extraction_unit_tests_passed) is not bool
            or type(self.total_states) is not int
            or self.total_states < 1
            or self.total_states != expected_total_states
            or type(self.finite_states) is not int
            or not 0 <= self.finite_states <= self.total_states
        ):
            raise RealityGateError("orientation eligibility counts are invalid")
        expected_fraction = self.finite_states / self.total_states
        if (
            not isinstance(self.finite_fraction, float)
            or not math.isfinite(self.finite_fraction)
            or not math.isclose(
                self.finite_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RealityGateError("orientation finite fraction disagrees with counts")
        if self.minimum_finite_fraction != ORIENTATION_MINIMUM_FINITE_FRACTION:
            raise RealityGateError("orientation finite threshold differs from protocol")
        if (
            self.minimum_physical_circular_sd_degrees
            != ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES
        ):
            raise RealityGateError(
                "orientation dispersion threshold differs from protocol"
            )
        if (
            self.weighting != ORIENTATION_WEIGHTING
            or self.cadence != ORIENTATION_CADENCE
        ):
            raise RealityGateError(
                "orientation cadence or weighting differs from protocol"
            )
        if type(self.physical_circular_sd_is_infinite) is not bool:
            raise RealityGateError("orientation infinite-SD flag must be boolean")
        if self.resultant_length is not None and (
            not isinstance(self.resultant_length, float)
            or not math.isfinite(self.resultant_length)
            or not 0.0 <= self.resultant_length <= 1.0
        ):
            raise RealityGateError("orientation resultant length is invalid")
        if self.physical_circular_sd_rad is not None and (
            not isinstance(self.physical_circular_sd_rad, float)
            or not math.isfinite(self.physical_circular_sd_rad)
            or self.physical_circular_sd_rad < 0.0
        ):
            raise RealityGateError("orientation circular SD is invalid")
        if self.finite_states == 0:
            if (
                self.resultant_length is not None
                or self.physical_circular_sd_rad is not None
                or self.physical_circular_sd_is_infinite
            ):
                raise RealityGateError(
                    "all-nonfinite orientation evidence cannot report dispersion"
                )
        elif self.physical_circular_sd_is_infinite:
            if (
                self.resultant_length != 0.0
                or self.physical_circular_sd_rad is not None
            ):
                raise RealityGateError(
                    "infinite orientation SD requires zero resultant and no finite SD"
                )
        elif self.resultant_length is None or self.physical_circular_sd_rad is None:
            raise RealityGateError(
                "finite orientation evidence must report resultant and circular SD"
            )
        threshold_rad = math.radians(ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES)
        dispersion_passes = self.physical_circular_sd_is_infinite or bool(
            self.physical_circular_sd_rad is not None
            and (
                self.physical_circular_sd_rad >= threshold_rad
                or math.isclose(
                    self.physical_circular_sd_rad,
                    threshold_rad,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        expected_eligible = bool(
            self.extraction_unit_tests_passed
            and self.finite_fraction >= ORIENTATION_MINIMUM_FINITE_FRACTION
            and dispersion_passes
        )
        if type(self.eligible) is not bool or self.eligible != expected_eligible:
            raise RealityGateError(
                "orientation eligible flag disagrees with preregistered criteria"
            )
        object.__setattr__(
            self,
            "_validation_marker",
            _factory_token,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_artifacts": [
                artifact.to_dict() for artifact in self.source_artifacts
            ],
            "terminal_control_steps": [
                {
                    "artifact": artifact.to_dict(),
                    "terminal_control_step": terminal_step,
                }
                for artifact, terminal_step in self.terminal_control_steps
            ],
            "state_sha256": self.state_sha256,
            "symmetry_order": self.symmetry_order,
            "extraction_unit_tests_passed": self.extraction_unit_tests_passed,
            "total_states": self.total_states,
            "finite_states": self.finite_states,
            "finite_fraction": self.finite_fraction,
            "resultant_length": self.resultant_length,
            "physical_circular_sd_rad": self.physical_circular_sd_rad,
            "physical_circular_sd_is_infinite": (self.physical_circular_sd_is_infinite),
            "minimum_finite_fraction": self.minimum_finite_fraction,
            "minimum_physical_circular_sd_degrees": (
                self.minimum_physical_circular_sd_degrees
            ),
            "weighting": self.weighting,
            "cadence": self.cadence,
            "eligible": self.eligible,
        }

    def canonical_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class RealityGateLockReceipt:
    """Final, hash-ready task/variable selection evidence for the lock file."""

    task_gate_receipt: RealityGateReceipt
    selected_task_artifacts: tuple[ArtifactIdentity, ...]
    orientation_eligibility: OrientationEligibility
    selected_variable: str = "orientation"
    schema_version: int = REALITY_GATE_LOCK_SCHEMA_VERSION
    protocol: str = REALITY_GATE_LOCK_PROTOCOL

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise RealityGateError("unsupported Reality Gate lock receipt schema")
        if self.protocol != REALITY_GATE_LOCK_PROTOCOL:
            raise RealityGateError("unsupported Reality Gate lock receipt protocol")
        if not _is_trusted_reality_gate_receipt(self.task_gate_receipt):
            raise RealityGateError(
                "task_gate_receipt must be produced by decide_reality_gate"
            )
        if self.task_gate_receipt.selected_task is None:
            raise RealityGateError(
                "cannot finalize the Reality Gate without a selected task"
            )
        selected_decision = self.task_gate_receipt.attempts[-1]
        if (
            selected_decision.task != self.task_gate_receipt.selected_task
            or not selected_decision.passes
        ):
            raise RealityGateError("selected task decision is not the final task pass")
        expected_artifacts = _selected_task_artifacts(selected_decision)
        if self.selected_task_artifacts != expected_artifacts:
            raise RealityGateError(
                "lock receipt must retain the exact selected-task 40-artifact set"
            )
        if not isinstance(self.orientation_eligibility, OrientationEligibility):
            raise RealityGateError("orientation_eligibility is invalid")
        if (
            self.orientation_eligibility._validation_marker
            is not _ORIENTATION_ROLLOUT_FACTORY_TOKEN
        ):
            raise RealityGateError(
                "orientation eligibility must be derived from RolloutArtifact evidence"
            )
        orientation_by_id = {
            item.episode_id: item
            for item in self.orientation_eligibility.source_artifacts
        }
        selected_by_id = {
            item.episode_id: item for item in self.selected_task_artifacts
        }
        if orientation_by_id != selected_by_id:
            raise RealityGateError(
                "orientation eligibility sources must equal the exact selected-task "
                "40-artifact set"
            )
        if (
            self.orientation_eligibility.symmetry_order
            != self.task_gate_receipt.selected_task.planar_symmetry_order
        ):
            raise RealityGateError(
                "orientation symmetry order differs from the selected task"
            )
        if not self.orientation_eligibility.eligible:
            raise RealityGateError(
                "orientation is ineligible; planar-position/contact eligibility was "
                "not prospectively operationalized and requires a protocol amendment"
            )
        if self.selected_variable != "orientation":
            raise RealityGateError(
                "only eligible orientation can be selected without a protocol amendment"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "task_gate_receipt_sha256": self.task_gate_receipt.sha256,
            "task_gate_receipt": self.task_gate_receipt.to_dict(),
            "selected_task": _task_metadata(self.task_gate_receipt.selected_task),
            "selected_task_artifacts": [
                artifact.to_dict() for artifact in self.selected_task_artifacts
            ],
            "orientation_eligibility_sha256": self.orientation_eligibility.sha256,
            "orientation_eligibility": self.orientation_eligibility.to_dict(),
            "selected_variable": self.selected_variable,
            "fallback_status": "not_invoked_orientation_eligible",
        }

    def canonical_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def to_guard_payload(
        self, failure_event_freeze: FailureEventFreezeManifest
    ) -> dict[str, Any]:
        """Build the exact tracked payload consumed by the Calibration guard.

        The internal receipt uses ``orientation`` as its variable label because
        that is the Reality-Gate decision vocabulary.  The protected-split guard
        uses the preregistered feature name ``theta_rel``.  This adapter performs
        that explicit translation and embeds every content-addressed Discovery
        input needed to audit the lock: the complete task-gate receipt,
        orientation eligibility, and deterministic failure-event freeze.
        """

        if not isinstance(failure_event_freeze, FailureEventFreezeManifest):
            raise RealityGateError(
                "failure_event_freeze must be a validated freeze manifest"
            )
        selected = self.task_gate_receipt.selected_task
        if selected is None:
            raise RealityGateError("cannot create a guard payload without a task")
        expected_task = {
            "suite": selected.suite,
            "task_id": selected.task_id,
            "task_rank": selected.rank,
            "language": selected.language,
            "primary_object": selected.primary_object,
            "planar_symmetry_order": selected.planar_symmetry_order,
        }
        if failure_event_freeze.task.to_dict() != expected_task:
            raise RealityGateError(
                "failure-event freeze task differs from the selected Reality-Gate task"
            )
        selected_task = _task_metadata(selected)
        selected_artifacts = [
            artifact.to_dict() for artifact in self.selected_task_artifacts
        ]
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "policy_revision": self.task_gate_receipt.policy_revision,
            "code_commit": self.task_gate_receipt.code_commit,
            "selected_task": selected_task,
            "selected_variable": {
                "name": "theta_rel",
                "symmetry_order": selected.planar_symmetry_order,
            },
            "reality_gate_lock_receipt_sha256": self.sha256,
            "reality_gate_lock_receipt": self.to_dict(),
            "reality_gate_receipt_sha256": self.task_gate_receipt.sha256,
            "reality_gate_receipt": self.task_gate_receipt.to_dict(),
            "orientation_eligibility_sha256": self.orientation_eligibility.sha256,
            "orientation_eligibility": self.orientation_eligibility.to_dict(),
            "selected_task_artifacts": selected_artifacts,
            "failure_event_freeze_sha256": failure_event_freeze.sha256,
            "failure_event_freeze": json.loads(
                failure_event_freeze.canonical_json().decode("utf-8")
            ),
        }


def decide_reality_gate(
    protocol: ProtocolConfig,
    attempts: tuple[TaskDiscoveryAttempt, ...],
) -> RealityGateReceipt:
    """Select the first complete task pass, or prove all configured tasks failed.

    Attempts must form an exact prefix of the configured order.  A final receipt
    is returned only after the first pass or after every configured task failed;
    an unresolved partial prefix fails closed.
    """

    _validate_protocol(protocol)
    if (
        not isinstance(attempts, tuple)
        or not attempts
        or any(not isinstance(attempt, TaskDiscoveryAttempt) for attempt in attempts)
    ):
        raise RealityGateError(
            "attempts must be a non-empty tuple of TaskDiscoveryAttempt values"
        )
    if any(
        attempt._validation_marker is not _DISCOVERY_ATTEMPT_FACTORY_TOKEN
        for attempt in attempts
    ):
        raise RealityGateError(
            "attempts must be produced from validated RolloutArtifact evidence"
        )
    tasks = protocol.task_order.tasks
    if len(attempts) > len(tasks):
        raise RealityGateError("more task attempts were supplied than preregistered")
    revision_pairs = tuple(
        _manifest_revision_pair(attempt.manifest) for attempt in attempts
    )
    if len(set(revision_pairs)) != 1:
        raise RealityGateError(
            "all task attempts must use the identical policy revision and code commit"
        )

    decisions: list[TaskGateDecision] = []
    selected: TaskSpec | None = None
    for index, attempt in enumerate(attempts):
        expected_task = tasks[index]
        if attempt.manifest.task != expected_task:
            raise RealityGateError(
                "task attempts must be the exact ordered prefix of the shortlist"
            )
        if selected is not None:
            raise RealityGateError("a later task was attempted after the first pass")
        decision = evaluate_task_attempt(protocol, attempt)
        decisions.append(decision)
        if decision.passes:
            selected = decision.task

    if selected is not None and len(decisions) != len(attempts):  # pragma: no cover
        raise RealityGateError("internal attempt accounting error")
    if selected is not None and not decisions[-1].passes:
        raise RealityGateError("a later task was attempted after the first pass")
    if selected is None and len(attempts) != len(tasks):
        raise RealityGateError(
            "task selection is unresolved: supply the next ordered task attempt"
        )

    return RealityGateReceipt(
        task_order=tasks,
        attempts=tuple(decisions),
        selected_task=selected,
        _factory_token=_REALITY_GATE_RECEIPT_FACTORY_TOKEN,
    )


def evaluate_task_attempt(
    protocol: ProtocolConfig,
    attempt: TaskDiscoveryAttempt,
) -> TaskGateDecision:
    """Evaluate one task, always deciding reproduction before perturbations."""

    _validate_protocol(protocol)
    if not isinstance(attempt, TaskDiscoveryAttempt):
        raise RealityGateError("attempt must be a validated TaskDiscoveryAttempt")
    if attempt._validation_marker is not _DISCOVERY_ATTEMPT_FACTORY_TOKEN:
        raise RealityGateError(
            "attempt must be produced from validated RolloutArtifact evidence"
        )
    if attempt.manifest.task not in protocol.task_order.tasks:
        raise RealityGateError("attempt task is absent from the task shortlist")
    expected_episodes = _validated_discovery_manifest(protocol, attempt.manifest)
    cells_by_id = _validate_exact_cell_coverage(expected_episodes, attempt.cells)
    ordered_cells = tuple(
        cells_by_id[episode.episode_id] for episode in expected_episodes
    )

    iid_cells = tuple(
        cells_by_id[episode.episode_id]
        for episode in expected_episodes
        if episode.condition_family == "iid"
    )
    perturbation_cells = tuple(
        cells_by_id[episode.episode_id]
        for episode in expected_episodes
        if episode.condition_family == "object_yaw"
    )
    if any(not cell.executed for cell in iid_cells):
        raise RealityGateError("all ten IID reproduction cells must be executed")

    gates = protocol.task_order.gates
    reproduction_successes = sum(bool(cell.success) for cell in iid_cells)
    reproduction = reproduction_gate_decision(
        reproduction_successes,
        len(iid_cells),
        minimum_successes=gates.baseline_min_successes,
    )
    if not reproduction.passes:
        if any(cell.executed for cell in perturbation_cells):
            raise RealityGateError(
                "perturbations were executed for a reproduction-failing task"
            )
        dynamic: DynamicRangeGateResult | None = None
    else:
        if any(not cell.executed for cell in perturbation_cells):
            raise RealityGateError(
                "all thirty perturbations are required after reproduction passes"
            )
        valid_count = sum(bool(cell.valid) for cell in perturbation_cells)
        failures = sum(
            bool(cell.valid) and not bool(cell.success) for cell in perturbation_cells
        )
        if valid_count == 0:
            raise RealityGateError(
                "dynamic-range failure rate is undefined with zero valid episodes"
            )
        try:
            dynamic = dynamic_range_gate_decision(
                valid_count,
                len(perturbation_cells),
                failures,
                minimum_valid_fraction=gates.min_valid_fraction,
                minimum_failure_rate=gates.min_failure_rate,
                maximum_failure_rate=gates.max_failure_rate,
            )
        except EvaluationError as exc:  # defensive normalization of public errors
            raise RealityGateError(f"invalid dynamic-range denominator: {exc}") from exc

    return TaskGateDecision(
        task=attempt.manifest.task,
        manifest_sha256=attempt.manifest.sha256,
        policy_revision=attempt.manifest.episodes[0].policy_revision,
        code_commit=attempt.manifest.episodes[0].code_commit,
        cells=ordered_cells,
        reproduction=reproduction,
        dynamic_range=dynamic,
        passes=bool(reproduction.passes and dynamic is not None and dynamic.passes),
        _factory_token=_TASK_GATE_DECISION_FACTORY_TOKEN,
    )


def finalize_reality_gate(
    task_gate_receipt: RealityGateReceipt,
    orientation_eligibility: OrientationEligibility,
) -> RealityGateLockReceipt:
    """Finalize the selected task and eligible primary variable, or fail closed.

    The two fallback variables have a frozen order but no prospective eligibility
    statistic.  Consequently this function cannot select either one without a
    recorded protocol amendment.
    """

    if not _is_trusted_reality_gate_receipt(task_gate_receipt):
        raise RealityGateError(
            "task_gate_receipt must be produced by decide_reality_gate"
        )
    if task_gate_receipt.selected_task is None:
        raise RealityGateError(
            "cannot finalize the Reality Gate without a selected task"
        )
    if not isinstance(orientation_eligibility, OrientationEligibility):
        raise RealityGateError(
            "orientation_eligibility must be an OrientationEligibility result"
        )
    selected_artifacts = _selected_task_artifacts(task_gate_receipt.attempts[-1])
    return RealityGateLockReceipt(
        task_gate_receipt=task_gate_receipt,
        selected_task_artifacts=selected_artifacts,
        orientation_eligibility=orientation_eligibility,
    )


def _evaluate_orientation_eligibility(
    source_artifacts: tuple[ArtifactIdentity, ...],
    states: tuple[OrientationState, ...],
    *,
    terminal_control_steps: tuple[tuple[ArtifactIdentity, int], ...],
    symmetry_order: int,
    extraction_unit_tests_passed: bool,
    weighting: str,
    cadence: str,
    _factory_token: object = _ORIENTATION_ARITHMETIC_FACTORY_TOKEN,
) -> OrientationEligibility:
    """Apply the fixed orientation-variable eligibility rule.

    Custom or implicit weights are deliberately unsupported.  Callers must name
    the fixed equal-state weighting and full-control-step cadence explicitly, and
    ``terminal_control_steps`` binds every exact source artifact identity to its
    content-derived terminal step.  The supplied state keys must equal the full
    ``0..terminal`` key set for every source episode; the last supplied state is
    never treated as evidence of where a trajectory ended.
    """

    if weighting != ORIENTATION_WEIGHTING:
        raise RealityGateError(
            f"weighting must be explicitly fixed to {ORIENTATION_WEIGHTING!r}"
        )
    if cadence != ORIENTATION_CADENCE:
        raise RealityGateError(
            f"cadence must be explicitly fixed to {ORIENTATION_CADENCE!r}"
        )
    if (
        not isinstance(symmetry_order, Integral)
        or isinstance(symmetry_order, bool)
        or symmetry_order < 1
    ):
        raise RealityGateError("symmetry_order must be a positive integer")
    symmetry_order = int(symmetry_order)
    if type(extraction_unit_tests_passed) is not bool:
        raise RealityGateError("extraction_unit_tests_passed must be boolean")
    if (
        not isinstance(source_artifacts, tuple)
        or not source_artifacts
        or any(
            not isinstance(artifact, ArtifactIdentity) for artifact in source_artifacts
        )
    ):
        raise RealityGateError(
            "source_artifacts must be a non-empty ArtifactIdentity tuple"
        )
    source_by_id = {artifact.episode_id: artifact for artifact in source_artifacts}
    if len(source_by_id) != len(source_artifacts):
        raise RealityGateError("orientation source episode IDs must be unique")
    if (
        not isinstance(terminal_control_steps, tuple)
        or not terminal_control_steps
    ):
        raise RealityGateError(
            "terminal_control_steps must be a non-empty tuple of artifact-step pairs"
        )
    terminal_by_id: dict[str, tuple[ArtifactIdentity, int]] = {}
    for item in terminal_control_steps:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RealityGateError(
                "each terminal-step entry must pair an ArtifactIdentity with an int"
            )
        artifact, terminal_step = item
        if not isinstance(artifact, ArtifactIdentity):
            raise RealityGateError(
                "terminal-step entries must contain ArtifactIdentity values"
            )
        if type(terminal_step) is not int or terminal_step < 0:
            raise RealityGateError(
                "terminal_control_step values must be nonnegative integers"
            )
        if artifact.episode_id in terminal_by_id:
            raise RealityGateError(
                "terminal-step evidence episode IDs must be unique"
            )
        terminal_by_id[artifact.episode_id] = (artifact, terminal_step)
    if set(terminal_by_id) != set(source_by_id) or any(
        terminal_by_id[episode_id][0] != artifact
        for episode_id, artifact in source_by_id.items()
        if episode_id in terminal_by_id
    ):
        raise RealityGateError(
            "terminal-step evidence must exactly map every source artifact identity"
        )
    if (
        not isinstance(states, tuple)
        or not states
        or any(not isinstance(state, OrientationState) for state in states)
    ):
        raise RealityGateError("states must be a non-empty OrientationState tuple")

    state_by_key: dict[tuple[str, int], OrientationState] = {}
    for state in states:
        if state.episode_id not in source_by_id:
            raise RealityGateError("orientation state has no supplied source artifact")
        key = (state.episode_id, state.control_step)
        if key in state_by_key:
            raise RealityGateError("orientation state keys must be unique")
        state_by_key[key] = state
    expected_state_keys = {
        (episode_id, step)
        for episode_id, (_, terminal_step) in terminal_by_id.items()
        for step in range(terminal_step + 1)
    }
    if set(state_by_key) != expected_state_keys:
        missing = sorted(expected_state_keys - set(state_by_key))
        extra = sorted(set(state_by_key) - expected_state_keys)
        raise RealityGateError(
            "orientation states must exactly cover the complete cadence 0..terminal step for every "
            f"source artifact: missing={missing}, extra={extra}"
        )

    ordered_states = tuple(state_by_key[key] for key in sorted(state_by_key))
    finite = tuple(
        float(state.theta_rel_rad)
        for state in ordered_states
        if state.theta_rel_rad is not None and math.isfinite(state.theta_rel_rad)
    )
    finite_fraction = len(finite) / len(ordered_states)
    resultant: float | None
    circular_sd: float | None
    circular_sd_infinite: bool
    if not finite:
        resultant = None
        circular_sd = None
        circular_sd_infinite = False
    else:
        cosine_mean = math.fsum(
            math.cos(symmetry_order * theta) for theta in finite
        ) / len(finite)
        sine_mean = math.fsum(
            math.sin(symmetry_order * theta) for theta in finite
        ) / len(finite)
        resultant = min(1.0, max(0.0, math.hypot(cosine_mean, sine_mean)))
        if resultant == 0.0:
            circular_sd = None
            circular_sd_infinite = True
        else:
            circular_sd = math.sqrt(max(0.0, -2.0 * math.log(resultant))) / (
                symmetry_order
            )
            circular_sd_infinite = False

    threshold_rad = math.radians(ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES)
    dispersion_passes = circular_sd_infinite or bool(
        circular_sd is not None
        and (
            circular_sd >= threshold_rad
            or math.isclose(circular_sd, threshold_rad, rel_tol=1e-12, abs_tol=1e-12)
        )
    )
    eligible = bool(
        extraction_unit_tests_passed
        and finite_fraction >= ORIENTATION_MINIMUM_FINITE_FRACTION
        and dispersion_passes
    )
    ordered_artifacts = tuple(source_by_id[key] for key in sorted(source_by_id))
    ordered_terminal_steps = tuple(
        terminal_by_id[key] for key in sorted(terminal_by_id)
    )
    return OrientationEligibility(
        source_artifacts=ordered_artifacts,
        terminal_control_steps=ordered_terminal_steps,
        state_sha256=_orientation_state_sha256(
            ordered_states,
            ordered_terminal_steps,
        ),
        symmetry_order=symmetry_order,
        extraction_unit_tests_passed=extraction_unit_tests_passed,
        total_states=len(ordered_states),
        finite_states=len(finite),
        finite_fraction=finite_fraction,
        resultant_length=resultant,
        physical_circular_sd_rad=circular_sd,
        physical_circular_sd_is_infinite=circular_sd_infinite,
        minimum_finite_fraction=ORIENTATION_MINIMUM_FINITE_FRACTION,
        minimum_physical_circular_sd_degrees=(ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES),
        weighting=weighting,
        cadence=cadence,
        eligible=eligible,
        _factory_token=_factory_token,
    )


def evaluate_orientation_eligibility(
    source_artifacts: tuple[ArtifactIdentity, ...],
    states: tuple[OrientationState, ...],
    *,
    terminal_control_steps: tuple[tuple[ArtifactIdentity, int], ...],
    symmetry_order: int,
    extraction_unit_tests_passed: bool,
    weighting: str,
    cadence: str,
) -> OrientationEligibility:
    """Evaluate caller-supplied orientation states at an arithmetic boundary.

    This helper is intentionally not sufficient to authorize a Reality-Gate
    lock: caller-supplied states are marked as arithmetic evidence.  Use
    :func:`orientation_eligibility_from_rollouts` for the only lock-authorizing
    constructor, which extracts the states directly from validated raw rollout
    arrays.
    """

    return _evaluate_orientation_eligibility(
        source_artifacts,
        states,
        terminal_control_steps=terminal_control_steps,
        symmetry_order=symmetry_order,
        extraction_unit_tests_passed=extraction_unit_tests_passed,
        weighting=weighting,
        cadence=cadence,
        _factory_token=_ORIENTATION_ARITHMETIC_FACTORY_TOKEN,
    )


def orientation_eligibility_from_rollouts(
    rollouts: tuple[RolloutArtifact, ...],
    *,
    symmetry_order: int,
    weighting: str,
    cadence: str,
) -> OrientationEligibility:
    """Extract and evaluate orientation eligibility from validated raw rollouts.

    The extraction contract is fixed to the stored EEF and primary-object yaw
    quaternions, with one state per stored frame (including the terminal frame).
    No caller-provided theta values or terminal lengths are accepted.  The
    resulting marker is the one accepted by :func:`finalize_reality_gate`.
    """

    if (
        not isinstance(rollouts, tuple)
        or not rollouts
        or any(not isinstance(rollout, RolloutArtifact) for rollout in rollouts)
    ):
        raise RealityGateError(
            "rollouts must be a non-empty tuple of validated RolloutArtifact values"
        )
    episode_ids = tuple(rollout.episode_id for rollout in rollouts)
    if len(set(episode_ids)) != len(episode_ids):
        raise RealityGateError("orientation rollout episode IDs must be unique")
    ordered_rollouts = tuple(sorted(rollouts, key=lambda rollout: rollout.episode_id))
    states: list[OrientationState] = []
    terminal_steps: list[tuple[ArtifactIdentity, int]] = []
    source_artifacts: list[ArtifactIdentity] = []
    for rollout in ordered_rollouts:
        try:
            eef = rollout.arrays["frame_eef_quaternion_xyzw"]
            primary = rollout.arrays["frame_primary_object_quaternion_wxyz"]
            control_steps = rollout.arrays["frame_control_step"]
        except (KeyError, TypeError) as exc:
            raise RealityGateError(
                f"{rollout.episode_id}: orientation quaternion arrays are missing"
            ) from exc
        if (
            not isinstance(eef, np.ndarray)
            or not isinstance(primary, np.ndarray)
            or not isinstance(control_steps, np.ndarray)
            or eef.ndim != 2
            or primary.ndim != 2
            or eef.shape != primary.shape
            or eef.shape[1] != 4
            or control_steps.ndim != 1
            or control_steps.shape[0] != eef.shape[0]
            or control_steps.dtype.kind not in "iu"
            or not np.array_equal(
                control_steps,
                np.arange(control_steps.size, dtype=control_steps.dtype),
            )
            or control_steps.size == 0
        ):
            raise RealityGateError(
                f"{rollout.episode_id}: orientation frame cadence is invalid"
            )
        identity = ArtifactIdentity(
            episode_id=rollout.episode_id,
            metadata_sha256=rollout.hashes.metadata_sha256,
            trajectory_sha256=rollout.hashes.trajectory_sha256,
        )
        source_artifacts.append(identity)
        terminal_step = int(control_steps[-1])
        terminal_steps.append((identity, terminal_step))
        for step, (eef_quaternion, primary_quaternion) in enumerate(
            zip(eef, primary, strict=True)
        ):
            eef_yaw = _yaw_from_xyzw(eef_quaternion)
            primary_yaw = _yaw_from_wxyz(primary_quaternion)
            states.append(
                OrientationState(
                    rollout.episode_id,
                    step,
                    _wrap_angle(eef_yaw - primary_yaw),
                )
            )

    unit_test_cases = (
        ((0.0, 0.0, 0.0, 1.0), 0.0),
        (
            (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            math.pi / 2.0,
        ),
        (
            (0.0, 0.0, math.sin(-math.pi / 4.0), math.cos(math.pi / 4.0)),
            -math.pi / 2.0,
        ),
        ((0.0, 0.0, 1.0, 0.0), math.pi),
    )
    extraction_unit_tests_passed = all(
        abs(_wrap_angle(_yaw_from_xyzw(quaternion) - expected)) < 1e-12
        for quaternion, expected in unit_test_cases
    )
    if not extraction_unit_tests_passed:  # pragma: no cover - defensive
        raise RealityGateError("orientation quaternion extraction unit tests failed")
    return _evaluate_orientation_eligibility(
        tuple(source_artifacts),
        tuple(states),
        terminal_control_steps=tuple(terminal_steps),
        symmetry_order=symmetry_order,
        extraction_unit_tests_passed=True,
        weighting=weighting,
        cadence=cadence,
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )


def variable_fallback_order() -> tuple[str, ...]:
    """Return the fixed order without making a performance-dependent selection."""

    return VARIABLE_FALLBACK_ORDER


def _mapping_exact(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealityGateError(f"{where} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise RealityGateError(f"{where} keys differ: {'; '.join(detail)}")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealityGateError(f"{where} must be a non-empty string")
    return value


def _git_sha(value: Any, where: str) -> str:
    text = _nonempty_string(value, where)
    if (
        len(text) != 40
        or any(character not in "0123456789abcdef" for character in text)
        or len(set(text)) == 1
    ):
        raise RealityGateError(f"{where} must be a lowercase 40-character Git SHA")
    return text


def _sha256_string(value: Any, where: str) -> str:
    if not _is_sha256(value) or len(set(value)) == 1:
        raise RealityGateError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise RealityGateError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise RealityGateError(f"{where} must be a nonnegative integer")
    return value


def _bool_value(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise RealityGateError(f"{where} must be boolean")
    return value


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RealityGateError(f"{where} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RealityGateError(f"{where} must be finite")
    return converted


def _task_from_metadata(value: Any, where: str) -> TaskSpec:
    root = _mapping_exact(
        value,
        {
            "rank",
            "suite",
            "task_id",
            "language",
            "primary_object",
            "planar_symmetry_order",
        },
        where,
    )
    rank = _positive_int(root["rank"], f"{where}.rank")
    task_id = _nonnegative_int(root["task_id"], f"{where}.task_id")
    symmetry_order = _positive_int(
        root["planar_symmetry_order"], f"{where}.planar_symmetry_order"
    )
    return TaskSpec(
        rank=rank,
        suite=_nonempty_string(root["suite"], f"{where}.suite"),
        task_id=task_id,
        language=_nonempty_string(root["language"], f"{where}.language"),
        primary_object=_nonempty_string(
            root["primary_object"], f"{where}.primary_object"
        ),
        planar_symmetry_order=symmetry_order,
    )


def _artifact_identity_from_metadata(value: Any, where: str) -> ArtifactIdentity:
    root = _mapping_exact(
        value,
        {"episode_id", "metadata_sha256", "trajectory_sha256"},
        where,
    )
    return ArtifactIdentity(
        episode_id=_nonempty_string(root["episode_id"], f"{where}.episode_id"),
        metadata_sha256=_sha256_string(
            root["metadata_sha256"], f"{where}.metadata_sha256"
        ),
        trajectory_sha256=_sha256_string(
            root["trajectory_sha256"], f"{where}.trajectory_sha256"
        ),
    )


def _retry_from_metadata(
    value: Any, *, valid: bool, where: str
) -> ValidityRetryEvidence:
    if not isinstance(value, Mapping):
        raise RealityGateError(f"{where} must be an object")
    if valid:
        _mapping_exact(value, {"performed"}, where)
        if value["performed"] is not False:
            raise RealityGateError(f"{where}.performed must be false for a valid reset")
        return ValidityRetryEvidence(False)
    root = _mapping_exact(
        value,
        {
            "performed",
            "same_reset_seed_and_condition",
            "agrees_on_invalidity",
            "agrees_on_reasons",
        },
        where,
    )
    return ValidityRetryEvidence(
        performed=_bool_value(root["performed"], f"{where}.performed"),
        same_reset_seed_and_condition=_bool_value(
            root["same_reset_seed_and_condition"],
            f"{where}.same_reset_seed_and_condition",
        ),
        agrees_on_invalidity=_bool_value(
            root["agrees_on_invalidity"], f"{where}.agrees_on_invalidity"
        ),
        agrees_on_reasons=_bool_value(
            root["agrees_on_reasons"], f"{where}.agrees_on_reasons"
        ),
    )


def _cell_from_metadata(value: Any, where: str) -> DiscoveryCellResult:
    root = _mapping_exact(
        value,
        {"episode_id", "executed", "artifact", "valid", "success", "validity_retry"},
        where,
    )
    episode_id = _nonempty_string(root["episode_id"], f"{where}.episode_id")
    executed = _bool_value(root["executed"], f"{where}.executed")
    if not executed:
        if any(root[key] is not None for key in ("artifact", "valid", "success", "validity_retry")):
            raise RealityGateError(f"{where}: unexecuted cell carries evidence")
        return DiscoveryCellResult(
            episode_id=episode_id,
            executed=False,
            artifact=None,
            valid=None,
            success=None,
            validity_retry=None,
            _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
        )
    if root["artifact"] is None:
        raise RealityGateError(f"{where}.artifact is required for an executed cell")
    artifact = _artifact_identity_from_metadata(root["artifact"], f"{where}.artifact")
    valid = _bool_value(root["valid"], f"{where}.valid")
    success = _bool_value(root["success"], f"{where}.success")
    retry = _retry_from_metadata(
        root["validity_retry"], valid=valid, where=f"{where}.validity_retry"
    )
    return DiscoveryCellResult(
        episode_id=episode_id,
        executed=True,
        artifact=artifact,
        valid=valid,
        success=success,
        validity_retry=retry,
        _factory_token=_DISCOVERY_CELL_FACTORY_TOKEN,
    )


def _rate_from_metadata(value: Any, where: str) -> RateInterval:
    root = _mapping_exact(
        value,
        {"rate", "lower", "upper", "confidence", "successes", "total"},
        where,
    )
    successes = _nonnegative_int(root["successes"], f"{where}.successes")
    total = _positive_int(root["total"], f"{where}.total")
    if successes > total:
        raise RealityGateError(f"{where}.successes cannot exceed total")
    rate = _finite_float(root["rate"], f"{where}.rate")
    lower = _finite_float(root["lower"], f"{where}.lower")
    upper = _finite_float(root["upper"], f"{where}.upper")
    confidence = _finite_float(root["confidence"], f"{where}.confidence")
    if not 0.0 <= lower <= rate <= upper <= 1.0:
        raise RealityGateError(f"{where} interval ordering is invalid")
    if not 0.0 < confidence < 1.0:
        raise RealityGateError(f"{where}.confidence must lie in (0, 1)")
    if not math.isclose(rate, successes / total, rel_tol=0.0, abs_tol=1e-15):
        raise RealityGateError(f"{where}.rate disagrees with successes/total")
    return RateInterval(
        rate=rate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        successes=successes,
        total=total,
    )


def _dynamic_from_metadata(value: Any, where: str) -> DynamicRangeGateResult:
    root = _mapping_exact(
        value,
        {
            "validity_interval",
            "failure_interval",
            "minimum_valid_fraction",
            "failure_rate_bounds",
            "passes_validity",
            "passes_failure_range",
            "passes",
        },
        where,
    )
    bounds = root["failure_rate_bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise RealityGateError(f"{where}.failure_rate_bounds must contain two values")
    minimum_valid_fraction = _finite_float(
        root["minimum_valid_fraction"], f"{where}.minimum_valid_fraction"
    )
    failure_bounds = (
        _finite_float(bounds[0], f"{where}.failure_rate_bounds[0]"),
        _finite_float(bounds[1], f"{where}.failure_rate_bounds[1]"),
    )
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise RealityGateError(f"{where}.minimum_valid_fraction is outside [0, 1]")
    if not 0.0 <= failure_bounds[0] <= failure_bounds[1] <= 1.0:
        raise RealityGateError(f"{where}.failure_rate_bounds are unordered")
    return DynamicRangeGateResult(
        validity_interval=_rate_from_metadata(
            root["validity_interval"], f"{where}.validity_interval"
        ),
        failure_interval=_rate_from_metadata(
            root["failure_interval"], f"{where}.failure_interval"
        ),
        minimum_valid_fraction=minimum_valid_fraction,
        failure_rate_bounds=failure_bounds,
        passes_validity=_bool_value(
            root["passes_validity"], f"{where}.passes_validity"
        ),
        passes_failure_range=_bool_value(
            root["passes_failure_range"], f"{where}.passes_failure_range"
        ),
        passes=_bool_value(root["passes"], f"{where}.passes"),
    )


def _task_gate_decision_from_metadata(
    value: Any,
    protocol: ProtocolConfig,
    *,
    expected_task: TaskSpec | None,
) -> TaskGateDecision:
    root = _mapping_exact(
        value,
        {
            "task",
            "manifest_sha256",
            "policy_revision",
            "code_commit",
            "cells",
            "reproduction",
            "dynamic_range",
            "passes",
        },
        "reality_gate_receipt.attempt",
    )
    task = _task_from_metadata(root["task"], "attempt.task")
    if expected_task is not None and task != expected_task:
        raise RealityGateError("Reality-Gate attempts are not the exact task prefix")
    policy_revision = _nonempty_string(root["policy_revision"], "attempt.policy_revision")
    code_commit = _git_sha(root["code_commit"], "attempt.code_commit")
    manifest = generate_episode_manifest(
        SplitName.DISCOVERY,
        task,
        protocol,
        policy_revision=policy_revision,
        code_commit=code_commit,
    )
    if _sha256_string(root["manifest_sha256"], "attempt.manifest_sha256") != manifest.sha256:
        raise RealityGateError("attempt manifest SHA-256 does not match ProtocolConfig")
    cells_value = root["cells"]
    if not isinstance(cells_value, list) or len(cells_value) != 40:
        raise RealityGateError("each task attempt must retain exactly 40 cells")
    cells = tuple(
        _cell_from_metadata(item, f"attempt.cells[{index}]")
        for index, item in enumerate(cells_value)
    )
    expected_episodes = _validated_discovery_manifest(protocol, manifest)
    if tuple(cell.episode_id for cell in cells) != tuple(
        episode.episode_id for episode in expected_episodes
    ):
        raise RealityGateError("attempt cells are not in exact manifest order")
    iid_cells = tuple(
        cell
        for cell, episode in zip(cells, expected_episodes, strict=True)
        if episode.condition_family == "iid"
    )
    perturbation_cells = tuple(
        cell
        for cell, episode in zip(cells, expected_episodes, strict=True)
        if episode.condition_family == "object_yaw"
    )
    if any(not cell.executed for cell in iid_cells):
        raise RealityGateError("attempt omits an IID reproduction cell")
    gates = protocol.task_order.gates
    reproduction = reproduction_gate_decision(
        sum(bool(cell.success) for cell in iid_cells),
        len(iid_cells),
        minimum_successes=gates.baseline_min_successes,
    )
    raw_dynamic = root["dynamic_range"]
    if not reproduction.passes:
        if any(cell.executed for cell in perturbation_cells) or raw_dynamic is not None:
            raise RealityGateError(
                "reproduction-failing attempt cannot contain perturbation evidence"
            )
        dynamic = None
    else:
        if any(not cell.executed for cell in perturbation_cells) or raw_dynamic is None:
            raise RealityGateError(
                "reproduction-passing attempt must contain all perturbation evidence"
            )
        valid_count = sum(bool(cell.valid) for cell in perturbation_cells)
        failures = sum(
            bool(cell.valid) and not bool(cell.success) for cell in perturbation_cells
        )
        if valid_count == 0:
            raise RealityGateError("dynamic-range denominator cannot be zero")
        try:
            dynamic = dynamic_range_gate_decision(
                valid_count,
                len(perturbation_cells),
                failures,
                minimum_valid_fraction=gates.min_valid_fraction,
                minimum_failure_rate=gates.min_failure_rate,
                maximum_failure_rate=gates.max_failure_rate,
            )
        except EvaluationError as exc:
            raise RealityGateError(f"invalid dynamic-range denominator: {exc}") from exc
        parsed_dynamic = _dynamic_from_metadata(raw_dynamic, "attempt.dynamic_range")
        if parsed_dynamic != dynamic:
            raise RealityGateError("dynamic-range receipt disagrees with recomputation")
    reproduction_root = _mapping_exact(
        root["reproduction"],
        {"success_interval", "minimum_successes", "passes"},
        "attempt.reproduction",
    )
    parsed_reproduction = _rate_from_metadata(
        reproduction_root["success_interval"],
        "attempt.reproduction.success_interval",
    )
    parsed_reproduction_result = ReproductionGateResult(
        success_interval=parsed_reproduction,
        minimum_successes=_nonnegative_int(
            reproduction_root["minimum_successes"],
            "attempt.reproduction.minimum_successes",
        ),
        passes=_bool_value(
            reproduction_root["passes"], "attempt.reproduction.passes"
        ),
    )
    if parsed_reproduction_result != reproduction:
        raise RealityGateError("reproduction receipt disagrees with recomputation")
    passes = _bool_value(root["passes"], "attempt.passes")
    decision = TaskGateDecision(
        task=task,
        manifest_sha256=manifest.sha256,
        policy_revision=policy_revision,
        code_commit=code_commit,
        cells=cells,
        reproduction=reproduction,
        dynamic_range=dynamic,
        passes=passes,
        _factory_token=_TASK_GATE_DECISION_FACTORY_TOKEN,
    )
    if decision.to_dict() != value:
        raise RealityGateError("task attempt is not exactly canonical or reconstructable")
    return decision


def reality_gate_receipt_from_metadata(
    value: Any, protocol: ProtocolConfig
) -> RealityGateReceipt:
    """Strictly rehydrate and recompute a persisted Reality-Gate receipt.

    This is the lock-boundary parser.  It does not trust serialized Wilson
    intervals or pass flags: it regenerates each Discovery manifest, reconstructs
    the exact cells, and recomputes the point-estimate reproduction and dynamic
    range gates before returning a trusted typed receipt.
    """

    _validate_protocol(protocol)
    root = _mapping_exact(value, {
        "schema_version", "protocol", "decision_basis", "wilson_intervals_role",
        "policy_revision", "code_commit", "task_order", "attempts",
        "selected_task", "all_attempts_retained", "variable_fallback_order",
    }, "reality_gate_receipt")
    if root["schema_version"] != REALITY_GATE_SCHEMA_VERSION:
        raise RealityGateError("unsupported Reality-Gate receipt schema")
    if root["protocol"] != REALITY_GATE_PROTOCOL:
        raise RealityGateError("unsupported Reality-Gate receipt protocol")
    if root["decision_basis"] != "point_estimates" or root["wilson_intervals_role"] != "descriptive_only":
        raise RealityGateError("Reality-Gate decision metadata has drifted")
    policy_revision = _nonempty_string(root["policy_revision"], "policy_revision")
    code_commit = _git_sha(root["code_commit"], "code_commit")
    task_order_values = root["task_order"]
    if not isinstance(task_order_values, list) or not task_order_values:
        raise RealityGateError("Reality-Gate task_order must be a nonempty list")
    task_order = tuple(
        _task_from_metadata(item, f"task_order[{index}]")
        for index, item in enumerate(task_order_values)
    )
    if task_order != protocol.task_order.tasks:
        raise RealityGateError("Reality-Gate task order differs from ProtocolConfig")
    if root["all_attempts_retained"] is not True or root["variable_fallback_order"] != list(VARIABLE_FALLBACK_ORDER):
        raise RealityGateError("Reality-Gate retention or fallback metadata has drifted")
    attempts_value = root["attempts"]
    if not isinstance(attempts_value, list) or not attempts_value:
        raise RealityGateError("Reality-Gate attempts must be a nonempty list")
    attempts: list[TaskGateDecision] = []
    for index, raw_attempt in enumerate(attempts_value):
        attempts.append(
            _task_gate_decision_from_metadata(
                raw_attempt,
                protocol,
                expected_task=task_order[index] if index < len(task_order) else None,
            )
        )
    if len(attempts) > len(task_order):
        raise RealityGateError("Reality-Gate attempts exceed the configured task order")
    selected_value = root["selected_task"]
    selected = None if selected_value is None else _task_from_metadata(selected_value, "selected_task")
    receipt = RealityGateReceipt(
        task_order=task_order,
        attempts=tuple(attempts),
        selected_task=selected,
        _factory_token=_REALITY_GATE_RECEIPT_FACTORY_TOKEN,
    )
    if receipt.policy_revision != policy_revision or receipt.code_commit != code_commit:
        raise RealityGateError("Reality-Gate receipt provenance is inconsistent")
    if receipt.to_dict() != value:
        raise RealityGateError("Reality-Gate receipt is not exactly canonical or reconstructable")
    return receipt


def orientation_eligibility_from_metadata(value: Any) -> OrientationEligibility:
    """Strictly rehydrate a persisted orientation eligibility result."""

    root = _mapping_exact(value, {
        "source_artifacts", "terminal_control_steps", "state_sha256", "symmetry_order",
        "extraction_unit_tests_passed", "total_states", "finite_states", "finite_fraction",
        "resultant_length", "physical_circular_sd_rad", "physical_circular_sd_is_infinite",
        "minimum_finite_fraction", "minimum_physical_circular_sd_degrees", "weighting",
        "cadence", "eligible",
    }, "orientation_eligibility")
    sources_value = root["source_artifacts"]
    if not isinstance(sources_value, list):
        raise RealityGateError("orientation source_artifacts must be a list")
    sources = tuple(_artifact_identity_from_metadata(item, f"source_artifacts[{i}]") for i, item in enumerate(sources_value))
    terminal_value = root["terminal_control_steps"]
    if not isinstance(terminal_value, list):
        raise RealityGateError("orientation terminal_control_steps must be a list")
    terminal: list[tuple[ArtifactIdentity, int]] = []
    for index, item in enumerate(terminal_value):
        entry = _mapping_exact(item, {"artifact", "terminal_control_step"}, f"terminal_control_steps[{index}]")
        step = entry["terminal_control_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise RealityGateError("orientation terminal control steps must be nonnegative integers")
        terminal.append((_artifact_identity_from_metadata(entry["artifact"], f"terminal_control_steps[{index}].artifact"), step))
    result = OrientationEligibility(
        source_artifacts=sources,
        terminal_control_steps=tuple(terminal),
        state_sha256=_sha256_string(root["state_sha256"], "orientation.state_sha256"),
        symmetry_order=_positive_int(root["symmetry_order"], "orientation.symmetry_order"),
        extraction_unit_tests_passed=_bool_value(root["extraction_unit_tests_passed"], "orientation.extraction_unit_tests_passed"),
        total_states=_nonnegative_int(root["total_states"], "orientation.total_states"),
        finite_states=_nonnegative_int(root["finite_states"], "orientation.finite_states"),
        finite_fraction=_finite_float(root["finite_fraction"], "orientation.finite_fraction"),
        resultant_length=None if root["resultant_length"] is None else _finite_float(root["resultant_length"], "orientation.resultant_length"),
        physical_circular_sd_rad=None if root["physical_circular_sd_rad"] is None else _finite_float(root["physical_circular_sd_rad"], "orientation.physical_circular_sd_rad"),
        physical_circular_sd_is_infinite=_bool_value(root["physical_circular_sd_is_infinite"], "orientation.physical_circular_sd_is_infinite"),
        minimum_finite_fraction=_finite_float(root["minimum_finite_fraction"], "orientation.minimum_finite_fraction"),
        minimum_physical_circular_sd_degrees=_finite_float(root["minimum_physical_circular_sd_degrees"], "orientation.minimum_physical_circular_sd_degrees"),
        weighting=_nonempty_string(root["weighting"], "orientation.weighting"),
        cadence=_nonempty_string(root["cadence"], "orientation.cadence"),
        eligible=_bool_value(root["eligible"], "orientation.eligible"),
        _factory_token=_ORIENTATION_ROLLOUT_FACTORY_TOKEN,
    )
    if result.to_dict() != value:
        raise RealityGateError("orientation eligibility is not exactly canonical or reconstructable")
    return result


def _validate_protocol(protocol: ProtocolConfig) -> None:
    if not isinstance(protocol, ProtocolConfig):
        raise RealityGateError("protocol must be a validated ProtocolConfig")
    task_order = protocol.task_order
    gates = task_order.gates
    if task_order.version != 1 or task_order.selection_rule != _SELECTION_RULE:
        raise RealityGateError("unsupported task-order Reality Gate protocol")
    if _task_identities(task_order.tasks) != _EXPECTED_TASKS:
        raise RealityGateError(
            "Reality Gate shortlist differs from the exact amended three-task protocol"
        )
    if (
        gates.baseline_iid_episodes != 10
        or gates.baseline_min_successes != 6
        or gates.perturbed_episodes != 30
        or gates.min_valid_fraction != 0.90
        or gates.min_failure_rate != 0.20
        or gates.max_failure_rate != 0.80
    ):
        raise RealityGateError("gate thresholds differ from the frozen protocol")
    if protocol.split.splits[SplitName.DISCOVERY].init_state_ids.ids() != (
        _DISCOVERY_INIT_IDS
    ):
        raise RealityGateError("Discovery init IDs must be exactly 0..9")
    perturbations = protocol.perturbations
    if (
        perturbations.version != 1
        or not perturbations.angles_in_degrees
        or perturbations.reality_gate_assignment
        != "balanced_hash_without_replacement_three_cells_per_init_state"
    ):
        raise RealityGateError("unsupported Reality Gate perturbation protocol")
    yaw_values: list[float] = []
    for cell in perturbations.reality_gate_cells:
        if cell.family != "object_yaw" or set(cell.parameters) != {"value"}:
            raise RealityGateError("Reality Gate cells must be pure object-yaw edits")
        value = cell.parameters["value"]
        if not isinstance(value, Real) or isinstance(value, bool):
            raise RealityGateError("Reality Gate yaw values must be numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise RealityGateError("Reality Gate yaw values must be finite")
        yaw_values.append(converted)
    if tuple(sorted(yaw_values)) != _EXPECTED_YAWS:
        raise RealityGateError("Reality Gate yaws must be +/-15, +/-30, and +/-45")


def _validated_discovery_manifest(
    protocol: ProtocolConfig, manifest: Manifest
) -> tuple[EpisodeSpec, ...]:
    if manifest.schema_version != 1 or manifest.split is not SplitName.DISCOVERY:
        raise RealityGateError("task attempt requires a schema-v1 Discovery manifest")
    if len(manifest.episodes) != 40:
        raise RealityGateError("Discovery manifest must contain exactly 40 episodes")
    if not manifest.episodes:
        raise RealityGateError("Discovery manifest cannot be empty")
    policy_revision, code_commit = _manifest_revision_pair(manifest)
    regenerated = generate_episode_manifest(
        SplitName.DISCOVERY,
        manifest.task,
        protocol,
        policy_revision=policy_revision,
        code_commit=code_commit,
    )
    if manifest.to_dict() != regenerated.to_dict():
        raise RealityGateError(
            "Discovery manifest differs from the deterministic assigned-yaw grid"
        )

    iid = [
        episode for episode in manifest.episodes if episode.condition_family == "iid"
    ]
    yaw = [
        episode
        for episode in manifest.episodes
        if episode.condition_family == "object_yaw"
    ]
    if len(iid) != 10 or len(yaw) != 30:
        raise RealityGateError("Discovery requires exactly 10 IID and 30 yaw cells")
    for init_id in _DISCOVERY_INIT_IDS:
        per_init = [
            episode
            for episode in manifest.episodes
            if episode.base_init_state_id == init_id
        ]
        if (
            sorted(episode.condition_index for episode in per_init) != [0, 1, 2, 3]
            or sum(episode.condition_family == "iid" for episode in per_init) != 1
        ):
            raise RealityGateError(
                "each Discovery init must contain IID cell 0 and three assigned yaws"
            )
        iid_episode = next(
            episode for episode in per_init if episode.condition_family == "iid"
        )
        if iid_episode.condition_index != 0:
            raise RealityGateError("the IID cell must have condition index zero")
    return manifest.episodes


def _validate_rollout_against_episode(
    raw: RolloutArtifact,
    task: TaskSpec,
    episode: EpisodeSpec,
) -> None:
    episode_id = episode.episode_id
    if not isinstance(raw, RolloutArtifact):
        raise RealityGateError(
            f"{episode_id}: expected a validated RolloutArtifact"
        )
    if not isinstance(raw.metadata, Mapping):
        raise RealityGateError(f"{episode_id}: rollout metadata is invalid")
    try:
        observed_episode = _plain_json_value(raw.metadata["episode"])
        observed_task = _plain_json_value(raw.metadata["task"])
        observed_condition = _plain_json_value(raw.metadata["condition"])
        task_language = raw.metadata["task_language"]
        model = raw.metadata["model"]
    except (KeyError, TypeError) as exc:
        raise RealityGateError(
            f"{episode_id}: rollout lacks manifested metadata"
        ) from exc
    expected_condition = {
        "name": episode.condition_name,
        "family": episode.condition_family,
        "index": episode.condition_index,
        "parameters": _plain_json_value(episode.condition_parameters),
    }
    if observed_episode != episode.to_dict():
        raise RealityGateError(
            f"{episode_id}: rollout episode metadata differs from the manifest"
        )
    if observed_task != _task_metadata(task) or task_language != task.language:
        raise RealityGateError(
            f"{episode_id}: rollout task metadata differs from the manifest task"
        )
    if observed_condition != expected_condition:
        raise RealityGateError(
            f"{episode_id}: rollout condition metadata differs from the manifest"
        )
    if not isinstance(model, Mapping) or (
        model.get("policy_revision") != episode.policy_revision
    ):
        raise RealityGateError(
            f"{episode_id}: rollout policy revision differs from the manifest"
        )
    base_vlm = model.get("base_vlm_revision")
    if not isinstance(base_vlm, str) or not base_vlm:
        raise RealityGateError(
            f"{episode_id}: rollout base-VLM revision must be non-empty"
        )
    if raw.episode_id != episode_id:
        raise RealityGateError(
            f"{episode_id}: rollout episode identity differs from the manifest"
        )
    _validate_rollout_frame_contract(raw)


def _validate_rollout_frame_contract(raw: RolloutArtifact) -> None:
    episode_id = raw.episode_id
    try:
        actions = raw.arrays["actions"]
        steps = raw.arrays["frame_control_step"]
        frame_success = raw.arrays["frame_task_success"]
        valid = raw.metadata["validity"]["valid"]
        outcome = raw.metadata["outcome"]
        success = outcome["success"]
        control_steps = outcome["control_steps"]
    except (KeyError, TypeError) as exc:
        raise RealityGateError(
            f"{episode_id}: rollout lacks validated frame/outcome evidence"
        ) from exc
    if not isinstance(actions, np.ndarray) or actions.ndim != 2:
        raise RealityGateError(f"{episode_id}: rollout actions are invalid")
    if (
        not isinstance(steps, np.ndarray)
        or steps.ndim != 1
        or steps.dtype.kind not in "iu"
        or not np.array_equal(steps, np.arange(steps.size, dtype=steps.dtype))
    ):
        raise RealityGateError(
            f"{episode_id}: frame control steps must be sequential from zero"
        )
    if (
        not isinstance(frame_success, np.ndarray)
        or frame_success.dtype != np.dtype(np.bool_)
        or frame_success.shape != steps.shape
        or steps.size != actions.shape[0] + 1
    ):
        raise RealityGateError(
            f"{episode_id}: rollout frame cadence or success evidence is invalid"
        )
    if (
        type(valid) is not bool
        or type(success) is not bool
        or type(control_steps) is not int
        or control_steps != actions.shape[0]
        or bool(frame_success[-1]) != success
    ):
        raise RealityGateError(
            f"{episode_id}: rollout outcome disagrees with trajectory evidence"
        )
    if (not valid and (actions.shape[0] != 0 or success)) or (
        valid and actions.shape[0] == 0
    ):
        raise RealityGateError(
            f"{episode_id}: rollout validity disagrees with trajectory length/outcome"
        )


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json_value(nested) for nested in value]
    return value


def _validate_exact_cell_coverage(
    episodes: tuple[EpisodeSpec, ...],
    cells: tuple[DiscoveryCellResult, ...],
) -> dict[str, DiscoveryCellResult]:
    if len(cells) != 40:
        raise RealityGateError("attempt must represent exactly 40 Discovery cells")
    by_id: dict[str, DiscoveryCellResult] = {}
    for cell in cells:
        if cell.episode_id in by_id:
            raise RealityGateError("Discovery cell episode IDs must be unique")
        by_id[cell.episode_id] = cell
    expected = {episode.episode_id for episode in episodes}
    actual = set(by_id)
    if actual != expected:
        raise RealityGateError(
            f"Discovery result coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return by_id


def _rate_metadata(interval: RateInterval) -> dict[str, Any]:
    return {
        "rate": interval.rate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence": interval.confidence,
        "successes": interval.successes,
        "total": interval.total,
    }


def _reproduction_metadata(result: ReproductionGateResult) -> dict[str, Any]:
    return {
        "success_interval": _rate_metadata(result.success_interval),
        "minimum_successes": result.minimum_successes,
        "passes": result.passes,
    }


def _dynamic_metadata(result: DynamicRangeGateResult) -> dict[str, Any]:
    return {
        "validity_interval": _rate_metadata(result.validity_interval),
        "failure_interval": _rate_metadata(result.failure_interval),
        "minimum_valid_fraction": result.minimum_valid_fraction,
        "failure_rate_bounds": list(result.failure_rate_bounds),
        "passes_validity": result.passes_validity,
        "passes_failure_range": result.passes_failure_range,
        "passes": result.passes,
    }


def _task_metadata(task: TaskSpec) -> dict[str, Any]:
    return {
        "rank": task.rank,
        "suite": task.suite,
        "task_id": task.task_id,
        "language": task.language,
        "primary_object": task.primary_object,
        "planar_symmetry_order": task.planar_symmetry_order,
    }


def _task_identities(tasks: tuple[TaskSpec, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            task.rank,
            task.suite,
            task.task_id,
            task.language,
            task.primary_object,
            task.planar_symmetry_order,
        )
        for task in tasks
    )


def _cell_metadata(cell: DiscoveryCellResult) -> dict[str, Any]:
    return {
        "episode_id": cell.episode_id,
        "executed": cell.executed,
        "artifact": None if cell.artifact is None else cell.artifact.to_dict(),
        "valid": cell.valid,
        "success": cell.success,
        "validity_retry": (
            None if cell.validity_retry is None else cell.validity_retry.to_dict()
        ),
    }


def _manifest_revision_pair(manifest: Manifest) -> tuple[str, str]:
    if not isinstance(manifest, Manifest) or not manifest.episodes:
        raise RealityGateError("task attempt requires a nonempty Manifest")
    policy_revisions = {episode.policy_revision for episode in manifest.episodes}
    code_commits = {episode.code_commit for episode in manifest.episodes}
    if len(policy_revisions) != 1 or len(code_commits) != 1:
        raise RealityGateError("Discovery manifest revisions must be uniform")
    return next(iter(policy_revisions)), next(iter(code_commits))


def _is_trusted_reality_gate_receipt(value: Any) -> bool:
    if (
        not isinstance(value, RealityGateReceipt)
        or getattr(value, "_validation_marker", None)
        is not _REALITY_GATE_RECEIPT_FACTORY_TOKEN
    ):
        return False
    attempts = getattr(value, "attempts", None)
    return isinstance(attempts, tuple) and bool(attempts) and all(
        isinstance(attempt, TaskGateDecision)
        and getattr(attempt, "_validation_marker", None)
        is _TASK_GATE_DECISION_FACTORY_TOKEN
        for attempt in attempts
    )


def _selected_task_artifacts(
    decision: TaskGateDecision,
) -> tuple[ArtifactIdentity, ...]:
    if not isinstance(decision, TaskGateDecision) or not decision.passes:
        raise RealityGateError("selected task must have a passing task decision")
    if any(not cell.executed or cell.artifact is None for cell in decision.cells):
        raise RealityGateError(
            "a selected task must have all 40 raw artifacts executed"
        )
    artifacts = tuple(
        sorted(
            (cell.artifact for cell in decision.cells if cell.artifact is not None),
            key=lambda artifact: artifact.episode_id,
        )
    )
    if len(artifacts) != 40 or len({item.episode_id for item in artifacts}) != 40:
        raise RealityGateError(
            "a selected task must bind exactly 40 unique raw artifacts"
        )
    return artifacts


def _wrap_angle(value: float) -> float:
    if not math.isfinite(value):
        raise RealityGateError("orientation yaw extraction must be finite")
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    # Keep the endpoint deterministic for exact +/-pi cases.
    return math.pi if wrapped == -math.pi and value > 0.0 else wrapped


def _yaw_from_xyzw(quaternion: Any) -> float:
    try:
        x, y, z, w = (float(component) for component in quaternion)
    except (TypeError, ValueError) as exc:
        raise RealityGateError("EEF quaternion is not numeric") from exc
    if not all(math.isfinite(component) for component in (x, y, z, w)):
        raise RealityGateError("EEF quaternion contains a non-finite component")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_from_wxyz(quaternion: Any) -> float:
    try:
        w, x, y, z = (float(component) for component in quaternion)
    except (TypeError, ValueError) as exc:
        raise RealityGateError("primary-object quaternion is not numeric") from exc
    if not all(math.isfinite(component) for component in (w, x, y, z)):
        raise RealityGateError(
            "primary-object quaternion contains a non-finite component"
        )
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _orientation_state_sha256(
    states: tuple[OrientationState, ...],
    terminal_control_steps: tuple[tuple[ArtifactIdentity, int], ...],
) -> str:
    records: list[dict[str, Any]] = []
    for state in states:
        theta = state.theta_rel_rad
        if theta is None:
            encoded_theta: float | str = "missing"
        elif math.isnan(theta):
            encoded_theta = "nan"
        elif theta == math.inf:
            encoded_theta = "positive_infinity"
        elif theta == -math.inf:
            encoded_theta = "negative_infinity"
        else:
            encoded_theta = theta
        records.append(
            {
                "episode_id": state.episode_id,
                "control_step": state.control_step,
                "theta_rel_rad": encoded_theta,
            }
        )
    terminal_records = [
        {
            "artifact": artifact.to_dict(),
            "terminal_control_step": terminal_step,
        }
        for artifact, terminal_step in terminal_control_steps
    ]
    payload = {
        "states": records,
        "terminal_control_steps": terminal_records,
    }
    return hashlib.sha256(
        b"mech-int-vla-orientation-states-v2\0" + _canonical_json(payload)
    ).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RealityGateError("receipt metadata is not finite canonical JSON") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ORIENTATION_CADENCE",
    "ORIENTATION_MINIMUM_CIRCULAR_SD_DEGREES",
    "ORIENTATION_MINIMUM_FINITE_FRACTION",
    "ORIENTATION_WEIGHTING",
    "REALITY_GATE_LOCK_PROTOCOL",
    "REALITY_GATE_LOCK_SCHEMA_VERSION",
    "REALITY_GATE_PROTOCOL",
    "REALITY_GATE_SCHEMA_VERSION",
    "VARIABLE_FALLBACK_ORDER",
    "DiscoveryCellResult",
    "OrientationEligibility",
    "OrientationState",
    "RealityGateError",
    "RealityGateLockReceipt",
    "RealityGateReceipt",
    "TaskDiscoveryAttempt",
    "TaskGateDecision",
    "ValidityRetryEvidence",
    "decide_reality_gate",
    "evaluate_orientation_eligibility",
    "evaluate_task_attempt",
    "finalize_reality_gate",
    "orientation_eligibility_from_metadata",
    "orientation_eligibility_from_rollouts",
    "reality_gate_receipt_from_metadata",
    "task_discovery_attempt_from_rollouts",
    "variable_fallback_order",
]
