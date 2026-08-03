"""Deterministic, terminal-state-preserving SmolVLA episode execution.

The runner in this module is deliberately narrower than the later scoring
pipeline.  It executes one already-manifested episode, captures the five frozen
activation candidates for every policy call, and atomically publishes a compact
raw artifact.  It never calls the auto-resetting LIBERO wrapper step method.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ConditionSpec
from .instrumentation import (
    EXPERT_EARLY,
    EXPERT_LATE,
    VLM_CONTEXT,
    ActivationRecord,
    CallPhase,
    SmolVLAInstrumentation,
)
from .libero_runtime import (
    RawLiberoEpisode,
    RawTraceFrame,
    ValidityResult,
    seed_runtime,
)
from .manifest import EpisodeSpec
from .snapshots import LoadedPolicyRuntime

ARTIFACT_SCHEMA_VERSION = 1
ACTION_DIMENSION = 7
PHASES = ("pregrasp", "grasped", "transport", "placed")

# Public names match the preregistered calibration-selection candidates.
ACTIVATION_CANDIDATES: Mapping[str, tuple[str, int | None]] = {
    "vlm_context": (VLM_CONTEXT, None),
    "early_expert_t1_0": (EXPERT_EARLY, 0),
    "early_expert_t0_5": (EXPERT_EARLY, 5),
    "late_expert_t1_0": (EXPERT_LATE, 0),
    "late_expert_t0_5": (EXPERT_LATE, 5),
}

FRAME_SCALAR_FEATURE_NAMES = (
    "normalized_step",
    "simulator_time_s",
    "eef_object_distance_m",
    "object_goal_distance_m",
    "gripper_opening",
    "primary_gripper_contact",
    "primary_grasped",
    "task_success",
    "phase_pregrasp",
    "phase_grasped",
    "phase_transport",
    "phase_placed",
    "symmetry_relative_yaw_sin",
    "symmetry_relative_yaw_cos",
)


class RolloutError(RuntimeError):
    """Raised when runtime behavior violates the single-episode contract."""


@dataclass(frozen=True)
class EpisodeRunResult:
    """Small in-memory result; full arrays live in ``artifact_path``."""

    artifact_path: Path
    status: str
    valid_reset: bool
    success: bool
    terminated: bool
    truncated: bool
    control_steps: int


def _assert_runtime_contract(
    policy_runtime: LoadedPolicyRuntime,
    episode: RawLiberoEpisode,
    instrumentation: SmolVLAInstrumentation,
    episode_spec: EpisodeSpec,
    condition: ConditionSpec,
) -> None:
    if episode._has_reset or episode.primary_object_name is not None:
        raise RolloutError(
            "RawLiberoEpisode has already been reset; construct a fresh runtime "
            "for every manifested condition"
        )
    if episode.execution.n_action_steps != 1:
        raise RolloutError("RawLiberoEpisode must use n_action_steps=1")
    policy_steps = getattr(
        getattr(policy_runtime.policy, "config", None), "n_action_steps", None
    )
    if policy_steps != 1:
        raise RolloutError(f"policy must use n_action_steps=1, got {policy_steps!r}")
    lock = policy_runtime.snapshots.lock
    if lock.policy_n_action_steps != 1:
        raise RolloutError("snapshot lock must use n_action_steps=1")
    if lock.policy_revision != episode_spec.policy_revision:
        raise RolloutError("manifest and loaded policy revisions differ")

    task = episode.task
    expected_task = (task.suite, task.task_id, task.rank)
    manifested_task = (
        episode_spec.suite,
        episode_spec.task_id,
        episode_spec.task_rank,
    )
    if expected_task != manifested_task:
        raise RolloutError("manifest task does not match RawLiberoEpisode task")
    if episode_spec.base_init_state_id < 0:
        raise RolloutError("manifest base init state must be nonnegative")
    if (
        condition.name != episode_spec.condition_name
        or condition.family != episode_spec.condition_family
        or condition.index != episode_spec.condition_index
        or dict(condition.parameters) != dict(episode_spec.condition_parameters)
    ):
        raise RolloutError("condition does not match manifested episode cell")
    if instrumentation.is_installed:
        raise RolloutError("instrumentation must not be installed before the run")

    expected_flow_model = getattr(policy_runtime.policy, "model", policy_runtime.policy)
    if instrumentation.flow_model is not expected_flow_model:
        raise RolloutError("instrumentation is attached to a different policy model")


def _as_action(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (1, ACTION_DIMENSION):
        action = action[0]
    if action.shape != (ACTION_DIMENSION,) or not np.isfinite(action).all():
        raise RolloutError(
            f"postprocessed policy action must be one finite {ACTION_DIMENSION}-vector, "
            f"got shape {action.shape}"
        )
    return action.copy()


def _record_array(record: ActivationRecord) -> np.ndarray:
    value = record.value
    if not isinstance(value, torch.Tensor):
        raise RolloutError("instrumentation record value is not a torch.Tensor")
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise RolloutError(
            "activation candidates must be finite single-example hidden vectors; "
            f"got shape {array.shape}"
        )
    return array.copy()


def _collect_activation_candidates(
    instrumentation: SmolVLAInstrumentation,
) -> dict[str, np.ndarray]:
    calls = instrumentation.calls
    if len(calls) != 11:
        raise RolloutError(
            f"one SmolVLA action must make 11 instrumented calls, got {len(calls)}"
        )
    if calls[0].phase is not CallPhase.PREFIX_CACHE:
        raise RolloutError("first instrumented call is not prefix-cache construction")
    denoising = calls[1:]
    if any(call.phase is not CallPhase.DENOISING for call in denoising) or [
        call.denoising_step for call in denoising
    ] != list(range(10)):
        raise RolloutError("instrumented denoising calls are not the frozen steps 0..9")

    records = instrumentation.records
    if len(records) != len(ACTIVATION_CANDIDATES):
        raise RolloutError(
            "one policy call must emit exactly five activation candidates, "
            f"got {len(records)}"
        )
    run_indices = {record.run_index for record in records}
    if len(run_indices) != 1:
        raise RolloutError("activation candidates came from multiple policy calls")
    if any(record.patched for record in records):
        raise RolloutError("the unpatched rollout runner received a patched activation")

    captured: dict[str, np.ndarray] = {}
    for candidate, (location, step) in ACTIVATION_CANDIDATES.items():
        matches = [
            record
            for record in records
            if record.location == location and record.denoising_step == step
        ]
        if len(matches) != 1:
            raise RolloutError(
                f"candidate {candidate} resolved to {len(matches)} activation records"
            )
        captured[candidate] = _record_array(matches[0])
    return captured


def _quaternion_yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _frame_scalar_features(
    frame: RawTraceFrame,
    *,
    max_steps: int,
    planar_symmetry_order: int,
) -> np.ndarray:
    if frame.phase not in PHASES:
        raise RolloutError(f"unknown deterministic task phase {frame.phase!r}")
    eef_object_distance = float(
        np.linalg.norm(frame.primary_object_position - frame.eef_position)
    )
    if frame.goal_position is None:
        object_goal_distance = math.nan
        relative_yaw_sin = math.nan
        relative_yaw_cos = math.nan
    else:
        object_goal_distance = float(
            np.linalg.norm(frame.goal_position - frame.primary_object_position)
        )
        if frame.goal_quaternion_wxyz is None:
            relative_yaw_sin = math.nan
            relative_yaw_cos = math.nan
        else:
            relative_yaw = _quaternion_yaw_wxyz(
                frame.primary_object_quaternion_wxyz
            ) - _quaternion_yaw_wxyz(frame.goal_quaternion_wxyz)
            symmetry_angle = planar_symmetry_order * relative_yaw
            relative_yaw_sin = math.sin(symmetry_angle)
            relative_yaw_cos = math.cos(symmetry_angle)
    phase_features = tuple(float(frame.phase == phase) for phase in PHASES)
    result = np.asarray(
        [
            frame.control_step / max_steps,
            frame.simulator_time,
            eef_object_distance,
            object_goal_distance,
            float(np.sum(frame.gripper_qpos)),
            float(frame.primary_gripper_contact),
            float(frame.primary_grasped),
            float(frame.task_success),
            *phase_features,
            relative_yaw_sin,
            relative_yaw_cos,
        ],
        dtype=np.float32,
    )
    if result.shape != (len(FRAME_SCALAR_FEATURE_NAMES),):
        raise AssertionError("frame scalar feature schema drifted")
    return result


def _stack_frame_field(
    frames: Sequence[RawTraceFrame], name: str, *, dtype: Any
) -> np.ndarray:
    try:
        return np.stack(
            [np.asarray(getattr(frame, name), dtype=dtype) for frame in frames]
        )
    except ValueError as exc:
        raise RolloutError(f"frame field {name} changed shape during episode") from exc


def _trajectory_arrays(
    *,
    episode: RawLiberoEpisode,
    frames: Sequence[RawTraceFrame],
    actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    activation_values: Mapping[str, Sequence[np.ndarray]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    if len(frames) != len(actions) + 1:
        raise RolloutError("trajectory must contain one more frame than action")
    frame_count = len(frames)
    arrays: dict[str, np.ndarray] = {
        "actions": (
            np.stack(actions).astype(np.float32, copy=False)
            if actions
            else np.empty((0, ACTION_DIMENSION), dtype=np.float32)
        ),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "frame_control_step": np.asarray(
            [frame.control_step for frame in frames], dtype=np.int32
        ),
        "frame_simulator_time": np.asarray(
            [frame.simulator_time for frame in frames], dtype=np.float64
        ),
        "frame_policy_state": _stack_frame_field(
            frames, "policy_state", dtype=np.float32
        ),
        "frame_eef_position": _stack_frame_field(
            frames, "eef_position", dtype=np.float64
        ),
        "frame_eef_quaternion_xyzw": _stack_frame_field(
            frames, "eef_quaternion_xyzw", dtype=np.float64
        ),
        "frame_primary_object_position": _stack_frame_field(
            frames, "primary_object_position", dtype=np.float64
        ),
        "frame_primary_object_quaternion_wxyz": _stack_frame_field(
            frames, "primary_object_quaternion_wxyz", dtype=np.float64
        ),
        "frame_gripper_qpos": _stack_frame_field(
            frames, "gripper_qpos", dtype=np.float64
        ),
        "frame_gripper_qvel": _stack_frame_field(
            frames, "gripper_qvel", dtype=np.float64
        ),
        "frame_primary_gripper_contact": np.asarray(
            [frame.primary_gripper_contact for frame in frames], dtype=np.bool_
        ),
        "frame_primary_grasped": np.asarray(
            [frame.primary_grasped for frame in frames], dtype=np.bool_
        ),
        "frame_task_success": np.asarray(
            [frame.task_success for frame in frames], dtype=np.bool_
        ),
        "frame_phase": np.asarray([frame.phase for frame in frames], dtype="U16"),
        "frame_scalar_features": np.stack(
            [
                _frame_scalar_features(
                    frame,
                    max_steps=episode.execution.max_steps,
                    planar_symmetry_order=episode.task.planar_symmetry_order,
                )
                for frame in frames
            ]
        ),
    }
    goal_present = np.asarray(
        [frame.goal_position is not None for frame in frames], dtype=np.bool_
    )
    arrays["frame_goal_present"] = goal_present
    arrays["frame_goal_position"] = np.stack(
        [
            np.full(3, np.nan, dtype=np.float64)
            if frame.goal_position is None
            else np.asarray(frame.goal_position, dtype=np.float64)
            for frame in frames
        ]
    )
    arrays["frame_goal_quaternion_wxyz"] = np.stack(
        [
            np.full(4, np.nan, dtype=np.float64)
            if frame.goal_quaternion_wxyz is None
            else np.asarray(frame.goal_quaternion_wxyz, dtype=np.float64)
            for frame in frames
        ]
    )

    predicate_names = tuple(sorted(frames[0].task_predicates))
    if any(tuple(sorted(frame.task_predicates)) != predicate_names for frame in frames):
        raise RolloutError("task predicate keys changed during episode")
    arrays["frame_task_predicates"] = np.asarray(
        [
            [bool(frame.task_predicates[name]) for name in predicate_names]
            for frame in frames
        ],
        dtype=np.bool_,
    ).reshape(frame_count, len(predicate_names))

    for candidate in ACTIVATION_CANDIDATES:
        values = activation_values[candidate]
        arrays[f"activation_{candidate}"] = (
            np.stack(values).astype(np.float32, copy=False)
            if values
            else np.empty((0, 0), dtype=np.float32)
        )
    return arrays, predicate_names


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _validity_payload(validity: ValidityResult) -> dict[str, Any]:
    return {
        "valid": validity.valid,
        "reasons": list(validity.reasons),
        "finite": validity.finite,
        "deepest_primary_penetration_m": _finite_or_none(
            validity.deepest_primary_penetration_m
        ),
        "linear_speed_m_s": _finite_or_none(validity.linear_speed_m_s),
        "angular_speed_rad_s": _finite_or_none(validity.angular_speed_rad_s),
        "in_workspace": validity.in_workspace,
        "initial_success": validity.initial_success,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_destination(
    artifact_root: Path, episode_spec: EpisodeSpec
) -> tuple[Path, Path]:
    root = artifact_root.expanduser().resolve()
    if any(part in {"locks", "configs"} for part in root.parts):
        raise RolloutError(
            "raw rollout artifacts may not be written into protocol or lock paths"
        )
    if Path(episode_spec.episode_id).name != episode_spec.episode_id:
        raise RolloutError("episode_id is not a safe artifact directory name")
    parent = root / episode_spec.split.value
    destination = parent / episode_spec.episode_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite episode artifact {destination}")
    return parent, destination


def _write_artifact_atomically(
    artifact_root: Path,
    *,
    episode_spec: EpisodeSpec,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    parent, destination = _artifact_destination(artifact_root, episode_spec)
    parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{episode_spec.episode_id}.tmp-", dir=parent)
    )
    try:
        trajectory_path = staging / "trajectory.npz"
        np.savez_compressed(trajectory_path, **arrays)
        with trajectory_path.open("rb") as stream:
            os.fsync(stream.fileno())

        metadata_path = staging / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(
                _json_value(metadata),
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def run_single_episode(
    policy_runtime: LoadedPolicyRuntime,
    episode: RawLiberoEpisode,
    instrumentation: SmolVLAInstrumentation,
    episode_spec: EpisodeSpec,
    condition: ConditionSpec,
    *,
    artifact_root: str | Path = Path("artifacts/raw"),
) -> EpisodeRunResult:
    """Execute and atomically record one frozen manifest episode.

    The supplied ``RawLiberoEpisode`` must be freshly constructed because the
    pinned LeRobot wrapper advances its init-state cursor during reset.  The
    policy is reset once, inference randomness is seeded once from the manifest,
    and closed-loop replanning occurs at every environment step.  An invalid
    post-settle reset is recorded without policy inference.  Completed success
    and truncation states are never advanced or reset by this function.
    """

    # Fail before the expensive reset/inference path if the destination is unsafe
    # or this manifested cell has already been published.
    _artifact_destination(Path(artifact_root), episode_spec)
    _assert_runtime_contract(
        policy_runtime, episode, instrumentation, episode_spec, condition
    )
    reset = episode.reset(seed=episode_spec.reset_seed, condition=condition)
    frames = [reset.frame]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    activation_values: dict[str, list[np.ndarray]] = {
        candidate: [] for candidate in ACTIVATION_CANDIDATES
    }

    final_terminated = False
    final_truncated = False
    if reset.validity.valid:
        seed_runtime(episode_spec.inference_seed)
        reset_policy = getattr(policy_runtime.policy, "reset", None)
        if not callable(reset_policy):
            raise RolloutError("loaded policy does not expose reset()")
        reset_policy()
        observation = reset.observation
        with instrumentation, torch.inference_mode():
            while not episode.terminal:
                instrumentation.clear()
                processed = policy_runtime.preprocess_observation(
                    dict(observation), task=episode.task.language
                )
                selected_action = policy_runtime.policy.select_action(processed)
                action = _as_action(policy_runtime.postprocessor(selected_action))
                captured = _collect_activation_candidates(instrumentation)
                step = episode.step(action)

                actions.append(action)
                rewards.append(step.reward)
                terminated_flags.append(step.terminated)
                truncated_flags.append(step.truncated)
                frames.append(step.frame)
                for candidate, value in captured.items():
                    activation_values[candidate].append(value)
                observation = step.observation
                final_terminated = step.terminated
                final_truncated = step.truncated

    success = bool(frames[-1].task_success)
    if not reset.validity.valid:
        status = "invalid_reset"
    elif success:
        status = "success"
    elif final_truncated:
        status = "truncated"
    elif final_terminated:
        status = "terminated_without_success"
    else:
        raise RolloutError("valid episode ended without a terminal or truncation state")

    arrays, predicate_names = _trajectory_arrays(
        episode=episode,
        frames=frames,
        actions=actions,
        rewards=rewards,
        terminated=terminated_flags,
        truncated=truncated_flags,
        activation_values=activation_values,
    )
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "episode": episode_spec.to_dict(),
        "task_language": episode.task.language,
        "condition": {
            "name": condition.name,
            "family": condition.family,
            "index": condition.index,
            "parameters": condition.parameters,
        },
        "model": {
            "policy_revision": policy_runtime.snapshots.lock.policy_revision,
            "base_vlm_revision": policy_runtime.snapshots.lock.base_vlm_revision,
        },
        "execution": {
            "n_action_steps": episode.execution.n_action_steps,
            "max_steps": episode.execution.max_steps,
            "reset_noop_steps": episode.execution.reset_noop_steps,
            "settle_actions": reset.settle_actions,
            "closed_loop_replanning": True,
        },
        "validity": _validity_payload(reset.validity),
        "outcome": {
            "status": status,
            "success": success,
            "terminated": final_terminated,
            "truncated": final_truncated,
            "control_steps": len(actions),
            "reward_sum": float(np.sum(rewards, dtype=np.float64)),
            "terminal_state_preserved": bool(episode.terminal),
        },
        "capture": {
            "policy_select_calls": len(actions),
            "instrumented_internal_calls": 11 * len(actions),
            "activation_candidates": list(ACTIVATION_CANDIDATES),
            "activation_dtype": "float32",
            "frame_count": len(frames),
            "frame_scalar_feature_names": FRAME_SCALAR_FEATURE_NAMES,
            "task_predicate_names": predicate_names,
            "raw_images_stored": False,
        },
        "files": {"trajectory": "trajectory.npz"},
    }
    artifact_path = _write_artifact_atomically(
        Path(artifact_root),
        episode_spec=episode_spec,
        metadata=metadata,
        arrays=arrays,
    )
    return EpisodeRunResult(
        artifact_path=artifact_path,
        status=status,
        valid_reset=reset.validity.valid,
        success=success,
        terminated=final_terminated,
        truncated=final_truncated,
        control_steps=len(actions),
    )


__all__ = [
    "ACTIVATION_CANDIDATES",
    "ARTIFACT_SCHEMA_VERSION",
    "FRAME_SCALAR_FEATURE_NAMES",
    "EpisodeRunResult",
    "RolloutError",
    "run_single_episode",
]
