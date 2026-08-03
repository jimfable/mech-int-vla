"""Pinned LIBERO/SmolVLA bridge for deterministic replay scoring.

Heavy rollout dependencies are imported only inside the methods which need
them.  Importing this module therefore remains safe in protocol-analysis and
artifact-validation processes without LeRobot, Transformers, or CUDA.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
import time
import zlib
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import RolloutArtifact
from .config import ConditionSpec
from .scoring import (
    ACTION_DIMENSION,
    CallCost,
    ContentLinks,
    FactualReplay,
    ReplayFrame,
    ReplayTransition,
    ScoredCall,
    ScoringError,
    ScoringTransform,
    SimulatorSnapshot,
    TransformValidity,
)

if TYPE_CHECKING:
    from .instrumentation import SmolVLAInstrumentation
    from .libero_runtime import RawLiberoEpisode
    from .probes import ProbeArtifact
    from .snapshots import LoadedPolicyRuntime

PHASES = ("pregrasp", "grasped", "transport", "placed")
FROZEN_LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
FROZEN_LEROBOT_PYTHON_SHA256 = (
    "79603648ff8d9889072449099da6e60b6a92fe0da84108e2bae1dc765b217ecd"
)
FROZEN_VALIDITY_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "settle_steps_after_edit": 10,
        "max_invalid_fraction": 0.10,
        "reject_nan_or_inf": True,
        "reject_workspace_exit": True,
        "reject_initial_success": True,
        "max_penetration_depth_m": 0.005,
        "max_linear_speed_m_s": 0.05,
        "max_angular_speed_rad_s": 0.5,
        "min_height_below_table_m": 0.02,
        "max_height_above_table_m": 0.50,
        "phase_min_object_displacement_m": 0.02,
    }
)
VLM_CONTEXT = "vlm_context"
EXPERT_EARLY = "expert_layer_4"
EXPERT_LATE = "expert_layer_12"
CANDIDATE_TARGETS: Mapping[str, tuple[str, int | None]] = MappingProxyType(
    {
        "vlm_context": (VLM_CONTEXT, None),
        "early_expert_t1_0": (EXPERT_EARLY, 0),
        "early_expert_t0_5": (EXPERT_EARLY, 5),
        "late_expert_t1_0": (EXPERT_LATE, 0),
        "late_expert_t0_5": (EXPERT_LATE, 5),
    }
)


class ScoringRuntimeError(ScoringError):
    """Raised when the pinned runtime bridge observes architectural drift."""


def candidate_target(candidate: str) -> tuple[str, int | None]:
    """Resolve one frozen probe-candidate name to its hook and flow step."""

    try:
        return CANDIDATE_TARGETS[candidate]
    except KeyError as exc:
        raise ScoringRuntimeError(
            f"unknown frozen probe candidate {candidate!r}"
        ) from exc


def _goal_array(value: NDArray[Any] | None, width: int) -> NDArray[np.float64]:
    if value is None:
        return np.full(width, np.nan, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,):
        raise ScoringRuntimeError(f"goal pose component must have shape ({width},)")
    return array.copy()


def _low_dimensional(
    *,
    simulator_time: float,
    policy_state: NDArray[Any],
    eef_position: NDArray[Any],
    eef_quaternion_xyzw: NDArray[Any],
    object_position: NDArray[Any],
    object_quaternion_wxyz: NDArray[Any],
    gripper_qpos: NDArray[Any],
    gripper_qvel: NDArray[Any],
    goal_present: bool,
    goal_position: NDArray[Any] | None,
    goal_quaternion_wxyz: NDArray[Any] | None,
    primary_gripper_contact: bool,
    primary_grasped: bool,
    task_success: bool,
    phase: str,
    task_predicate_names: tuple[str, ...],
    task_predicates: NDArray[Any],
) -> dict[str, NDArray[Any]]:
    if phase not in PHASES:
        raise ScoringRuntimeError(f"unknown deterministic phase {phase!r}")
    phase_one_hot = np.asarray([phase == value for value in PHASES], dtype=np.bool_)
    predicate_schema = np.frombuffer(
        hashlib.sha256(
            json.dumps(
                list(task_predicate_names),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest(),
        dtype=np.uint8,
    ).copy()
    values: dict[str, NDArray[Any]] = {
        "simulator_time": np.asarray(simulator_time, dtype=np.float64),
        "policy_state": np.asarray(policy_state, dtype=np.float32),
        "eef_position": np.asarray(eef_position, dtype=np.float64),
        "eef_quaternion_xyzw": np.asarray(eef_quaternion_xyzw, dtype=np.float64),
        "primary_object_position": np.asarray(object_position, dtype=np.float64),
        "primary_object_quaternion_wxyz": np.asarray(
            object_quaternion_wxyz, dtype=np.float64
        ),
        "gripper_qpos": np.asarray(gripper_qpos, dtype=np.float64),
        "gripper_qvel": np.asarray(gripper_qvel, dtype=np.float64),
        "goal_present": np.asarray(goal_present, dtype=np.bool_),
        "goal_position": _goal_array(goal_position, 3),
        "goal_quaternion_wxyz": _goal_array(goal_quaternion_wxyz, 4),
        "primary_gripper_contact": np.asarray(primary_gripper_contact, dtype=np.bool_),
        "primary_grasped": np.asarray(primary_grasped, dtype=np.bool_),
        "task_success": np.asarray(task_success, dtype=np.bool_),
        "phase_one_hot": phase_one_hot,
        "task_predicate_schema_sha256": predicate_schema,
        "task_predicates": np.asarray(task_predicates, dtype=np.bool_),
    }
    expected_shapes = {
        "policy_state": (8,),
        "eef_position": (3,),
        "eef_quaternion_xyzw": (4,),
        "primary_object_position": (3,),
        "primary_object_quaternion_wxyz": (4,),
        "goal_position": (3,),
        "goal_quaternion_wxyz": (4,),
        "phase_one_hot": (4,),
    }
    for name, shape in expected_shapes.items():
        if values[name].shape != shape:
            raise ScoringRuntimeError(
                f"low-dimensional {name} expected shape {shape}, got {values[name].shape}"
            )
    if (
        values["gripper_qpos"].ndim != 1
        or values["gripper_qvel"].shape != values["gripper_qpos"].shape
    ):
        raise ScoringRuntimeError("gripper position/velocity shapes are inconsistent")
    if values["task_predicates"].ndim != 1:
        raise ScoringRuntimeError("task predicates must be a numeric vector")
    if len(task_predicate_names) != values["task_predicates"].size:
        raise ScoringRuntimeError("task predicate names and vector size disagree")
    for name, value in values.items():
        if (
            name not in {"goal_position", "goal_quaternion_wxyz"}
            and not np.isfinite(value).all()
        ):
            raise ScoringRuntimeError(f"low-dimensional field {name} is nonfinite")
    if goal_present:
        if (
            not np.isfinite(values["goal_position"]).all()
            or not np.isfinite(values["goal_quaternion_wxyz"]).all()
        ):
            raise ScoringRuntimeError("present goal pose must be finite")
    elif not (
        np.isnan(values["goal_position"]).all()
        and np.isnan(values["goal_quaternion_wxyz"]).all()
    ):
        raise ScoringRuntimeError("absent goal pose must be encoded as NaN")
    return values


def _artifact_frame(
    artifact: RolloutArtifact, index: int, predicate_names: tuple[str, ...]
) -> ReplayFrame:
    arrays = artifact.arrays
    phase = str(arrays["frame_phase"][index])
    goal_present = bool(arrays["frame_goal_present"][index])
    return ReplayFrame(
        control_step=int(arrays["frame_control_step"][index]),
        camera_images={
            "agentview_image": arrays["frame_agentview_image"][index],
            "robot0_eye_in_hand_image": arrays["frame_robot0_eye_in_hand_image"][index],
        },
        low_dimensional=_low_dimensional(
            simulator_time=float(arrays["frame_simulator_time"][index]),
            policy_state=arrays["frame_policy_state"][index],
            eef_position=arrays["frame_eef_position"][index],
            eef_quaternion_xyzw=arrays["frame_eef_quaternion_xyzw"][index],
            object_position=arrays["frame_primary_object_position"][index],
            object_quaternion_wxyz=arrays["frame_primary_object_quaternion_wxyz"][
                index
            ],
            gripper_qpos=arrays["frame_gripper_qpos"][index],
            gripper_qvel=arrays["frame_gripper_qvel"][index],
            goal_present=goal_present,
            goal_position=(
                arrays["frame_goal_position"][index] if goal_present else None
            ),
            goal_quaternion_wxyz=(
                arrays["frame_goal_quaternion_wxyz"][index] if goal_present else None
            ),
            primary_gripper_contact=bool(
                arrays["frame_primary_gripper_contact"][index]
            ),
            primary_grasped=bool(arrays["frame_primary_grasped"][index]),
            task_success=bool(arrays["frame_task_success"][index]),
            phase=phase,
            task_predicate_names=predicate_names,
            task_predicates=arrays["frame_task_predicates"][index],
        ),
        success=bool(arrays["frame_task_success"][index]),
    )


def factual_replay_from_artifact(artifact: RolloutArtifact) -> FactualReplay:
    """Convert one already-validated valid rollout into an immutable replay."""

    if not isinstance(artifact, RolloutArtifact):
        raise ScoringRuntimeError("artifact must be a validated RolloutArtifact")
    if not artifact.valid_reset or artifact.action_count < 1:
        raise ScoringRuntimeError("replay scoring requires a valid nonempty artifact")
    arrays = artifact.arrays
    frame_count = int(arrays["frame_control_step"].shape[0])
    predicate_names = tuple(artifact.metadata["capture"]["task_predicate_names"])
    if predicate_names != tuple(sorted(predicate_names)):
        raise ScoringRuntimeError("artifact task predicate names are not sorted")
    if arrays["frame_task_predicates"].shape[1] != len(predicate_names):
        raise ScoringRuntimeError("task predicate names and saved vectors disagree")
    frames = tuple(
        _artifact_frame(artifact, index, predicate_names)
        for index in range(frame_count)
    )
    outcome = artifact.metadata["outcome"]
    episode = artifact.metadata["episode"]
    return FactualReplay(
        episode_id=artifact.episode_id,
        split=str(episode["split"]),
        frames=frames,
        actions=np.asarray(arrays["actions"], dtype=np.float32).copy(),
        terminated=np.asarray(arrays["terminated"], dtype=np.bool_).copy(),
        truncated=np.asarray(arrays["truncated"], dtype=np.bool_).copy(),
        terminal_status=str(outcome["status"]),
        success=bool(outcome["success"]),
    )


def _raw_trace_frame(frame: Any) -> ReplayFrame:
    predicate_names = tuple(sorted(frame.task_predicates))
    goal_present = frame.goal_position is not None
    if goal_present != (frame.goal_quaternion_wxyz is not None):
        raise ScoringRuntimeError("runtime goal position/quaternion presence disagrees")
    return ReplayFrame(
        control_step=int(frame.control_step),
        camera_images={
            "agentview_image": frame.raw_observation["agentview_image"],
            "robot0_eye_in_hand_image": frame.raw_observation[
                "robot0_eye_in_hand_image"
            ],
        },
        low_dimensional=_low_dimensional(
            simulator_time=float(frame.simulator_time),
            policy_state=frame.policy_state,
            eef_position=frame.eef_position,
            eef_quaternion_xyzw=frame.eef_quaternion_xyzw,
            object_position=frame.primary_object_position,
            object_quaternion_wxyz=frame.primary_object_quaternion_wxyz,
            gripper_qpos=frame.gripper_qpos,
            gripper_qvel=frame.gripper_qvel,
            goal_present=goal_present,
            goal_position=frame.goal_position,
            goal_quaternion_wxyz=frame.goal_quaternion_wxyz,
            primary_gripper_contact=bool(frame.primary_gripper_contact),
            primary_grasped=bool(frame.primary_grasped),
            task_success=bool(frame.task_success),
            phase=str(frame.phase),
            task_predicate_names=predicate_names,
            task_predicates=np.asarray(
                [frame.task_predicates[name] for name in predicate_names],
                dtype=np.bool_,
            ),
        ),
        success=bool(frame.task_success),
    )


def _libero_runtime_module() -> Any:
    from . import libero_runtime

    return libero_runtime


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - rollout-host integration
        raise ScoringRuntimeError("PyTorch is required for SmolVLA scoring") from exc
    return torch


def _clone_batch(value: Any) -> Any:
    clone = getattr(value, "detach", None)
    if callable(clone) and callable(getattr(value, "clone", None)):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    return copy.deepcopy(value)


def _runtime_value_equal(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second, equal_nan=True)
    if isinstance(first, Mapping):
        return set(first) == set(second) and all(
            _runtime_value_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _runtime_value_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, deque):
        return first.maxlen == second.maxlen and _runtime_value_equal(
            tuple(first), tuple(second)
        )
    equal = getattr(first, "equal", None)
    if callable(equal):
        return bool(equal(second))
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _compressed_activation_bytes(activation: NDArray[np.float32]) -> int:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(activation, dtype=np.float32),
        allow_pickle=False,
    )
    return len(zlib.compress(buffer.getvalue(), level=9))


def _capture_process_rng(torch: Any) -> tuple[Any, tuple[Any, ...], Any, Any | None]:
    cuda = None
    if torch.cuda.is_available():
        cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        cuda,
    )


def _restore_process_rng(torch: Any, state: tuple[Any, ...]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(list(state[3]))


@dataclass
class _DefaultTimerBackend:
    device: Any
    torch: Any

    @property
    def cuda(self) -> bool:
        return getattr(self.device, "type", str(self.device).split(":")[0]) == "cuda"

    def synchronize(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)

    def reset_peak_memory_stats(self) -> None:
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory_allocated(self) -> int:
        return int(self.torch.cuda.memory_allocated(self.device)) if self.cuda else 0

    def max_memory_allocated(self) -> int:
        return (
            int(self.torch.cuda.max_memory_allocated(self.device)) if self.cuda else 0
        )

    def create_event(self) -> Any | None:
        return self.torch.cuda.Event(enable_timing=True) if self.cuda else None

    def record_event(self, event: Any | None) -> None:
        if event is not None:
            event.record()

    def elapsed_ms(self, start: Any | None, end: Any | None) -> float:
        return float(start.elapsed_time(end)) if start is not None else 0.0

    def perf_counter_ns(self) -> int:
        return time.perf_counter_ns()


class SmolVLAScoringAdapter:
    """Pinned private-inference bridge implementing :class:`ScoringAdapter`."""

    def __init__(
        self,
        episode: RawLiberoEpisode,
        policy_runtime: LoadedPolicyRuntime,
        artifact: RolloutArtifact,
        probe: ProbeArtifact,
        instrumentation: SmolVLAInstrumentation,
        *,
        reset_seed: int,
        original_condition: ConditionSpec,
        timer_backend: Any | None = None,
    ) -> None:
        self.episode = episode
        self.policy_runtime = policy_runtime
        self.artifact = artifact
        self.factual = factual_replay_from_artifact(artifact)
        self.probe = probe
        self.instrumentation = instrumentation
        self.reset_seed = int(reset_seed)
        self.original_condition = original_condition
        self._active_transform: ScoringTransform | None = None
        self._original_activation_by_noise: dict[int, Any] = {}
        self._peak_baseline: int | None = None
        self._validate_runtime()
        torch = _torch_module()
        self.device = self._policy_device(torch)
        self.timer = timer_backend or _DefaultTimerBackend(self.device, torch)

    def _validate_runtime(self) -> None:
        episode = self.episode
        policy = self.policy_runtime.policy
        metadata = self.artifact.metadata
        manifested = metadata["episode"]
        condition = metadata["condition"]
        task = metadata["task"]
        if (
            getattr(episode, "_has_reset", True)
            or getattr(episode, "primary_object_name", object()) is not None
        ):
            raise ScoringRuntimeError("scoring requires a fresh RawLiberoEpisode")
        runtime_task = episode.task
        if (
            runtime_task.suite,
            runtime_task.task_id,
            runtime_task.rank,
            runtime_task.language,
            runtime_task.primary_object,
            runtime_task.planar_symmetry_order,
        ) != (
            task["suite"],
            task["task_id"],
            task["rank"],
            task["language"],
            task["primary_object"],
            task["planar_symmetry_order"],
        ):
            raise ScoringRuntimeError("runtime task differs from the raw artifact")
        if self.reset_seed != int(manifested["reset_seed"]):
            raise ScoringRuntimeError("reset seed differs from the raw artifact")
        if (
            self.original_condition.name != condition["name"]
            or self.original_condition.family != condition["family"]
            or self.original_condition.index != condition["index"]
            or dict(self.original_condition.parameters) != dict(condition["parameters"])
        ):
            raise ScoringRuntimeError(
                "original condition differs from the raw artifact"
            )
        execution = episode.execution
        if (
            execution.n_action_steps != 1
            or execution.max_steps != metadata["execution"]["max_steps"]
            or execution.reset_noop_steps != metadata["execution"]["reset_noop_steps"]
        ):
            raise ScoringRuntimeError("runtime execution config differs from artifact")
        validity_config = episode.validity_config
        actual_validity = {
            name: getattr(validity_config, name, object())
            for name in FROZEN_VALIDITY_VALUES
        }
        if actual_validity != dict(FROZEN_VALIDITY_VALUES):
            raise ScoringRuntimeError(
                "runtime counterfactual validity config differs from frozen values"
            )
        config = getattr(policy, "config", None)
        if config is None:
            raise ScoringRuntimeError("policy has no SmolVLA config")
        model_config = getattr(getattr(policy, "model", None), "config", None)
        if (
            getattr(config, "n_action_steps", None) != 1
            or getattr(config, "chunk_size", None) != 50
            or getattr(model_config, "num_steps", None) != 10
        ):
            raise ScoringRuntimeError(
                "policy action/flow dimensions differ from pinned values"
            )
        if getattr(config, "adapt_to_pi_aloha", None) is not False:
            raise ScoringRuntimeError("scoring requires adapt_to_pi_aloha=False")
        action_feature = getattr(config, "action_feature", None)
        if tuple(getattr(action_feature, "shape", ())) != (ACTION_DIMENSION,):
            raise ScoringRuntimeError("policy action feature must have shape (7,)")
        rtc_enabled = getattr(policy, "_rtc_enabled", None)
        if not callable(rtc_enabled) or rtc_enabled():
            raise ScoringRuntimeError("RTC must be explicitly disabled")
        rtc_config = getattr(config, "rtc_config", None)
        if rtc_config is not None and getattr(rtc_config, "enabled", True):
            raise ScoringRuntimeError("policy RTC config must be disabled")
        model_rtc_enabled = getattr(policy.model, "_rtc_enabled", None)
        if callable(model_rtc_enabled) and model_rtc_enabled():
            raise ScoringRuntimeError("flow-model RTC must be disabled")
        if not callable(getattr(policy, "_get_action_chunk", None)):
            raise ScoringRuntimeError("policy lacks pinned private _get_action_chunk")
        if getattr(policy, "training", True) or getattr(policy.model, "training", True):
            raise ScoringRuntimeError(
                "policy and flow model must already be in eval mode"
            )
        if not hasattr(policy, "_queues"):
            raise ScoringRuntimeError("policy queues are not observable")
        snapshots = getattr(self.policy_runtime, "snapshots", None)
        lock = getattr(snapshots, "lock", None)
        if lock is None:
            raise ScoringRuntimeError(
                "loaded policy runtime has no immutable input lock"
            )
        model_metadata = metadata["model"]
        if (
            lock.policy_revision != manifested["policy_revision"]
            or lock.base_vlm_revision != model_metadata["base_vlm_revision"]
            or lock.policy_num_flow_steps != 10
            or lock.policy_chunk_size != 50
            or lock.policy_n_action_steps != 1
            or lock.lerobot_commit != FROZEN_LEROBOT_COMMIT
            or lock.lerobot_python_sha256 != FROZEN_LEROBOT_PYTHON_SHA256
        ):
            raise ScoringRuntimeError(
                "loaded model input lock differs from the artifact"
            )
        if tuple(getattr(self.policy_runtime, "original_state_shape", ())) != (
            6,
        ) or tuple(getattr(self.policy_runtime, "runtime_state_shape", ())) != (8,):
            raise ScoringRuntimeError("loaded policy state-shape amendment drifted")
        normalization_shapes = getattr(
            self.policy_runtime, "normalization_state_shapes", {}
        )
        vector_shapes = {
            name: tuple(shape)
            for name, shape in normalization_shapes.items()
            if name != "count"
        }
        if not {"mean", "std"}.issubset(vector_shapes) or any(
            shape != (8,) for shape in vector_shapes.values()
        ):
            raise ScoringRuntimeError(
                "policy state normalizer is not eight-dimensional"
            )
        if getattr(self.instrumentation, "is_installed", True):
            raise ScoringRuntimeError("instrumentation must not already be installed")
        if self.instrumentation.flow_model is not policy.model:
            raise ScoringRuntimeError("instrumentation is attached to another model")
        if getattr(self.instrumentation, "capture_steps", None) != frozenset((0, 5)):
            raise ScoringRuntimeError("instrumentation capture steps must be 0 and 5")
        if getattr(self.instrumentation, "expected_action_tokens", None) != 50:
            raise ScoringRuntimeError("instrumentation must capture 50 action tokens")
        if not callable(getattr(self.instrumentation, "_append_record", None)):
            raise ScoringRuntimeError(
                "pinned instrumentation lacks selected-only capture interception"
            )
        from .probes import ProbeArtifact

        if not isinstance(self.probe, ProbeArtifact):
            raise ScoringRuntimeError("probe must be a validated ProbeArtifact")
        self.location, self.denoising_step = candidate_target(self.probe.candidate)
        coefficient = np.asarray(self.probe.model.coefficient, dtype=np.float64)
        feature_center = np.asarray(self.probe.model.feature_center, dtype=np.float64)
        target_center = np.asarray(self.probe.model.target_center, dtype=np.float64)
        if (
            coefficient.ndim != 2
            or coefficient.shape[0] != 2
            or feature_center.shape != (coefficient.shape[1],)
            or target_center.shape != (2,)
            or not np.isfinite(coefficient).all()
            or not np.isfinite(feature_center).all()
            or not np.isfinite(target_center).all()
            or np.linalg.matrix_rank(coefficient) != 2
        ):
            raise ScoringRuntimeError(
                "frozen probe must be finite and rank exactly two"
            )
        if self.probe.model.symmetry_order != task["planar_symmetry_order"]:
            raise ScoringRuntimeError("probe and task symmetry orders differ")
        self._coefficient = coefficient.copy()
        self._feature_center = feature_center.copy()
        self._target_center = target_center.copy()

    def _policy_device(self, torch: Any) -> Any:
        policy = self.policy_runtime.policy
        candidate = getattr(policy, "device", None)
        if candidate is None:
            try:
                candidate = next(policy.parameters()).device
            except (AttributeError, StopIteration, TypeError) as exc:
                candidate = getattr(policy.config, "device", None)
                if candidate is None:
                    raise ScoringRuntimeError("cannot resolve policy device") from exc
        return torch.device(candidate)

    def reset_replay(self) -> ReplayFrame:
        torch = _torch_module()
        rng = _capture_process_rng(torch)
        try:
            reset = self.episode.reset(
                seed=self.reset_seed, condition=self.original_condition
            )
        finally:
            _restore_process_rng(torch, rng)
        if not reset.validity.valid:
            raise ScoringRuntimeError("factual replay reset became invalid")
        return _raw_trace_frame(reset.frame)

    def step_replay(self, action: NDArray[np.float32]) -> ReplayTransition:
        step = self.episode.step(np.asarray(action, dtype=np.float32).copy())
        return ReplayTransition(
            frame=_raw_trace_frame(step.frame),
            terminated=bool(step.terminated),
            truncated=bool(step.truncated),
        )

    def current_frame(self) -> ReplayFrame:
        return _raw_trace_frame(self.episode.current_raw_trace())

    def capture_snapshot(self) -> SimulatorSnapshot:
        runtime = _libero_runtime_module()
        snapshot = runtime.capture_simulator_snapshot(self.episode.wrapper)
        return SimulatorSnapshot(
            state=np.asarray(snapshot.state, dtype=np.float64),
            camera_positions=np.asarray(snapshot.camera_positions, dtype=np.float64),
            camera_quaternions=np.asarray(
                snapshot.camera_quaternions, dtype=np.float64
            ),
        )

    def restore_snapshot(self, snapshot: SimulatorSnapshot) -> None:
        runtime = _libero_runtime_module()
        restored = runtime.SimulatorSnapshot(
            state=snapshot.state.copy(),
            camera_positions=snapshot.camera_positions.copy(),
            camera_quaternions=snapshot.camera_quaternions.copy(),
        )
        runtime.restore_simulator_snapshot(self.episode.wrapper, restored)
        self._active_transform = None

    def apply_transform(self, transform: ScoringTransform) -> None:
        runtime = _libero_runtime_module()
        if transform.family == "camera_render":
            condition = ConditionSpec(
                name=transform.name,
                family=transform.family,
                parameters=MappingProxyType({"yaw": transform.value, "lateral_m": 0.0}),
            )
        elif transform.family == "object_pose":
            condition = ConditionSpec(
                name=transform.name,
                family=transform.family,
                parameters=MappingProxyType({"yaw": transform.value}),
            )
        else:
            raise ScoringRuntimeError(
                "simulator bridge received a photometric transform"
            )
        runtime.apply_condition(
            self.episode.wrapper,
            condition,
            primary_object_name=self.episode.primary_object_name,
        )
        self._active_transform = transform

    def transformed_validity(self) -> TransformValidity:
        if self._active_transform is None:
            raise ScoringRuntimeError("no simulator transform is active")
        runtime = _libero_runtime_module()
        reasons = runtime.counterfactual_validity_reasons(
            self.episode.wrapper,
            self.episode.primary_object_name,
            self.episode.validity_config,
            check_object_edit=self._active_transform.family == "object_pose",
        )
        allowed = {
            "nonfinite_counterfactual_state",
            "counterfactual_primary_object_penetration",
            "counterfactual_primary_object_outside_workspace",
        }
        if set(reasons) - allowed:
            raise ScoringRuntimeError(
                f"unknown counterfactual validity reasons {reasons}"
            )
        finite = "nonfinite_counterfactual_state" not in reasons
        if self._active_transform.family == "camera_render":
            return TransformValidity(finite, True, True)
        return TransformValidity(
            finite,
            "counterfactual_primary_object_penetration" not in reasons,
            "counterfactual_primary_object_outside_workspace" not in reasons,
        )

    def process_observation(self, frame: ReplayFrame) -> Any:
        raw = _clone_batch(self.episode.current_raw_trace().raw_observation)
        low = frame.low_dimensional
        raw["agentview_image"] = frame.camera_images["agentview_image"].copy()
        raw["robot0_eye_in_hand_image"] = frame.camera_images[
            "robot0_eye_in_hand_image"
        ].copy()
        raw["robot0_eef_pos"] = np.asarray(low["eef_position"], dtype=np.float64).copy()
        raw["robot0_eef_quat"] = np.asarray(
            low["eef_quaternion_xyzw"], dtype=np.float64
        ).copy()
        raw["robot0_gripper_qpos"] = np.asarray(
            low["gripper_qpos"], dtype=np.float64
        ).copy()
        raw["robot0_gripper_qvel"] = np.asarray(
            low["gripper_qvel"], dtype=np.float64
        ).copy()
        formatted = self.episode.format_observation(raw)
        return self.policy_runtime.preprocess_observation(
            dict(formatted), task=self.episode.task.language
        )

    def noise_for_seed(self, seed: int) -> Any:
        if isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ScoringRuntimeError("noise seed must be in [0, 2**63)")
        torch = _torch_module()
        maximum = getattr(self.policy_runtime.policy.config, "max_action_dim", None)
        if type(maximum) is not int or maximum < ACTION_DIMENSION:
            raise ScoringRuntimeError("policy max_action_dim is invalid")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return torch.randn(
            (1, 50, maximum),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def begin_score_state(self) -> None:
        self.timer.synchronize()
        self.timer.reset_peak_memory_stats()
        self._peak_baseline = int(self.timer.memory_allocated())
        self._original_activation_by_noise.clear()

    def validate_content_links(self, links: ContentLinks) -> None:
        """Bind the replay and fitted probe bytes to the published score sidecar."""

        if not isinstance(links, ContentLinks):
            raise ScoringRuntimeError("score links must be a validated ContentLinks")
        if links.raw_metadata_sha256 != self.artifact.hashes.metadata_sha256:
            raise ScoringRuntimeError("score raw metadata hash link is stale")
        if links.raw_trajectory_sha256 != self.artifact.hashes.trajectory_sha256:
            raise ScoringRuntimeError("score raw trajectory hash link is stale")
        if links.probe_sha256 != self.probe.sha256():
            raise ScoringRuntimeError("score probe hash link is stale")

    def policy_queue_state(self) -> Any:
        try:
            return copy.deepcopy(self.policy_runtime.policy._queues)
        except (TypeError, ValueError) as exc:
            raise ScoringRuntimeError("policy queues cannot be deep-copied") from exc

    def intervention_available(self, activation: NDArray[np.float32]) -> bool:
        value = np.asarray(activation, dtype=np.float64)
        if value.shape != self._feature_center.shape or not np.isfinite(value).all():
            raise ScoringRuntimeError("activation does not match the frozen probe")
        if np.linalg.matrix_rank(self._coefficient) != 2:
            raise ScoringRuntimeError("probe coefficient lost numerical rank two")
        raw = (value - self._feature_center) @ self._coefficient.T + self._target_center
        return bool(np.linalg.norm(raw) > 1e-12)

    def _selected_activation(self, *, patched: bool) -> Any:
        records = self.instrumentation.activations(
            self.location, denoising_step=self.denoising_step
        )
        if len(records) != 1:
            raise ScoringRuntimeError(
                f"instrumented forward emitted {len(records)} selected activations"
            )
        calls = self.instrumentation.calls
        if len(calls) != 11:
            raise ScoringRuntimeError(
                "one private action chunk must make 11 model calls"
            )
        phases = [
            getattr(getattr(call, "phase", None), "value", getattr(call, "phase", None))
            for call in calls
        ]
        steps = [getattr(call, "denoising_step", None) for call in calls]
        if phases != ["prefix_cache", *(["denoising"] * 10)] or steps != [
            None,
            *range(10),
        ]:
            raise ScoringRuntimeError(
                "private action chunk call phases are not prefix plus denoising 0..9"
            )
        record = records[0]
        if bool(record.patched) != patched:
            raise ScoringRuntimeError("selected activation patch marker disagrees")
        value = record.value
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1 or value.numel() != self._feature_center.size:
            raise ScoringRuntimeError("selected activation has the wrong hidden width")
        return value.detach().to(device="cpu", dtype=_torch_module().float32).clone()

    def _probe_shift(self, original: Any, degrees: float) -> Any:
        torch = _torch_module()
        from .instrumentation import minimum_norm_circular_probe_shift

        return minimum_norm_circular_probe_shift(
            torch.as_tensor(self._coefficient),
            original,
            torch.as_tensor(self._feature_center),
            torch.as_tensor(self._target_center),
            scaled_angle_radians=(
                self.probe.model.symmetry_order * math.radians(float(degrees))
            ),
        )

    @contextmanager
    def _capture_selected_only(self):
        """Suppress all non-selected activation copies during this one forward."""

        original_append = self.instrumentation._append_record

        def selected_append(**kwargs: Any) -> None:
            if kwargs.get("location") != self.location:
                return
            frame = kwargs.get("frame")
            step = getattr(frame, "denoising_step", None)
            if step != self.denoising_step:
                return
            original_append(**kwargs)

        self.instrumentation._append_record = selected_append
        try:
            yield
        finally:
            if self.instrumentation._append_record is not selected_append:
                raise ScoringRuntimeError(
                    "instrumentation capture method changed concurrently"
                )
            self.instrumentation._append_record = original_append

    def predict_action_chunk(
        self,
        processed_observation: Any,
        *,
        noise: Any,
        intervention_degrees: float | None,
    ) -> ScoredCall:
        if self._peak_baseline is None:
            raise ScoringRuntimeError("begin_score_state must precede inference")
        torch = _torch_module()
        expected_noise_shape = (
            1,
            50,
            self.policy_runtime.policy.config.max_action_dim,
        )
        if (
            not isinstance(noise, torch.Tensor)
            or tuple(noise.shape) != expected_noise_shape
            or noise.dtype != torch.float32
            or noise.device != self.device
            or not torch.isfinite(noise).all().item()
        ):
            raise ScoringRuntimeError("explicit noise tensor violates the pinned shape")
        patch_context: Any = nullcontext()
        patched = intervention_degrees is not None
        queue_before = self.policy_queue_state()
        if patched:
            original = self._original_activation_by_noise.get(id(noise))
            if original is None:
                raise ScoringRuntimeError(
                    "intervention has no cached original for this noise identity"
                )
            if not self.intervention_available(
                original.detach().cpu().numpy().astype(np.float32, copy=False)
            ):
                raise ScoringRuntimeError(
                    "zero probe resultant cannot be intervened on"
                )
            shift = self._probe_shift(original, float(intervention_degrees))

        self.timer.synchronize()
        start_event = self.timer.create_event()
        end_event = self.timer.create_event()
        self.timer.record_event(start_event)
        wall_start = int(self.timer.perf_counter_ns())
        self.instrumentation.clear()
        try:
            with (
                torch.inference_mode(),
                self._capture_selected_only(),
                self.instrumentation,
            ):
                if patched:
                    patch_context = self.instrumentation.patch(
                        self.location,
                        shift,
                        denoising_step=self.denoising_step,
                    )
                with patch_context:
                    normalized = self.policy_runtime.policy._get_action_chunk(
                        _clone_batch(processed_observation), noise=noise
                    )
            if (
                not isinstance(normalized, torch.Tensor)
                or tuple(normalized.shape) != (1, 50, ACTION_DIMENSION)
                or normalized.dtype != torch.float32
                or normalized.device != self.device
                or not torch.isfinite(normalized).all().item()
            ):
                raise ScoringRuntimeError(
                    "private policy chunk must be finite float32 shape (1, 50, 7)"
                )
            postprocessed = self.policy_runtime.postprocessor(normalized)
            if isinstance(postprocessed, torch.Tensor):
                actions = (
                    postprocessed.detach().to(device="cpu", dtype=torch.float32).numpy()
                )
            else:
                actions = np.asarray(postprocessed)
            if actions.shape == (1, 50, ACTION_DIMENSION):
                actions = actions[0]
            if actions.dtype != np.float32 or actions.shape != (50, ACTION_DIMENSION):
                raise ScoringRuntimeError(
                    "postprocessor must emit float32 shape (1,50,7) or (50,7)"
                )
            if not np.isfinite(actions).all():
                raise ScoringRuntimeError("postprocessed action chunk is nonfinite")
            activation_tensor = self._selected_activation(patched=patched)
            activation = activation_tensor.numpy().astype(np.float32, copy=True)
            if not patched:
                self._original_activation_by_noise.setdefault(
                    id(noise), activation_tensor.clone()
                )
        finally:
            self.timer.record_event(end_event)
            self.timer.synchronize()
        wall_end = int(self.timer.perf_counter_ns())
        peak = int(self.timer.max_memory_allocated())
        if not _runtime_value_equal(queue_before, self.policy_queue_state()):
            raise ScoringRuntimeError(
                "private action-chunk inference mutated policy queues"
            )
        logical_bytes = int(activation.nbytes)
        cost = CallCost(
            cuda_event_ms=float(self.timer.elapsed_ms(start_event, end_event)),
            wall_time_ns=wall_end - wall_start,
            forward_count=1,
            intervention_count=int(patched),
            peak_allocated_bytes=peak,
            incremental_peak_allocated_bytes=max(0, peak - self._peak_baseline),
            logical_activation_bytes=logical_bytes,
            compressed_activation_bytes=_compressed_activation_bytes(activation),
            cuda_synchronized_before_after=True,
        )
        return ScoredCall(actions=actions.copy(), activation=activation, cost=cost)


__all__ = [
    "CANDIDATE_TARGETS",
    "PHASES",
    "ScoringRuntimeError",
    "SmolVLAScoringAdapter",
    "candidate_target",
    "factual_replay_from_artifact",
]
