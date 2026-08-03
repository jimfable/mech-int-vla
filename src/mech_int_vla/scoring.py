"""Deterministic replay orchestration for raw counterfactual score primitives.

The GPU and simulator bindings intentionally live behind :class:`ScoringAdapter`.
This module owns the protocol checks, not those optional dependencies: it replays
an already-recorded factual trajectory, supplies explicit common-noise objects to
every policy call, restores simulator snapshots after temporary edits, and only
then publishes a content-addressed sidecar.

The sidecar contains raw action chunks, the one Calibration-selected activation,
availability masks, seeds, and per-call costs.  Feature reduction belongs in a
separate module.  A replay, restoration, queue-state, or RNG mismatch leaves no
published directory.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import io
import json
import os
import random
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ParamSpec, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .determinism import hash_seed

SCORING_SCHEMA_VERSION = 1
ACTION_DIMENSION = 7
CHUNK_ACTIONS = 10
ORIGINAL_DRAWS = 8
COMMON_DRAWS = 4
SCORE_STRIDE_STEPS = 5
INTERVENTION_DEGREES = (-10.0, 10.0)
CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
COST_FIELDS = (
    "cuda_event_ms",
    "wall_time_ns",
    "forward_count",
    "intervention_count",
    "peak_allocated_bytes",
    "incremental_peak_allocated_bytes",
    "logical_activation_bytes",
    "compressed_activation_bytes",
)


class ScoringError(RuntimeError):
    """Raised before publication when the replay contract is violated."""


class ScoringValidationError(ValueError):
    """Raised when a published sidecar is malformed or has been tampered with."""


@dataclass(frozen=True)
class ScoringTransform:
    """One frozen transform in its protocol-significant order."""

    name: str
    family: str
    value: float
    action_transform: str | None = None
    terminology: str | None = None


FROZEN_TRANSFORMS = (
    ScoringTransform("brightness_0_85", "photometric", 0.85, "identity"),
    ScoringTransform("brightness_1_15", "photometric", 1.15, "identity"),
    ScoringTransform("camera_yaw_neg_3", "camera_render", -3.0, "identity"),
    ScoringTransform("camera_yaw_pos_3", "camera_render", 3.0, "identity"),
    ScoringTransform(
        "object_yaw_neg_15",
        "object_pose",
        -15.0,
        terminology="counterfactual_action_drift",
    ),
    ScoringTransform(
        "object_yaw_pos_15",
        "object_pose",
        15.0,
        terminology="counterfactual_action_drift",
    ),
)


@dataclass(frozen=True)
class ContentLinks:
    """Immutable hashes of every input that determines a score sidecar."""

    raw_metadata_sha256: str
    raw_trajectory_sha256: str
    probe_sha256: str
    config_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not _is_sha256(value):
                raise ScoringError(f"{name} must be a lowercase SHA-256")

    def as_dict(self) -> dict[str, str]:
        return {
            "raw_metadata_sha256": self.raw_metadata_sha256,
            "raw_trajectory_sha256": self.raw_trajectory_sha256,
            "probe_sha256": self.probe_sha256,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True)
class ReplayFrame:
    """Factual pre-action state sufficient for exact replay verification."""

    control_step: int
    camera_images: Mapping[str, NDArray[np.uint8]]
    low_dimensional: Mapping[str, NDArray[Any]]
    success: bool

    def __post_init__(self) -> None:
        images: dict[str, NDArray[np.uint8]] = {}
        if set(self.camera_images) != set(CAMERA_KEYS):
            raise ScoringError(f"camera_images must contain exactly {CAMERA_KEYS}")
        image_shape: tuple[int, ...] | None = None
        for key in CAMERA_KEYS:
            image = np.asarray(self.camera_images[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ScoringError(f"{key} must be an HxWx3 uint8 image")
            if image_shape is None:
                image_shape = image.shape
            elif image.shape != image_shape:
                raise ScoringError("the two factual camera images must share a shape")
            frozen = image.copy()
            frozen.flags.writeable = False
            images[key] = frozen
        low_dimensional: dict[str, NDArray[Any]] = {}
        if not self.low_dimensional:
            raise ScoringError("low_dimensional must contain replay state")
        for key, value in self.low_dimensional.items():
            if not isinstance(key, str) or not key:
                raise ScoringError(
                    "low-dimensional state keys must be nonempty strings"
                )
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise ScoringError(f"low-dimensional field {key!r} is not numeric")
            frozen = array.copy()
            frozen.flags.writeable = False
            low_dimensional[key] = frozen
        if self.control_step < 0:
            raise ScoringError("control_step must be nonnegative")
        object.__setattr__(self, "camera_images", MappingProxyType(images))
        object.__setattr__(self, "low_dimensional", MappingProxyType(low_dimensional))


@dataclass(frozen=True)
class FactualReplay:
    """The complete saved trajectory consumed by deterministic replay."""

    episode_id: str
    split: str
    frames: tuple[ReplayFrame, ...]
    actions: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    terminal_status: str
    success: bool

    def __post_init__(self) -> None:
        if Path(self.episode_id).name != self.episode_id or not self.episode_id:
            raise ScoringError("episode_id is not a safe directory name")
        if Path(self.split).name != self.split or not self.split:
            raise ScoringError("split is not a safe directory name")
        actions = np.asarray(self.actions)
        terminated = np.asarray(self.terminated)
        truncated = np.asarray(self.truncated)
        if actions.dtype != np.float32 or actions.ndim != 2:
            raise ScoringError("factual actions must be a float32 matrix")
        if actions.shape[1:] != (ACTION_DIMENSION,) or not np.isfinite(actions).all():
            raise ScoringError("factual actions must be finite T-by-7")
        if terminated.dtype != np.bool_ or terminated.shape != (actions.shape[0],):
            raise ScoringError("terminated flags are not aligned to factual actions")
        if truncated.dtype != np.bool_ or truncated.shape != (actions.shape[0],):
            raise ScoringError("truncated flags are not aligned to factual actions")
        if len(self.frames) != actions.shape[0] + 1:
            raise ScoringError("factual replay must have one more frame than action")
        steps = [frame.control_step for frame in self.frames]
        if steps != list(range(len(self.frames))):
            raise ScoringError(
                "factual frame control steps must be sequential from zero"
            )
        if actions.shape[0] == 0:
            raise ScoringError("score sidecars require a valid, nonempty episode")
        if np.any(terminated & truncated):
            raise ScoringError("one factual action cannot be terminated and truncated")
        if np.any(terminated[:-1]) or np.any(truncated[:-1]):
            raise ScoringError(
                "factual terminal flags may only occur on the last action"
            )
        if not (terminated[-1] or truncated[-1]):
            raise ScoringError("factual replay does not end in a terminal state")
        if bool(self.frames[-1].success) != bool(self.success):
            raise ScoringError("factual terminal frame and outcome success disagree")
        if any(frame.success for frame in self.frames[:-1]):
            raise ScoringError("factual replay continued after task success")
        expected_status = _terminal_status(
            success=bool(self.success),
            terminated=bool(terminated[-1]),
            truncated=bool(truncated[-1]),
        )
        if self.terminal_status != expected_status:
            raise ScoringError("factual terminal_status disagrees with terminal flags")
        actions = actions.copy()
        terminated = terminated.copy()
        truncated = truncated.copy()
        actions.flags.writeable = False
        terminated.flags.writeable = False
        truncated.flags.writeable = False
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)


@dataclass(frozen=True)
class ReplayTransition:
    """One replayed transition returned after issuing a saved factual action."""

    frame: ReplayFrame
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Exact dynamic and camera state used to prove edit restoration."""

    state: NDArray[np.float64]
    camera_positions: NDArray[np.float64]
    camera_quaternions: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("state", "camera_positions", "camera_quaternions"):
            array = np.asarray(getattr(self, name))
            if array.dtype != np.float64 or array.size == 0:
                raise ScoringError(f"snapshot {name} must be nonempty float64")
            if not np.isfinite(array).all():
                raise ScoringError(f"snapshot {name} must be finite")
            copied = array.copy()
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class TransformValidity:
    """Frozen finite/penetration/workspace decision for a temporary edit."""

    finite: bool
    penetration_ok: bool
    workspace_ok: bool

    @property
    def available(self) -> bool:
        return self.finite and self.penetration_ok and self.workspace_ok


@dataclass(frozen=True)
class CallCost:
    """Physical and logical cost recorded by one synchronized policy call."""

    cuda_event_ms: float
    wall_time_ns: int
    forward_count: int
    intervention_count: int
    peak_allocated_bytes: int
    incremental_peak_allocated_bytes: int
    logical_activation_bytes: int
    compressed_activation_bytes: int
    cuda_synchronized_before_after: bool = True

    def as_array(self) -> NDArray[np.float64]:
        values = np.asarray(
            [
                self.cuda_event_ms,
                self.wall_time_ns,
                self.forward_count,
                self.intervention_count,
                self.peak_allocated_bytes,
                self.incremental_peak_allocated_bytes,
                self.logical_activation_bytes,
                self.compressed_activation_bytes,
            ],
            dtype=np.float64,
        )
        if not self.cuda_synchronized_before_after:
            raise ScoringError("policy cost interval was not CUDA-synchronized")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ScoringError("policy call costs must be finite and nonnegative")
        integer_values = values[1:]
        if not np.equal(integer_values, np.floor(integer_values)).all():
            raise ScoringError("count, wall-time, and byte costs must be integers")
        if np.any(integer_values > 2**53):
            raise ScoringError("integer costs exceed exact float64 representation")
        return values


@dataclass(frozen=True)
class ScoredCall:
    """Postprocessed unnormalized actions and the selected representation."""

    actions: NDArray[Any]
    activation: NDArray[Any]
    cost: CallCost


@runtime_checkable
class ScoringAdapter(Protocol):
    """Runtime boundary implemented by the pinned LIBERO/SmolVLA adapter.

    ``noise_for_seed`` must construct the complete explicit flow-noise tensor.
    The orchestrator reuses the same returned objects for common-noise calls.
    ``predict_action_chunk`` must bypass the action queue, perform no warmup, and
    return postprocessed *unnormalized* actions.
    """

    def reset_replay(self) -> ReplayFrame: ...

    def step_replay(self, action: NDArray[np.float32]) -> ReplayTransition: ...

    def capture_snapshot(self) -> SimulatorSnapshot: ...

    def restore_snapshot(self, snapshot: SimulatorSnapshot) -> None: ...

    def apply_transform(self, transform: ScoringTransform) -> None: ...

    def transformed_validity(self) -> TransformValidity: ...

    def current_frame(self) -> ReplayFrame: ...

    def process_observation(self, frame: ReplayFrame) -> Any: ...

    def noise_for_seed(self, seed: int) -> Any: ...

    def predict_action_chunk(
        self,
        processed_observation: Any,
        *,
        noise: Any,
        intervention_degrees: float | None,
    ) -> ScoredCall: ...

    def intervention_available(self, activation: NDArray[np.float32]) -> bool: ...

    def begin_score_state(self) -> None: ...

    def policy_queue_state(self) -> Any | None: ...

    def validate_content_links(self, links: ContentLinks) -> None: ...


@dataclass(frozen=True)
class ScoringResult:
    path: Path
    scored_control_steps: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class LoadedScoringSidecar:
    path: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, NDArray[Any]]
    metadata_sha256: str
    primitives_sha256: str


def validate_frozen_transform_order(
    transforms: Sequence[ScoringTransform],
) -> tuple[ScoringTransform, ...]:
    """Refuse reordered, renamed, or numerically changed score transforms."""

    supplied = tuple(transforms)
    if supplied != FROZEN_TRANSFORMS:
        raise ScoringError(
            "score transforms differ from the frozen brightness/camera/object order"
        )
    return supplied


def _transform_metadata(transform: ScoringTransform) -> dict[str, Any]:
    return {
        "name": transform.name,
        "family": transform.family,
        "value": transform.value,
        "action_transform": transform.action_transform,
        "terminology": transform.terminology,
    }


def scoring_transforms_from_conditions(
    conditions: Sequence[Any],
) -> tuple[ScoringTransform, ...]:
    """Convert typed frozen config conditions and validate every relevant field.

    The deliberately small structural interface avoids coupling the scoring
    sidecar to a simulator package.  Each item must expose ``name``, ``family``,
    ``index``, and ``parameters`` like :class:`config.ConditionSpec`.
    """

    converted: list[ScoringTransform] = []
    for index, condition in enumerate(conditions):
        try:
            name = condition.name
            family = condition.family
            condition_index = condition.index
            parameters = dict(condition.parameters)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ScoringError(f"score condition {index} is malformed") from exc
        if condition_index is not None:
            raise ScoringError("counterfactual score conditions must not have indices")
        if family == "photometric":
            expected_keys = {"multiplier", "action_transform"}
            value = parameters.get("multiplier")
        elif family == "camera_render":
            expected_keys = {"yaw", "action_transform"}
            value = parameters.get("yaw")
        elif family == "object_pose":
            expected_keys = {"yaw", "terminology"}
            value = parameters.get("yaw")
        else:
            raise ScoringError(f"unsupported score transform family {family!r}")
        if set(parameters) != expected_keys or type(value) not in {int, float}:
            raise ScoringError(f"score condition {name!r} parameters drifted")
        converted.append(
            ScoringTransform(
                name=str(name),
                family=str(family),
                value=float(value),
                action_transform=parameters.get("action_transform"),
                terminology=parameters.get("terminology"),
            )
        )
    return validate_frozen_transform_order(converted)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _terminal_status(*, success: bool, terminated: bool, truncated: bool) -> str:
    if success:
        return "success"
    if truncated:
        return "truncated"
    if terminated:
        return "terminated_without_success"
    raise ScoringError("replay ended without a terminal or truncation state")


def _compare_frames(actual: ReplayFrame, expected: ReplayFrame, where: str) -> None:
    if actual.control_step != expected.control_step:
        raise ScoringError(f"{where}: control step differs from factual replay")
    if bool(actual.success) != bool(expected.success):
        raise ScoringError(f"{where}: task success differs from factual replay")
    if set(actual.camera_images) != set(expected.camera_images):
        raise ScoringError(f"{where}: camera keys differ from factual replay")
    for key in CAMERA_KEYS:
        if not np.array_equal(actual.camera_images[key], expected.camera_images[key]):
            raise ScoringError(f"{where}: uint8 camera {key!r} is not exactly equal")
    if set(actual.low_dimensional) != set(expected.low_dimensional):
        raise ScoringError(f"{where}: low-dimensional keys differ")
    for key in sorted(expected.low_dimensional):
        observed = np.asarray(actual.low_dimensional[key])
        factual = np.asarray(expected.low_dimensional[key])
        if observed.shape != factual.shape:
            raise ScoringError(f"{where}: low-dimensional field {key!r} changed shape")
        finite = np.isfinite(factual)
        if not np.array_equal(np.isfinite(observed), finite):
            raise ScoringError(f"{where}: field {key!r} changed finiteness")
        if finite.any() and not np.allclose(
            observed[finite], factual[finite], rtol=0.0, atol=1e-10
        ):
            raise ScoringError(f"{where}: field {key!r} exceeds absolute tolerance")
        if (~finite).any() and not np.array_equal(
            observed[~finite], factual[~finite], equal_nan=True
        ):
            raise ScoringError(f"{where}: nonfinite field {key!r} differs")


def _same_snapshot(first: SimulatorSnapshot, second: SimulatorSnapshot) -> bool:
    return all(
        np.array_equal(getattr(first, field), getattr(second, field))
        for field in ("state", "camera_positions", "camera_quaternions")
    )


def _brightness_frame(frame: ReplayFrame, multiplier: float) -> ReplayFrame:
    images = {
        key: np.rint(
            np.clip(frame.camera_images[key].astype(np.float32) * multiplier, 0, 255)
        ).astype(np.uint8)
        for key in CAMERA_KEYS
    }
    return ReplayFrame(
        control_step=frame.control_step,
        camera_images=images,
        low_dimensional=frame.low_dimensional,
        success=frame.success,
    )


def _rng_state() -> tuple[Any, tuple[Any, ...], Any | None, Any | None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu: Any | None = None
    torch_cuda: Any | None = None
    try:
        import torch  # Imported only when the optional runtime is installed.

        torch_cpu = torch.random.get_rng_state().clone()
        if torch.cuda.is_available():
            torch_cuda = tuple(
                state.clone() for state in torch.cuda.get_rng_state_all()
            )
    except ImportError:
        pass
    return python_state, numpy_state, torch_cpu, torch_cuda


def _rng_equal(first: Any, second: Any) -> bool:
    if first[0] != second[0]:
        return False
    first_np, second_np = first[1], second[1]
    if first_np[0] != second_np[0] or first_np[2:] != second_np[2:]:
        return False
    if not np.array_equal(first_np[1], second_np[1]):
        return False
    for left, right in zip(first[2:], second[2:], strict=True):
        if left is None or right is None:
            if left is not right:
                return False
        elif isinstance(left, tuple):
            if not isinstance(right, tuple) or len(left) != len(right):
                return False
            if any(not _tensor_equal(a, b) for a, b in zip(left, right, strict=True)):
                return False
        elif not _tensor_equal(left, right):
            return False
    return True


def _restore_rng_state(state: Any) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    if state[2] is None:
        return
    try:
        import torch

        torch.random.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(list(state[3]))
    except ImportError as exc:  # pragma: no cover - impossible after capture
        raise ScoringError("Torch disappeared while restoring its RNG state") from exc


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _require_rng_nonmutation(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Restore and reject any accidental use of process-global RNG streams."""

    @functools.wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        initial = _rng_state()
        try:
            result = function(*args, **kwargs)
        except BaseException:
            if not _rng_equal(initial, _rng_state()):
                _restore_rng_state(initial)
            raise
        if not _rng_equal(initial, _rng_state()):
            _restore_rng_state(initial)
            raise ScoringError("global Python, NumPy, or Torch RNG state changed")
        return result

    return guarded


def _tensor_equal(first: Any, second: Any) -> bool:
    try:
        return bool(first.equal(second))
    except AttributeError:
        return bool(np.array_equal(first, second))


def _opaque_equal(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second, equal_nan=True)
    if isinstance(first, Mapping):
        return set(first) == set(second) and all(
            _opaque_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _opaque_equal(a, b) for a, b in zip(first, second, strict=True)
        )
    equal = getattr(first, "equal", None)
    if callable(equal):
        try:
            return bool(equal(second))
        except (TypeError, ValueError):
            return False
    try:
        result = first == second
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _validate_call(
    call: ScoredCall,
    *,
    activation_width: int | None,
    intervention: bool,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    actions = np.asarray(call.actions)
    activation = np.asarray(call.activation)
    if (
        actions.dtype != np.float32
        or actions.ndim != 2
        or actions.shape[0] < CHUNK_ACTIONS
    ):
        raise ScoringError("policy call must return at least ten postprocessed actions")
    actions = actions[:CHUNK_ACTIONS]
    if actions.shape != (CHUNK_ACTIONS, ACTION_DIMENSION):
        raise ScoringError("policy action chunk must have seven action dimensions")
    if not np.isfinite(actions).all():
        raise ScoringError("policy action chunk contains a nonfinite value")
    if activation.dtype != np.float32 or activation.ndim != 1 or activation.size == 0:
        raise ScoringError("selected activation must be a nonempty float32 vector")
    if not np.isfinite(activation).all():
        raise ScoringError("selected activation contains a nonfinite value")
    if activation_width is not None and activation.shape != (activation_width,):
        raise ScoringError("selected activation width changed between score calls")
    cost = call.cost.as_array()
    if int(cost[2]) != 1 or int(cost[3]) != int(intervention):
        raise ScoringError("call forward/intervention count violates the protocol")
    if int(cost[6]) != activation.nbytes:
        raise ScoringError("logical activation bytes disagree with selected activation")
    return actions.copy(), activation.copy(), cost


def _safe_queue_state(adapter: ScoringAdapter) -> tuple[Any | None, bool]:
    method = getattr(adapter, "policy_queue_state", None)
    if not callable(method):
        return None, False
    state = method()
    if state is None:
        return None, False
    try:
        return copy.deepcopy(state), True
    except (TypeError, ValueError) as exc:
        raise ScoringError(
            "policy queue state cannot be copied for comparison"
        ) from exc


def _copy_noise(noise: Any) -> Any:
    if isinstance(noise, np.ndarray):
        return noise.copy()
    clone = getattr(noise, "clone", None)
    if callable(clone):
        return clone()
    try:
        return copy.deepcopy(noise)
    except (TypeError, ValueError) as exc:
        raise ScoringError("explicit noise cannot be copied for comparison") from exc


def _predict_with_noise(
    adapter: ScoringAdapter,
    processed_observation: Any,
    *,
    noise: Any,
    intervention_degrees: float | None,
) -> ScoredCall:
    frozen_noise = _copy_noise(noise)
    result = adapter.predict_action_chunk(
        processed_observation,
        noise=noise,
        intervention_degrees=intervention_degrees,
    )
    if not _opaque_equal(noise, frozen_noise):
        raise ScoringError("policy call mutated its explicit common-noise tensor")
    if not isinstance(result, ScoredCall):
        raise ScoringError("adapter returned an invalid scored-call record")
    return result


def _array_schema(
    state_count: int, width: int
) -> dict[str, tuple[tuple[int, ...], Any]]:
    transformed_action_shape = (
        state_count,
        len(FROZEN_TRANSFORMS),
        COMMON_DRAWS,
        CHUNK_ACTIONS,
        ACTION_DIMENSION,
    )
    transformed_activation_shape = (
        state_count,
        len(FROZEN_TRANSFORMS),
        COMMON_DRAWS,
        width,
    )
    transformed_cost_shape = (
        state_count,
        len(FROZEN_TRANSFORMS),
        COMMON_DRAWS,
        len(COST_FIELDS),
    )
    return {
        "control_step": ((state_count,), np.int32),
        "noise_seed": ((state_count, ORIGINAL_DRAWS), np.int64),
        "transform_available": ((state_count, len(FROZEN_TRANSFORMS)), np.bool_),
        "intervention_available": ((state_count, COMMON_DRAWS), np.bool_),
        "original_actions": (
            (state_count, ORIGINAL_DRAWS, CHUNK_ACTIONS, ACTION_DIMENSION),
            np.float32,
        ),
        "transformed_actions": (transformed_action_shape, np.float32),
        "intervention_minus_actions": (
            (state_count, COMMON_DRAWS, CHUNK_ACTIONS, ACTION_DIMENSION),
            np.float32,
        ),
        "intervention_plus_actions": (
            (state_count, COMMON_DRAWS, CHUNK_ACTIONS, ACTION_DIMENSION),
            np.float32,
        ),
        "original_activation": (
            (state_count, ORIGINAL_DRAWS, width),
            np.float32,
        ),
        "transformed_activation": (transformed_activation_shape, np.float32),
        "intervention_minus_activation": (
            (state_count, COMMON_DRAWS, width),
            np.float32,
        ),
        "intervention_plus_activation": (
            (state_count, COMMON_DRAWS, width),
            np.float32,
        ),
        "original_cost": (
            (state_count, ORIGINAL_DRAWS, len(COST_FIELDS)),
            np.float64,
        ),
        "transformed_cost": (transformed_cost_shape, np.float64),
        "intervention_minus_cost": (
            (state_count, COMMON_DRAWS, len(COST_FIELDS)),
            np.float64,
        ),
        "intervention_plus_cost": (
            (state_count, COMMON_DRAWS, len(COST_FIELDS)),
            np.float64,
        ),
    }


def _allocate_arrays(state_count: int, width: int) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {}
    for name, (shape, dtype) in _array_schema(state_count, width).items():
        if name in {"control_step", "noise_seed"}:
            arrays[name] = np.empty(shape, dtype=dtype)
        elif name in {"transform_available", "intervention_available"}:
            arrays[name] = np.zeros(shape, dtype=dtype)
        else:
            arrays[name] = np.full(shape, np.nan, dtype=dtype)
    return arrays


def _score_state(
    adapter: ScoringAdapter,
    *,
    replay: FactualReplay,
    frame: ReplayFrame,
    state_index: int,
    arrays: dict[str, NDArray[Any]] | None,
    transforms: tuple[ScoringTransform, ...],
) -> tuple[dict[str, NDArray[Any]], int]:
    begin_state = getattr(adapter, "begin_score_state", None)
    if not callable(begin_state):
        raise ScoringError("adapter cannot reset per-state peak allocation counters")
    begin_state()
    seeds = np.asarray(
        [
            hash_seed("score-noise-v1", replay.episode_id, frame.control_step, draw)
            for draw in range(ORIGINAL_DRAWS)
        ],
        dtype=np.int64,
    )
    noise = [adapter.noise_for_seed(int(seed)) for seed in seeds]
    processed = adapter.process_observation(frame)

    first_call = _predict_with_noise(
        adapter,
        processed,
        noise=noise[0],
        intervention_degrees=None,
    )
    first_actions, first_activation, first_cost = _validate_call(
        first_call, activation_width=None, intervention=False
    )
    width = int(first_activation.size)
    if arrays is None:
        state_count = sum(
            factual.control_step % SCORE_STRIDE_STEPS == 0
            for factual in replay.frames[:-1]
        )
        arrays = _allocate_arrays(state_count, width)
    elif arrays["original_activation"].shape[-1] != width:
        raise ScoringError("selected activation width changed between score states")

    arrays["control_step"][state_index] = frame.control_step
    arrays["noise_seed"][state_index] = seeds
    arrays["original_actions"][state_index, 0] = first_actions
    arrays["original_activation"][state_index, 0] = first_activation
    arrays["original_cost"][state_index, 0] = first_cost
    for draw in range(1, ORIGINAL_DRAWS):
        result = _predict_with_noise(
            adapter,
            processed,
            noise=noise[draw],
            intervention_degrees=None,
        )
        actions, activation, cost = _validate_call(
            result, activation_width=width, intervention=False
        )
        arrays["original_actions"][state_index, draw] = actions
        arrays["original_activation"][state_index, draw] = activation
        arrays["original_cost"][state_index, draw] = cost

    for transform_index, transform in enumerate(transforms):
        available = True
        if transform.family == "photometric":
            transformed_frame = _brightness_frame(frame, transform.value)
        else:
            before = adapter.capture_snapshot()
            saved = SimulatorSnapshot(
                state=before.state,
                camera_positions=before.camera_positions,
                camera_quaternions=before.camera_quaternions,
            )
            try:
                adapter.apply_transform(transform)
                validity = adapter.transformed_validity()
                if not isinstance(validity, TransformValidity):
                    raise ScoringError("adapter returned an invalid transform validity")
                available = validity.available
                if transform.family == "camera_render" and not available:
                    raise ScoringError(
                        "a render-only camera edit cannot be unavailable"
                    )
                transformed_frame = adapter.current_frame() if available else frame
            finally:
                adapter.restore_snapshot(saved)
                restored = adapter.capture_snapshot()
                if not _same_snapshot(saved, restored):
                    raise ScoringError(
                        "simulator/camera snapshot restoration was not exact"
                    )
        arrays["transform_available"][state_index, transform_index] = available
        if not available:
            continue
        transformed_processed = adapter.process_observation(transformed_frame)
        for draw in range(COMMON_DRAWS):
            result = _predict_with_noise(
                adapter,
                transformed_processed,
                noise=noise[draw],
                intervention_degrees=None,
            )
            actions, activation, cost = _validate_call(
                result, activation_width=width, intervention=False
            )
            arrays["transformed_actions"][state_index, transform_index, draw] = actions
            arrays["transformed_activation"][state_index, transform_index, draw] = (
                activation
            )
            arrays["transformed_cost"][state_index, transform_index, draw] = cost

    intervention_check = getattr(adapter, "intervention_available", None)
    if not callable(intervention_check):
        raise ScoringError("adapter cannot evaluate the fitted probe-vector norm")
    for draw in range(COMMON_DRAWS):
        intervention_available = intervention_check(
            arrays["original_activation"][state_index, draw].copy()
        )
        if type(intervention_available) is not bool:
            raise ScoringError("intervention availability must be a boolean")
        arrays["intervention_available"][state_index, draw] = intervention_available
    for direction_name, direction in zip(
        ("minus", "plus"), INTERVENTION_DEGREES, strict=True
    ):
        for draw in range(COMMON_DRAWS):
            if arrays["intervention_available"][state_index, draw]:
                result = _predict_with_noise(
                    adapter,
                    processed,
                    noise=noise[draw],
                    intervention_degrees=direction,
                )
                actions, activation, cost = _validate_call(
                    result, activation_width=width, intervention=True
                )
                arrays[f"intervention_{direction_name}_actions"][state_index, draw] = (
                    actions
                )
                arrays[f"intervention_{direction_name}_activation"][
                    state_index, draw
                ] = activation
                arrays[f"intervention_{direction_name}_cost"][state_index, draw] = cost
    return arrays, width


@_require_rng_nonmutation
def score_replay_to_sidecar(
    adapter: ScoringAdapter,
    replay: FactualReplay,
    links: ContentLinks,
    *,
    transforms: Sequence[ScoringTransform],
    output_root: str | Path,
) -> ScoringResult:
    """Replay, score, validate, and atomically publish one episode sidecar."""

    frozen_transforms = validate_frozen_transform_order(transforms)
    link_validator = getattr(adapter, "validate_content_links", None)
    if link_validator is not None:
        if not callable(link_validator):
            raise ScoringError("adapter content-link validator is not callable")
        link_validator(links)
    parent, destination = _sidecar_destination(Path(output_root), replay)
    initial_rng = _rng_state()
    initial_queue, queue_accessible = _safe_queue_state(adapter)
    arrays: dict[str, NDArray[Any]] | None = None
    width: int | None = None
    scored_index = 0

    actual = adapter.reset_replay()
    _compare_frames(actual, replay.frames[0], "reset")
    for action_index, factual_action in enumerate(replay.actions):
        expected_frame = replay.frames[action_index]
        _compare_frames(actual, expected_frame, f"frame {action_index}")
        if expected_frame.control_step % SCORE_STRIDE_STEPS == 0:
            arrays, width = _score_state(
                adapter,
                replay=replay,
                frame=expected_frame,
                state_index=scored_index,
                arrays=arrays,
                transforms=frozen_transforms,
            )
            scored_index += 1
        transition = adapter.step_replay(factual_action.copy())
        if bool(transition.terminated) != bool(replay.terminated[action_index]):
            raise ScoringError(f"transition {action_index}: terminated flag differs")
        if bool(transition.truncated) != bool(replay.truncated[action_index]):
            raise ScoringError(f"transition {action_index}: truncated flag differs")
        actual = transition.frame
        _compare_frames(
            actual, replay.frames[action_index + 1], f"frame {action_index + 1}"
        )

    status = _terminal_status(
        success=actual.success,
        terminated=bool(replay.terminated[-1]),
        truncated=bool(replay.truncated[-1]),
    )
    if status != replay.terminal_status:
        raise ScoringError("replayed terminal status differs from factual outcome")
    if arrays is None or width is None or scored_index != arrays["control_step"].size:
        raise ScoringError("scored-state cardinality is inconsistent")
    final_queue, final_queue_accessible = _safe_queue_state(adapter)
    if queue_accessible != final_queue_accessible:
        raise ScoringError("policy queue observability changed during scoring")
    if queue_accessible and not _opaque_equal(initial_queue, final_queue):
        raise ScoringError("policy action queue changed during scoring")
    if not _rng_equal(initial_rng, _rng_state()):
        _restore_rng_state(initial_rng)
        raise ScoringError("global Python, NumPy, or Torch RNG state changed")

    metadata = _metadata_payload(
        replay=replay,
        links=links,
        arrays=arrays,
        width=width,
        queue_checked=queue_accessible,
    )
    path, digest = _write_sidecar_atomically(
        parent=parent,
        destination=destination,
        metadata=metadata,
        arrays=arrays,
    )
    return ScoringResult(
        path=path,
        scored_control_steps=tuple(int(value) for value in arrays["control_step"]),
        sha256=digest,
    )


def _metadata_payload(
    *,
    replay: FactualReplay,
    links: ContentLinks,
    arrays: Mapping[str, NDArray[Any]],
    width: int,
    queue_checked: bool,
) -> dict[str, Any]:
    state_count = int(arrays["control_step"].size)
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "episode_id": replay.episode_id,
        "split": replay.split,
        "links": links.as_dict(),
        "protocol": {
            "score_stride_steps": SCORE_STRIDE_STEPS,
            "original_draws": ORIGINAL_DRAWS,
            "common_draws": COMMON_DRAWS,
            "chunk_actions": CHUNK_ACTIONS,
            "action_dimension": ACTION_DIMENSION,
            "noise_namespace": "score-noise-v1",
            "transform_order": [
                _transform_metadata(item) for item in FROZEN_TRANSFORMS
            ],
            "intervention_degrees": list(INTERVENTION_DEGREES),
            "brightness_formula": "rint(clip(float32_pixel * multiplier, 0, 255))",
            "replay_image_equality": "uint8_exact",
            "replay_low_dimensional_atol": 1e-10,
        },
        "capture": {
            "state_count": state_count,
            "selected_activation_width": width,
            "selected_activation_only": True,
            "postprocessed_unnormalized_actions": True,
            "rng_checked": True,
            "policy_queue_checked": queue_checked,
            "peak_allocation_reset_per_state": True,
            "cuda_synchronized_outside_measured_interval": True,
            "unreported_warmup_calls": 0,
            "cost_fields": list(COST_FIELDS),
        },
        "factual_outcome": {
            "status": replay.terminal_status,
            "success": replay.success,
            "control_steps": int(replay.actions.shape[0]),
        },
        "files": {"primitives": "primitives.npz", "primitives_sha256": None},
    }


def _sidecar_destination(output_root: Path, replay: FactualReplay) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    if _has_symlink_component(root):
        raise ScoringError("output path may not contain a symlink")
    if any(part in {"locks", "configs"} for part in root.parts):
        raise ScoringError(
            "score sidecars may not be written into protocol or lock paths"
        )
    if root.exists() and not root.is_dir():
        raise ScoringError("output root is not a directory")
    parent = root / replay.split
    if parent.exists() and parent.is_symlink():
        raise ScoringError("sidecar split directory may not be a symlink")
    destination = parent / replay.episode_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite score sidecar {destination}")
    return parent, destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _canonical_metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            metadata,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_sidecar_atomically(
    *,
    parent: Path,
    destination: Path,
    metadata: dict[str, Any],
    arrays: Mapping[str, NDArray[Any]],
) -> tuple[Path, str]:
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ScoringError("sidecar split directory may not be a symlink")
    lock_path = parent / f".{destination.name}.publish.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"another writer is publishing score sidecar {destination}"
        ) from exc
    os.close(lock_descriptor)
    staging: Path | None = None
    try:
        _fsync_directory(parent)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
        primitive_bytes = _deterministic_npz_bytes(arrays)
        primitive_digest = hashlib.sha256(primitive_bytes).hexdigest()
        metadata["files"]["primitives_sha256"] = primitive_digest
        primitive_path = staging / "primitives.npz"
        with primitive_path.open("xb") as stream:
            stream.write(primitive_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        metadata_bytes = _canonical_metadata_bytes(metadata)
        with (staging / "metadata.json").open("xb") as stream:
            stream.write(metadata_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite score sidecar {destination}")
        os.rename(staging, destination)
        _fsync_directory(parent)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
        _fsync_directory(parent)
    digest = hashlib.sha256(metadata_bytes + primitive_bytes).hexdigest()
    return destination, digest


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScoringValidationError(f"metadata.json has duplicate key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise ScoringValidationError(
            f"{where} keys differ: missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _regular_file(path: Path) -> None:
    if path.is_symlink():
        raise ScoringValidationError(f"{path.name} may not be a symlink")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ScoringValidationError(f"cannot stat {path.name}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ScoringValidationError(f"{path.name} is not a regular file")


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        if current.is_symlink():
            return True
    return False


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    result = path.stat()
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def load_scoring_sidecar(
    path: str | Path,
    *,
    expected_links: ContentLinks | None = None,
    expected_episode_id: str | None = None,
) -> LoadedScoringSidecar:
    """Load and fully validate a published score sidecar without pickle."""

    directory = Path(path)
    if _has_symlink_component(directory) or not directory.is_dir():
        raise ScoringValidationError("sidecar path must be a real directory")
    entries = {item.name for item in directory.iterdir()}
    if entries != {"metadata.json", "primitives.npz"}:
        raise ScoringValidationError(
            "sidecar must contain only metadata and primitives"
        )
    metadata_path = directory / "metadata.json"
    primitive_path = directory / "primitives.npz"
    _regular_file(metadata_path)
    _regular_file(primitive_path)
    metadata_stat = _file_identity(metadata_path)
    primitive_stat = _file_identity(primitive_path)
    metadata_bytes = metadata_path.read_bytes()
    primitive_bytes = primitive_path.read_bytes()
    if (
        _file_identity(metadata_path) != metadata_stat
        or _file_identity(primitive_path) != primitive_stat
    ):
        raise ScoringValidationError("sidecar changed while it was being read")
    try:
        metadata = json.loads(
            metadata_bytes.decode("utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoringValidationError(f"invalid metadata JSON: {exc}") from exc
    _validate_metadata(metadata, directory)
    if (
        expected_episode_id is not None
        and metadata["episode_id"] != expected_episode_id
    ):
        raise ScoringValidationError(
            "sidecar episode_id differs from the requested one"
        )
    if expected_links is not None and metadata["links"] != expected_links.as_dict():
        raise ScoringValidationError(
            "sidecar content links differ from expected hashes"
        )
    expected_digest = metadata["files"]["primitives_sha256"]
    primitive_digest = hashlib.sha256(primitive_bytes).hexdigest()
    if primitive_digest != expected_digest:
        raise ScoringValidationError("primitives SHA-256 does not match metadata")
    try:
        with np.load(io.BytesIO(primitive_bytes), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ScoringValidationError("primitives contain duplicate array names")
            arrays = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ScoringValidationError(f"invalid safe NumPy archive: {exc}") from exc
    _validate_arrays(metadata, arrays)
    frozen_arrays: dict[str, NDArray[Any]] = {}
    for name, value in arrays.items():
        value.flags.writeable = False
        frozen_arrays[name] = value
    return LoadedScoringSidecar(
        path=directory,
        metadata=_freeze_json(metadata),
        arrays=MappingProxyType(frozen_arrays),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        primitives_sha256=primitive_digest,
    )


def _validate_metadata(metadata: Any, directory: Path) -> None:
    if not isinstance(metadata, dict):
        raise ScoringValidationError("metadata root must be an object")
    _exact_keys(
        metadata,
        {
            "schema_version",
            "episode_id",
            "split",
            "links",
            "protocol",
            "capture",
            "factual_outcome",
            "files",
        },
        "metadata",
    )
    if metadata["schema_version"] != SCORING_SCHEMA_VERSION:
        raise ScoringValidationError("unsupported scoring schema version")
    for key in ("episode_id", "split"):
        value = metadata[key]
        if not isinstance(value, str) or Path(value).name != value or not value:
            raise ScoringValidationError(f"metadata {key} is not a safe name")
    if directory.name != metadata["episode_id"]:
        raise ScoringValidationError("directory name differs from episode_id")
    if directory.parent.name != metadata["split"]:
        raise ScoringValidationError("parent directory differs from split")
    links = metadata["links"]
    if not isinstance(links, dict):
        raise ScoringValidationError("links must be an object")
    _exact_keys(links, set(ContentLinks.__dataclass_fields__), "links")
    if any(not _is_sha256(value) for value in links.values()):
        raise ScoringValidationError("all content links must be lowercase SHA-256")
    protocol = metadata["protocol"]
    if not isinstance(protocol, dict):
        raise ScoringValidationError("protocol must be an object")
    expected_protocol = {
        "score_stride_steps": SCORE_STRIDE_STEPS,
        "original_draws": ORIGINAL_DRAWS,
        "common_draws": COMMON_DRAWS,
        "chunk_actions": CHUNK_ACTIONS,
        "action_dimension": ACTION_DIMENSION,
        "noise_namespace": "score-noise-v1",
        "transform_order": [_transform_metadata(item) for item in FROZEN_TRANSFORMS],
        "intervention_degrees": list(INTERVENTION_DEGREES),
        "brightness_formula": "rint(clip(float32_pixel * multiplier, 0, 255))",
        "replay_image_equality": "uint8_exact",
        "replay_low_dimensional_atol": 1e-10,
    }
    if protocol != expected_protocol:
        raise ScoringValidationError("protocol constants differ from frozen scoring")
    capture = metadata["capture"]
    if not isinstance(capture, dict):
        raise ScoringValidationError("capture must be an object")
    _exact_keys(
        capture,
        {
            "state_count",
            "selected_activation_width",
            "selected_activation_only",
            "postprocessed_unnormalized_actions",
            "rng_checked",
            "policy_queue_checked",
            "peak_allocation_reset_per_state",
            "cuda_synchronized_outside_measured_interval",
            "unreported_warmup_calls",
            "cost_fields",
        },
        "capture",
    )
    state_count = capture["state_count"]
    width = capture["selected_activation_width"]
    if type(state_count) is not int or state_count < 1:
        raise ScoringValidationError("capture state_count must be positive")
    if type(width) is not int or width < 1:
        raise ScoringValidationError("activation width must be positive")
    for key in (
        "selected_activation_only",
        "postprocessed_unnormalized_actions",
        "rng_checked",
        "peak_allocation_reset_per_state",
        "cuda_synchronized_outside_measured_interval",
    ):
        if capture[key] is not True:
            raise ScoringValidationError(f"capture {key} must be true")
    if type(capture["policy_queue_checked"]) is not bool:
        raise ScoringValidationError("policy_queue_checked must be boolean")
    if capture["unreported_warmup_calls"] != 0:
        raise ScoringValidationError("unreported_warmup_calls must be zero")
    if capture["cost_fields"] != list(COST_FIELDS):
        raise ScoringValidationError("cost field order differs from frozen schema")
    outcome = metadata["factual_outcome"]
    if not isinstance(outcome, dict):
        raise ScoringValidationError("factual_outcome must be an object")
    _exact_keys(outcome, {"status", "success", "control_steps"}, "factual_outcome")
    if outcome["status"] not in {
        "success",
        "truncated",
        "terminated_without_success",
    }:
        raise ScoringValidationError("unknown factual terminal status")
    if type(outcome["success"]) is not bool:
        raise ScoringValidationError("factual success must be boolean")
    if (outcome["status"] == "success") != outcome["success"]:
        raise ScoringValidationError("factual status and success disagree")
    if type(outcome["control_steps"]) is not int or outcome["control_steps"] < 1:
        raise ScoringValidationError("factual control_steps must be positive")
    files = metadata["files"]
    if not isinstance(files, dict):
        raise ScoringValidationError("files must be an object")
    _exact_keys(files, {"primitives", "primitives_sha256"}, "files")
    if files["primitives"] != "primitives.npz" or not _is_sha256(
        files["primitives_sha256"]
    ):
        raise ScoringValidationError("invalid primitives file declaration")


def _validate_arrays(
    metadata: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]
) -> None:
    state_count = int(metadata["capture"]["state_count"])
    width = int(metadata["capture"]["selected_activation_width"])
    schema = _array_schema(state_count, width)
    if set(arrays) != set(schema):
        raise ScoringValidationError("primitive array names differ from schema")
    for name, (shape, dtype) in schema.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ScoringValidationError(
                f"array {name} expected {np.dtype(dtype)} {shape}, "
                f"got {value.dtype} {value.shape}"
            )
    steps = arrays["control_step"]
    expected_steps = np.arange(
        0,
        int(metadata["factual_outcome"]["control_steps"]),
        SCORE_STRIDE_STEPS,
        dtype=np.int32,
    )
    if not np.array_equal(steps, expected_steps):
        raise ScoringValidationError(
            "control steps do not cover every required stride state"
        )
    expected_seeds = np.asarray(
        [
            [
                hash_seed("score-noise-v1", metadata["episode_id"], int(step), draw)
                for draw in range(ORIGINAL_DRAWS)
            ]
            for step in steps
        ],
        dtype=np.int64,
    )
    if not np.array_equal(arrays["noise_seed"], expected_seeds):
        raise ScoringValidationError("noise seeds differ from the frozen hash plan")
    for name in ("original_actions", "original_activation", "original_cost"):
        if not np.isfinite(arrays[name]).all():
            raise ScoringValidationError(f"{name} must be entirely finite")
    transform_mask = arrays["transform_available"]
    if not np.all(transform_mask[:, :4]):
        raise ScoringValidationError(
            "photometric and camera transforms must be available"
        )
    for name in ("transformed_actions", "transformed_activation", "transformed_cost"):
        value = arrays[name]
        expanded = transform_mask
        while expanded.ndim < value.ndim:
            expanded = expanded[..., None]
        finite = np.isfinite(value)
        if not np.all(finite == np.broadcast_to(expanded, value.shape)):
            raise ScoringValidationError(f"{name} finiteness disagrees with its mask")
    intervention_mask = arrays["intervention_available"]
    for direction in ("minus", "plus"):
        for suffix in ("actions", "activation", "cost"):
            value = arrays[f"intervention_{direction}_{suffix}"]
            expanded = intervention_mask
            while expanded.ndim < value.ndim:
                expanded = expanded[..., None]
            if not np.all(np.isfinite(value) == np.broadcast_to(expanded, value.shape)):
                raise ScoringValidationError(
                    f"intervention_{direction}_{suffix} finiteness disagrees with mask"
                )
    for name in (
        "original_cost",
        "transformed_cost",
        "intervention_minus_cost",
        "intervention_plus_cost",
    ):
        cost = arrays[name]
        finite_rows = np.isfinite(cost).all(axis=-1)
        if np.any(cost[finite_rows] < 0):
            raise ScoringValidationError(f"{name} contains a negative cost")
        if finite_rows.any():
            integer_values = cost[finite_rows][:, 1:]
            if not np.equal(integer_values, np.floor(integer_values)).all():
                raise ScoringValidationError(
                    f"{name} integer cost columns are fractional"
                )
            expected_interventions = int(name.startswith("intervention_"))
            if not np.all(cost[finite_rows][:, 2] == 1) or not np.all(
                cost[finite_rows][:, 3] == expected_interventions
            ):
                raise ScoringValidationError(
                    f"{name} forward/intervention counts violate the protocol"
                )
    for name in (
        "original_activation",
        "transformed_activation",
        "intervention_minus_activation",
        "intervention_plus_activation",
    ):
        activation = arrays[name]
        costs = arrays[name.replace("activation", "cost")]
        valid = np.isfinite(activation).all(axis=-1)
        if valid.any():
            logical = costs[..., COST_FIELDS.index("logical_activation_bytes")]
            if not np.all(logical[valid] == width * np.dtype(np.float32).itemsize):
                raise ScoringValidationError(
                    f"{name} logical byte cost disagrees with activation width"
                )


__all__ = [
    "ACTION_DIMENSION",
    "CAMERA_KEYS",
    "CHUNK_ACTIONS",
    "COMMON_DRAWS",
    "COST_FIELDS",
    "FROZEN_TRANSFORMS",
    "INTERVENTION_DEGREES",
    "ORIGINAL_DRAWS",
    "SCORE_STRIDE_STEPS",
    "SCORING_SCHEMA_VERSION",
    "CallCost",
    "ContentLinks",
    "FactualReplay",
    "LoadedScoringSidecar",
    "ReplayFrame",
    "ReplayTransition",
    "ScoredCall",
    "ScoringAdapter",
    "ScoringError",
    "ScoringResult",
    "ScoringTransform",
    "ScoringValidationError",
    "SimulatorSnapshot",
    "TransformValidity",
    "load_scoring_sidecar",
    "score_replay_to_sidecar",
    "scoring_transforms_from_conditions",
    "validate_frozen_transform_order",
]
