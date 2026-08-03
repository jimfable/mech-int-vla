"""Frozen, outcome-independent M0/M1/M2 feature arithmetic.

This module consumes already validated replay primitives.  It never runs a
policy or simulator and never discovers cohort membership.  Array inputs are
defensively copied into immutable NumPy arrays, fixed grid axes are checked
exactly, and undefined quantities are represented by NaN so the predictor's
prespecified missingness indicators can handle them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

FEATURE_SCHEMA_VERSION: Final = 1
ACTION_DIMENSION: Final = 7
ACTION_DRAWS: Final = 8
COMMON_DRAWS: Final = 4
ACTION_STEPS: Final = 10
ACTION_SCALE_FLOOR: Final = 1e-8
COVERAGE_SCALE_FLOOR: Final = 1e-8
ROBUST_SCALE_FLOOR: Final = 1e-8
PROBE_NORM_FLOOR: Final = 1e-12
YAW_ACTION_INDEX: Final = 5
PHASE_ORDER: Final = ("pregrasp", "grasped", "transport", "placed")
TRANSFORM_ORDER: Final = (
    "brightness_0_85",
    "brightness_1_15",
    "camera_yaw_neg_3",
    "camera_yaw_pos_3",
    "object_yaw_neg_15",
    "object_yaw_pos_15",
)
TRANSFORM_FAMILIES: Final = ("brightness", "camera", "object")
OBJECT_YAW_DELTAS_RAD: Final = (
    math.radians(-15.0),
    math.radians(15.0),
)
INTERVENTION_ORDER: Final = ("probe_yaw_neg_10", "probe_yaw_pos_10")
INTERVENTION_SPAN_RAD: Final = math.radians(20.0)

M0_FEATURE_NAMES: Final = (
    "m0_brightness_action_drift_mean",
    "m0_brightness_action_drift_max",
    "m0_camera_action_drift_mean",
    "m0_camera_action_drift_max",
    "m0_object_action_drift_mean",
    "m0_object_action_drift_max",
    "m0_brightness_render_equivariance_error",
    "m0_camera_render_equivariance_error",
    "m0_output_covariance_trace",
    "m0_output_pairwise_distance_mean",
    "m0_output_pairwise_distance_p90",
    "m0_action_chunk_norm_mean",
    "m0_action_temporal_roughness_mean",
)

M1_RAW_FEATURE_NAMES: Final = (
    "m1_object_minus_eef_x",
    "m1_object_minus_eef_y",
    "m1_object_minus_eef_z",
    "m1_goal_minus_object_x",
    "m1_goal_minus_object_y",
    "m1_goal_minus_object_z",
    "m1_q_eef_inverse_q_object_w",
    "m1_q_eef_inverse_q_object_x",
    "m1_q_eef_inverse_q_object_y",
    "m1_q_eef_inverse_q_object_z",
    "m1_q_object_inverse_q_goal_w",
    "m1_q_object_inverse_q_goal_x",
    "m1_q_object_inverse_q_goal_y",
    "m1_q_object_inverse_q_goal_z",
    "m1_symmetry_eef_object_yaw_sin",
    "m1_symmetry_eef_object_yaw_cos",
    "m1_symmetry_object_goal_yaw_sin",
    "m1_symmetry_object_goal_yaw_cos",
    "m1_mean_two_finger_opening",
    "m1_primary_contact",
    "m1_backend_grasp",
    "m1_normalized_step",
    "m1_goal_present",
    "m1_phase_pregrasp",
    "m1_phase_grasped",
    "m1_phase_transport",
    "m1_phase_placed",
)

COVERAGE_VECTOR_NAMES: Final = (
    *M1_RAW_FEATURE_NAMES[0:6],
    *M1_RAW_FEATURE_NAMES[14:27],
)
COVERAGE_FEATURE_NAMES: Final = (
    "m1_success_same_phase_five_nn_median_distance",
    "m1_all_phase_nearest_distance",
    "m1_all_phase_25nn_failure_fraction",
)

M2_BASE_FEATURE_NAMES: Final = (
    "m2_object_probe_equivariance_error_mean_rad",
    "m2_brightness_probe_circular_dispersion",
    "m2_camera_probe_circular_dispersion",
    "m2_probe_resultant_norm_mean",
    "m2_probe_norm_success_same_phase_robust_z_abs",
)
M2_EXPERT_NOISE_FEATURE_NAME: Final = "m2_probe_flow_noise_circular_dispersion"
M2_INTERVENTION_FEATURE_NAMES: Final = (
    "m2_probe_controllability_gain_per_rad",
    "m2_probe_intervention_specificity_ratio",
)
M1_FEATURE_NAMES: Final = (
    *M0_FEATURE_NAMES,
    *M1_RAW_FEATURE_NAMES,
    *COVERAGE_FEATURE_NAMES,
)
M2_VLM_FEATURE_NAMES: Final = (
    *M1_FEATURE_NAMES,
    *M2_BASE_FEATURE_NAMES,
    *M2_INTERVENTION_FEATURE_NAMES,
)
M2_EXPERT_FEATURE_NAMES: Final = (
    *M1_FEATURE_NAMES,
    *M2_BASE_FEATURE_NAMES,
    M2_EXPERT_NOISE_FEATURE_NAME,
    *M2_INTERVENTION_FEATURE_NAMES,
)


class FeatureValidationError(ValueError):
    """Raised when inputs cannot satisfy the frozen feature contract."""


def _readonly_array(
    value: ArrayLike,
    *,
    dtype: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> NDArray[Any]:
    try:
        result = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(f"{name} must be a numeric array") from exc
    if shape is not None and result.shape != shape:
        raise FeatureValidationError(
            f"{name} has shape {result.shape}; expected {shape}"
        )
    result.setflags(write=False)
    return result


def _readonly_bool_mask(
    value: ArrayLike, *, name: str, shape: tuple[int, ...]
) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.dtype != np.bool_:
        raise FeatureValidationError(f"{name} must have boolean dtype")
    return _readonly_array(value, dtype=np.bool_, name=name, shape=shape)


def _finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise FeatureValidationError(f"{name} must be a finite scalar")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise FeatureValidationError(f"{name} must be a positive integer")
    return int(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(nested) for nested in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _metadata_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NamedFeatureRow:
    """One immutable feature row with canonical schema provenance."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise FeatureValidationError("feature names must be nonempty strings")
        if len(names) != len(set(names)):
            raise FeatureValidationError("feature names must be unique")
        values = _readonly_array(
            self.values, dtype=np.float64, name="feature values", shape=(len(names),)
        )
        if np.isinf(values).any():
            raise FeatureValidationError(
                "feature values may contain NaN but not infinity"
            )
        metadata = _freeze_json(json.loads(_canonical_json(dict(self.metadata))))
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", metadata)

    @property
    def metadata_sha256(self) -> str:
        return _metadata_hash(dict(self.metadata))

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist(), strict=True))


@dataclass(frozen=True)
class ActionScale:
    """Episode-equal population scale for the seven unnormalized actions."""

    values: NDArray[np.float64]
    replaced_by_one: NDArray[np.bool_]
    episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _readonly_array(
            self.values,
            dtype=np.float64,
            name="action scale",
            shape=(ACTION_DIMENSION,),
        )
        replaced = _readonly_array(
            self.replaced_by_one,
            dtype=np.bool_,
            name="action scale replacement mask",
            shape=(ACTION_DIMENSION,),
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise FeatureValidationError("action scales must be finite and positive")
        if np.any(replaced & (values != 1.0)):
            raise FeatureValidationError("replaced action scales must equal one")
        episode_ids = tuple(self.episode_ids)
        if not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise FeatureValidationError("action-scale episode IDs must be unique")
        if any(not isinstance(episode, str) or not episode for episode in episode_ids):
            raise FeatureValidationError(
                "action-scale episode IDs must be nonempty strings"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "replaced_by_one", replaced)
        object.__setattr__(self, "episode_ids", episode_ids)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "definition": "episode_equal_weighted_population_standard_deviation",
            "source": "eight_by_ten_original_unnormalized_calibration_actions",
            "floor": ACTION_SCALE_FLOOR,
            "values": self.values.tolist(),
            "replaced_by_one": self.replaced_by_one.tolist(),
            "episode_ids": list(self.episode_ids),
        }

    @property
    def metadata_sha256(self) -> str:
        return _metadata_hash(self.to_metadata())


def fit_episode_equal_action_scale(
    original_actions: ArrayLike, episode_id: ArrayLike
) -> ActionScale:
    """Fit the frozen scale, giving every Calibration episode total weight one.

    ``original_actions`` is state x 8 draws x 10 actions x 7 dimensions.  Within
    an episode every scalar action occurrence has equal weight; episode means and
    second moments are then averaged equally across episodes.
    """

    actions = _readonly_array(
        original_actions, dtype=np.float64, name="original calibration actions"
    )
    if actions.ndim != 4 or actions.shape[1:] != (
        ACTION_DRAWS,
        ACTION_STEPS,
        ACTION_DIMENSION,
    ):
        raise FeatureValidationError(
            "original calibration actions must have shape (state,8,10,7)"
        )
    if actions.shape[0] == 0 or not np.isfinite(actions).all():
        raise FeatureValidationError(
            "original calibration actions must be nonempty and finite"
        )
    episodes = np.asarray(episode_id, dtype=np.str_)
    if episodes.ndim != 1 or episodes.shape[0] != actions.shape[0]:
        raise FeatureValidationError("episode_id must align with action states")
    if any(not episode for episode in episodes.tolist()):
        raise FeatureValidationError("episode IDs must be nonempty")
    unique_episodes = tuple(sorted(np.unique(episodes).tolist()))
    episode_means = []
    episode_second_moments = []
    for episode in unique_episodes:
        episode_actions = actions[episodes == episode]
        flat = episode_actions.reshape(-1, ACTION_DIMENSION)
        episode_means.append(np.mean(flat, axis=0))
        episode_second_moments.append(np.mean(np.square(flat), axis=0))
    mean = np.mean(np.stack(episode_means), axis=0)
    second_moment = np.mean(np.stack(episode_second_moments), axis=0)
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if not np.isfinite(scale).all():
        raise FeatureValidationError("computed action scale is nonfinite")
    replaced = scale < ACTION_SCALE_FLOOR
    scale[replaced] = 1.0
    return ActionScale(scale, replaced, unique_episodes)


@dataclass(frozen=True)
class M0Primitives:
    """Fixed original/transformed action grid for one scored state."""

    original_actions: NDArray[np.float64]
    transformed_actions: NDArray[np.float64]
    transformed_available: NDArray[np.bool_]
    transform_order: tuple[str, ...] = TRANSFORM_ORDER

    def __post_init__(self) -> None:
        original = _readonly_array(
            self.original_actions,
            dtype=np.float64,
            name="original actions",
            shape=(ACTION_DRAWS, ACTION_STEPS, ACTION_DIMENSION),
        )
        transformed = _readonly_array(
            self.transformed_actions,
            dtype=np.float64,
            name="transformed actions",
            shape=(len(TRANSFORM_ORDER), COMMON_DRAWS, ACTION_STEPS, ACTION_DIMENSION),
        )
        available = _readonly_array(
            self.transformed_available,
            dtype=np.bool_,
            name="transformed availability",
            shape=(len(TRANSFORM_ORDER),),
        )
        order = tuple(self.transform_order)
        if order != TRANSFORM_ORDER:
            raise FeatureValidationError(
                f"transform order must equal the frozen order {TRANSFORM_ORDER!r}"
            )
        if not np.isfinite(original).all():
            raise FeatureValidationError("original actions must be finite")
        for index, is_available in enumerate(available):
            block = transformed[index]
            if is_available and not np.isfinite(block).all():
                raise FeatureValidationError(
                    f"available transform {order[index]} must contain finite actions"
                )
            if not is_available and not np.isnan(block).all():
                raise FeatureValidationError(
                    f"unavailable transform {order[index]} must contain only NaN"
                )
        object.__setattr__(self, "original_actions", original)
        object.__setattr__(self, "transformed_actions", transformed)
        object.__setattr__(self, "transformed_available", available)
        object.__setattr__(self, "transform_order", order)

    @classmethod
    def from_arrays(
        cls,
        *,
        original_actions: ArrayLike,
        transformed_actions: ArrayLike,
        transformed_available: ArrayLike,
        transform_order: Sequence[str] = TRANSFORM_ORDER,
    ) -> M0Primitives:
        return cls(
            np.asarray(original_actions),
            np.asarray(transformed_actions),
            np.asarray(transformed_available),
            tuple(transform_order),
        )


def compute_m0_features(
    primitives: M0Primitives, scale: ActionScale
) -> NamedFeatureRow:
    """Compute the thirteen fixed output-only features for one state."""

    standardized_original = primitives.original_actions / scale.values
    family_mean_max: list[float] = []
    render_errors: dict[str, float] = {}
    for family_index, family in enumerate(TRANSFORM_FAMILIES):
        transform_slice = slice(2 * family_index, 2 * family_index + 2)
        available = primitives.transformed_available[transform_slice]
        if not bool(np.all(available)):
            family_mean_max.extend((math.nan, math.nan))
            if family != "object":
                render_errors[family] = math.nan
            continue
        transformed = primitives.transformed_actions[transform_slice]
        delta = transformed - primitives.original_actions[None, :COMMON_DRAWS]
        standardized_delta = delta / scale.values
        per_action = np.linalg.norm(standardized_delta, axis=-1) / math.sqrt(
            ACTION_DIMENSION
        )
        family_mean_max.extend((float(np.mean(per_action)), float(np.max(per_action))))
        if family != "object":
            flat_distances = np.linalg.norm(
                standardized_delta.reshape(2, COMMON_DRAWS, -1), axis=-1
            ) / math.sqrt(ACTION_STEPS * ACTION_DIMENSION)
            render_errors[family] = float(np.mean(flat_distances))

    flat_standardized = standardized_original.reshape(ACTION_DRAWS, -1)
    centered = flat_standardized - np.mean(flat_standardized, axis=0, keepdims=True)
    covariance_trace = float(np.sum(np.square(centered)) / ACTION_DRAWS)
    pairwise = np.asarray(
        [
            np.linalg.norm(flat_standardized[first] - flat_standardized[second])
            / math.sqrt(ACTION_STEPS * ACTION_DIMENSION)
            for first in range(ACTION_DRAWS)
            for second in range(first + 1, ACTION_DRAWS)
        ],
        dtype=np.float64,
    )
    chunk_norm = float(
        np.mean(
            np.linalg.norm(
                primitives.original_actions.reshape(ACTION_DRAWS, -1), axis=1
            )
            / math.sqrt(ACTION_STEPS * ACTION_DIMENSION)
        )
    )
    roughness = float(
        np.mean(
            np.linalg.norm(np.diff(primitives.original_actions, axis=1), axis=-1)
            / math.sqrt(ACTION_DIMENSION)
        )
    )
    values = np.asarray(
        [
            *family_mean_max,
            render_errors["brightness"],
            render_errors["camera"],
            covariance_trace,
            float(np.mean(pairwise)),
            float(np.percentile(pairwise, 90, method="linear")),
            chunk_norm,
            roughness,
        ],
        dtype=np.float64,
    )
    metadata = canonical_feature_metadata("M0")
    metadata["action_scale_sha256"] = scale.metadata_sha256
    return NamedFeatureRow(M0_FEATURE_NAMES, values, metadata)


@dataclass(frozen=True)
class M1PoseState:
    """Typed simulator pose/state input for one pre-action feature row."""

    eef_position: NDArray[np.float64]
    eef_quaternion_xyzw: NDArray[np.float64]
    object_position: NDArray[np.float64]
    object_quaternion_wxyz: NDArray[np.float64]
    goal_position: NDArray[np.float64] | None
    goal_quaternion_wxyz: NDArray[np.float64] | None
    gripper_qpos: NDArray[np.float64]
    primary_contact: bool
    backend_grasp: bool
    normalized_step: float
    phase: str
    symmetry_order: int

    def __post_init__(self) -> None:
        if (self.goal_position is None) != (self.goal_quaternion_wxyz is None):
            raise FeatureValidationError(
                "goal position and quaternion must both be present or both be absent"
            )
        if not isinstance(self.primary_contact, bool) or not isinstance(
            self.backend_grasp, bool
        ):
            raise FeatureValidationError("contact and grasp flags must be boolean")
        step = _finite_scalar(self.normalized_step, "normalized_step")
        if not 0.0 <= step <= 1.0:
            raise FeatureValidationError("normalized_step must lie in [0, 1]")
        if self.phase not in PHASE_ORDER:
            raise FeatureValidationError(f"phase must be one of {PHASE_ORDER!r}")
        eef_position = _finite_vector(self.eef_position, 3, "eef_position")
        eef_quaternion = _unit_quaternion(
            self.eef_quaternion_xyzw, "eef_quaternion_xyzw"
        )
        object_position = _finite_vector(self.object_position, 3, "object_position")
        object_quaternion = _unit_quaternion(
            self.object_quaternion_wxyz, "object_quaternion_wxyz"
        )
        goal_position = (
            None
            if self.goal_position is None
            else _finite_vector(self.goal_position, 3, "goal_position")
        )
        goal_quaternion = (
            None
            if self.goal_quaternion_wxyz is None
            else _unit_quaternion(self.goal_quaternion_wxyz, "goal_quaternion_wxyz")
        )
        gripper = _finite_vector(self.gripper_qpos, 2, "gripper_qpos")
        object.__setattr__(self, "eef_position", eef_position)
        object.__setattr__(self, "eef_quaternion_xyzw", eef_quaternion)
        object.__setattr__(self, "object_position", object_position)
        object.__setattr__(self, "object_quaternion_wxyz", object_quaternion)
        object.__setattr__(self, "goal_position", goal_position)
        object.__setattr__(self, "goal_quaternion_wxyz", goal_quaternion)
        object.__setattr__(self, "gripper_qpos", gripper)
        object.__setattr__(self, "normalized_step", step)
        object.__setattr__(
            self,
            "symmetry_order",
            _positive_integer(self.symmetry_order, "symmetry_order"),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        eef_position: ArrayLike,
        eef_quaternion_xyzw: ArrayLike,
        object_position: ArrayLike,
        object_quaternion_wxyz: ArrayLike,
        goal_position: ArrayLike | None,
        goal_quaternion_wxyz: ArrayLike | None,
        gripper_qpos: ArrayLike,
        primary_contact: bool,
        backend_grasp: bool,
        normalized_step: float,
        phase: str,
        symmetry_order: int,
    ) -> M1PoseState:
        if (goal_position is None) != (goal_quaternion_wxyz is None):
            raise FeatureValidationError(
                "goal position and quaternion must both be present or both be absent"
            )
        if not isinstance(primary_contact, bool) or not isinstance(backend_grasp, bool):
            raise FeatureValidationError("contact and grasp flags must be boolean")
        step = _finite_scalar(normalized_step, "normalized_step")
        if not 0.0 <= step <= 1.0:
            raise FeatureValidationError("normalized_step must lie in [0, 1]")
        if phase not in PHASE_ORDER:
            raise FeatureValidationError(f"phase must be one of {PHASE_ORDER!r}")
        symmetry = _positive_integer(symmetry_order, "symmetry_order")
        eef_pos = _finite_vector(eef_position, 3, "eef_position")
        object_pos = _finite_vector(object_position, 3, "object_position")
        eef_quat = _unit_quaternion(eef_quaternion_xyzw, "eef_quaternion_xyzw")
        object_quat = _unit_quaternion(object_quaternion_wxyz, "object_quaternion_wxyz")
        goal_pos = (
            None
            if goal_position is None
            else _finite_vector(goal_position, 3, "goal_position")
        )
        goal_quat = (
            None
            if goal_quaternion_wxyz is None
            else _unit_quaternion(goal_quaternion_wxyz, "goal_quaternion_wxyz")
        )
        gripper = _finite_vector(gripper_qpos, 2, "gripper_qpos")
        return cls(
            eef_pos,
            eef_quat,
            object_pos,
            object_quat,
            goal_pos,
            goal_quat,
            gripper,
            primary_contact,
            backend_grasp,
            step,
            phase,
            symmetry,
        )


def _finite_vector(value: ArrayLike, width: int, name: str) -> NDArray[np.float64]:
    result = _readonly_array(value, dtype=np.float64, name=name, shape=(width,))
    if not np.isfinite(result).all():
        raise FeatureValidationError(f"{name} must contain only finite values")
    return result


def _unit_quaternion(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise FeatureValidationError(f"{name} must be unit length within 1e-6")
    normalized = np.array(result / norm, dtype=np.float64, copy=True)
    normalized.setflags(write=False)
    return normalized


def _xyzw_to_wxyz(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray([value[3], value[0], value[1], value[2]], dtype=np.float64)


def _quaternion_product(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _relative_quaternion(
    source_wxyz: NDArray[np.float64], target_wxyz: NDArray[np.float64]
) -> NDArray[np.float64]:
    inverse = source_wxyz * np.asarray([1.0, -1.0, -1.0, -1.0])
    result = _quaternion_product(inverse, target_wxyz)
    result /= np.linalg.norm(result)
    # Canonicalize the double-cover sign: positive W, then lexicographic XYZ at W=0.
    for value in result:
        if value > 0.0:
            break
        if value < 0.0:
            result *= -1.0
            break
    return result


def _yaw_wxyz(quaternion: NDArray[np.float64]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def construct_m1_raw_pose(state: M1PoseState) -> NamedFeatureRow:
    """Construct the fixed 27-column raw M1 pose block."""

    eef_wxyz = _xyzw_to_wxyz(state.eef_quaternion_xyzw)
    q_eef_object = _relative_quaternion(eef_wxyz, state.object_quaternion_wxyz)
    eef_object_yaw = _yaw_wxyz(q_eef_object)
    object_minus_eef = state.object_position - state.eef_position
    if state.goal_position is None or state.goal_quaternion_wxyz is None:
        goal_minus_object = np.full(3, np.nan)
        q_object_goal = np.full(4, np.nan)
        object_goal_pair = (math.nan, math.nan)
        goal_present = 0.0
    else:
        goal_minus_object = state.goal_position - state.object_position
        q_object_goal = _relative_quaternion(
            state.object_quaternion_wxyz, state.goal_quaternion_wxyz
        )
        object_goal_yaw = _yaw_wxyz(q_object_goal)
        object_goal_pair = (
            math.sin(state.symmetry_order * object_goal_yaw),
            math.cos(state.symmetry_order * object_goal_yaw),
        )
        goal_present = 1.0
    values = np.asarray(
        [
            *object_minus_eef,
            *goal_minus_object,
            *q_eef_object,
            *q_object_goal,
            math.sin(state.symmetry_order * eef_object_yaw),
            math.cos(state.symmetry_order * eef_object_yaw),
            *object_goal_pair,
            float(np.mean(state.gripper_qpos)),
            float(state.primary_contact),
            float(state.backend_grasp),
            state.normalized_step,
            goal_present,
            *(float(state.phase == phase) for phase in PHASE_ORDER),
        ],
        dtype=np.float64,
    )
    return NamedFeatureRow(
        M1_RAW_FEATURE_NAMES, values, canonical_feature_metadata("M1_raw")
    )


def m1_coverage_vector(raw_pose: NamedFeatureRow) -> NDArray[np.float64]:
    """Select the frozen continuous coverage vector from an M1 raw row."""

    if raw_pose.names != M1_RAW_FEATURE_NAMES:
        raise FeatureValidationError("coverage input must use the frozen M1 raw schema")
    indices = [M1_RAW_FEATURE_NAMES.index(name) for name in COVERAGE_VECTOR_NAMES]
    return _readonly_array(
        raw_pose.values[indices],
        dtype=np.float64,
        name="coverage vector",
        shape=(len(COVERAGE_VECTOR_NAMES),),
    )


@dataclass(frozen=True)
class CoverageState:
    """One explicitly identified Calibration or Locked Test coverage row."""

    episode_id: str
    base_init_state_id: int
    control_step: int
    split: str
    phase: str
    success: bool
    vector: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FeatureValidationError("coverage episode_id must be nonempty")
        if (
            isinstance(self.base_init_state_id, bool)
            or not isinstance(self.base_init_state_id, Integral)
            or self.base_init_state_id < 0
        ):
            raise FeatureValidationError("base_init_state_id must be nonnegative")
        if (
            isinstance(self.control_step, bool)
            or not isinstance(self.control_step, Integral)
            or self.control_step < 0
            or self.control_step % 5 != 0
        ):
            raise FeatureValidationError(
                "coverage control_step must be nonnegative and divisible by five"
            )
        if self.split not in {"calibration", "locked_test"}:
            raise FeatureValidationError(
                "coverage split must be calibration or locked_test"
            )
        if self.phase not in PHASE_ORDER:
            raise FeatureValidationError(f"phase must be one of {PHASE_ORDER!r}")
        if not isinstance(self.success, bool):
            raise FeatureValidationError("coverage success must be boolean")
        vector = _readonly_array(
            self.vector,
            dtype=np.float64,
            name="coverage vector",
            shape=(len(COVERAGE_VECTOR_NAMES),),
        )
        if np.isinf(vector).any():
            raise FeatureValidationError(
                "coverage vector may contain NaN but not infinity"
            )
        object.__setattr__(self, "base_init_state_id", int(self.base_init_state_id))
        object.__setattr__(self, "control_step", int(self.control_step))
        object.__setattr__(self, "vector", vector)

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        base_init_state_id: int,
        control_step: int,
        split: str,
        phase: str,
        success: bool,
        raw_pose: NamedFeatureRow,
    ) -> CoverageState:
        if not isinstance(episode_id, str) or not episode_id:
            raise FeatureValidationError("coverage episode_id must be nonempty")
        if (
            isinstance(base_init_state_id, bool)
            or not isinstance(base_init_state_id, Integral)
            or base_init_state_id < 0
        ):
            raise FeatureValidationError("base_init_state_id must be nonnegative")
        if (
            isinstance(control_step, bool)
            or not isinstance(control_step, Integral)
            or control_step < 0
            or control_step % 5 != 0
        ):
            raise FeatureValidationError(
                "coverage control_step must be nonnegative and divisible by five"
            )
        if split not in {"calibration", "locked_test"}:
            raise FeatureValidationError(
                "coverage split must be calibration or locked_test"
            )
        if phase not in PHASE_ORDER:
            raise FeatureValidationError(f"phase must be one of {PHASE_ORDER!r}")
        if not isinstance(success, bool):
            raise FeatureValidationError("coverage success must be boolean")
        vector = m1_coverage_vector(raw_pose)
        if np.isinf(vector).any():
            raise FeatureValidationError(
                "coverage vector may contain NaN but not infinity"
            )
        return cls(
            episode_id,
            int(base_init_state_id),
            int(control_step),
            split,
            phase,
            success,
            vector,
        )


@dataclass(frozen=True)
class CoverageFit:
    """Immutable episode-equal standardizer and eligible Calibration references."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    scale_replaced_by_one: NDArray[np.bool_]
    references: tuple[CoverageState, ...]
    excluded_episode_id: str | None
    excluded_base_init_state_id: int | None

    def __post_init__(self) -> None:
        mean = _readonly_array(
            self.mean,
            dtype=np.float64,
            name="coverage mean",
            shape=(len(COVERAGE_VECTOR_NAMES),),
        )
        scale = _readonly_array(
            self.scale,
            dtype=np.float64,
            name="coverage scale",
            shape=(len(COVERAGE_VECTOR_NAMES),),
        )
        replaced = _readonly_array(
            self.scale_replaced_by_one,
            dtype=np.bool_,
            name="coverage scale replacement mask",
            shape=(len(COVERAGE_VECTOR_NAMES),),
        )
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise FeatureValidationError("coverage standardizer must be finite")
        if np.any(scale <= 0.0) or np.any(replaced & (scale != 1.0)):
            raise FeatureValidationError(
                "coverage scales must be positive and flagged exactly"
            )
        references = tuple(self.references)
        if not references:
            raise FeatureValidationError("coverage fit requires eligible references")
        if any(reference.split != "calibration" for reference in references):
            raise FeatureValidationError(
                "coverage references must all be Calibration rows"
            )
        if any(not np.isfinite(reference.vector).all() for reference in references):
            raise FeatureValidationError("coverage references must have finite vectors")
        identities = [
            (reference.episode_id, reference.control_step) for reference in references
        ]
        if len(identities) != len(set(identities)):
            raise FeatureValidationError("coverage fit contains duplicate references")
        if (self.excluded_episode_id is None) != (
            self.excluded_base_init_state_id is None
        ):
            raise FeatureValidationError(
                "coverage exclusions must specify both episode and base init or neither"
            )
        if self.excluded_episode_id is not None and any(
            reference.episode_id == self.excluded_episode_id
            or reference.base_init_state_id == self.excluded_base_init_state_id
            for reference in references
        ):
            raise FeatureValidationError(
                "coverage references violate the declared group exclusions"
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "scale_replaced_by_one", replaced)
        object.__setattr__(
            self,
            "references",
            tuple(
                sorted(
                    references,
                    key=lambda reference: (
                        reference.episode_id,
                        reference.control_step,
                    ),
                )
            ),
        )

    @property
    def reference_records_sha256(self) -> str:
        records = [
            {
                "episode_id": reference.episode_id,
                "base_init_state_id": reference.base_init_state_id,
                "control_step": reference.control_step,
                "phase": reference.phase,
                "success": reference.success,
                "vector_float64": reference.vector.tolist(),
            }
            for reference in self.references
        ]
        return _metadata_hash(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "record_order": ["episode_id", "control_step"],
                "records": records,
            }
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "vector_names": list(COVERAGE_VECTOR_NAMES),
            "standardization": "episode_equal_population_mean_sd",
            "distance": "euclidean_rms",
            "nonfinite_rows": "ineligible_reference_or_nan_query",
            "tie_order": ["distance", "episode_id", "control_step"],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "scale_replaced_by_one": self.scale_replaced_by_one.tolist(),
            "excluded_episode_id": self.excluded_episode_id,
            "excluded_base_init_state_id": self.excluded_base_init_state_id,
            "reference_records_sha256": self.reference_records_sha256,
            "reference_rows": [
                [reference.episode_id, reference.control_step]
                for reference in self.references
            ],
        }

    @property
    def metadata_sha256(self) -> str:
        return _metadata_hash(self.to_metadata())


def fit_coverage_reference(
    calibration_states: Sequence[CoverageState],
    *,
    exclude_episode_id: str | None = None,
    exclude_base_init_state_id: int | None = None,
) -> CoverageFit:
    """Fit coverage scaling after applying the frozen group exclusions."""

    states = tuple(calibration_states)
    if not states:
        raise FeatureValidationError("coverage calibration states must be nonempty")
    if (exclude_episode_id is None) != (exclude_base_init_state_id is None):
        raise FeatureValidationError(
            "coverage exclusions must specify both episode and base init or neither"
        )
    if any(state.split != "calibration" for state in states):
        raise FeatureValidationError("coverage references must all be Calibration rows")
    identities = [(state.episode_id, state.control_step) for state in states]
    if len(identities) != len(set(identities)):
        raise FeatureValidationError(
            "coverage references contain duplicate state identities"
        )
    eligible = tuple(
        sorted(
            (
                state
                for state in states
                if state.episode_id != exclude_episode_id
                and state.base_init_state_id != exclude_base_init_state_id
                and np.isfinite(state.vector).all()
            ),
            key=lambda state: (state.episode_id, state.control_step),
        )
    )
    if not eligible:
        raise FeatureValidationError("no finite Calibration coverage references remain")
    episodes = sorted({state.episode_id for state in eligible})
    episode_means = []
    episode_second_moments = []
    for episode in episodes:
        matrix = np.stack(
            [state.vector for state in eligible if state.episode_id == episode]
        )
        episode_means.append(np.mean(matrix, axis=0))
        episode_second_moments.append(np.mean(np.square(matrix), axis=0))
    mean = np.mean(np.stack(episode_means), axis=0)
    second_moment = np.mean(np.stack(episode_second_moments), axis=0)
    scale = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
    if not np.isfinite(scale).all():
        raise FeatureValidationError("coverage scale is nonfinite")
    replaced = scale < COVERAGE_SCALE_FLOOR
    scale[replaced] = 1.0
    return CoverageFit(
        mean,
        scale,
        replaced,
        eligible,
        exclude_episode_id,
        exclude_base_init_state_id,
    )


def query_coverage_features(query: CoverageState, fit: CoverageFit) -> NamedFeatureRow:
    """Query the three frozen coverage features with deterministic neighbor ties."""

    if query.split == "calibration":
        if fit.excluded_episode_id != query.episode_id:
            raise FeatureValidationError(
                "Calibration query fit must exclude the query's entire episode"
            )
        if fit.excluded_base_init_state_id != query.base_init_state_id:
            raise FeatureValidationError(
                "Calibration query fit must exclude the query's base init group"
            )
    elif query.split == "locked_test":
        if (
            fit.excluded_episode_id is not None
            or fit.excluded_base_init_state_id is not None
        ):
            raise FeatureValidationError(
                "Locked Test queries require the full Calibration fit"
            )
    else:  # pragma: no cover - constructors already reject this
        raise FeatureValidationError("unsupported coverage query split")
    if not np.isfinite(query.vector).all():
        values = np.full(len(COVERAGE_FEATURE_NAMES), np.nan)
    else:
        standardized_query = (query.vector - fit.mean) / fit.scale
        neighbors: list[tuple[float, str, int, CoverageState]] = []
        for reference in fit.references:
            standardized_reference = (reference.vector - fit.mean) / fit.scale
            distance = float(
                np.linalg.norm(standardized_query - standardized_reference)
                / math.sqrt(len(COVERAGE_VECTOR_NAMES))
            )
            neighbors.append(
                (distance, reference.episode_id, reference.control_step, reference)
            )
        neighbors.sort(key=lambda item: (item[0], item[1], item[2]))
        successful_same_phase = [
            item
            for item in neighbors
            if item[3].success and item[3].phase == query.phase
        ]
        median_five = (
            float(np.median([item[0] for item in successful_same_phase[:5]]))
            if len(successful_same_phase) >= 5
            else math.nan
        )
        nearest = neighbors[0][0] if neighbors else math.nan
        failure_fraction = (
            float(np.mean([not item[3].success for item in neighbors[:25]]))
            if len(neighbors) >= 25
            else math.nan
        )
        values = np.asarray([median_five, nearest, failure_fraction])
    metadata = canonical_feature_metadata("M1_coverage")
    metadata["coverage_fit_sha256"] = fit.metadata_sha256
    return NamedFeatureRow(COVERAGE_FEATURE_NAMES, values, metadata)


def compute_coverage_features(
    query: CoverageState, calibration_states: Sequence[CoverageState]
) -> NamedFeatureRow:
    """Convenience wrapper that enforces OOF Calibration versus full Test fitting."""

    if query.split == "calibration":
        fit = fit_coverage_reference(
            calibration_states,
            exclude_episode_id=query.episode_id,
            exclude_base_init_state_id=query.base_init_state_id,
        )
    else:
        fit = fit_coverage_reference(calibration_states)
    return query_coverage_features(query, fit)


@dataclass(frozen=True)
class ProbeNormState:
    """State-level mean probe norm used by the successful same-phase reference."""

    episode_id: str
    base_init_state_id: int
    control_step: int
    split: str
    phase: str
    success: bool
    mean_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FeatureValidationError("probe-norm episode_id must be nonempty")
        if (
            isinstance(self.base_init_state_id, bool)
            or not isinstance(self.base_init_state_id, Integral)
            or self.base_init_state_id < 0
        ):
            raise FeatureValidationError("probe-norm base init ID must be nonnegative")
        if (
            isinstance(self.control_step, bool)
            or not isinstance(self.control_step, Integral)
            or self.control_step < 0
            or self.control_step % 5 != 0
        ):
            raise FeatureValidationError(
                "probe-norm control step must be nonnegative and divisible by five"
            )
        if self.split not in {"calibration", "locked_test"}:
            raise FeatureValidationError(
                "probe-norm split must be calibration or locked_test"
            )
        if self.phase not in PHASE_ORDER:
            raise FeatureValidationError(f"phase must be one of {PHASE_ORDER!r}")
        if not isinstance(self.success, bool):
            raise FeatureValidationError("probe-norm success must be boolean")
        _finite_scalar(self.mean_norm, "probe-norm mean")
        if self.mean_norm < 0.0:
            raise FeatureValidationError("probe-norm mean must be nonnegative")


def robust_probe_norm_distance(
    query: ProbeNormState, calibration_states: Sequence[ProbeNormState]
) -> float:
    """Compute the frozen OOF/full-fit robust norm distance (minimum five refs)."""

    states = tuple(calibration_states)
    if any(state.split != "calibration" for state in states):
        raise FeatureValidationError(
            "probe-norm references must all be Calibration rows"
        )
    identities = [(state.episode_id, state.control_step) for state in states]
    if len(identities) != len(set(identities)):
        raise FeatureValidationError("probe-norm references contain duplicate states")
    references = [
        state
        for state in states
        if state.success
        and state.phase == query.phase
        and (
            query.split == "locked_test"
            or (
                state.episode_id != query.episode_id
                and state.base_init_state_id != query.base_init_state_id
            )
        )
    ]
    if len(references) < 5:
        return math.nan
    values = np.asarray([reference.mean_norm for reference in references])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, ROBUST_SCALE_FLOOR)
    return abs(query.mean_norm - median) / scale


@dataclass(frozen=True)
class M2Primitives:
    """Probe vectors and intervention actions for one selected representation."""

    original_probe_vectors: NDArray[np.float64]
    transformed_probe_vectors: NDArray[np.float64]
    transformed_available: NDArray[np.bool_]
    intervention_actions: NDArray[np.float64]
    intervention_available: NDArray[np.bool_]
    symmetry_order: int
    noise_dependent: bool
    transform_order: tuple[str, ...] = TRANSFORM_ORDER
    intervention_order: tuple[str, ...] = INTERVENTION_ORDER

    def __post_init__(self) -> None:
        original = _readonly_array(
            self.original_probe_vectors,
            dtype=np.float64,
            name="original probe vectors",
            shape=(ACTION_DRAWS, 2),
        )
        transformed = _readonly_array(
            self.transformed_probe_vectors,
            dtype=np.float64,
            name="transformed probe vectors",
            shape=(len(TRANSFORM_ORDER), COMMON_DRAWS, 2),
        )
        available = _readonly_array(
            self.transformed_available,
            dtype=np.bool_,
            name="probe transform availability",
            shape=(len(TRANSFORM_ORDER),),
        )
        intervention = _readonly_array(
            self.intervention_actions,
            dtype=np.float64,
            name="intervention actions",
            shape=(2, COMMON_DRAWS, ACTION_STEPS, ACTION_DIMENSION),
        )
        intervention_available = _readonly_bool_mask(
            self.intervention_available,
            name="intervention availability",
            shape=(COMMON_DRAWS,),
        )
        if tuple(self.transform_order) != TRANSFORM_ORDER:
            raise FeatureValidationError(
                "probe transform order differs from frozen order"
            )
        if tuple(self.intervention_order) != INTERVENTION_ORDER:
            raise FeatureValidationError(
                "intervention order differs from frozen negative/positive order"
            )
        if not isinstance(self.noise_dependent, bool):
            raise FeatureValidationError("noise_dependent must be boolean")
        if not np.isfinite(original).all():
            raise FeatureValidationError("original probe vectors must be finite")
        for index, is_available in enumerate(available):
            block = transformed[index]
            if is_available and not np.isfinite(block).all():
                raise FeatureValidationError(
                    "available transformed probes must be finite"
                )
            if not is_available and not np.isnan(block).all():
                raise FeatureValidationError(
                    "unavailable transformed probes must be NaN"
                )
        for draw, is_available in enumerate(intervention_available):
            draw_actions = intervention[:, draw]
            if is_available and not np.isfinite(draw_actions).all():
                raise FeatureValidationError(
                    "available intervention draw actions must be finite"
                )
            if not is_available and not np.isnan(draw_actions).all():
                raise FeatureValidationError(
                    "unavailable intervention draw actions must be NaN"
                )
        object.__setattr__(self, "original_probe_vectors", original)
        object.__setattr__(self, "transformed_probe_vectors", transformed)
        object.__setattr__(self, "transformed_available", available)
        object.__setattr__(self, "intervention_actions", intervention)
        object.__setattr__(self, "intervention_available", intervention_available)
        object.__setattr__(
            self,
            "symmetry_order",
            _positive_integer(self.symmetry_order, "symmetry_order"),
        )
        object.__setattr__(self, "transform_order", tuple(self.transform_order))
        object.__setattr__(self, "intervention_order", tuple(self.intervention_order))

    @classmethod
    def from_arrays(
        cls,
        *,
        original_probe_vectors: ArrayLike,
        transformed_probe_vectors: ArrayLike,
        transformed_available: ArrayLike,
        intervention_actions: ArrayLike,
        intervention_available: ArrayLike,
        symmetry_order: int,
        noise_dependent: bool,
        transform_order: Sequence[str] = TRANSFORM_ORDER,
        intervention_order: Sequence[str] = INTERVENTION_ORDER,
    ) -> M2Primitives:
        original = _readonly_array(
            original_probe_vectors,
            dtype=np.float64,
            name="original probe vectors",
            shape=(ACTION_DRAWS, 2),
        )
        transformed = _readonly_array(
            transformed_probe_vectors,
            dtype=np.float64,
            name="transformed probe vectors",
            shape=(len(TRANSFORM_ORDER), COMMON_DRAWS, 2),
        )
        available = _readonly_array(
            transformed_available,
            dtype=np.bool_,
            name="probe transform availability",
            shape=(len(TRANSFORM_ORDER),),
        )
        intervention = _readonly_array(
            intervention_actions,
            dtype=np.float64,
            name="intervention actions",
            shape=(2, COMMON_DRAWS, ACTION_STEPS, ACTION_DIMENSION),
        )
        if tuple(transform_order) != TRANSFORM_ORDER:
            raise FeatureValidationError(
                "probe transform order differs from frozen order"
            )
        if tuple(intervention_order) != INTERVENTION_ORDER:
            raise FeatureValidationError(
                "intervention order differs from frozen negative/positive order"
            )
        if not isinstance(noise_dependent, bool):
            raise FeatureValidationError("noise_dependent must be boolean")
        if not np.isfinite(original).all():
            raise FeatureValidationError("original probe vectors must be finite")
        for index, is_available in enumerate(available):
            block = transformed[index]
            if is_available and not np.isfinite(block).all():
                raise FeatureValidationError(
                    "available transformed probes must be finite"
                )
            if not is_available and not np.isnan(block).all():
                raise FeatureValidationError(
                    "unavailable transformed probes must be NaN"
                )
        return cls(
            original,
            transformed,
            available,
            intervention,
            np.asarray(intervention_available),
            _positive_integer(symmetry_order, "symmetry_order"),
            noise_dependent,
            tuple(transform_order),
            tuple(intervention_order),
        )


def _probe_angles(vectors: NDArray[np.float64]) -> NDArray[np.float64] | None:
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(norms <= PROBE_NORM_FLOOR):
        return None
    return np.arctan2(vectors[..., 1], vectors[..., 0])


def _wrapped_angle(value: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.arctan2(np.sin(value), np.cos(value))


def _family_probe_dispersion(primitives: M2Primitives, family_index: int) -> float:
    transform_slice = slice(2 * family_index, 2 * family_index + 2)
    if not bool(np.all(primitives.transformed_available[transform_slice])):
        return math.nan
    original_angles = _probe_angles(primitives.original_probe_vectors[:COMMON_DRAWS])
    transformed_angles = _probe_angles(
        primitives.transformed_probe_vectors[transform_slice]
    )
    if original_angles is None or transformed_angles is None:
        return math.nan
    values = []
    for draw in range(COMMON_DRAWS):
        angles = np.asarray(
            [
                original_angles[draw],
                transformed_angles[0, draw],
                transformed_angles[1, draw],
            ]
        )
        values.append(1.0 - abs(np.mean(np.exp(1j * angles))))
    return float(np.mean(values))


def compute_m2_features(
    primitives: M2Primitives,
    scale: ActionScale,
    *,
    robust_norm_distance: float,
) -> NamedFeatureRow:
    """Compute the selected-probe geometry and intervention feature increment."""

    robust_distance = float(robust_norm_distance)
    if math.isinf(robust_distance):
        raise FeatureValidationError("robust norm distance may be NaN but not infinity")
    original_angles = _probe_angles(primitives.original_probe_vectors[:COMMON_DRAWS])
    object_available = bool(np.all(primitives.transformed_available[4:6]))
    transformed_object_angles = (
        _probe_angles(primitives.transformed_probe_vectors[4:6])
        if object_available
        else None
    )
    if original_angles is None or transformed_object_angles is None:
        object_error = math.nan
    else:
        errors = []
        for transform_index, delta in enumerate(OBJECT_YAW_DELTAS_RAD):
            expected = original_angles - primitives.symmetry_order * delta
            error_scaled = _wrapped_angle(
                transformed_object_angles[transform_index] - expected
            )
            errors.extend((np.abs(error_scaled) / primitives.symmetry_order).tolist())
        object_error = float(np.mean(errors))

    brightness_dispersion = _family_probe_dispersion(primitives, 0)
    camera_dispersion = _family_probe_dispersion(primitives, 1)
    mean_norm = float(
        np.mean(np.linalg.norm(primitives.original_probe_vectors, axis=1))
    )
    values = [
        object_error,
        brightness_dispersion,
        camera_dispersion,
        mean_norm,
        robust_distance,
    ]
    names = list(M2_BASE_FEATURE_NAMES)

    all_original_angles = _probe_angles(primitives.original_probe_vectors)
    if primitives.noise_dependent:
        noise_dispersion = (
            math.nan
            if all_original_angles is None
            else float(1.0 - abs(np.mean(np.exp(1j * all_original_angles))))
        )
        names.append(M2_EXPERT_NOISE_FEATURE_NAME)
        values.append(noise_dispersion)

    zero_current_probe = original_angles is None
    if not bool(np.all(primitives.intervention_available)) or zero_current_probe:
        controllability = math.nan
        specificity = math.nan
    else:
        standardized = primitives.intervention_actions / scale.values
        mean_by_direction = np.mean(standardized, axis=(1, 2))
        central_difference = (
            mean_by_direction[1] - mean_by_direction[0]
        ) / INTERVENTION_SPAN_RAD
        yaw_difference = float(central_difference[YAW_ACTION_INDEX])
        controllability = abs(yaw_difference)
        non_yaw = np.delete(central_difference, YAW_ACTION_INDEX)
        specificity = (
            math.nan
            if yaw_difference == 0.0
            else float(np.linalg.norm(non_yaw) / abs(yaw_difference))
        )
    names.extend(M2_INTERVENTION_FEATURE_NAMES)
    values.extend((controllability, specificity))
    metadata = canonical_feature_metadata(
        "M2_expert" if primitives.noise_dependent else "M2_vlm"
    )
    metadata["action_scale_sha256"] = scale.metadata_sha256
    return NamedFeatureRow(tuple(names), np.asarray(values), metadata)


@dataclass(frozen=True)
class FeatureHierarchy:
    """Exact nested M0, M1, and M2 rows consumed by predictor fitting."""

    m0: NamedFeatureRow
    m1: NamedFeatureRow
    m2: NamedFeatureRow

    def __post_init__(self) -> None:
        if self.m0.names != M0_FEATURE_NAMES:
            raise FeatureValidationError("feature hierarchy M0 columns drifted")
        if self.m1.names != M1_FEATURE_NAMES:
            raise FeatureValidationError("feature hierarchy M1 columns drifted")
        if self.m2.names not in {M2_VLM_FEATURE_NAMES, M2_EXPERT_FEATURE_NAMES}:
            raise FeatureValidationError("feature hierarchy M2 columns drifted")
        if not np.array_equal(
            self.m0.values, self.m1.values[: len(M0_FEATURE_NAMES)], equal_nan=True
        ):
            raise FeatureValidationError("M1 does not preserve the exact M0 prefix")
        if not np.array_equal(
            self.m1.values, self.m2.values[: len(M1_FEATURE_NAMES)], equal_nan=True
        ):
            raise FeatureValidationError("M2 does not preserve the exact M1 prefix")

    @property
    def metadata_sha256(self) -> str:
        return _metadata_hash(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "m0_sha256": self.m0.metadata_sha256,
                "m1_sha256": self.m1.metadata_sha256,
                "m2_sha256": self.m2.metadata_sha256,
                "column_order": {
                    "M0": list(self.m0.names),
                    "M1": list(self.m1.names),
                    "M2": list(self.m2.names),
                },
            }
        )


def assemble_feature_hierarchy(
    m0: NamedFeatureRow,
    m1_raw: NamedFeatureRow,
    coverage: NamedFeatureRow,
    m2_increment: NamedFeatureRow,
) -> FeatureHierarchy:
    """Append the component rows in the one frozen model-column order."""

    if m0.names != M0_FEATURE_NAMES:
        raise FeatureValidationError("m0 row does not use the frozen schema")
    if m1_raw.names != M1_RAW_FEATURE_NAMES:
        raise FeatureValidationError("m1 raw row does not use the frozen schema")
    if coverage.names != COVERAGE_FEATURE_NAMES:
        raise FeatureValidationError("coverage row does not use the frozen schema")
    m2_vlm_increment = (*M2_BASE_FEATURE_NAMES, *M2_INTERVENTION_FEATURE_NAMES)
    m2_expert_increment = (
        *M2_BASE_FEATURE_NAMES,
        M2_EXPERT_NOISE_FEATURE_NAME,
        *M2_INTERVENTION_FEATURE_NAMES,
    )
    if m2_increment.names not in {m2_vlm_increment, m2_expert_increment}:
        raise FeatureValidationError("m2 increment does not use a frozen schema")
    m1_values = np.concatenate((m0.values, m1_raw.values, coverage.values))
    m2_values = np.concatenate((m1_values, m2_increment.values))
    component_hashes = {
        "m0": m0.metadata_sha256,
        "m1_raw": m1_raw.metadata_sha256,
        "coverage": coverage.metadata_sha256,
        "m2_increment": m2_increment.metadata_sha256,
    }
    m1 = NamedFeatureRow(
        M1_FEATURE_NAMES,
        m1_values,
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "component": "M1_full",
            "feature_names": list(M1_FEATURE_NAMES),
            "component_hashes": component_hashes,
        },
    )
    m2_names = (
        M2_EXPERT_FEATURE_NAMES
        if m2_increment.names == m2_expert_increment
        else M2_VLM_FEATURE_NAMES
    )
    m2 = NamedFeatureRow(
        m2_names,
        m2_values,
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "component": "M2_full",
            "feature_names": list(m2_names),
            "component_hashes": component_hashes,
        },
    )
    return FeatureHierarchy(m0, m1, m2)


def canonical_feature_metadata(component: str) -> dict[str, Any]:
    """Return the canonical numerical contract for one feature component."""

    schemas: dict[str, tuple[str, ...]] = {
        "M0": M0_FEATURE_NAMES,
        "M1_raw": M1_RAW_FEATURE_NAMES,
        "M1_coverage": COVERAGE_FEATURE_NAMES,
        "M2_vlm": (*M2_BASE_FEATURE_NAMES, *M2_INTERVENTION_FEATURE_NAMES),
        "M2_expert": (
            *M2_BASE_FEATURE_NAMES,
            M2_EXPERT_NOISE_FEATURE_NAME,
            *M2_INTERVENTION_FEATURE_NAMES,
        ),
    }
    if component not in schemas:
        raise FeatureValidationError(f"unknown feature component {component!r}")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "component": component,
        "feature_names": list(schemas[component]),
        "action_shape": [ACTION_DRAWS, ACTION_STEPS, ACTION_DIMENSION],
        "common_draws": COMMON_DRAWS,
        "transform_order": list(TRANSFORM_ORDER),
        "transform_family_order": list(TRANSFORM_FAMILIES),
        "intervention_order": list(INTERVENTION_ORDER),
        "phase_order": list(PHASE_ORDER),
        "action_distance": "l2(delta/action_scale)/sqrt(7)",
        "output_covariance": "population_divisor_8_ordinary_trace",
        "percentile": "numpy_linear",
        "coverage_distance": "euclidean_rms",
        "coverage_ties": ["distance", "episode_id", "control_step"],
        "probe_circle": "phi=s*theta",
        "probe_norm_floor": PROBE_NORM_FLOOR,
        "robust_norm_scale": "max(1.4826*MAD,1e-8)",
        "robust_norm_minimum_references": 5,
        "intervention_span_rad": INTERVENTION_SPAN_RAD,
        "intervention_availability": "all_four_common_draws_required",
    }


def canonical_feature_metadata_sha256(component: str) -> str:
    """Hash the canonical schema metadata without runtime-dependent values."""

    return _metadata_hash(canonical_feature_metadata(component))


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_DRAWS",
    "ACTION_SCALE_FLOOR",
    "ACTION_STEPS",
    "COMMON_DRAWS",
    "COVERAGE_FEATURE_NAMES",
    "COVERAGE_VECTOR_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "INTERVENTION_ORDER",
    "M0_FEATURE_NAMES",
    "M1_FEATURE_NAMES",
    "M1_RAW_FEATURE_NAMES",
    "M2_BASE_FEATURE_NAMES",
    "M2_EXPERT_FEATURE_NAMES",
    "M2_EXPERT_NOISE_FEATURE_NAME",
    "M2_INTERVENTION_FEATURE_NAMES",
    "M2_VLM_FEATURE_NAMES",
    "OBJECT_YAW_DELTAS_RAD",
    "PHASE_ORDER",
    "PROBE_NORM_FLOOR",
    "TRANSFORM_ORDER",
    "ActionScale",
    "CoverageFit",
    "CoverageState",
    "FeatureHierarchy",
    "FeatureValidationError",
    "M0Primitives",
    "M1PoseState",
    "M2Primitives",
    "NamedFeatureRow",
    "ProbeNormState",
    "assemble_feature_hierarchy",
    "canonical_feature_metadata",
    "canonical_feature_metadata_sha256",
    "compute_coverage_features",
    "compute_m0_features",
    "compute_m2_features",
    "construct_m1_raw_pose",
    "fit_coverage_reference",
    "fit_episode_equal_action_scale",
    "m1_coverage_vector",
    "query_coverage_features",
    "robust_probe_norm_distance",
]
