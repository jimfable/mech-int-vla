"""Fail-closed loading and probe-cohort assembly for rollout artifacts.

This module is deliberately a consumer of already-completed artifacts.  It does
not discover files, choose episodes from their outcomes, or run policy code.
Callers must provide both the artifact directories and the complete requested
episode-ID set.  Valid episodes contribute pre-action rows on a deterministic
control-step stride; invalid resets are reported and excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import TaskSpec
from .probes import ProbeSamples

ARTIFACT_SCHEMA_VERSION = 1
ACTION_DIMENSION = 7
POLICY_STATE_DIMENSION = 8
ACTIVATION_SCORE_STRIDE_STEPS = 5
PHASES = ("pregrasp", "grasped", "transport", "placed")
RAW_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
RAW_IMAGE_SHAPE = (360, 360, 3)
ACTIVATION_CANDIDATES = (
    "vlm_context",
    "early_expert_t1_0",
    "early_expert_t0_5",
    "late_expert_t1_0",
    "late_expert_t0_5",
)
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
    "symmetry_eef_object_yaw_sin",
    "symmetry_eef_object_yaw_cos",
    "symmetry_object_goal_yaw_sin",
    "symmetry_object_goal_yaw_cos",
)
SPLITS = frozenset({"discovery", "calibration", "locked_test"})

_BASE_ARRAY_NAMES = frozenset(
    {
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "frame_control_step",
        "frame_simulator_time",
        "frame_policy_state",
        "frame_agentview_image",
        "frame_robot0_eye_in_hand_image",
        "frame_eef_position",
        "frame_eef_quaternion_xyzw",
        "frame_primary_object_position",
        "frame_primary_object_quaternion_wxyz",
        "frame_gripper_qpos",
        "frame_gripper_qvel",
        "frame_primary_gripper_contact",
        "frame_primary_grasped",
        "frame_task_success",
        "frame_phase",
        "frame_scalar_features",
        "frame_goal_present",
        "frame_goal_position",
        "frame_goal_quaternion_wxyz",
        "frame_task_predicates",
        "activation_control_step",
    }
)
_ARRAY_NAMES = _BASE_ARRAY_NAMES | frozenset(
    f"activation_{candidate}" for candidate in ACTIVATION_CANDIDATES
)


class ArtifactValidationError(ValueError):
    """Raised when an artifact or requested cohort fails the frozen contract."""


@dataclass(frozen=True)
class ArtifactHashes:
    """Content hashes of the two inputs consumed by the loader."""

    metadata_sha256: str
    trajectory_sha256: str


@dataclass(frozen=True)
class RolloutArtifact:
    """A validated, immutable in-memory view of one rollout artifact."""

    path: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, NDArray[Any]]
    hashes: ArtifactHashes

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode"]["episode_id"])

    @property
    def valid_reset(self) -> bool:
        return bool(self.metadata["validity"]["valid"])

    @property
    def success(self) -> bool:
        return bool(self.metadata["outcome"]["success"])

    @property
    def action_count(self) -> int:
        return int(self.arrays["actions"].shape[0])


@dataclass(frozen=True)
class CohortManifest:
    """Portable provenance for an explicitly requested probe cohort."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    def canonical_json(self) -> str:
        return json.dumps(
            _plain_json(self.payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProbeCohort:
    """Aligned probe rows assembled without outcome-dependent row selection."""

    samples: ProbeSamples
    activation_features: Mapping[str, NDArray[np.float32]]
    control_step: NDArray[np.int64]
    failure_label: NDArray[np.bool_]
    valid_episode_ids: tuple[str, ...]
    invalid_reset_episode_ids: tuple[str, ...]
    manifest: CohortManifest

    @property
    def episode_id(self) -> NDArray[np.str_]:
        return self.samples.episode_id

    @property
    def base_init_state_id(self) -> NDArray[np.int64]:
        return self.samples.base_init_state_id

    @property
    def theta_rel(self) -> NDArray[np.float64]:
        return self.samples.theta_rel

    @property
    def manifest_sha256(self) -> str:
        return self.manifest.sha256


def probe_cohort_array_sha256(value: NDArray[Any]) -> str:
    """Hash one logical probe-cohort array with dtype and shape framing."""

    array = np.array(value, copy=True, order="C")
    if array.dtype.kind == "f" and np.isnan(array).any():
        array[np.isnan(array)] = np.nan
    canonical = np.ascontiguousarray(
        array.astype(array.dtype.newbyteorder("<"), copy=False)
    )
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"mech-int-vla/probe-training-array/v1\0")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _fail(where: str, message: str) -> None:
    raise ArtifactValidationError(f"{where}: {message}")


def _freeze_json(value: Any) -> Any:
    """Recursively copy JSON containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _plain_json(value: Any) -> Any:
    """Convert recursively frozen JSON values back to encoder-native values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(nested) for nested in value]
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(stat_result: Any) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"metadata.json: duplicate key {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        _fail(where, "; ".join(details))


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _fail(where, "must be a nonempty string" if nonempty else "must be a string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, "must be a boolean")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(where, f"must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(where, "must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        _fail(where, "must be a finite number")
    return converted


def _optional_finite_number(value: Any, where: str) -> float | None:
    return None if value is None else _finite_number(value, where)


def _validate_settle_actions(value: Any, where: str) -> None:
    if not isinstance(value, list) or len(value) != 10:
        _fail(where, "must contain exactly ten action vectors")
    for action_index, action in enumerate(value):
        if not isinstance(action, list) or len(action) != ACTION_DIMENSION:
            _fail(
                f"{where}[{action_index}]",
                f"must be a {ACTION_DIMENSION}-element action vector",
            )
        for value_index, component in enumerate(action):
            _finite_number(component, f"{where}[{action_index}][{value_index}]")


def _string_list(value: Any, where: str, *, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(where, "must be a JSON array")
    result = tuple(
        _string(item, f"{where}[{index}]") for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        _fail(where, "must not contain duplicates")
    return result


def _safe_episode_id(value: Any, where: str = "episode.episode_id") -> str:
    episode_id = _string(value, where)
    if (
        Path(episode_id).name != episode_id
        or episode_id in {".", ".."}
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in episode_id
        )
    ):
        _fail(where, "is not a safe artifact directory name")
    return episode_id


def _validate_metadata(metadata: Any, directory: Path) -> dict[str, Any]:
    root = _mapping(metadata, "metadata")
    _exact_keys(
        root,
        {
            "schema_version",
            "episode",
            "task_language",
            "task",
            "condition",
            "model",
            "execution",
            "validity",
            "validity_retry",
            "outcome",
            "capture",
            "files",
        },
        "metadata",
    )
    if (
        _integer(root["schema_version"], "schema_version", minimum=1)
        != ARTIFACT_SCHEMA_VERSION
    ):
        _fail("schema_version", f"must equal {ARTIFACT_SCHEMA_VERSION}")

    episode = _mapping(root["episode"], "episode")
    _exact_keys(
        episode,
        {
            "episode_id",
            "suite",
            "task_id",
            "task_rank",
            "split",
            "base_init_state_id",
            "condition_index",
            "condition_name",
            "condition_family",
            "condition_parameters",
            "reset_seed",
            "inference_seed",
            "policy_revision",
            "code_commit",
        },
        "episode",
    )
    episode_id = _safe_episode_id(episode["episode_id"])
    suite = _string(episode["suite"], "episode.suite")
    task_id = _integer(episode["task_id"], "episode.task_id")
    _integer(episode["task_rank"], "episode.task_rank", minimum=1)
    split = _string(episode["split"], "episode.split")
    if split not in SPLITS:
        _fail("episode.split", f"must be one of {sorted(SPLITS)}")
    base_init = _integer(episode["base_init_state_id"], "episode.base_init_state_id")
    condition_index = _integer(episode["condition_index"], "episode.condition_index")
    if condition_index > 9:
        _fail("episode.condition_index", "must be in [0, 9]")
    condition_name = _string(episode["condition_name"], "episode.condition_name")
    condition_family = _string(episode["condition_family"], "episode.condition_family")
    _mapping(episode["condition_parameters"], "episode.condition_parameters")
    _integer(episode["reset_seed"], "episode.reset_seed")
    _integer(episode["inference_seed"], "episode.inference_seed")
    policy_revision = _string(episode["policy_revision"], "episode.policy_revision")
    _string(episode["code_commit"], "episode.code_commit")

    expected_id = (
        f"{suite}-task{task_id}-{split}-init{base_init:02d}-cell{condition_index}"
    )
    if episode_id != expected_id:
        _fail("episode.episode_id", f"must equal generated ID {expected_id!r}")
    if directory.name != episode_id:
        _fail("episode.episode_id", "does not match the artifact directory name")
    if directory.parent.name != split:
        _fail("episode.split", "does not match the artifact parent directory name")
    task_language = _string(root["task_language"], "task_language")

    task = _mapping(root["task"], "task")
    _exact_keys(
        task,
        {
            "rank",
            "suite",
            "task_id",
            "language",
            "primary_object",
            "planar_symmetry_order",
        },
        "task",
    )
    if _integer(task["rank"], "task.rank", minimum=1) != episode["task_rank"]:
        _fail("task.rank", "does not match episode.task_rank")
    if _string(task["suite"], "task.suite") != suite:
        _fail("task.suite", "does not match episode.suite")
    if _integer(task["task_id"], "task.task_id") != task_id:
        _fail("task.task_id", "does not match episode.task_id")
    if _string(task["language"], "task.language") != task_language:
        _fail("task.language", "does not match task_language")
    _string(task["primary_object"], "task.primary_object")
    _integer(task["planar_symmetry_order"], "task.planar_symmetry_order", minimum=1)

    condition = _mapping(root["condition"], "condition")
    _exact_keys(condition, {"name", "family", "index", "parameters"}, "condition")
    if _string(condition["name"], "condition.name") != condition_name:
        _fail("condition.name", "does not match episode.condition_name")
    if _string(condition["family"], "condition.family") != condition_family:
        _fail("condition.family", "does not match episode.condition_family")
    if _integer(condition["index"], "condition.index") != condition_index:
        _fail("condition.index", "does not match episode.condition_index")
    parameters = _mapping(condition["parameters"], "condition.parameters")
    if parameters != episode["condition_parameters"]:
        _fail("condition.parameters", "does not match episode.condition_parameters")

    model = _mapping(root["model"], "model")
    _exact_keys(model, {"policy_revision", "base_vlm_revision"}, "model")
    if _string(model["policy_revision"], "model.policy_revision") != policy_revision:
        _fail("model.policy_revision", "does not match episode.policy_revision")
    _string(model["base_vlm_revision"], "model.base_vlm_revision")

    execution = _mapping(root["execution"], "execution")
    _exact_keys(
        execution,
        {
            "n_action_steps",
            "max_steps",
            "reset_noop_steps",
            "settle_actions",
            "closed_loop_replanning",
        },
        "execution",
    )
    if (
        _integer(execution["n_action_steps"], "execution.n_action_steps", minimum=1)
        != 1
    ):
        _fail("execution.n_action_steps", "must equal 1")
    _integer(execution["max_steps"], "execution.max_steps", minimum=1)
    _integer(execution["reset_noop_steps"], "execution.reset_noop_steps")
    _validate_settle_actions(execution["settle_actions"], "execution.settle_actions")
    if not _boolean(
        execution["closed_loop_replanning"], "execution.closed_loop_replanning"
    ):
        _fail("execution.closed_loop_replanning", "must be true")

    validity = _mapping(root["validity"], "validity")
    _exact_keys(
        validity,
        {
            "valid",
            "reasons",
            "finite",
            "deepest_primary_penetration_m",
            "linear_speed_m_s",
            "angular_speed_rad_s",
            "in_workspace",
            "initial_success",
        },
        "validity",
    )
    valid = _boolean(validity["valid"], "validity.valid")
    reasons = _string_list(validity["reasons"], "validity.reasons", unique=True)
    finite_validity = _boolean(validity["finite"], "validity.finite")
    _optional_finite_number(
        validity["deepest_primary_penetration_m"],
        "validity.deepest_primary_penetration_m",
    )
    _optional_finite_number(validity["linear_speed_m_s"], "validity.linear_speed_m_s")
    _optional_finite_number(
        validity["angular_speed_rad_s"], "validity.angular_speed_rad_s"
    )
    in_workspace = _boolean(validity["in_workspace"], "validity.in_workspace")
    initial_success = _boolean(validity["initial_success"], "validity.initial_success")
    if valid and reasons:
        _fail("validity.reasons", "must be empty for a valid reset")
    if not valid and not reasons:
        _fail("validity.reasons", "must explain an invalid reset")
    if valid and (not finite_validity or not in_workspace or initial_success):
        _fail(
            "validity",
            "valid reset must be finite, inside the workspace, and initially unsuccessful",
        )

    validity_retry = _mapping(root["validity_retry"], "validity_retry")
    retry_performed = _boolean(
        validity_retry.get("performed"), "validity_retry.performed"
    )
    if valid:
        _exact_keys(validity_retry, {"performed"}, "validity_retry")
        if retry_performed:
            _fail("validity_retry.performed", "must be false for a valid reset")
    else:
        _exact_keys(
            validity_retry,
            {
                "performed",
                "same_reset_seed_and_condition",
                "validity",
                "settle_actions",
                "agrees_on_invalidity",
                "agrees_on_reasons",
            },
            "validity_retry",
        )
        if not retry_performed:
            _fail("validity_retry.performed", "must be true for an invalid reset")
        if not _boolean(
            validity_retry["same_reset_seed_and_condition"],
            "validity_retry.same_reset_seed_and_condition",
        ):
            _fail(
                "validity_retry.same_reset_seed_and_condition",
                "must be true",
            )
        retry_validity = _mapping(validity_retry["validity"], "validity_retry.validity")
        _exact_keys(retry_validity, set(validity), "validity_retry.validity")
        retry_valid = _boolean(retry_validity["valid"], "validity_retry.validity.valid")
        retry_reasons = _string_list(
            retry_validity["reasons"],
            "validity_retry.validity.reasons",
            unique=True,
        )
        _boolean(retry_validity["finite"], "validity_retry.validity.finite")
        for field in (
            "deepest_primary_penetration_m",
            "linear_speed_m_s",
            "angular_speed_rad_s",
        ):
            _optional_finite_number(
                retry_validity[field], f"validity_retry.validity.{field}"
            )
        _boolean(
            retry_validity["in_workspace"],
            "validity_retry.validity.in_workspace",
        )
        _boolean(
            retry_validity["initial_success"],
            "validity_retry.validity.initial_success",
        )
        _validate_settle_actions(
            validity_retry["settle_actions"], "validity_retry.settle_actions"
        )
        agrees_invalidity = _boolean(
            validity_retry["agrees_on_invalidity"],
            "validity_retry.agrees_on_invalidity",
        )
        agrees_reasons = _boolean(
            validity_retry["agrees_on_reasons"],
            "validity_retry.agrees_on_reasons",
        )
        if agrees_invalidity != (not retry_valid):
            _fail(
                "validity_retry.agrees_on_invalidity",
                "is inconsistent with retry validity",
            )
        if agrees_reasons != (retry_reasons == reasons):
            _fail(
                "validity_retry.agrees_on_reasons",
                "is inconsistent with retry reasons",
            )

    outcome = _mapping(root["outcome"], "outcome")
    _exact_keys(
        outcome,
        {
            "status",
            "success",
            "terminated",
            "truncated",
            "control_steps",
            "reward_sum",
            "terminal_state_preserved",
        },
        "outcome",
    )
    status = _string(outcome["status"], "outcome.status")
    if status not in {
        "invalid_reset",
        "success",
        "truncated",
        "terminated_without_success",
    }:
        _fail("outcome.status", "is not a recognized rollout status")
    success = _boolean(outcome["success"], "outcome.success")
    terminated = _boolean(outcome["terminated"], "outcome.terminated")
    truncated = _boolean(outcome["truncated"], "outcome.truncated")
    _integer(outcome["control_steps"], "outcome.control_steps")
    _finite_number(outcome["reward_sum"], "outcome.reward_sum")
    terminal_preserved = _boolean(
        outcome["terminal_state_preserved"], "outcome.terminal_state_preserved"
    )
    expected_status = (
        "invalid_reset"
        if not valid
        else "success"
        if success
        else "truncated"
        if truncated
        else "terminated_without_success"
        if terminated
        else ""
    )
    if status != expected_status:
        _fail("outcome.status", "is inconsistent with validity and terminal flags")
    if success and not terminated:
        _fail("outcome", "a successful episode must be terminated")
    if terminated and truncated:
        _fail("outcome", "terminated and truncated cannot both be true")
    if valid != terminal_preserved:
        _fail("outcome.terminal_state_preserved", "must equal validity.valid")
    if not valid and (success or terminated or truncated):
        _fail("outcome", "invalid reset cannot have a terminal outcome")

    capture = _mapping(root["capture"], "capture")
    _exact_keys(
        capture,
        {
            "policy_select_calls",
            "scored_policy_select_calls",
            "score_stride_steps",
            "instrumented_internal_calls",
            "activation_candidates",
            "activation_dtype",
            "frame_count",
            "frame_scalar_feature_names",
            "task_predicate_names",
            "raw_images_stored",
            "raw_image_encoding",
            "raw_image_observation_keys",
            "raw_image_shape",
        },
        "capture",
    )
    _integer(capture["policy_select_calls"], "capture.policy_select_calls")
    _integer(
        capture["scored_policy_select_calls"],
        "capture.scored_policy_select_calls",
    )
    if (
        _integer(
            capture["score_stride_steps"],
            "capture.score_stride_steps",
            minimum=1,
        )
        != ACTIVATION_SCORE_STRIDE_STEPS
    ):
        _fail(
            "capture.score_stride_steps",
            f"must equal {ACTIVATION_SCORE_STRIDE_STEPS}",
        )
    _integer(
        capture["instrumented_internal_calls"],
        "capture.instrumented_internal_calls",
    )
    candidates = _string_list(
        capture["activation_candidates"],
        "capture.activation_candidates",
        unique=True,
    )
    if candidates != ACTIVATION_CANDIDATES:
        _fail("capture.activation_candidates", "must equal the frozen five candidates")
    if _string(capture["activation_dtype"], "capture.activation_dtype") != "float32":
        _fail("capture.activation_dtype", "must equal 'float32'")
    _integer(capture["frame_count"], "capture.frame_count", minimum=1)
    scalar_names = _string_list(
        capture["frame_scalar_feature_names"],
        "capture.frame_scalar_feature_names",
        unique=True,
    )
    if scalar_names != FRAME_SCALAR_FEATURE_NAMES:
        _fail(
            "capture.frame_scalar_feature_names",
            "must equal the frozen scalar-feature schema",
        )
    _string_list(
        capture["task_predicate_names"], "capture.task_predicate_names", unique=True
    )
    if not _boolean(capture["raw_images_stored"], "capture.raw_images_stored"):
        _fail("capture.raw_images_stored", "must be true")
    if (
        _string(capture["raw_image_encoding"], "capture.raw_image_encoding")
        != "lossless_uint8_npz_deflate"
    ):
        _fail(
            "capture.raw_image_encoding",
            "must equal 'lossless_uint8_npz_deflate'",
        )
    raw_image_keys = _string_list(
        capture["raw_image_observation_keys"],
        "capture.raw_image_observation_keys",
        unique=True,
    )
    if raw_image_keys != RAW_IMAGE_KEYS:
        _fail(
            "capture.raw_image_observation_keys",
            "must equal the frozen camera-key order",
        )
    raw_image_shape = capture["raw_image_shape"]
    if not isinstance(raw_image_shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_image_shape
    ):
        _fail("capture.raw_image_shape", "must be an integer shape array")
    if tuple(raw_image_shape) != RAW_IMAGE_SHAPE:
        _fail(
            "capture.raw_image_shape",
            f"must equal {list(RAW_IMAGE_SHAPE)}",
        )

    files = _mapping(root["files"], "files")
    _exact_keys(files, {"trajectory"}, "files")
    if _string(files["trajectory"], "files.trajectory") != "trajectory.npz":
        _fail("files.trajectory", "must equal 'trajectory.npz'")
    return dict(root)


def _expect_array(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int | None, ...],
) -> NDArray[Any]:
    value = arrays[name]
    if value.dtype != dtype:
        _fail(f"trajectory.{name}", f"must have dtype {dtype}, got {value.dtype}")
    if value.ndim != len(shape) or any(
        expected is not None and value.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        rendered = tuple("*" if item is None else item for item in shape)
        _fail(f"trajectory.{name}", f"must have shape {rendered}, got {value.shape}")
    return value


def _assert_finite(value: NDArray[Any], where: str) -> None:
    if not np.isfinite(value).all():
        _fail(where, "must contain only finite values")


def _validate_arrays(
    arrays: Mapping[str, NDArray[Any]], metadata: Mapping[str, Any]
) -> None:
    actual_names = set(arrays)
    if actual_names != _ARRAY_NAMES:
        missing = sorted(_ARRAY_NAMES - actual_names)
        unexpected = sorted(actual_names - _ARRAY_NAMES)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        _fail("trajectory", "; ".join(details))

    actions = _expect_array(
        arrays, "actions", dtype=np.dtype(np.float32), shape=(None, ACTION_DIMENSION)
    )
    action_count = actions.shape[0]
    frame_count = action_count + 1
    _assert_finite(actions, "trajectory.actions")
    for name in ("frame_agentview_image", "frame_robot0_eye_in_hand_image"):
        _expect_array(
            arrays,
            name,
            dtype=np.dtype(np.uint8),
            shape=(frame_count, *RAW_IMAGE_SHAPE),
        )
    rewards = _expect_array(
        arrays, "rewards", dtype=np.dtype(np.float32), shape=(action_count,)
    )
    terminated = _expect_array(
        arrays, "terminated", dtype=np.dtype(np.bool_), shape=(action_count,)
    )
    truncated = _expect_array(
        arrays, "truncated", dtype=np.dtype(np.bool_), shape=(action_count,)
    )
    _assert_finite(rewards, "trajectory.rewards")
    if np.any(terminated & truncated):
        _fail("trajectory", "no action may be both terminated and truncated")

    activation_steps = _expect_array(
        arrays,
        "activation_control_step",
        dtype=np.dtype(np.int32),
        shape=(None,),
    )
    expected_activation_steps = np.arange(
        0, action_count, ACTIVATION_SCORE_STRIDE_STEPS, dtype=np.int32
    )
    if not np.array_equal(activation_steps, expected_activation_steps):
        _fail(
            "trajectory.activation_control_step",
            "must equal the strictly increasing pre-action stride-5 steps",
        )
    activation_count = int(activation_steps.size)

    steps = _expect_array(
        arrays, "frame_control_step", dtype=np.dtype(np.int32), shape=(frame_count,)
    )
    if not np.array_equal(steps, np.arange(frame_count, dtype=np.int32)):
        _fail("trajectory.frame_control_step", "must be sequential from zero")
    simulator_time = _expect_array(
        arrays,
        "frame_simulator_time",
        dtype=np.dtype(np.float64),
        shape=(frame_count,),
    )
    _assert_finite(simulator_time, "trajectory.frame_simulator_time")
    if np.any(np.diff(simulator_time) < 0.0):
        _fail("trajectory.frame_simulator_time", "must be nondecreasing")

    finite_fields = {
        "frame_policy_state": (
            np.dtype(np.float32),
            (frame_count, POLICY_STATE_DIMENSION),
        ),
        "frame_eef_position": (np.dtype(np.float64), (frame_count, 3)),
        "frame_eef_quaternion_xyzw": (np.dtype(np.float64), (frame_count, 4)),
        "frame_primary_object_position": (np.dtype(np.float64), (frame_count, 3)),
        "frame_primary_object_quaternion_wxyz": (
            np.dtype(np.float64),
            (frame_count, 4),
        ),
        "frame_gripper_qpos": (np.dtype(np.float64), (frame_count, None)),
        "frame_gripper_qvel": (np.dtype(np.float64), (frame_count, None)),
    }
    for name, (dtype, shape) in finite_fields.items():
        value = _expect_array(arrays, name, dtype=dtype, shape=shape)
        if value.ndim == 2 and value.shape[1] == 0:
            _fail(f"trajectory.{name}", "feature width must be nonzero")
        _assert_finite(value, f"trajectory.{name}")
    if arrays["frame_gripper_qpos"].shape != arrays["frame_gripper_qvel"].shape:
        _fail("trajectory", "gripper position and velocity shapes must match")
    for name in ("frame_eef_quaternion_xyzw", "frame_primary_object_quaternion_wxyz"):
        norms = np.linalg.norm(arrays[name], axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            _fail(f"trajectory.{name}", "must contain unit quaternions")

    for name in (
        "frame_primary_gripper_contact",
        "frame_primary_grasped",
        "frame_task_success",
        "frame_goal_present",
    ):
        _expect_array(arrays, name, dtype=np.dtype(np.bool_), shape=(frame_count,))
    phase = arrays["frame_phase"]
    if phase.dtype != np.dtype("U16") or phase.shape != (frame_count,):
        _fail(
            "trajectory.frame_phase",
            f"must have dtype {np.dtype('U16')} and shape {(frame_count,)}",
        )
    if not set(phase.tolist()).issubset(PHASES):
        _fail("trajectory.frame_phase", "contains an unknown task phase")

    scalar = _expect_array(
        arrays,
        "frame_scalar_features",
        dtype=np.dtype(np.float32),
        shape=(frame_count, len(FRAME_SCALAR_FEATURE_NAMES)),
    )
    finite_columns = [
        index for index in range(scalar.shape[1]) if index not in {3, 14, 15}
    ]
    _assert_finite(scalar[:, finite_columns], "trajectory.frame_scalar_features")
    max_steps = int(metadata["execution"]["max_steps"])
    if not np.allclose(scalar[:, 0], steps / max_steps, rtol=0.0, atol=1e-7):
        _fail(
            "trajectory.frame_scalar_features", "normalized step column is inconsistent"
        )
    if not np.allclose(scalar[:, 1], simulator_time, rtol=0.0, atol=1e-6):
        _fail(
            "trajectory.frame_scalar_features", "simulator time column is inconsistent"
        )
    boolean_scalar = np.column_stack(
        (
            arrays["frame_primary_gripper_contact"],
            arrays["frame_primary_grasped"],
            arrays["frame_task_success"],
        )
    ).astype(np.float32)
    if not np.array_equal(scalar[:, 5:8], boolean_scalar):
        _fail(
            "trajectory.frame_scalar_features", "contact/grasp/success columns disagree"
        )
    phase_scalar = np.column_stack([phase == name for name in PHASES]).astype(
        np.float32
    )
    if not np.array_equal(scalar[:, 8:12], phase_scalar):
        _fail("trajectory.frame_scalar_features", "phase columns disagree")

    goal_position = _expect_array(
        arrays,
        "frame_goal_position",
        dtype=np.dtype(np.float64),
        shape=(frame_count, 3),
    )
    goal_quaternion = _expect_array(
        arrays,
        "frame_goal_quaternion_wxyz",
        dtype=np.dtype(np.float64),
        shape=(frame_count, 4),
    )
    goal_present = arrays["frame_goal_present"]
    if np.any(goal_present):
        _assert_finite(goal_position[goal_present], "trajectory.frame_goal_position")
    quaternion_finite = np.isfinite(goal_quaternion).all(axis=1)
    quaternion_absent = np.isnan(goal_quaternion).all(axis=1)
    if not np.all(quaternion_finite | quaternion_absent):
        _fail(
            "trajectory.frame_goal_quaternion_wxyz",
            "each row must be entirely finite or entirely NaN",
        )
    if np.any(quaternion_finite) and not np.allclose(
        np.linalg.norm(goal_quaternion[quaternion_finite], axis=1),
        1.0,
        rtol=0.0,
        atol=1e-5,
    ):
        _fail(
            "trajectory.frame_goal_quaternion_wxyz",
            "finite rows must contain unit quaternions",
        )
    if np.any(~goal_present) and (
        not np.isnan(goal_position[~goal_present]).all()
        or not quaternion_absent[~goal_present].all()
    ):
        _fail("trajectory", "absent goals must be represented by NaN pose rows")

    predicate_names = tuple(metadata["capture"]["task_predicate_names"])
    _expect_array(
        arrays,
        "frame_task_predicates",
        dtype=np.dtype(np.bool_),
        shape=(frame_count, len(predicate_names)),
    )

    valid = bool(metadata["validity"]["valid"])
    for candidate in ACTIVATION_CANDIDATES:
        activation = _expect_array(
            arrays,
            f"activation_{candidate}",
            dtype=np.dtype(np.float32),
            shape=(activation_count, None),
        )
        if valid and activation.shape[1] == 0:
            _fail(
                f"trajectory.activation_{candidate}",
                "valid episodes need nonzero width",
            )
        if not valid and activation.shape != (0, 0):
            _fail(
                f"trajectory.activation_{candidate}",
                "invalid resets require a 0x0 activation matrix",
            )
        _assert_finite(activation, f"trajectory.activation_{candidate}")

    outcome = metadata["outcome"]
    capture = metadata["capture"]
    if int(outcome["control_steps"]) != action_count:
        _fail("outcome.control_steps", "does not match trajectory action count")
    if int(capture["policy_select_calls"]) != action_count:
        _fail("capture.policy_select_calls", "does not match trajectory action count")
    if int(capture["scored_policy_select_calls"]) != activation_count:
        _fail(
            "capture.scored_policy_select_calls",
            "does not match activation-control-step count",
        )
    if int(capture["instrumented_internal_calls"]) != 11 * activation_count:
        _fail("capture.instrumented_internal_calls", "must equal 11 per scored call")
    if int(capture["frame_count"]) != frame_count:
        _fail("capture.frame_count", "does not match trajectory frame count")
    if action_count > int(metadata["execution"]["max_steps"]):
        _fail("trajectory.actions", "exceeds execution.max_steps")
    if not math.isclose(
        float(outcome["reward_sum"]),
        float(np.sum(rewards, dtype=np.float64)),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        _fail("outcome.reward_sum", "does not match trajectory rewards")
    if bool(outcome["success"]) != bool(arrays["frame_task_success"][-1]):
        _fail("outcome.success", "does not match the final task-success frame")
    if bool(metadata["validity"]["initial_success"]) != bool(
        arrays["frame_task_success"][0]
    ):
        _fail(
            "validity.initial_success",
            "does not match the initial task-success frame",
        )

    if valid:
        if action_count == 0:
            _fail("trajectory.actions", "valid episodes must contain an action")
        if not bool(terminated[-1] or truncated[-1]):
            _fail("trajectory", "valid episode must end on its final action")
        if np.any(terminated[:-1] | truncated[:-1]):
            _fail("trajectory", "terminal flags may only appear on the final action")
        if bool(outcome["terminated"]) != bool(terminated[-1]):
            _fail("outcome.terminated", "does not match the final action")
        if bool(outcome["truncated"]) != bool(truncated[-1]):
            _fail("outcome.truncated", "does not match the final action")
    else:
        if action_count != 0 or frame_count != 1:
            _fail("trajectory", "invalid resets require one frame and zero actions")
        if bool(arrays["frame_task_success"][0]):
            _fail("trajectory.frame_task_success", "invalid reset cannot be successful")


def _validate_expected_task(
    metadata: Mapping[str, Any], expected_task: TaskSpec
) -> None:
    if not isinstance(expected_task, TaskSpec):
        _fail("expected_task", "must be a TaskSpec")
    episode = metadata["episode"]
    task = metadata["task"]
    observed = (
        episode["suite"],
        episode["task_id"],
        episode["task_rank"],
        metadata["task_language"],
        task["primary_object"],
        task["planar_symmetry_order"],
    )
    expected = (
        expected_task.suite,
        expected_task.task_id,
        expected_task.rank,
        expected_task.language,
        expected_task.primary_object,
        expected_task.planar_symmetry_order,
    )
    if observed != expected:
        _fail(
            "expected_task",
            "suite, task ID, rank, language, object, or symmetry does not match artifact metadata",
        )


def load_rollout_artifact(
    path: str | Path, *, expected_task: TaskSpec | None = None
) -> RolloutArtifact:
    """Load and fully validate one explicit atomic artifact directory."""

    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        _fail(str(directory), "artifact directory does not exist")
    entries = {entry.name for entry in directory.iterdir()}
    if entries != {"metadata.json", "trajectory.npz"}:
        missing = sorted({"metadata.json", "trajectory.npz"} - entries)
        unexpected = sorted(entries - {"metadata.json", "trajectory.npz"})
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        _fail(str(directory), "; ".join(details))
    metadata_path = directory / "metadata.json"
    trajectory_path = directory / "trajectory.npz"
    if metadata_path.is_symlink() or trajectory_path.is_symlink():
        _fail(str(directory), "artifact inputs must not be symbolic links")

    metadata_bytes = metadata_path.read_bytes()
    trajectory_stat = trajectory_path.stat()
    trajectory_sha256 = _sha256_file(trajectory_path)
    try:
        decoded = metadata_bytes.decode("utf-8")
        metadata_json = json.loads(decoded, object_pairs_hook=_json_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"metadata.json: invalid UTF-8 JSON: {error}"
        ) from error
    metadata = _validate_metadata(metadata_json, directory)
    if expected_task is not None:
        _validate_expected_task(metadata, expected_task)

    try:
        with np.load(trajectory_path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                _fail("trajectory.npz", "contains duplicate array names")
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactValidationError(
            f"trajectory.npz: invalid safe NumPy archive: {error}"
        ) from error
    if _file_identity(trajectory_path.stat()) != _file_identity(trajectory_stat):
        _fail("trajectory.npz", "changed while it was being loaded")
    _validate_arrays(arrays, metadata)
    for value in arrays.values():
        value.setflags(write=False)
    return RolloutArtifact(
        path=directory,
        metadata=metadata,
        arrays=MappingProxyType(arrays),
        hashes=ArtifactHashes(
            metadata_sha256=_sha256(metadata_bytes),
            trajectory_sha256=trajectory_sha256,
        ),
    )


def _yaw_xyzw(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = quaternion.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_wxyz(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = quaternion.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _readonly(value: NDArray[Any]) -> NDArray[Any]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _common_cohort_key(artifact: RolloutArtifact) -> tuple[Any, ...]:
    metadata = artifact.metadata
    episode = metadata["episode"]
    model = metadata["model"]
    return (
        metadata["schema_version"],
        episode["split"],
        episode["suite"],
        episode["task_id"],
        episode["task_rank"],
        metadata["task_language"],
        metadata["task"]["primary_object"],
        metadata["task"]["planar_symmetry_order"],
        episode["policy_revision"],
        model["base_vlm_revision"],
        episode["code_commit"],
    )


def _load_probe_artifact(path: str | Path, expected_task: TaskSpec) -> RolloutArtifact:
    """Validate all arrays, then release camera payloads unused by probes."""

    artifact = load_rollout_artifact(path, expected_task=expected_task)
    retained = {
        name: value
        for name, value in artifact.arrays.items()
        if name not in {"frame_agentview_image", "frame_robot0_eye_in_hand_image"}
    }
    return RolloutArtifact(
        path=artifact.path,
        metadata=artifact.metadata,
        arrays=MappingProxyType(retained),
        hashes=artifact.hashes,
    )


def assemble_probe_cohort(
    artifact_directories: Sequence[str | Path],
    *,
    requested_episode_ids: Sequence[str],
    expected_task: TaskSpec,
    stride: int = 5,
) -> ProbeCohort:
    """Validate an exact requested set and assemble aligned pre-action rows.

    Row eligibility depends only on reset validity and control step, never on an
    episode's success or failure.  The failure label is attached only after the
    eligible rows have been selected.
    """

    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        _fail("stride", "must be a positive integer")
    if stride % ACTIVATION_SCORE_STRIDE_STEPS:
        _fail(
            "stride",
            f"must be a multiple of the stored stride {ACTIVATION_SCORE_STRIDE_STEPS}",
        )
    if not isinstance(expected_task, TaskSpec):
        _fail("expected_task", "must be a TaskSpec")
    if expected_task.planar_symmetry_order < 1:
        _fail("expected_task.planar_symmetry_order", "must be positive")
    requested = tuple(
        _safe_episode_id(value, "requested_episode_ids")
        for value in requested_episode_ids
    )
    if not requested:
        _fail("requested_episode_ids", "must not be empty")
    if len(requested) != len(set(requested)):
        _fail("requested_episode_ids", "contains duplicates")

    artifacts = tuple(
        _load_probe_artifact(path, expected_task) for path in artifact_directories
    )
    identifiers = tuple(artifact.episode_id for artifact in artifacts)
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier
            for identifier in set(identifiers)
            if identifiers.count(identifier) > 1
        )
        _fail("artifact_directories", "duplicate episode IDs: " + ", ".join(duplicates))
    missing = sorted(set(requested) - set(identifiers))
    unexpected = sorted(set(identifiers) - set(requested))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing requested artifacts: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected artifacts: " + ", ".join(unexpected))
        _fail("cohort", "; ".join(details))

    ordered = tuple(sorted(artifacts, key=lambda artifact: artifact.episode_id))
    common = _common_cohort_key(ordered[0])
    if any(_common_cohort_key(artifact) != common for artifact in ordered[1:]):
        _fail(
            "cohort",
            "artifacts mix schema, split, task, policy, base VLM, or code metadata",
        )

    feature_widths: dict[str, int] = {}
    valid_ids: list[str] = []
    invalid_ids: list[str] = []
    theta_parts: list[NDArray[np.float64]] = []
    base_parts: list[NDArray[np.int64]] = []
    episode_parts: list[NDArray[np.str_]] = []
    step_parts: list[NDArray[np.int64]] = []
    label_parts: list[NDArray[np.bool_]] = []
    activation_parts: dict[str, list[NDArray[np.float32]]] = {
        candidate: [] for candidate in ACTIVATION_CANDIDATES
    }

    for artifact in ordered:
        if not artifact.valid_reset:
            invalid_ids.append(artifact.episode_id)
            continue
        valid_ids.append(artifact.episode_id)
        for candidate in ACTIVATION_CANDIDATES:
            width = int(artifact.arrays[f"activation_{candidate}"].shape[1])
            previous = feature_widths.setdefault(candidate, width)
            if width != previous:
                _fail(
                    f"cohort.activation_{candidate}",
                    f"feature width changed from {previous} to {width}",
                )

        activation_steps = artifact.arrays["activation_control_step"]
        selected = np.flatnonzero(activation_steps % int(stride) == 0)
        if selected.size == 0:
            _fail(artifact.episode_id, "valid episode produced no stride-selected rows")
        selected_steps = activation_steps[selected]
        eef_yaw = _yaw_xyzw(
            artifact.arrays["frame_eef_quaternion_xyzw"][selected_steps]
        )
        object_yaw = _yaw_wxyz(
            artifact.arrays["frame_primary_object_quaternion_wxyz"][selected_steps]
        )
        theta = np.arctan2(np.sin(eef_yaw - object_yaw), np.cos(eef_yaw - object_yaw))
        row_count = selected.size
        episode = artifact.metadata["episode"]
        theta_parts.append(theta.astype(np.float64, copy=False))
        base_parts.append(
            np.full(row_count, int(episode["base_init_state_id"]), dtype=np.int64)
        )
        episode_parts.append(
            np.asarray([artifact.episode_id] * row_count, dtype=np.str_)
        )
        step_parts.append(selected_steps.astype(np.int64, copy=False))
        label_parts.append(np.full(row_count, not artifact.success, dtype=np.bool_))
        for candidate in ACTIVATION_CANDIDATES:
            activation_parts[candidate].append(
                artifact.arrays[f"activation_{candidate}"][selected]
            )

    if not valid_ids:
        _fail("cohort", "contains no valid rollout episodes")
    theta_rel = _readonly(np.concatenate(theta_parts))
    base_init_state_id = _readonly(np.concatenate(base_parts))
    episode_id = _readonly(np.concatenate(episode_parts))
    control_step = _readonly(np.concatenate(step_parts))
    failure_label = _readonly(np.concatenate(label_parts))
    samples = ProbeSamples.from_arrays(
        theta_rel=theta_rel,
        base_init_state_id=base_init_state_id,
        episode_id=episode_id,
        symmetry_order=expected_task.planar_symmetry_order,
    )
    activation_features = MappingProxyType(
        {
            candidate: _readonly(np.concatenate(activation_parts[candidate], axis=0))
            for candidate in ACTIVATION_CANDIDATES
        }
    )

    training_content = {
        "row_count": samples.n_rows,
        "episode_id_sha256": probe_cohort_array_sha256(episode_id),
        "base_init_state_id_sha256": probe_cohort_array_sha256(base_init_state_id),
        "control_step_sha256": probe_cohort_array_sha256(control_step),
        "theta_rel_sha256": probe_cohort_array_sha256(theta_rel),
        "failure_label_sha256": probe_cohort_array_sha256(failure_label),
        "activation_features": {
            candidate: {
                "shape": list(activation_features[candidate].shape),
                "dtype": activation_features[candidate].dtype.str,
                "logical_sha256": probe_cohort_array_sha256(
                    activation_features[candidate]
                ),
            }
            for candidate in ACTIVATION_CANDIDATES
        },
    }

    episode_entries = [
        {
            "episode_id": artifact.episode_id,
            "metadata_sha256": artifact.hashes.metadata_sha256,
            "trajectory_sha256": artifact.hashes.trajectory_sha256,
        }
        for artifact in ordered
    ]
    episode0 = ordered[0].metadata["episode"]
    model0 = ordered[0].metadata["model"]
    manifest = CohortManifest(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": "probe_cohort",
            "split": episode0["split"],
            "task": {
                "suite": episode0["suite"],
                "task_id": episode0["task_id"],
                "task_rank": episode0["task_rank"],
                "language": ordered[0].metadata["task_language"],
                "primary_object": expected_task.primary_object,
                "planar_symmetry_order": expected_task.planar_symmetry_order,
            },
            "model": {
                "policy_revision": episode0["policy_revision"],
                "base_vlm_revision": model0["base_vlm_revision"],
                "code_commit": episode0["code_commit"],
            },
            "selection": {
                "kind": "valid_pre_action_control_step_stride",
                "stride": int(stride),
                "outcome_conditioned": False,
            },
            "activation_candidates": list(ACTIVATION_CANDIDATES),
            "training_content": training_content,
            "episodes": episode_entries,
            "invalid_reset_episode_ids": invalid_ids,
        }
    )
    return ProbeCohort(
        samples=samples,
        activation_features=activation_features,
        control_step=control_step,
        failure_label=failure_label,
        valid_episode_ids=tuple(valid_ids),
        invalid_reset_episode_ids=tuple(invalid_ids),
        manifest=manifest,
    )


__all__ = [
    "ACTIVATION_CANDIDATES",
    "ACTIVATION_SCORE_STRIDE_STEPS",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactHashes",
    "ArtifactValidationError",
    "CohortManifest",
    "ProbeCohort",
    "RolloutArtifact",
    "assemble_probe_cohort",
    "load_rollout_artifact",
    "probe_cohort_array_sha256",
]
