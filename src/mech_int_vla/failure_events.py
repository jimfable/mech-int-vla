"""Deterministic, outcome-contained failure-event annotation.

This module implements the prospectively frozen event rules without importing
simulator or model code.  Discovery bounds are derived only after exact artifact
coverage has been validated, and every public container defensively copies its
inputs into immutable, finite representations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .artifacts import RolloutArtifact

ACTION_DIMENSION = 7
GRIPPER_ACTION_INDEX = 6
MISSED_GRASP_WINDOW = 10
DISTANCE_NET_INCREASE_M = 0.005
DROP_LANDING_FRAMES = 3
DROP_SUPPORT_TOLERANCE_M = 0.005
WORKSPACE_MARGIN_M = 0.05
TERMINAL_HORIZON = 520
PHASES = ("pregrasp", "grasped", "transport", "placed")
EVENT_PRECEDENCE = (
    "missed_grasp",
    "dropped_object",
    "irrecoverable_workspace_exit",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}")


class FailureEventError(ValueError):
    """Raised when trace or freeze inputs violate the frozen protocol."""


class AnnotationStatus(str, Enum):
    EXCLUDED_INVALID_RESET = "excluded_invalid_reset"
    SUCCESS_NO_EVENT = "success_no_event"
    ANNOTATED = "annotated"
    UNANNOTATABLE_EARLY_TERMINAL = "unannotatable_early_terminal"


class FailureEventType(str, Enum):
    MISSED_GRASP = "missed_grasp"
    DROPPED_OBJECT = "dropped_object"
    IRRECOVERABLE_WORKSPACE_EXIT = "irrecoverable_workspace_exit"
    TERMINAL_HORIZON = "terminal_horizon"


def _readonly_finite_matrix(
    values: ArrayLike, name: str, *, columns: int
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise FailureEventError(f"{name} must have shape (n, {columns})")
    if not np.isfinite(array).all():
        raise FailureEventError(f"{name} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _readonly_bool_vector(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind != "b":
        raise FailureEventError(f"{name} must be a one-dimensional boolean array")
    result = raw.astype(np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _readonly_steps(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise FailureEventError("frame_control_step must be an integer vector")
    result = raw.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_phases(values: ArrayLike) -> NDArray[np.str_]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "US":
        raise FailureEventError("frame_phase must be a one-dimensional string array")
    result = raw.astype(np.str_, copy=True)
    if any(value not in PHASES for value in result.tolist()):
        raise FailureEventError("frame_phase contains an unknown phase")
    result.setflags(write=False)
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise FailureEventError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class ArtifactIdentity:
    """Content identity of one raw rollout artifact."""

    episode_id: str
    metadata_sha256: str
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FailureEventError("episode_id must be a non-empty string")
        for name, value in (
            ("metadata_sha256", self.metadata_sha256),
            ("trajectory_sha256", self.trajectory_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise FailureEventError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "metadata_sha256": self.metadata_sha256,
            "trajectory_sha256": self.trajectory_sha256,
        }


@dataclass(frozen=True)
class FailureEventTrace:
    """Immutable arrays for one completed raw episode."""

    episode_id: str
    valid_reset: bool
    success: bool
    validity_reasons: tuple[str, ...]
    actions: NDArray[np.float64]
    frame_control_step: NDArray[np.int64]
    frame_eef_position: NDArray[np.float64]
    frame_object_position: NDArray[np.float64]
    frame_contact: NDArray[np.bool_]
    frame_grasped: NDArray[np.bool_]
    frame_task_success: NDArray[np.bool_]
    frame_phase: NDArray[np.str_]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FailureEventError("episode_id must be a non-empty string")
        valid = _bool(self.valid_reset, "valid_reset")
        success = _bool(self.success, "success")
        if not isinstance(self.validity_reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.validity_reasons
        ):
            raise FailureEventError(
                "validity_reasons must be non-empty strings in a tuple"
            )
        if len(set(self.validity_reasons)) != len(self.validity_reasons):
            raise FailureEventError("validity_reasons must be unique")
        if valid == bool(self.validity_reasons):
            raise FailureEventError(
                "valid reset must have no reasons; invalid reset must have reasons"
            )
        if not valid and success:
            raise FailureEventError("an invalid reset cannot be successful")

        actions = _readonly_finite_matrix(
            self.actions, "actions", columns=ACTION_DIMENSION
        )
        steps = _readonly_steps(self.frame_control_step)
        eef = _readonly_finite_matrix(
            self.frame_eef_position, "frame_eef_position", columns=3
        )
        obj = _readonly_finite_matrix(
            self.frame_object_position, "frame_object_position", columns=3
        )
        contact = _readonly_bool_vector(self.frame_contact, "frame_contact")
        grasped = _readonly_bool_vector(self.frame_grasped, "frame_grasped")
        task_success = _readonly_bool_vector(
            self.frame_task_success, "frame_task_success"
        )
        phase = _readonly_phases(self.frame_phase)
        terminated = _readonly_bool_vector(self.terminated, "terminated")
        truncated = _readonly_bool_vector(self.truncated, "truncated")

        action_count = actions.shape[0]
        frame_count = action_count + 1
        frame_arrays = (steps, eef, obj, contact, grasped, task_success, phase)
        if any(array.shape[0] != frame_count for array in frame_arrays):
            raise FailureEventError("every frame array must have len(actions) + 1 rows")
        if terminated.size != action_count or truncated.size != action_count:
            raise FailureEventError(
                "terminal arrays must align one-to-one with actions"
            )
        if not np.array_equal(steps, np.arange(frame_count, dtype=np.int64)):
            raise FailureEventError(
                "frame_control_step must be exactly 0..len(actions)"
            )
        if action_count > TERMINAL_HORIZON:
            raise FailureEventError("trace exceeds the frozen 520-action horizon")
        if bool(task_success[-1]) != success or np.any(task_success[:-1]):
            raise FailureEventError(
                "task-success frames must be false except for a successful final frame"
            )
        if np.any(terminated & truncated) or np.any(terminated[:-1] | truncated[:-1]):
            raise FailureEventError(
                "only the final action may be terminal, and not both ways"
            )
        if valid:
            if action_count == 0 or not bool(terminated[-1] or truncated[-1]):
                raise FailureEventError(
                    "a valid trace must contain actions and end terminally"
                )
            if success and not bool(terminated[-1]):
                raise FailureEventError("a successful trace must terminate on success")
        elif action_count != 0:
            raise FailureEventError("an invalid reset trace must contain zero actions")

        for name, value in (
            ("actions", actions),
            ("frame_control_step", steps),
            ("frame_eef_position", eef),
            ("frame_object_position", obj),
            ("frame_contact", contact),
            ("frame_grasped", grasped),
            ("frame_task_success", task_success),
            ("frame_phase", phase),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            object.__setattr__(self, name, value)

    @property
    def action_count(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class DiscoveryEpisode:
    artifact: ArtifactIdentity
    trace: FailureEventTrace

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactIdentity) or not isinstance(
            self.trace, FailureEventTrace
        ):
            raise FailureEventError(
                "DiscoveryEpisode requires validated artifact and trace types"
            )
        if self.artifact.episode_id != self.trace.episode_id:
            raise FailureEventError("artifact and trace episode IDs differ")


def _require_rollout_artifact(artifact: Any) -> RolloutArtifact:
    if not isinstance(artifact, RolloutArtifact):
        raise FailureEventError(
            "failure-event conversion requires a validated RolloutArtifact"
        )
    return artifact


def artifact_identity_from_rollout(artifact: RolloutArtifact) -> ArtifactIdentity:
    """Copy both loader-verified raw-artifact hashes into event provenance."""

    rollout = _require_rollout_artifact(artifact)
    return ArtifactIdentity(
        episode_id=rollout.episode_id,
        metadata_sha256=rollout.hashes.metadata_sha256,
        trajectory_sha256=rollout.hashes.trajectory_sha256,
    )


def failure_event_trace_from_artifact(
    artifact: RolloutArtifact,
) -> FailureEventTrace:
    """Convert an already validated raw artifact into an immutable event trace."""

    rollout = _require_rollout_artifact(artifact)
    try:
        validity = rollout.metadata["validity"]
        outcome = rollout.metadata["outcome"]
        reasons = validity["reasons"]
        if not isinstance(reasons, tuple):
            raise FailureEventError(
                "validated artifact validity reasons must be a frozen tuple"
            )
        arrays = rollout.arrays
        return FailureEventTrace(
            episode_id=rollout.episode_id,
            valid_reset=validity["valid"],
            success=outcome["success"],
            validity_reasons=reasons,
            actions=arrays["actions"],
            frame_control_step=arrays["frame_control_step"],
            frame_eef_position=arrays["frame_eef_position"],
            frame_object_position=arrays["frame_primary_object_position"],
            frame_contact=arrays["frame_primary_gripper_contact"],
            frame_grasped=arrays["frame_primary_grasped"],
            frame_task_success=arrays["frame_task_success"],
            frame_phase=arrays["frame_phase"],
            terminated=arrays["terminated"],
            truncated=arrays["truncated"],
        )
    except (KeyError, TypeError) as exc:
        raise FailureEventError(
            "RolloutArtifact is missing the validated failure-event schema"
        ) from exc


def discovery_episode_from_artifact(artifact: RolloutArtifact) -> DiscoveryEpisode:
    """Create one exact-hash Discovery input from a validated raw artifact."""

    rollout = _require_rollout_artifact(artifact)
    return DiscoveryEpisode(
        artifact=artifact_identity_from_rollout(rollout),
        trace=failure_event_trace_from_artifact(rollout),
    )


@dataclass(frozen=True)
class ReachableBounds:
    lower_xyz: tuple[float, float, float]
    upper_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.lower_xyz, tuple) or not isinstance(
            self.upper_xyz, tuple
        ):
            raise FailureEventError("bounds must be immutable XYZ tuples")
        lower = np.asarray(self.lower_xyz, dtype=np.float64)
        upper = np.asarray(self.upper_xyz, dtype=np.float64)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all()
        ):
            raise FailureEventError("bounds must contain six finite XYZ values")
        if np.any(lower > upper):
            raise FailureEventError("lower bounds must not exceed upper bounds")
        object.__setattr__(self, "lower_xyz", tuple(float(value) for value in lower))
        object.__setattr__(self, "upper_xyz", tuple(float(value) for value in upper))

    def to_dict(self) -> dict[str, list[float]]:
        return {"lower_xyz": list(self.lower_xyz), "upper_xyz": list(self.upper_xyz)}

    def outside(self, positions: ArrayLike) -> NDArray[np.bool_]:
        array = _readonly_finite_matrix(positions, "positions", columns=3)
        lower = np.asarray(self.lower_xyz, dtype=np.float64)
        upper = np.asarray(self.upper_xyz, dtype=np.float64)
        result = np.any((array < lower) | (array > upper), axis=1)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class DiscoveryEpisodeProvenance:
    artifact: ArtifactIdentity
    valid_reset: bool
    success: bool
    validity_reasons: tuple[str, ...]
    frame_zero_included: bool
    successful_path_included: bool

    def __post_init__(self) -> None:
        for name in (
            "valid_reset",
            "success",
            "frame_zero_included",
            "successful_path_included",
        ):
            _bool(getattr(self, name), f"provenance.{name}")
        if not isinstance(self.validity_reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.validity_reasons
        ):
            raise FailureEventError(
                "provenance validity reasons must be non-empty strings in a tuple"
            )
        if len(set(self.validity_reasons)) != len(self.validity_reasons):
            raise FailureEventError("provenance validity reasons must be unique")
        if self.valid_reset == bool(self.validity_reasons):
            raise FailureEventError(
                "provenance validity flag and reasons are inconsistent"
            )
        if self.success and not self.valid_reset:
            raise FailureEventError("invalid Discovery provenance cannot be successful")
        if self.frame_zero_included != self.valid_reset:
            raise FailureEventError("only valid Discovery frame zeros may be included")
        if self.successful_path_included != (self.valid_reset and self.success):
            raise FailureEventError(
                "only valid successful Discovery paths may be included"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.artifact.to_dict(),
            "valid_reset": self.valid_reset,
            "success": self.success,
            "validity_reasons": list(self.validity_reasons),
            "frame_zero_included": self.frame_zero_included,
            "successful_path_included": self.successful_path_included,
        }


@dataclass(frozen=True)
class DiscoveryBoundsDerivation:
    raw_bounds: ReachableBounds
    expanded_bounds: ReachableBounds
    margin_m: float
    expected_artifacts: tuple[ArtifactIdentity, ...]
    provenance: tuple[DiscoveryEpisodeProvenance, ...]

    def __post_init__(self) -> None:
        if self.margin_m != WORKSPACE_MARGIN_M:
            raise FailureEventError("workspace margin must be exactly 0.05 m")
        expected_ids = tuple(item.episode_id for item in self.expected_artifacts)
        provenance_ids = tuple(item.artifact.episode_id for item in self.provenance)
        if expected_ids != tuple(sorted(expected_ids)) or len(set(expected_ids)) != len(
            expected_ids
        ):
            raise FailureEventError(
                "expected artifacts must have unique sorted episode IDs"
            )
        if provenance_ids != expected_ids:
            raise FailureEventError(
                "bounds provenance must exactly cover expected artifacts"
            )
        if any(
            p.artifact != e
            for p, e in zip(self.provenance, self.expected_artifacts, strict=True)
        ):
            raise FailureEventError(
                "bounds provenance hashes differ from expected artifacts"
            )
        if not any(item.frame_zero_included for item in self.provenance):
            raise FailureEventError("bounds need at least one valid frame zero")
        if not any(item.successful_path_included for item in self.provenance):
            raise FailureEventError("bounds need at least one valid successful path")
        raw_low = np.asarray(self.raw_bounds.lower_xyz)
        raw_high = np.asarray(self.raw_bounds.upper_xyz)
        expanded_low = np.asarray(self.expanded_bounds.lower_xyz)
        expanded_high = np.asarray(self.expanded_bounds.upper_xyz)
        if not np.array_equal(
            expanded_low, raw_low - self.margin_m
        ) or not np.array_equal(expanded_high, raw_high + self.margin_m):
            raise FailureEventError("expanded bounds do not equal raw bounds plus 5 cm")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": "all_valid_frame0_plus_valid_successful_frames1_to_terminal",
            "margin_m": self.margin_m,
            "raw_bounds": self.raw_bounds.to_dict(),
            "expanded_bounds": self.expanded_bounds.to_dict(),
            "expected_artifacts": [item.to_dict() for item in self.expected_artifacts],
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True)
class FailureEventResult:
    episode_id: str
    valid_reset: bool
    success: bool
    status: AnnotationStatus
    event_type: FailureEventType | None
    onset_step: int | None
    confirmation_step: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FailureEventError("result episode_id must be non-empty")
        _bool(self.valid_reset, "result.valid_reset")
        _bool(self.success, "result.success")
        if not isinstance(self.status, AnnotationStatus):
            raise FailureEventError("result status must be an AnnotationStatus")
        if self.event_type is not None and not isinstance(
            self.event_type, FailureEventType
        ):
            raise FailureEventError("result event_type must be a FailureEventType")
        for name, value in (
            ("onset_step", self.onset_step),
            ("confirmation_step", self.confirmation_step),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral)
            ):
                raise FailureEventError(f"result {name} must be an integer or null")
        has_event = self.event_type is not None
        if has_event != (
            self.onset_step is not None and self.confirmation_step is not None
        ):
            raise FailureEventError(
                "event type and onset/confirmation must appear together"
            )
        if self.status is AnnotationStatus.ANNOTATED:
            if not has_event or not self.valid_reset or self.success:
                raise FailureEventError("only valid failed episodes may be annotated")
            assert self.onset_step is not None and self.confirmation_step is not None
            if not (0 <= self.onset_step <= self.confirmation_step <= TERMINAL_HORIZON):
                raise FailureEventError("event steps are outside the frozen horizon")
        elif has_event:
            raise FailureEventError("non-annotated results cannot contain an event")
        allowed_statuses = (
            {AnnotationStatus.EXCLUDED_INVALID_RESET}
            if not self.valid_reset
            else {AnnotationStatus.SUCCESS_NO_EVENT}
            if self.success
            else {
                AnnotationStatus.ANNOTATED,
                AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL,
            }
        )
        if self.status not in allowed_statuses:
            raise FailureEventError("result status disagrees with validity/success")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "valid_reset": self.valid_reset,
            "success": self.success,
            "status": self.status.value,
            "event_type": None if self.event_type is None else self.event_type.value,
            "onset_step": self.onset_step,
            "confirmation_step": self.confirmation_step,
        }


@dataclass(frozen=True)
class TaskIdentity:
    suite: str
    task_id: int
    task_rank: int
    language: str
    primary_object: str
    planar_symmetry_order: int

    def __post_init__(self) -> None:
        for name in ("suite", "language", "primary_object"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise FailureEventError(f"task {name} must be a non-empty string")
        for name in ("task_id", "task_rank", "planar_symmetry_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise FailureEventError(f"task {name} must be a nonnegative integer")
        if self.task_rank < 1 or self.planar_symmetry_order < 1:
            raise FailureEventError("task rank and symmetry order must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "task_id": int(self.task_id),
            "task_rank": int(self.task_rank),
            "language": self.language,
            "primary_object": self.primary_object,
            "planar_symmetry_order": int(self.planar_symmetry_order),
        }


@dataclass(frozen=True)
class FailureEventFreezeManifest:
    task: TaskIdentity
    bounds: DiscoveryBoundsDerivation
    primary_placement_predicate_keys: tuple[str, ...]
    annotations: tuple[FailureEventResult, ...]
    video_audit_episode_ids: tuple[str, ...]
    implementation_commit: str
    reality_gate_tag: str = "prereg-locked-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        keys = self.primary_placement_predicate_keys
        if (
            not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or keys != tuple(sorted(set(keys)))
        ):
            raise FailureEventError(
                "primary placement predicate keys must be non-empty, sorted, unique"
            )
        if any(not isinstance(item, FailureEventResult) for item in self.annotations):
            raise FailureEventError("freeze annotations must be validated results")
        annotation_ids = tuple(item.episode_id for item in self.annotations)
        expected_ids = tuple(item.episode_id for item in self.bounds.expected_artifacts)
        if annotation_ids != expected_ids:
            raise FailureEventError(
                "freeze annotations must exactly cover expected Discovery episodes"
            )
        audit = self.video_audit_episode_ids
        if (
            any(not isinstance(item, str) or not item for item in audit)
            or audit != expected_ids
        ):
            raise FailureEventError(
                "video audit IDs must exactly cover all expected Discovery episodes"
            )
        provenance_by_id = {
            item.artifact.episode_id: item for item in self.bounds.provenance
        }
        for annotation in self.annotations:
            provenance = provenance_by_id[annotation.episode_id]
            if (
                annotation.valid_reset != provenance.valid_reset
                or annotation.success != provenance.success
            ):
                raise FailureEventError(
                    "freeze annotation validity/success disagrees with bounds provenance"
                )
        if _GIT_SHA1.fullmatch(self.implementation_commit) is None:
            raise FailureEventError(
                "implementation_commit must be a lowercase 40-character Git SHA"
            )
        if self.reality_gate_tag != "prereg-locked-v1" or self.schema_version != 1:
            raise FailureEventError("unsupported failure-event freeze version or tag")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reality_gate_tag": self.reality_gate_tag,
            "task": self.task.to_dict(),
            "protocol": failure_event_protocol_metadata(),
            "bounds": self.bounds.to_dict(),
            "primary_placement_predicate_keys": list(
                self.primary_placement_predicate_keys
            ),
            "annotations": [item.to_dict() for item in self.annotations],
            "video_audit_episode_ids": list(self.video_audit_episode_ids),
            "implementation_commit": self.implementation_commit,
        }

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class _Candidate:
    event_type: FailureEventType
    onset: int
    confirmation: int


def derive_discovery_bounds(
    episodes: tuple[DiscoveryEpisode, ...] | list[DiscoveryEpisode],
    *,
    expected_artifacts: tuple[ArtifactIdentity, ...] | list[ArtifactIdentity],
) -> DiscoveryBoundsDerivation:
    """Derive the noncircular Discovery XYZ envelope after exact coverage checks."""

    episode_list = list(episodes)
    expected_list = list(expected_artifacts)
    if not expected_list:
        raise FailureEventError("expected Discovery artifact set cannot be empty")
    expected_by_id: dict[str, ArtifactIdentity] = {}
    for artifact in expected_list:
        if artifact.episode_id in expected_by_id:
            raise FailureEventError("expected Discovery artifact IDs must be unique")
        expected_by_id[artifact.episode_id] = artifact
    episode_by_id: dict[str, DiscoveryEpisode] = {}
    for episode in episode_list:
        if episode.trace.episode_id in episode_by_id:
            raise FailureEventError("Discovery episode IDs must be unique")
        episode_by_id[episode.trace.episode_id] = episode
    if set(episode_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(episode_by_id))
        extra = sorted(set(episode_by_id) - set(expected_by_id))
        raise FailureEventError(
            f"Discovery coverage mismatch: missing={missing}, extra={extra}"
        )
    for episode_id, expected in expected_by_id.items():
        if episode_by_id[episode_id].artifact != expected:
            raise FailureEventError(
                f"Discovery artifact hash mismatch for {episode_id!r}"
            )

    included: list[NDArray[np.float64]] = []
    provenance: list[DiscoveryEpisodeProvenance] = []
    for episode_id in sorted(episode_by_id):
        episode = episode_by_id[episode_id]
        trace = episode.trace
        if trace.valid_reset:
            included.append(trace.frame_object_position[0:1])
            if trace.success:
                included.append(trace.frame_object_position[1:])
        provenance.append(
            DiscoveryEpisodeProvenance(
                artifact=episode.artifact,
                valid_reset=trace.valid_reset,
                success=trace.success,
                validity_reasons=trace.validity_reasons,
                frame_zero_included=trace.valid_reset,
                successful_path_included=trace.valid_reset and trace.success,
            )
        )
    if not included or not any(item.successful_path_included for item in provenance):
        raise FailureEventError(
            "Discovery bounds require valid starts and a valid successful path"
        )
    positions = np.concatenate(included, axis=0, dtype=np.float64)
    raw_low = np.min(positions, axis=0)
    raw_high = np.max(positions, axis=0)
    raw = ReachableBounds(tuple(raw_low.tolist()), tuple(raw_high.tolist()))
    expanded = ReachableBounds(
        tuple((raw_low - WORKSPACE_MARGIN_M).tolist()),
        tuple((raw_high + WORKSPACE_MARGIN_M).tolist()),
    )
    expected_sorted = tuple(expected_by_id[key] for key in sorted(expected_by_id))
    return DiscoveryBoundsDerivation(
        raw_bounds=raw,
        expanded_bounds=expanded,
        margin_m=WORKSPACE_MARGIN_M,
        expected_artifacts=expected_sorted,
        provenance=tuple(provenance),
    )


def _missed_grasp(trace: FailureEventTrace) -> _Candidate | None:
    close_indices = np.flatnonzero(trace.actions[:, GRIPPER_ACTION_INDEX] > 0.0)
    if close_indices.size == 0:
        return None
    close = int(close_indices[0])
    if close + MISSED_GRASP_WINDOW > trace.action_count:
        return None
    start = close + 1
    stop = close + MISSED_GRASP_WINDOW + 1
    if np.any(trace.frame_contact[start:stop]):
        return None
    distances = np.linalg.norm(
        trace.frame_eef_position[start:stop] - trace.frame_object_position[start:stop],
        axis=1,
    )
    x = np.arange(MISSED_GRASP_WINDOW, dtype=np.float64)
    centered = x - np.mean(x)
    slope = float(
        np.dot(centered, distances - np.mean(distances)) / np.dot(centered, centered)
    )
    if not slope > 0.0 or distances[-1] - distances[0] < DISTANCE_NET_INCREASE_M:
        return None
    return _Candidate(FailureEventType.MISSED_GRASP, start, stop - 1)


def _drop(trace: FailureEventTrace) -> _Candidate | None:
    loss_frames = (
        np.flatnonzero(
            trace.frame_contact[:-1]
            & ~trace.frame_contact[1:]
            & trace.frame_grasped[:-1]
        )
        + 1
    )
    support_ceiling = float(
        trace.frame_object_position[0, 2] + DROP_SUPPORT_TOLERANCE_M
    )
    for raw_loss in loss_frames:
        loss = int(raw_loss)
        for landing in range(loss, trace.action_count - DROP_LANDING_FRAMES + 2):
            stop = landing + DROP_LANDING_FRAMES
            if np.any(trace.frame_contact[loss:stop]):
                break
            landing_slice = slice(landing, stop)
            if np.all(
                trace.frame_object_position[landing_slice, 2] <= support_ceiling
            ) and np.all(trace.frame_phase[landing_slice] != "placed"):
                return _Candidate(FailureEventType.DROPPED_OBJECT, loss, stop - 1)
    return None


def _workspace_exit(
    trace: FailureEventTrace, bounds: ReachableBounds
) -> _Candidate | None:
    outside = bounds.outside(trace.frame_object_position)
    if not bool(outside[-1]):
        return None
    inside = np.flatnonzero(~outside)
    onset = 0 if inside.size == 0 else int(inside[-1] + 1)
    return _Candidate(
        FailureEventType.IRRECOVERABLE_WORKSPACE_EXIT, onset, trace.action_count
    )


def annotate_failure_event(
    trace: FailureEventTrace, bounds: ReachableBounds
) -> FailureEventResult:
    """Annotate one trace, keeping success and invalid-reset outcomes contained."""

    if not trace.valid_reset:
        return FailureEventResult(
            trace.episode_id,
            False,
            False,
            AnnotationStatus.EXCLUDED_INVALID_RESET,
            None,
            None,
            None,
        )
    if trace.success:
        return FailureEventResult(
            trace.episode_id,
            True,
            True,
            AnnotationStatus.SUCCESS_NO_EVENT,
            None,
            None,
            None,
        )
    candidates = [
        candidate
        for candidate in (
            _missed_grasp(trace),
            _drop(trace),
            _workspace_exit(trace, bounds),
        )
        if candidate is not None
    ]
    if candidates:
        priority = {name: index for index, name in enumerate(EVENT_PRECEDENCE)}
        chosen = min(
            candidates, key=lambda item: (item.onset, priority[item.event_type.value])
        )
        return FailureEventResult(
            trace.episode_id,
            True,
            False,
            AnnotationStatus.ANNOTATED,
            chosen.event_type,
            chosen.onset,
            chosen.confirmation,
        )
    if trace.action_count == TERMINAL_HORIZON:
        return FailureEventResult(
            trace.episode_id,
            True,
            False,
            AnnotationStatus.ANNOTATED,
            FailureEventType.TERMINAL_HORIZON,
            TERMINAL_HORIZON,
            TERMINAL_HORIZON,
        )
    return FailureEventResult(
        trace.episode_id,
        True,
        False,
        AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL,
        None,
        None,
        None,
    )


def create_freeze_manifest(
    *,
    task: TaskIdentity,
    bounds: DiscoveryBoundsDerivation,
    primary_placement_predicate_keys: tuple[str, ...] | list[str],
    annotations: tuple[FailureEventResult, ...] | list[FailureEventResult],
    video_audit_episode_ids: tuple[str, ...] | list[str],
    implementation_commit: str,
) -> FailureEventFreezeManifest:
    """Canonicalize order and construct a validated Reality-Gate freeze."""

    predicate_keys = tuple(primary_placement_predicate_keys)
    audit_ids = tuple(video_audit_episode_ids)
    annotation_items = tuple(annotations)
    if any(not isinstance(item, str) for item in (*predicate_keys, *audit_ids)):
        raise FailureEventError("freeze predicate and video-audit IDs must be strings")
    if any(not isinstance(item, FailureEventResult) for item in annotation_items):
        raise FailureEventError("freeze annotations must be validated results")
    return FailureEventFreezeManifest(
        task=task,
        bounds=bounds,
        primary_placement_predicate_keys=tuple(sorted(predicate_keys)),
        annotations=tuple(sorted(annotation_items, key=lambda item: item.episode_id)),
        video_audit_episode_ids=tuple(sorted(audit_ids)),
        implementation_commit=implementation_commit,
    )


def failure_event_protocol_metadata() -> dict[str, Any]:
    """Return the complete finite numerical contract embedded in every freeze."""

    return {
        "scope": {
            "invalid_reset": "excluded",
            "success": "no_failure_event",
            "failed": "annotate",
        },
        "timing": "action_c_maps_frame_c_to_frame_c_plus_1",
        "missed_grasp": {
            "gripper_action_index": GRIPPER_ACTION_INDEX,
            "close_comparison": ">0",
            "zero_is_neither": True,
            "only_first_close": True,
            "post_action_frames": MISSED_GRASP_WINDOW,
            "contact_required_absent": True,
            "distance": "float64_euclidean_3d_eef_center_to_object_center",
            "trend": "ordinary_least_squares_slope_strictly_positive",
            "net_increase_m": DISTANCE_NET_INCREASE_M,
            "onset": "c+1",
            "confirmation": "c+10",
        },
        "dropped_object": {
            "loss": "contact_true_to_false_and_prior_backend_grasp_true",
            "landing_frames": DROP_LANDING_FRAMES,
            "proxy": "initial-support-height proxy",
            "support_center_z": "post_settle_frame_0_object_center_z",
            "support_tolerance_m": DROP_SUPPORT_TOLERANCE_M,
            "no_recontact_through_confirmation": True,
            "outside_goal": "phase_not_placed_on_all_landing_frames",
            "onset": "contact_loss_frame_r",
            "confirmation": "landing_u+2",
        },
        "workspace_exit": {
            "bounds_source": "all_valid_frame0_plus_valid_successful_frames1_to_terminal",
            "shape": "closed_axis_aligned_xyz",
            "margin_m": WORKSPACE_MARGIN_M,
            "rule": "first_frame_of_final_contiguous_outside_suffix",
            "confirmation": "terminal_frame",
        },
        "selection": {
            "key": "earliest_onset_then_textual_precedence",
            "precedence": list(EVENT_PRECEDENCE),
        },
        "fallback": {
            "event": FailureEventType.TERMINAL_HORIZON.value,
            "actions_required": TERMINAL_HORIZON,
            "shorter_failed_trace": AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL.value,
        },
    }


__all__ = [
    "AnnotationStatus",
    "ArtifactIdentity",
    "DiscoveryBoundsDerivation",
    "DiscoveryEpisode",
    "DiscoveryEpisodeProvenance",
    "FailureEventError",
    "FailureEventFreezeManifest",
    "FailureEventResult",
    "FailureEventTrace",
    "FailureEventType",
    "ReachableBounds",
    "TaskIdentity",
    "annotate_failure_event",
    "artifact_identity_from_rollout",
    "create_freeze_manifest",
    "derive_discovery_bounds",
    "discovery_episode_from_artifact",
    "failure_event_protocol_metadata",
    "failure_event_trace_from_artifact",
]
