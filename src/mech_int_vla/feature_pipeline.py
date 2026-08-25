"""Validated artifact-to-feature integration for the frozen M0/M1/M2 study.

The public functions in this module consume already loaded, validated in-memory
containers.  They nevertheless re-check every cross-artifact identity and the
critical array contracts before doing any reduction.  Calibration outcomes are
contained in a frozen reference bundle; Locked Test feature arithmetic can only
query that bundle and never fits, selects, or rescales on test rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .allocation import (
    AllocationError,
    ScoreAllocationReceipt,
    revalidate_score_receipt,
)
from .artifacts import RolloutArtifact
from .config import ProtocolConfig
from .determinism import hash_seed
from .features import (
    ACTION_DIMENSION,
    ACTION_DRAWS,
    ACTION_STEPS,
    COMMON_DRAWS,
    M0_FEATURE_NAMES,
    M1_FEATURE_NAMES,
    M2_EXPERT_FEATURE_NAMES,
    M2_VLM_FEATURE_NAMES,
    PHASE_ORDER,
    TRANSFORM_ORDER,
    ActionScale,
    CoverageState,
    FeatureHierarchy,
    M0Primitives,
    M1PoseState,
    M2Primitives,
    NamedFeatureRow,
    ProbeNormState,
    assemble_feature_hierarchy,
    compute_coverage_features,
    compute_m0_features,
    compute_m2_features,
    construct_m1_raw_pose,
    fit_episode_equal_action_scale,
    robust_probe_norm_distance,
)
from .probes import DEFAULT_CANDIDATE_PREFERENCE, ProbeArtifact
from .scoring import COST_FIELDS, FROZEN_TRANSFORMS, LoadedScoringSidecar

PIPELINE_SCHEMA_VERSION: Final = 1
SCORE_STRIDE_STEPS: Final = 5
SCORE_ARRAY_NAMES: Final = frozenset(
    {
        "control_step",
        "noise_seed",
        "transform_available",
        "intervention_available",
        "original_actions",
        "transformed_actions",
        "intervention_minus_actions",
        "intervention_plus_actions",
        "original_activation",
        "transformed_activation",
        "intervention_minus_activation",
        "intervention_plus_activation",
        "original_cost",
        "transformed_cost",
        "intervention_minus_cost",
        "intervention_plus_cost",
    }
)


class FeaturePipelineError(ValueError):
    """Raised before feature reduction when artifact integration fails closed."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise FeaturePipelineError(f"{name} must be a lowercase SHA-256")
    return str(value)


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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _metadata_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: NDArray[Any], *, dtype: Any) -> str:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.kind == "f" and np.isnan(array).any():
        array[np.isnan(array)] = np.nan
    dtype_little = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype_little, copy=False))
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _readonly(value: NDArray[Any], *, dtype: Any, name: str) -> NDArray[Any]:
    try:
        result = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise FeaturePipelineError(f"{name} cannot be copied") from exc
    result.setflags(write=False)
    return result


@dataclass(frozen=True, order=True)
class TaskIdentity:
    """Task fields that must be identical between Calibration and Locked Test."""

    rank: int
    suite: str
    task_id: int
    language: str
    primary_object: str
    symmetry_order: int

    def __post_init__(self) -> None:
        for name in ("rank", "symmetry_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise FeaturePipelineError(f"task {name} must be a positive integer")
        if (
            isinstance(self.task_id, bool)
            or not isinstance(self.task_id, Integral)
            or self.task_id < 0
        ):
            raise FeaturePipelineError("task task_id must be a nonnegative integer")
        for name in ("suite", "language", "primary_object"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise FeaturePipelineError(f"task {name} must be nonempty")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "suite": self.suite,
            "task_id": self.task_id,
            "language": self.language,
            "primary_object": self.primary_object,
            "symmetry_order": self.symmetry_order,
        }


@dataclass(frozen=True, order=True)
class CohortIdentity:
    """Policy, code, and scoring links shared by one feature cohort."""

    policy_revision: str
    base_vlm_revision: str
    code_commit: str
    config_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for name in ("policy_revision", "base_vlm_revision", "code_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise FeaturePipelineError(f"cohort {name} must be nonempty")
        _require_sha256(self.config_sha256, "cohort config_sha256")
        _require_sha256(self.code_sha256, "cohort code_sha256")

    def to_metadata(self) -> dict[str, str]:
        return {
            "policy_revision": self.policy_revision,
            "base_vlm_revision": self.base_vlm_revision,
            "code_commit": self.code_commit,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
        }

    def is_cross_split_compatible(self, other: object) -> bool:
        """Compare immutable execution inputs while retaining distinct commits.

        Calibration is collected at ``prereg-locked-v1`` and Locked Test at the
        later ``calibration-locked-v1`` commit.  Requiring the raw collection
        commits to be equal would therefore make every legitimate Locked Test
        cohort impossible.  The commits remain recorded in each cohort; this
        compatibility relation instead freezes the policy, base VLM, protocol
        configuration, and scoring-source content across the split boundary.
        """

        return isinstance(other, CohortIdentity) and (
            self.policy_revision,
            self.base_vlm_revision,
            self.config_sha256,
            self.code_sha256,
        ) == (
            other.policy_revision,
            other.base_vlm_revision,
            other.config_sha256,
            other.code_sha256,
        )


@dataclass(frozen=True, order=True)
class EpisodeSourceHashes:
    """Content identities for one raw-rollout/score-sidecar pair."""

    episode_id: str
    raw_metadata_sha256: str
    raw_trajectory_sha256: str
    score_metadata_sha256: str
    score_primitives_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FeaturePipelineError("source episode_id must be nonempty")
        for name in (
            "raw_metadata_sha256",
            "raw_trajectory_sha256",
            "score_metadata_sha256",
            "score_primitives_sha256",
        ):
            _require_sha256(getattr(self, name), f"source {name}")

    def to_metadata(self) -> dict[str, str]:
        return {
            "episode_id": self.episode_id,
            "raw_metadata_sha256": self.raw_metadata_sha256,
            "raw_trajectory_sha256": self.raw_trajectory_sha256,
            "score_metadata_sha256": self.score_metadata_sha256,
            "score_primitives_sha256": self.score_primitives_sha256,
        }


def _copy_action_scale(value: ActionScale) -> ActionScale:
    if not isinstance(value, ActionScale):
        raise FeaturePipelineError("reference action_scale has the wrong type")
    return ActionScale(value.values, value.replaced_by_one, value.episode_ids)


def _copy_coverage_state(value: CoverageState) -> CoverageState:
    if not isinstance(value, CoverageState):
        raise FeaturePipelineError("reference coverage state has the wrong type")
    return CoverageState(
        value.episode_id,
        value.base_init_state_id,
        value.control_step,
        value.split,
        value.phase,
        value.success,
        value.vector,
    )


def _copy_probe_norm_state(value: ProbeNormState) -> ProbeNormState:
    if not isinstance(value, ProbeNormState):
        raise FeaturePipelineError("reference probe-norm state has the wrong type")
    return ProbeNormState(
        value.episode_id,
        value.base_init_state_id,
        value.control_step,
        value.split,
        value.phase,
        value.success,
        value.mean_norm,
    )


def _reference_state_sha256(
    coverage_states: Sequence[CoverageState],
    probe_norm_states: Sequence[ProbeNormState],
) -> str:
    digest = hashlib.sha256()
    for coverage, probe_norm in zip(coverage_states, probe_norm_states, strict=True):
        identity = (
            coverage.episode_id,
            coverage.base_init_state_id,
            coverage.control_step,
            coverage.split,
            coverage.phase,
            coverage.success,
            probe_norm.mean_norm,
        )
        digest.update(_canonical_json({"identity": identity}))
        vector = np.array(coverage.vector, dtype="<f8", copy=True, order="C")
        if np.isnan(vector).any():
            vector[np.isnan(vector)] = np.nan
        digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureReferenceBundle:
    """Frozen full-Calibration state used by every Locked Test feature query."""

    action_scale: ActionScale
    coverage_states: tuple[CoverageState, ...]
    probe_norm_states: tuple[ProbeNormState, ...]
    probe_sha256: str
    selected_candidate: str
    task_identity: TaskIdentity
    cohort_identity: CohortIdentity
    source_hashes: tuple[EpisodeSourceHashes, ...]

    def __post_init__(self) -> None:
        action_scale = _copy_action_scale(self.action_scale)
        coverage = tuple(
            sorted(
                (_copy_coverage_state(state) for state in self.coverage_states),
                key=lambda state: (state.episode_id, state.control_step),
            )
        )
        probe_norm = tuple(
            sorted(
                (_copy_probe_norm_state(state) for state in self.probe_norm_states),
                key=lambda state: (state.episode_id, state.control_step),
            )
        )
        if not coverage or len(coverage) != len(probe_norm):
            raise FeaturePipelineError(
                "reference coverage and probe-norm states must be nonempty and aligned"
            )
        coverage_ids = [(state.episode_id, state.control_step) for state in coverage]
        probe_ids = [(state.episode_id, state.control_step) for state in probe_norm]
        if coverage_ids != probe_ids or len(coverage_ids) != len(set(coverage_ids)):
            raise FeaturePipelineError("reference state identities are not one-to-one")
        if any(state.split != "calibration" for state in coverage) or any(
            state.split != "calibration" for state in probe_norm
        ):
            raise FeaturePipelineError("reference states must all be Calibration")
        if any(
            coverage_state.phase != norm_state.phase
            or coverage_state.success != norm_state.success
            or coverage_state.base_init_state_id != norm_state.base_init_state_id
            for coverage_state, norm_state in zip(coverage, probe_norm, strict=True)
        ):
            raise FeaturePipelineError("reference coverage/probe metadata disagree")
        probe_sha = _require_sha256(self.probe_sha256, "reference probe_sha256")
        if self.selected_candidate not in DEFAULT_CANDIDATE_PREFERENCE:
            raise FeaturePipelineError("reference selected candidate is not frozen")
        if not isinstance(self.task_identity, TaskIdentity) or not isinstance(
            self.cohort_identity, CohortIdentity
        ):
            raise FeaturePipelineError("reference identities have the wrong type")
        sources = tuple(sorted(self.source_hashes, key=lambda item: item.episode_id))
        if not sources or len({item.episode_id for item in sources}) != len(sources):
            raise FeaturePipelineError("reference source hashes are empty or duplicate")
        source_episode_ids = tuple(item.episode_id for item in sources)
        if source_episode_ids != tuple(sorted(action_scale.episode_ids)):
            raise FeaturePipelineError(
                "reference action scale and source episode sets disagree"
            )
        if set(source_episode_ids) != {state.episode_id for state in coverage}:
            raise FeaturePipelineError(
                "reference state and source episode sets disagree"
            )
        object.__setattr__(self, "action_scale", action_scale)
        object.__setattr__(self, "coverage_states", coverage)
        object.__setattr__(self, "probe_norm_states", probe_norm)
        object.__setattr__(self, "probe_sha256", probe_sha)
        object.__setattr__(self, "source_hashes", sources)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "kind": "full_calibration_feature_reference",
            "probe_sha256": self.probe_sha256,
            "selected_candidate": self.selected_candidate,
            "task_identity": self.task_identity.to_metadata(),
            "cohort_identity": self.cohort_identity.to_metadata(),
            "action_scale": self.action_scale.to_metadata(),
            "reference_state_count": len(self.coverage_states),
            "reference_state_sha256": _reference_state_sha256(
                self.coverage_states, self.probe_norm_states
            ),
            "sources": [source.to_metadata() for source in self.source_hashes],
        }

    @property
    def metadata_sha256(self) -> str:
        return _metadata_sha256(self.to_metadata())


@dataclass(frozen=True)
class FeatureStateRecord:
    """One ordered scored state with nested feature rows and source identity."""

    episode_id: str
    base_init_state_id: int
    split: str
    control_step: int
    terminal_failure_label: bool
    phase: str
    hierarchy: FeatureHierarchy
    source_hashes: EpisodeSourceHashes

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise FeaturePipelineError("feature state episode_id must be nonempty")
        if (
            isinstance(self.base_init_state_id, bool)
            or not isinstance(self.base_init_state_id, Integral)
            or self.base_init_state_id < 0
        ):
            raise FeaturePipelineError("feature state base init ID is invalid")
        if self.split not in {"calibration", "locked_test"}:
            raise FeaturePipelineError("feature state split is invalid")
        if (
            isinstance(self.control_step, bool)
            or not isinstance(self.control_step, Integral)
            or self.control_step < 0
            or self.control_step % SCORE_STRIDE_STEPS != 0
        ):
            raise FeaturePipelineError("feature state control step is off cadence")
        if not isinstance(self.terminal_failure_label, bool):
            raise FeaturePipelineError("terminal failure label must be boolean")
        if self.phase not in PHASE_ORDER:
            raise FeaturePipelineError("feature state phase is invalid")
        if not isinstance(self.hierarchy, FeatureHierarchy):
            raise FeaturePipelineError("feature hierarchy has the wrong type")
        if not isinstance(self.source_hashes, EpisodeSourceHashes):
            raise FeaturePipelineError("feature source hashes have the wrong type")
        if self.source_hashes.episode_id != self.episode_id:
            raise FeaturePipelineError("feature state/source episode IDs disagree")
        object.__setattr__(self, "base_init_state_id", int(self.base_init_state_id))
        object.__setattr__(self, "control_step", int(self.control_step))

    @property
    def m0(self) -> NamedFeatureRow:
        return self.hierarchy.m0

    @property
    def m1(self) -> NamedFeatureRow:
        return self.hierarchy.m1

    @property
    def m2(self) -> NamedFeatureRow:
        return self.hierarchy.m2


@dataclass(frozen=True)
class FeatureCohort:
    """Deterministically ordered feature matrices and their complete provenance."""

    split: str
    records: tuple[FeatureStateRecord, ...]
    m0_names: tuple[str, ...]
    m1_names: tuple[str, ...]
    m2_names: tuple[str, ...]
    m0_matrix: NDArray[np.float64]
    m1_matrix: NDArray[np.float64]
    m2_matrix: NDArray[np.float64]
    probe_sha256: str
    reference_bundle_sha256: str
    task_identity: TaskIdentity
    cohort_identity: CohortIdentity

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "locked_test"}:
            raise FeaturePipelineError("feature cohort split is invalid")
        records = tuple(self.records)
        if not records:
            raise FeaturePipelineError("feature cohort cannot be empty")
        identities = [(record.episode_id, record.control_step) for record in records]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise FeaturePipelineError("feature states must be unique and sorted")
        if any(record.split != self.split for record in records):
            raise FeaturePipelineError("feature record split differs from cohort")
        if tuple(self.m0_names) != M0_FEATURE_NAMES:
            raise FeaturePipelineError("M0 column order differs from frozen schema")
        if tuple(self.m1_names) != M1_FEATURE_NAMES:
            raise FeaturePipelineError("M1 column order differs from frozen schema")
        if tuple(self.m2_names) not in {
            M2_VLM_FEATURE_NAMES,
            M2_EXPERT_FEATURE_NAMES,
        }:
            raise FeaturePipelineError("M2 column order differs from frozen schema")
        matrices = {}
        for name, value, width in (
            ("m0_matrix", self.m0_matrix, len(self.m0_names)),
            ("m1_matrix", self.m1_matrix, len(self.m1_names)),
            ("m2_matrix", self.m2_matrix, len(self.m2_names)),
        ):
            matrix = _readonly(value, dtype=np.float64, name=name)
            if matrix.shape != (len(records), width) or np.isinf(matrix).any():
                raise FeaturePipelineError(
                    f"{name} must align with records and contain no infinity"
                )
            matrices[name] = matrix
        expected = {
            "m0_matrix": np.stack([record.m0.values for record in records]),
            "m1_matrix": np.stack([record.m1.values for record in records]),
            "m2_matrix": np.stack([record.m2.values for record in records]),
        }
        if any(
            not np.array_equal(matrices[name], expected[name], equal_nan=True)
            for name in matrices
        ):
            raise FeaturePipelineError("feature matrices differ from state records")
        probe_sha = _require_sha256(self.probe_sha256, "cohort probe_sha256")
        reference_sha = _require_sha256(
            self.reference_bundle_sha256, "cohort reference_bundle_sha256"
        )
        if not isinstance(self.task_identity, TaskIdentity) or not isinstance(
            self.cohort_identity, CohortIdentity
        ):
            raise FeaturePipelineError("feature cohort identities have the wrong type")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "m0_names", tuple(self.m0_names))
        object.__setattr__(self, "m1_names", tuple(self.m1_names))
        object.__setattr__(self, "m2_names", tuple(self.m2_names))
        object.__setattr__(self, "m0_matrix", matrices["m0_matrix"])
        object.__setattr__(self, "m1_matrix", matrices["m1_matrix"])
        object.__setattr__(self, "m2_matrix", matrices["m2_matrix"])
        object.__setattr__(self, "probe_sha256", probe_sha)
        object.__setattr__(self, "reference_bundle_sha256", reference_sha)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "split": self.split,
            "probe_sha256": self.probe_sha256,
            "reference_bundle_sha256": self.reference_bundle_sha256,
            "task_identity": self.task_identity.to_metadata(),
            "cohort_identity": self.cohort_identity.to_metadata(),
            "rows": [
                {
                    "episode_id": record.episode_id,
                    "base_init_state_id": record.base_init_state_id,
                    "control_step": record.control_step,
                    "terminal_failure_label": record.terminal_failure_label,
                    "phase": record.phase,
                    "source_hashes": record.source_hashes.to_metadata(),
                }
                for record in self.records
            ],
            "columns": {
                "M0": list(self.m0_names),
                "M1": list(self.m1_names),
                "M2": list(self.m2_names),
            },
            "matrix_sha256": {
                "M0": _array_sha256(self.m0_matrix, dtype=np.float64),
                "M1": _array_sha256(self.m1_matrix, dtype=np.float64),
                "M2": _array_sha256(self.m2_matrix, dtype=np.float64),
            },
        }

    @property
    def provenance_sha256(self) -> str:
        return _metadata_sha256(self.to_metadata())


@dataclass(frozen=True)
class _EpisodePair:
    episode_id: str
    raw: RolloutArtifact
    score: LoadedScoringSidecar
    task_identity: TaskIdentity
    cohort_identity: CohortIdentity
    source_hashes: EpisodeSourceHashes
    base_init_state_id: int
    success: bool


@dataclass(frozen=True)
class _PreparedState:
    pair: _EpisodePair
    index: int
    control_step: int
    phase: str
    m1_raw: NamedFeatureRow
    original_probe_vectors: NDArray[np.float64]
    transformed_probe_vectors: NDArray[np.float64]
    mean_probe_norm: float


def _task_identity(raw: RolloutArtifact) -> TaskIdentity:
    try:
        episode = raw.metadata["episode"]
        task = raw.metadata["task"]
        identity = TaskIdentity(
            rank=int(task["rank"]),
            suite=str(task["suite"]),
            task_id=int(task["task_id"]),
            language=str(task["language"]),
            primary_object=str(task["primary_object"]),
            symmetry_order=int(task["planar_symmetry_order"]),
        )
        if (
            episode["suite"] != identity.suite
            or int(episode["task_id"]) != identity.task_id
            or int(episode["task_rank"]) != identity.rank
            or raw.metadata["task_language"] != identity.language
        ):
            raise FeaturePipelineError("raw episode and task identities disagree")
        return identity
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, FeaturePipelineError):
            raise
        raise FeaturePipelineError("raw task identity is malformed") from exc


def _cohort_identity(
    raw: RolloutArtifact, score: LoadedScoringSidecar
) -> CohortIdentity:
    try:
        episode = raw.metadata["episode"]
        model = raw.metadata["model"]
        links = score.metadata["links"]
        if episode["policy_revision"] != model["policy_revision"]:
            raise FeaturePipelineError("raw policy revisions disagree")
        return CohortIdentity(
            policy_revision=str(model["policy_revision"]),
            base_vlm_revision=str(model["base_vlm_revision"]),
            code_commit=str(episode["code_commit"]),
            config_sha256=str(links["config_sha256"]),
            code_sha256=str(links["code_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, FeaturePipelineError):
            raise
        raise FeaturePipelineError("cohort identity is malformed") from exc


def _require_array(
    arrays: Mapping[str, NDArray[Any]],
    name: str,
    *,
    dtype: Any,
    shape: tuple[int, ...],
) -> NDArray[Any]:
    if name not in arrays:
        raise FeaturePipelineError(f"score sidecar is missing {name}")
    value = np.asarray(arrays[name])
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        raise FeaturePipelineError(
            f"score {name} has dtype/shape {value.dtype}/{value.shape}; "
            f"expected {np.dtype(dtype)}/{shape}"
        )
    return value


def _validate_masked_finiteness(
    value: NDArray[Any], mask: NDArray[np.bool_], name: str
) -> None:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    expected = np.broadcast_to(expanded, value.shape)
    if not np.array_equal(np.isfinite(value), expected):
        raise FeaturePipelineError(f"score {name} finiteness disagrees with its mask")


def _validate_score_arrays(
    pair_episode_id: str,
    arrays: Mapping[str, NDArray[Any]],
    *,
    state_count: int,
    width: int,
) -> None:
    if set(arrays) != SCORE_ARRAY_NAMES:
        raise FeaturePipelineError("score sidecar array names differ from schema")
    steps = _require_array(arrays, "control_step", dtype=np.int32, shape=(state_count,))
    seeds = _require_array(
        arrays, "noise_seed", dtype=np.int64, shape=(state_count, ACTION_DRAWS)
    )
    expected_seeds = np.asarray(
        [
            [
                hash_seed("score-noise-v1", pair_episode_id, int(step), draw)
                for draw in range(ACTION_DRAWS)
            ]
            for step in steps
        ],
        dtype=np.int64,
    )
    if not np.array_equal(seeds, expected_seeds):
        raise FeaturePipelineError("score noise seeds differ from frozen plan")
    transform_mask = _require_array(
        arrays, "transform_available", dtype=np.bool_, shape=(state_count, 6)
    )
    if not np.all(transform_mask[:, :4]):
        raise FeaturePipelineError(
            "brightness and camera transforms must always be available"
        )
    intervention_mask = _require_array(
        arrays,
        "intervention_available",
        dtype=np.bool_,
        shape=(state_count, COMMON_DRAWS),
    )
    original_shapes = {
        "original_actions": (state_count, ACTION_DRAWS, ACTION_STEPS, ACTION_DIMENSION),
        "original_activation": (state_count, ACTION_DRAWS, width),
        "original_cost": (state_count, ACTION_DRAWS, len(COST_FIELDS)),
    }
    for name, shape in original_shapes.items():
        dtype = np.float64 if name.endswith("cost") else np.float32
        value = _require_array(arrays, name, dtype=dtype, shape=shape)
        if not np.isfinite(value).all():
            raise FeaturePipelineError(f"score {name} must be entirely finite")
    transformed_shapes = {
        "transformed_actions": (
            state_count,
            6,
            COMMON_DRAWS,
            ACTION_STEPS,
            ACTION_DIMENSION,
        ),
        "transformed_activation": (state_count, 6, COMMON_DRAWS, width),
        "transformed_cost": (state_count, 6, COMMON_DRAWS, len(COST_FIELDS)),
    }
    for name, shape in transformed_shapes.items():
        dtype = np.float64 if name.endswith("cost") else np.float32
        value = _require_array(arrays, name, dtype=dtype, shape=shape)
        _validate_masked_finiteness(value, transform_mask, name)
    for direction in ("minus", "plus"):
        for suffix, tail, dtype in (
            ("actions", (ACTION_STEPS, ACTION_DIMENSION), np.float32),
            ("activation", (width,), np.float32),
            ("cost", (len(COST_FIELDS),), np.float64),
        ):
            name = f"intervention_{direction}_{suffix}"
            value = _require_array(
                arrays,
                name,
                dtype=dtype,
                shape=(state_count, COMMON_DRAWS, *tail),
            )
            _validate_masked_finiteness(value, intervention_mask, name)
    for name in (
        "original_cost",
        "transformed_cost",
        "intervention_minus_cost",
        "intervention_plus_cost",
    ):
        cost = np.asarray(arrays[name])
        finite_rows = np.isfinite(cost).all(axis=-1)
        finite_cost = cost[finite_rows]
        if finite_cost.size and (
            np.any(finite_cost < 0.0)
            or not np.equal(finite_cost[:, 1:], np.floor(finite_cost[:, 1:])).all()
            or not np.all(finite_cost[:, 2] == 1.0)
            or not np.all(finite_cost[:, 3] == float(name.startswith("intervention_")))
        ):
            raise FeaturePipelineError(f"score {name} violates the cost contract")


def _validate_pair(
    raw: RolloutArtifact,
    score: LoadedScoringSidecar,
    *,
    expected_split: str,
    probe: ProbeArtifact,
    probe_sha256: str,
) -> _EpisodePair:
    if not isinstance(raw, RolloutArtifact):
        raise FeaturePipelineError("raw input is not a RolloutArtifact")
    if not isinstance(score, LoadedScoringSidecar):
        raise FeaturePipelineError("score input is not a LoadedScoringSidecar")
    episode_id = raw.episode_id
    try:
        raw_split = str(raw.metadata["episode"]["split"])
        score_episode_id = str(score.metadata["episode_id"])
        score_split = str(score.metadata["split"])
        links = score.metadata["links"]
    except (KeyError, TypeError) as exc:
        raise FeaturePipelineError("episode linkage metadata is malformed") from exc
    if score_episode_id != episode_id:
        raise FeaturePipelineError("raw and score episode IDs disagree")
    if raw_split != expected_split or score_split != expected_split:
        raise FeaturePipelineError(f"feature inputs must all be {expected_split}")
    if (
        raw.path.name != episode_id
        or score.path.name != episode_id
        or raw.path.parent.name != expected_split
        or score.path.parent.name != expected_split
    ):
        raise FeaturePipelineError("container paths disagree with episode IDs")
    if raw.hashes.metadata_sha256 != links["raw_metadata_sha256"]:
        raise FeaturePipelineError("score raw metadata hash link is stale")
    if raw.hashes.trajectory_sha256 != links["raw_trajectory_sha256"]:
        raise FeaturePipelineError("score raw trajectory hash link is stale")
    if links["probe_sha256"] != probe_sha256:
        raise FeaturePipelineError("score sidecar was built with a different probe")
    try:
        protocol = score.metadata["protocol"]
        transform_order = tuple(
            str(transform["name"]) for transform in protocol["transform_order"]
        )
        intervention_degrees = tuple(
            float(value) for value in protocol["intervention_degrees"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeaturePipelineError("score protocol metadata is malformed") from exc
    if (
        protocol.get("score_stride_steps") != SCORE_STRIDE_STEPS
        or protocol.get("original_draws") != ACTION_DRAWS
        or protocol.get("common_draws") != COMMON_DRAWS
        or protocol.get("chunk_actions") != ACTION_STEPS
        or protocol.get("action_dimension") != ACTION_DIMENSION
        or transform_order != tuple(transform.name for transform in FROZEN_TRANSFORMS)
        or intervention_degrees != (-10.0, 10.0)
    ):
        raise FeaturePipelineError("score protocol constants differ from frozen values")
    for name, value in (
        ("raw metadata", raw.hashes.metadata_sha256),
        ("raw trajectory", raw.hashes.trajectory_sha256),
        ("score metadata", score.metadata_sha256),
        ("score primitives", score.primitives_sha256),
    ):
        _require_sha256(value, f"{episode_id} {name}")
    if not raw.valid_reset or raw.action_count == 0:
        raise FeaturePipelineError(
            "invalid or empty rollouts must be excluded upstream"
        )
    outcome = raw.metadata["outcome"]
    factual = score.metadata["factual_outcome"]
    if (
        bool(outcome["success"]) != bool(factual["success"])
        or str(outcome["status"]) != str(factual["status"])
        or int(outcome["control_steps"]) != int(factual["control_steps"])
        or raw.action_count != int(factual["control_steps"])
    ):
        raise FeaturePipelineError("raw and score factual outcomes disagree")
    raw_steps = np.asarray(raw.arrays["activation_control_step"])
    score_steps = np.asarray(score.arrays["control_step"])
    expected_steps = np.arange(0, raw.action_count, SCORE_STRIDE_STEPS, dtype=np.int32)
    if (
        raw_steps.dtype != np.int32
        or score_steps.dtype != np.int32
        or not np.array_equal(raw_steps, expected_steps)
        or not np.array_equal(score_steps, expected_steps)
    ):
        raise FeaturePipelineError("raw/score stride control steps disagree")
    state_count = int(expected_steps.size)
    capture = score.metadata["capture"]
    if int(capture["state_count"]) != state_count:
        raise FeaturePipelineError("score state cardinality differs from raw rollout")
    width = int(capture["selected_activation_width"])
    probe_width = int(probe.model.feature_center.size)
    if width != probe_width or probe.model.coefficient.shape != (2, probe_width):
        raise FeaturePipelineError("score activation width differs from probe width")
    _validate_score_arrays(
        episode_id, score.arrays, state_count=state_count, width=width
    )
    task = _task_identity(raw)
    if task.symmetry_order != probe.model.symmetry_order:
        raise FeaturePipelineError("task symmetry order differs from probe")
    cohort = _cohort_identity(raw, score)
    source = EpisodeSourceHashes(
        episode_id,
        raw.hashes.metadata_sha256,
        raw.hashes.trajectory_sha256,
        score.metadata_sha256,
        score.primitives_sha256,
    )
    base_init = raw.metadata["episode"]["base_init_state_id"]
    if (
        isinstance(base_init, bool)
        or not isinstance(base_init, Integral)
        or base_init < 0
    ):
        raise FeaturePipelineError("base init state ID is invalid")
    return _EpisodePair(
        episode_id,
        raw,
        score,
        task,
        cohort,
        source,
        int(base_init),
        bool(outcome["success"]),
    )


def _validate_and_pair(
    raw_artifacts: Sequence[RolloutArtifact],
    score_sidecars: Sequence[LoadedScoringSidecar],
    probe: ProbeArtifact,
    *,
    expected_split: str,
    probe_sha256: str,
    allocation_receipt: ScoreAllocationReceipt,
) -> tuple[_EpisodePair, ...]:
    if not isinstance(probe, ProbeArtifact):
        raise FeaturePipelineError("probe has the wrong artifact type")
    if probe.candidate not in DEFAULT_CANDIDATE_PREFERENCE:
        raise FeaturePipelineError("probe selected candidate is not frozen")
    probe_sha = _require_sha256(probe_sha256, "bound probe sha256")
    try:
        raws = tuple(raw_artifacts)
    except TypeError as exc:
        raise FeaturePipelineError("raw feature cohort must be an iterable") from exc
    try:
        scores = tuple(score_sidecars)
    except TypeError as exc:
        raise FeaturePipelineError("score feature cohort must be an iterable") from exc
    if not raws or not scores:
        raise FeaturePipelineError("raw and score cohorts must be nonempty")
    raw_by_id: dict[str, RolloutArtifact] = {}
    for raw in raws:
        if not isinstance(raw, RolloutArtifact):
            raise FeaturePipelineError("raw cohort contains the wrong type")
        if raw.episode_id in raw_by_id:
            raise FeaturePipelineError("raw cohort contains duplicate episode IDs")
        raw_by_id[raw.episode_id] = raw
    score_by_id: dict[str, LoadedScoringSidecar] = {}
    for score in scores:
        if not isinstance(score, LoadedScoringSidecar):
            raise FeaturePipelineError("score cohort contains the wrong type")
        try:
            episode_id = str(score.metadata["episode_id"])
        except (KeyError, TypeError) as exc:
            raise FeaturePipelineError("score episode ID is malformed") from exc
        if episode_id in score_by_id:
            raise FeaturePipelineError("score cohort contains duplicate episode IDs")
        score_by_id[episode_id] = score
    if set(raw_by_id) != set(score_by_id):
        raise FeaturePipelineError("raw and score episode sets are not one-to-one")

    expected_raw = {
        identity.episode_id: identity
        for identity in allocation_receipt.rollout.valid_artifacts
    }
    expected_scores = {
        identity.episode_id: identity for identity in allocation_receipt.score_artifacts
    }
    if set(raw_by_id) != set(expected_raw):
        raise FeaturePipelineError(
            "raw feature cohort does not exactly match the score allocation receipt"
        )
    if set(score_by_id) != set(expected_scores):
        raise FeaturePipelineError(
            "score feature cohort does not exactly match the score allocation receipt"
        )
    for episode_id in sorted(expected_raw):
        raw = raw_by_id[episode_id]
        score = score_by_id[episode_id]
        raw_identity = expected_raw[episode_id]
        score_identity = expected_scores[episode_id]
        if (
            raw.hashes.metadata_sha256 != raw_identity.metadata_sha256
            or raw.hashes.trajectory_sha256 != raw_identity.trajectory_sha256
        ):
            raise FeaturePipelineError(
                f"{episode_id}: raw hashes differ from score allocation receipt"
            )
        if (
            score.metadata_sha256 != score_identity.metadata_sha256
            or score.primitives_sha256 != score_identity.primitives_sha256
        ):
            raise FeaturePipelineError(
                f"{episode_id}: score hashes differ from score allocation receipt"
            )
        links = score.metadata.get("links")
        expected_links = {
            "raw_metadata_sha256": score_identity.raw_metadata_sha256,
            "raw_trajectory_sha256": score_identity.raw_trajectory_sha256,
            "probe_sha256": allocation_receipt.probe_sha256,
            "config_sha256": allocation_receipt.config_sha256,
            "code_sha256": allocation_receipt.code_sha256,
        }
        if (
            not isinstance(links, Mapping)
            or set(links) != set(expected_links)
            or any(links.get(name) != value for name, value in expected_links.items())
        ):
            raise FeaturePipelineError(
                f"{episode_id}: score links differ from score allocation receipt"
            )
    pairs = tuple(
        _validate_pair(
            raw_by_id[episode_id],
            score_by_id[episode_id],
            expected_split=expected_split,
            probe=probe,
            probe_sha256=probe_sha,
        )
        for episode_id in sorted(raw_by_id)
    )
    task = pairs[0].task_identity
    cohort = pairs[0].cohort_identity
    if any(pair.task_identity != task for pair in pairs):
        raise FeaturePipelineError("feature cohort mixes task identities")
    if any(pair.cohort_identity != cohort for pair in pairs):
        raise FeaturePipelineError("feature cohort mixes policy/config/code links")
    return pairs


def _validated_feature_allocation(
    score_receipt: ScoreAllocationReceipt,
    bound_probe: object,
    *,
    expected_split: str,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> tuple[ScoreAllocationReceipt, ProbeArtifact, str]:
    """Resolve the sole trusted identity context for a feature cohort."""

    try:
        receipt = revalidate_score_receipt(
            score_receipt,
            bound_probe,
            protocol=protocol,
            repo_root=repo_root,
        )
    except (AllocationError, TypeError, ValueError) as exc:
        raise FeaturePipelineError(
            f"score allocation receipt validation failed: {exc}"
        ) from exc
    if receipt.rollout.source.split.value != expected_split:
        raise FeaturePipelineError(
            f"score allocation receipt must target {expected_split}"
        )
    probe = getattr(bound_probe, "probe", None)
    if not isinstance(probe, ProbeArtifact):
        raise FeaturePipelineError("bound probe has the wrong numerical artifact type")
    probe_sha256 = _require_sha256(receipt.probe_sha256, "bound probe sha256")
    return receipt, probe, probe_sha256


def _predict_probe_block(
    probe: ProbeArtifact, activation: NDArray[Any]
) -> NDArray[np.float64]:
    matrix = np.asarray(activation, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise FeaturePipelineError("probe activation block must be a finite matrix")
    try:
        result = np.asarray(probe.model.predict_raw(matrix), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FeaturePipelineError("probe prediction failed") from exc
    if result.shape != (matrix.shape[0], 2) or not np.isfinite(result).all():
        raise FeaturePipelineError("probe prediction returned an invalid matrix")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _prepare_states(
    pairs: Sequence[_EpisodePair], probe: ProbeArtifact
) -> tuple[_PreparedState, ...]:
    prepared: list[_PreparedState] = []
    for pair in pairs:
        raw = pair.raw
        score = pair.score
        arrays = raw.arrays
        score_arrays = score.arrays
        steps = np.asarray(score_arrays["control_step"])
        transform_masks = np.asarray(score_arrays["transform_available"])
        for index, step_value in enumerate(steps):
            step = int(step_value)
            phase = str(arrays["frame_phase"][step])
            if phase not in PHASE_ORDER:
                raise FeaturePipelineError("raw state contains an unknown task phase")
            goal_present = bool(arrays["frame_goal_present"][step])
            pose = M1PoseState.from_arrays(
                eef_position=arrays["frame_eef_position"][step],
                eef_quaternion_xyzw=arrays["frame_eef_quaternion_xyzw"][step],
                object_position=arrays["frame_primary_object_position"][step],
                object_quaternion_wxyz=arrays["frame_primary_object_quaternion_wxyz"][
                    step
                ],
                goal_position=(
                    arrays["frame_goal_position"][step] if goal_present else None
                ),
                goal_quaternion_wxyz=(
                    arrays["frame_goal_quaternion_wxyz"][step] if goal_present else None
                ),
                gripper_qpos=arrays["frame_gripper_qpos"][step],
                primary_contact=bool(arrays["frame_primary_gripper_contact"][step]),
                backend_grasp=bool(arrays["frame_primary_grasped"][step]),
                normalized_step=step / int(raw.metadata["execution"]["max_steps"]),
                phase=phase,
                symmetry_order=pair.task_identity.symmetry_order,
            )
            m1_raw = construct_m1_raw_pose(pose)
            original_vectors = _predict_probe_block(
                probe, score_arrays["original_activation"][index]
            )
            transformed_vectors = np.full((6, COMMON_DRAWS, 2), np.nan)
            for transform_index, available in enumerate(transform_masks[index]):
                if available:
                    transformed_vectors[transform_index] = _predict_probe_block(
                        probe,
                        score_arrays["transformed_activation"][index, transform_index],
                    )
            transformed_vectors.setflags(write=False)
            prepared.append(
                _PreparedState(
                    pair,
                    index,
                    step,
                    phase,
                    m1_raw,
                    original_vectors,
                    transformed_vectors,
                    float(np.mean(np.linalg.norm(original_vectors, axis=1))),
                )
            )
    return tuple(
        sorted(prepared, key=lambda state: (state.pair.episode_id, state.control_step))
    )


def _reference_states(
    prepared: Sequence[_PreparedState],
) -> tuple[tuple[CoverageState, ...], tuple[ProbeNormState, ...]]:
    coverage = []
    probe_norm = []
    for state in prepared:
        pair = state.pair
        coverage.append(
            CoverageState.create(
                episode_id=pair.episode_id,
                base_init_state_id=pair.base_init_state_id,
                control_step=state.control_step,
                split="calibration",
                phase=state.phase,
                success=pair.success,
                raw_pose=state.m1_raw,
            )
        )
        probe_norm.append(
            ProbeNormState(
                pair.episode_id,
                pair.base_init_state_id,
                state.control_step,
                "calibration",
                state.phase,
                pair.success,
                state.mean_probe_norm,
            )
        )
    return tuple(coverage), tuple(probe_norm)


def _query_states(
    state: _PreparedState, *, split: str
) -> tuple[CoverageState, ProbeNormState]:
    pair = state.pair
    coverage = CoverageState.create(
        episode_id=pair.episode_id,
        base_init_state_id=pair.base_init_state_id,
        control_step=state.control_step,
        split=split,
        phase=state.phase,
        success=pair.success,
        raw_pose=state.m1_raw,
    )
    norm = ProbeNormState(
        pair.episode_id,
        pair.base_init_state_id,
        state.control_step,
        split,
        state.phase,
        pair.success,
        state.mean_probe_norm,
    )
    return coverage, norm


def _state_hierarchy(
    state: _PreparedState,
    *,
    split: str,
    probe: ProbeArtifact,
    action_scale: ActionScale,
    coverage_references: Sequence[CoverageState],
    probe_norm_references: Sequence[ProbeNormState],
) -> FeatureHierarchy:
    score_arrays = state.pair.score.arrays
    index = state.index
    transform_mask = score_arrays["transform_available"][index]
    m0 = compute_m0_features(
        M0Primitives.from_arrays(
            original_actions=score_arrays["original_actions"][index],
            transformed_actions=score_arrays["transformed_actions"][index],
            transformed_available=transform_mask,
            transform_order=TRANSFORM_ORDER,
        ),
        action_scale,
    )
    coverage_query, norm_query = _query_states(state, split=split)
    coverage = compute_coverage_features(coverage_query, coverage_references)
    robust_norm = robust_probe_norm_distance(norm_query, probe_norm_references)
    intervention_actions = np.stack(
        (
            score_arrays["intervention_minus_actions"][index],
            score_arrays["intervention_plus_actions"][index],
        )
    )
    m2 = compute_m2_features(
        M2Primitives.from_arrays(
            original_probe_vectors=state.original_probe_vectors,
            transformed_probe_vectors=state.transformed_probe_vectors,
            transformed_available=transform_mask,
            intervention_actions=intervention_actions,
            intervention_available=score_arrays["intervention_available"][index],
            symmetry_order=state.pair.task_identity.symmetry_order,
            noise_dependent=probe.candidate != "vlm_context",
        ),
        action_scale,
        robust_norm_distance=robust_norm,
    )
    return assemble_feature_hierarchy(m0, state.m1_raw, coverage, m2)


def _build_cohort(
    prepared: Sequence[_PreparedState],
    *,
    split: str,
    probe: ProbeArtifact,
    probe_sha256: str,
    action_scale: ActionScale,
    coverage_references: Sequence[CoverageState],
    probe_norm_references: Sequence[ProbeNormState],
    reference_sha256: str,
) -> FeatureCohort:
    records = []
    for state in prepared:
        pair = state.pair
        hierarchy = _state_hierarchy(
            state,
            split=split,
            probe=probe,
            action_scale=action_scale,
            coverage_references=coverage_references,
            probe_norm_references=probe_norm_references,
        )
        records.append(
            FeatureStateRecord(
                pair.episode_id,
                pair.base_init_state_id,
                split,
                state.control_step,
                not pair.success,
                state.phase,
                hierarchy,
                pair.source_hashes,
            )
        )
    ordered = tuple(records)
    m0_matrix = np.stack([record.m0.values for record in ordered])
    m1_matrix = np.stack([record.m1.values for record in ordered])
    m2_matrix = np.stack([record.m2.values for record in ordered])
    return FeatureCohort(
        split,
        ordered,
        ordered[0].m0.names,
        ordered[0].m1.names,
        ordered[0].m2.names,
        m0_matrix,
        m1_matrix,
        m2_matrix,
        _require_sha256(probe_sha256, "bound probe sha256"),
        reference_sha256,
        prepared[0].pair.task_identity,
        prepared[0].pair.cohort_identity,
    )


def build_calibration_features(
    raw_artifacts: Sequence[RolloutArtifact],
    score_sidecars: Sequence[LoadedScoringSidecar],
    bound_probe: object,
    score_receipt: ScoreAllocationReceipt,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
    split: str = "calibration",
) -> tuple[FeatureReferenceBundle, FeatureCohort]:
    """Build OOF features from one exact score allocation for a frozen split.

    ``split`` defaults to "calibration" and is threaded mechanically through
    the split-bound validators; the default path is byte-identical to the
    pre-parameterization behavior. Only "calibration" (Calibration scoring)
    and "locked_test" (Locked Test scoring) are accepted by the validators.
    """

    receipt, probe, bound_probe_sha256 = _validated_feature_allocation(
        score_receipt,
        bound_probe,
        expected_split=split,
        protocol=protocol,
        repo_root=repo_root,
    )

    pairs = _validate_and_pair(
        raw_artifacts,
        score_sidecars,
        probe,
        expected_split=split,
        probe_sha256=bound_probe_sha256,
        allocation_receipt=receipt,
    )
    prepared = _prepare_states(pairs, probe)
    original_actions = np.stack(
        [state.pair.score.arrays["original_actions"][state.index] for state in prepared]
    )
    episode_ids = np.asarray([state.pair.episode_id for state in prepared])
    action_scale = fit_episode_equal_action_scale(original_actions, episode_ids)
    coverage_states, probe_norm_states = _reference_states(prepared)
    bundle = FeatureReferenceBundle(
        action_scale,
        coverage_states,
        probe_norm_states,
        bound_probe_sha256,
        probe.candidate,
        pairs[0].task_identity,
        pairs[0].cohort_identity,
        tuple(pair.source_hashes for pair in pairs),
    )
    cohort = _build_cohort(
        prepared,
        split="calibration",
        probe=probe,
        probe_sha256=bound_probe_sha256,
        action_scale=bundle.action_scale,
        coverage_references=bundle.coverage_states,
        probe_norm_references=bundle.probe_norm_states,
        reference_sha256=bundle.metadata_sha256,
    )
    return bundle, cohort


def build_locked_test_features(
    raw_artifacts: Sequence[RolloutArtifact],
    score_sidecars: Sequence[LoadedScoringSidecar],
    bound_probe: object,
    score_receipt: ScoreAllocationReceipt,
    reference_bundle: FeatureReferenceBundle,
    *,
    protocol: ProtocolConfig,
    repo_root: str | Path,
) -> FeatureCohort:
    """Build Test features from one exact score allocation and frozen references."""

    if not isinstance(reference_bundle, FeatureReferenceBundle):
        raise FeaturePipelineError("reference bundle has the wrong type")
    receipt, probe, bound_probe_sha256 = _validated_feature_allocation(
        score_receipt,
        bound_probe,
        expected_split="locked_test",
        protocol=protocol,
        repo_root=repo_root,
    )
    pairs = _validate_and_pair(
        raw_artifacts,
        score_sidecars,
        probe,
        expected_split="locked_test",
        probe_sha256=bound_probe_sha256,
        allocation_receipt=receipt,
    )
    if bound_probe_sha256 != reference_bundle.probe_sha256:
        raise FeaturePipelineError("Locked Test probe differs from reference bundle")
    if probe.candidate != reference_bundle.selected_candidate:
        raise FeaturePipelineError(
            "Locked Test selected candidate differs from reference bundle"
        )
    if pairs[0].task_identity != reference_bundle.task_identity:
        raise FeaturePipelineError("Locked Test task differs from reference bundle")
    if not pairs[0].cohort_identity.is_cross_split_compatible(
        reference_bundle.cohort_identity
    ):
        raise FeaturePipelineError(
            "Locked Test policy/config/code cohort differs from reference bundle"
        )
    prepared = _prepare_states(pairs, probe)
    return _build_cohort(
        prepared,
        split="locked_test",
        probe=probe,
        probe_sha256=bound_probe_sha256,
        action_scale=reference_bundle.action_scale,
        coverage_references=reference_bundle.coverage_states,
        probe_norm_references=reference_bundle.probe_norm_states,
        reference_sha256=reference_bundle.metadata_sha256,
    )


__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "CohortIdentity",
    "EpisodeSourceHashes",
    "FeatureCohort",
    "FeaturePipelineError",
    "FeatureReferenceBundle",
    "FeatureStateRecord",
    "TaskIdentity",
    "build_calibration_features",
    "build_locked_test_features",
]
