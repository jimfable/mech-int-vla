"""Symmetry-aware circular ridge probes for the frozen Calibration protocol.

The implementation is deliberately NumPy-only so probe fitting can run without
the policy stack.  Rows may contain multiple states from an episode; fitting and
evaluation give every episode total weight one regardless of its row count.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import CalibrationSelectionConfig

DEFAULT_CANDIDATE_PREFERENCE = (
    "vlm_context",
    "early_expert_t1_0",
    "early_expert_t0_5",
    "late_expert_t1_0",
    "late_expert_t0_5",
)
FROZEN_RIDGE_ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
ARTIFACT_SCHEMA_VERSION = 1


class ProbeError(ValueError):
    """Raised when probe data or a frozen selection rule is invalid."""


@dataclass(frozen=True)
class ProbeSamples:
    """Targets and grouping metadata shared by aligned feature matrices."""

    theta_rel: NDArray[np.float64]
    base_init_state_id: NDArray[np.int64]
    episode_id: NDArray[np.str_]
    symmetry_order: int

    @classmethod
    def from_arrays(
        cls,
        *,
        theta_rel: ArrayLike,
        base_init_state_id: ArrayLike,
        episode_id: ArrayLike,
        symmetry_order: int,
    ) -> ProbeSamples:
        theta = _readonly_vector(theta_rel, np.float64, "theta_rel")
        groups = _readonly_vector(base_init_state_id, np.int64, "base_init_state_id")
        episodes = _readonly_vector(episode_id, np.str_, "episode_id")
        if not (theta.size == groups.size == episodes.size):
            raise ProbeError("target, group, and episode arrays must have equal length")
        if theta.size == 0:
            raise ProbeError("probe data must contain at least one row")
        if not np.isfinite(theta).all():
            raise ProbeError("theta_rel must contain only finite values")
        if (
            isinstance(symmetry_order, bool)
            or not isinstance(symmetry_order, Integral)
            or symmetry_order < 1
        ):
            raise ProbeError("symmetry_order must be a positive integer")

        for episode in np.unique(episodes):
            episode_groups = np.unique(groups[episodes == episode])
            if episode_groups.size != 1:
                raise ProbeError(
                    f"episode {episode!r} maps to more than one base init ID"
                )
        return cls(theta, groups, episodes, int(symmetry_order))

    @property
    def n_rows(self) -> int:
        return int(self.theta_rel.size)

    @property
    def n_episodes(self) -> int:
        return int(np.unique(self.episode_id).size)


@dataclass(frozen=True)
class GroupFold:
    """One deterministic held-out base-init fold."""

    index: int
    train_rows: NDArray[np.int64]
    test_rows: NDArray[np.int64]
    test_groups: tuple[int, ...]


@dataclass(frozen=True)
class CenteredCircularRidge:
    """A weighted ridge mapping to an unnormalized circular two-vector."""

    alpha: float
    symmetry_order: int
    feature_center: NDArray[np.float64]
    target_center: NDArray[np.float64]
    coefficient: NDArray[np.float64]

    def __post_init__(self) -> None:
        alpha = _validate_alpha(self.alpha)
        symmetry_order = _positive_integer(self.symmetry_order, "symmetry_order")
        feature_center = _readonly_float_array(
            self.feature_center, "feature_center", ndim=1
        )
        target_center = _readonly_float_array(
            self.target_center, "target_center", shape=(2,)
        )
        coefficient = _readonly_float_array(self.coefficient, "coefficient", ndim=2)
        if feature_center.size == 0:
            raise ProbeError("feature_center must not be empty")
        if coefficient.shape != (2, feature_center.size):
            raise ProbeError("coefficient must have shape (2, feature_center width)")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "symmetry_order", symmetry_order)
        object.__setattr__(self, "feature_center", feature_center)
        object.__setattr__(self, "target_center", target_center)
        object.__setattr__(self, "coefficient", coefficient)

    def predict_raw(self, features: ArrayLike) -> NDArray[np.float64]:
        matrix = _feature_matrix(features, expected_dim=self.feature_center.size)
        return (matrix - self.feature_center) @ self.coefficient.T + self.target_center

    def predict_unit(self, features: ArrayLike) -> NDArray[np.float64]:
        """Normalize predicted vectors, and only predicted vectors, to unit length.

        An exactly zero raw vector stays zero.  Its angle is deterministically
        interpreted as zero by :func:`numpy.arctan2`.
        """

        raw = self.predict_raw(features)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0.0)

    def predict_angle(self, features: ArrayLike) -> NDArray[np.float64]:
        unit = self.predict_unit(features)
        return np.arctan2(unit[:, 1], unit[:, 0]) / self.symmetry_order


@dataclass(frozen=True)
class AlphaCVResult:
    alpha: float
    fold_mae_rad: tuple[float, ...]
    mean_mae_rad: float
    standard_error_rad: float

    def __post_init__(self) -> None:
        alpha = _validate_alpha(self.alpha)
        fold_scores = _finite_nonnegative_tuple(
            self.fold_mae_rad, "fold_mae_rad", minimum_length=2
        )
        mean = _finite_nonnegative(self.mean_mae_rad, "mean_mae_rad")
        standard_error = _finite_nonnegative(
            self.standard_error_rad, "standard_error_rad"
        )
        expected_mean, expected_standard_error = _mean_and_standard_error(fold_scores)
        if not _floats_match(mean, expected_mean):
            raise ProbeError("mean_mae_rad is inconsistent with fold_mae_rad")
        if not _floats_match(standard_error, expected_standard_error):
            raise ProbeError("standard_error_rad is inconsistent with fold_mae_rad")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "fold_mae_rad", fold_scores)
        object.__setattr__(self, "mean_mae_rad", mean)
        object.__setattr__(self, "standard_error_rad", standard_error)


@dataclass(frozen=True)
class CandidateCVResult:
    candidate: str
    alpha_results: tuple[AlphaCVResult, ...]
    selected_alpha: float
    mean_mae_rad: float
    standard_error_rad: float

    def __post_init__(self) -> None:
        candidate = _nonempty_string(self.candidate, "candidate")
        alpha_results = tuple(self.alpha_results)
        if not alpha_results or not all(
            isinstance(result, AlphaCVResult) for result in alpha_results
        ):
            raise ProbeError("alpha_results must contain AlphaCVResult values")
        alphas = tuple(result.alpha for result in alpha_results)
        if len(alphas) != len(set(alphas)):
            raise ProbeError("alpha_results must contain unique alphas")
        selected_alpha = _validate_alpha(self.selected_alpha)
        selected = [
            result for result in alpha_results if result.alpha == selected_alpha
        ]
        if len(selected) != 1:
            raise ProbeError("selected_alpha must identify exactly one alpha result")
        empirical_best = min(
            enumerate(alpha_results),
            key=lambda item: (item[1].mean_mae_rad, item[0]),
        )[1]
        if selected[0] is not empirical_best:
            raise ProbeError("selected_alpha is not the first empirical-best alpha")
        mean = _finite_nonnegative(self.mean_mae_rad, "mean_mae_rad")
        standard_error = _finite_nonnegative(
            self.standard_error_rad, "standard_error_rad"
        )
        if not _floats_match(mean, selected[0].mean_mae_rad):
            raise ProbeError("candidate mean does not match selected alpha result")
        if not _floats_match(standard_error, selected[0].standard_error_rad):
            raise ProbeError(
                "candidate standard error does not match selected alpha result"
            )
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "alpha_results", alpha_results)
        object.__setattr__(self, "selected_alpha", selected_alpha)
        object.__setattr__(self, "mean_mae_rad", mean)
        object.__setattr__(self, "standard_error_rad", standard_error)


@dataclass(frozen=True)
class MeanPredictionCVResult:
    fold_mae_rad: tuple[float, ...]
    mean_mae_rad: float
    standard_error_rad: float

    def __post_init__(self) -> None:
        fold_scores = _finite_nonnegative_tuple(
            self.fold_mae_rad, "fold_mae_rad", minimum_length=2
        )
        mean = _finite_nonnegative(self.mean_mae_rad, "mean_mae_rad")
        standard_error = _finite_nonnegative(
            self.standard_error_rad, "standard_error_rad"
        )
        expected_mean, expected_standard_error = _mean_and_standard_error(fold_scores)
        if not _floats_match(mean, expected_mean):
            raise ProbeError("mean_mae_rad is inconsistent with fold_mae_rad")
        if not _floats_match(standard_error, expected_standard_error):
            raise ProbeError("standard_error_rad is inconsistent with fold_mae_rad")
        object.__setattr__(self, "fold_mae_rad", fold_scores)
        object.__setattr__(self, "mean_mae_rad", mean)
        object.__setattr__(self, "standard_error_rad", standard_error)


@dataclass(frozen=True)
class ProbeArtifact:
    """Final fitted probe plus canonical, hash-ready provenance metadata."""

    model: CenteredCircularRidge
    candidate: str
    alpha_grid: tuple[float, ...]
    candidate_preference: tuple[str, ...]
    candidate_results: tuple[CandidateCVResult, ...]
    one_standard_error_threshold_rad: float
    fold_test_groups: tuple[tuple[int, ...], ...]
    training_rows: int
    training_episodes: int
    training_base_init_state_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, CenteredCircularRidge):
            raise ProbeError("model must be a CenteredCircularRidge")
        candidate = _nonempty_string(self.candidate, "candidate")
        alpha_grid = tuple(_validate_alpha(alpha) for alpha in self.alpha_grid)
        if alpha_grid != FROZEN_RIDGE_ALPHA_GRID:
            raise ProbeError("alpha_grid differs from the frozen ridge grid")
        candidate_preference = tuple(self.candidate_preference)
        if candidate_preference != DEFAULT_CANDIDATE_PREFERENCE:
            raise ProbeError("candidate_preference differs from the frozen order")
        if len(candidate_preference) != len(set(candidate_preference)):
            raise ProbeError("candidate_preference must contain unique names")
        if candidate not in candidate_preference:
            raise ProbeError("selected candidate is not in candidate_preference")

        candidate_results = tuple(self.candidate_results)
        if not all(
            isinstance(result, CandidateCVResult) for result in candidate_results
        ):
            raise ProbeError("candidate_results must contain CandidateCVResult values")
        if tuple(result.candidate for result in candidate_results) != (
            candidate_preference
        ):
            raise ProbeError(
                "candidate_results must match the frozen candidate order exactly"
            )
        for result in candidate_results:
            if tuple(item.alpha for item in result.alpha_results) != alpha_grid:
                raise ProbeError(
                    "every candidate result must cover the frozen alpha grid in order"
                )
            if any(len(item.fold_mae_rad) != 5 for item in result.alpha_results):
                raise ProbeError(
                    "every artifact alpha result must contain exactly five folds"
                )

        selected_result = candidate_results[candidate_preference.index(candidate)]
        if not _floats_match(self.model.alpha, selected_result.selected_alpha):
            raise ProbeError("model alpha does not match the selected candidate result")
        best = min(
            enumerate(candidate_results),
            key=lambda item: (item[1].mean_mae_rad, item[0]),
        )[1]
        expected_threshold = best.mean_mae_rad + best.standard_error_rad
        threshold = _finite_nonnegative(
            self.one_standard_error_threshold_rad,
            "one_standard_error_threshold_rad",
        )
        if not _floats_match(threshold, expected_threshold):
            raise ProbeError("one-standard-error threshold is inconsistent")
        eligible = tuple(
            name
            for name, result in zip(
                candidate_preference, candidate_results, strict=True
            )
            if result.mean_mae_rad <= threshold
        )
        if not eligible or candidate != eligible[0]:
            raise ProbeError("candidate is inconsistent with the frozen selection rule")

        training_rows = _positive_integer(self.training_rows, "training_rows")
        training_episodes = _positive_integer(
            self.training_episodes, "training_episodes"
        )
        training_groups = _integer_tuple(
            self.training_base_init_state_ids,
            "training_base_init_state_ids",
            nonempty=True,
        )
        if training_groups != tuple(sorted(set(training_groups))):
            raise ProbeError("training_base_init_state_ids must be sorted and unique")
        if not (training_rows >= training_episodes >= len(training_groups)):
            raise ProbeError(
                "training counts must satisfy rows >= episodes >= base init groups"
            )

        fold_test_groups = tuple(
            _integer_tuple(groups, "fold_test_groups", nonempty=True)
            for groups in self.fold_test_groups
        )
        if len(fold_test_groups) != 5:
            raise ProbeError("fold_test_groups must contain exactly five folds")
        expected_folds = tuple(training_groups[index::5] for index in range(5))
        if fold_test_groups != expected_folds:
            raise ProbeError(
                "fold_test_groups are inconsistent with deterministic grouped CV"
            )

        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "alpha_grid", alpha_grid)
        object.__setattr__(self, "candidate_preference", candidate_preference)
        object.__setattr__(self, "candidate_results", candidate_results)
        object.__setattr__(self, "one_standard_error_threshold_rad", threshold)
        object.__setattr__(self, "fold_test_groups", fold_test_groups)
        object.__setattr__(self, "training_rows", training_rows)
        object.__setattr__(self, "training_episodes", training_episodes)
        object.__setattr__(self, "training_base_init_state_ids", training_groups)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "estimator": "episode_weighted_centered_ridge",
            "target": "[cos(s*theta_rel),sin(s*theta_rel)]",
            "prediction_vector_normalization": "prediction_time_only",
            "metric": "episode_equal_symmetry_aware_circular_mae_rad",
            "cv": {
                "kind": "group_5_fold",
                "group": "base_init_state_id",
                "fold_test_groups": [list(groups) for groups in self.fold_test_groups],
            },
            "selection": {
                "candidate": self.candidate,
                "ridge_alpha": self.model.alpha,
                "alpha_grid": list(self.alpha_grid),
                "candidate_preference": list(self.candidate_preference),
                "rule": "lowest_mean_then_candidate_one_standard_error",
                "one_standard_error_threshold_rad": (
                    self.one_standard_error_threshold_rad
                ),
            },
            "training": {
                "rows": self.training_rows,
                "episodes": self.training_episodes,
                "base_init_state_ids": list(self.training_base_init_state_ids),
                "symmetry_order": self.model.symmetry_order,
                "feature_dim": int(self.model.feature_center.size),
            },
            "parameters": {
                "feature_center": self.model.feature_center.tolist(),
                "target_center": self.model.target_center.tolist(),
                "coefficient": self.model.coefficient.tolist(),
            },
            "candidate_results": [
                _candidate_result_metadata(result) for result in self.candidate_results
            ],
        }

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.to_metadata(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class ProbeSelection:
    artifact: ProbeArtifact
    eligible_candidates: tuple[str, ...]


def write_probe_artifact(artifact: ProbeArtifact, output_root: str | Path) -> Path:
    """Publish one canonical probe in a content-addressed directory.

    Publication is fail-closed: an existing digest directory is never reused or
    replaced, and a staging directory is renamed only after its file and
    directory entries have reached stable storage.
    """

    if not isinstance(artifact, ProbeArtifact):
        raise ProbeError("artifact must be a validated ProbeArtifact")
    root = _normalized_absolute_path(output_root)
    if any(part in {"locks", "configs"} for part in root.parts):
        raise ProbeError("probe artifacts may not be written into lock/config paths")
    if _has_symlink_component(root):
        raise ProbeError("output path may not contain a symlink component")
    if root.exists() and not root.is_dir():
        raise ProbeError("output_root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(root):
        raise ProbeError("output path may not contain a symlink component")

    canonical = artifact.canonical_json()
    digest = hashlib.sha256(canonical).hexdigest()
    destination = root / digest
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite probe artifact {destination}")

    lock_path = root / f".{digest}.publish.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"another writer is publishing probe artifact {destination}"
        ) from exc
    os.close(lock_descriptor)

    staging: Path | None = None
    try:
        _fsync_directory(root)
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        probe_path = staging / "probe.json"
        with probe_path.open("xb") as stream:
            stream.write(canonical)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        if _lexists(destination):
            raise FileExistsError(f"refusing to overwrite probe artifact {destination}")
        os.rename(staging, destination)
        staging = None
        _fsync_directory(root)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
        _fsync_directory(root)
    return destination


def load_probe_artifact(
    path: str | Path, expected_sha256: str | None = None
) -> ProbeArtifact:
    """Load and fully validate a canonical probe artifact without pickle."""

    if expected_sha256 is not None and not _is_lower_sha256(expected_sha256):
        raise ProbeError("expected_sha256 must be 64 lowercase hexadecimal digits")
    directory = _normalized_absolute_path(path)
    if _has_symlink_component(directory):
        raise ProbeError("probe artifact path may not contain a symlink component")
    if not directory.is_dir():
        raise ProbeError("probe artifact path must be a directory")
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise ProbeError(f"could not inspect probe artifact: {exc}") from exc
    if len(entries) != 1 or entries[0].name != "probe.json":
        raise ProbeError("probe artifact must contain exactly one probe.json")
    entry = entries[0]
    try:
        entry_stat = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProbeError(f"could not inspect probe.json: {exc}") from exc
    if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
        raise ProbeError("probe.json must be a regular file, not a symlink")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(entry.path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ProbeError("probe.json must be a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                canonical = stream.read()
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProbeError(f"could not read probe.json safely: {exc}") from exc

    try:
        metadata = json.loads(
            canonical.decode("utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"probe.json is not strict UTF-8 JSON: {exc}") from exc
    artifact = _probe_artifact_from_metadata(metadata)
    reconstructed = artifact.canonical_json()
    if reconstructed != canonical:
        raise ProbeError("probe.json bytes are not the exact canonical encoding")
    digest = hashlib.sha256(reconstructed).hexdigest()
    if directory.name != digest:
        raise ProbeError("probe artifact directory does not match its SHA-256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProbeError("probe artifact SHA-256 does not match expected_sha256")
    return artifact


def probe_artifact_from_metadata(value: Any) -> ProbeArtifact:
    """Strictly reconstruct a numerical probe from canonical metadata.

    Bound provenance containers embed probe metadata rather than a filesystem
    path.  This supported parser keeps their loader off the private implementation
    helper while retaining exact reconstruction checks.
    """

    artifact = _probe_artifact_from_metadata(value)
    if artifact.to_metadata() != value:
        raise ProbeError("probe metadata is not exactly reconstructable")
    return artifact


def episode_equal_weights(episode_id: ArrayLike) -> NDArray[np.float64]:
    """Return row weights such that every episode has total weight one."""

    episodes = _readonly_vector(episode_id, np.str_, "episode_id")
    if episodes.size == 0:
        raise ProbeError("episode_id must not be empty")
    _, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def make_group_folds(
    base_init_state_id: ArrayLike, *, n_splits: int = 5
) -> tuple[GroupFold, ...]:
    """Assign sorted base-init IDs round-robin to deterministic group folds."""

    groups = _readonly_vector(base_init_state_id, np.int64, "base_init_state_id")
    if n_splits < 2:
        raise ProbeError("n_splits must be at least two")
    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ProbeError(
            f"group {n_splits}-fold CV needs at least {n_splits} base init IDs"
        )

    folds: list[GroupFold] = []
    all_rows = np.arange(groups.size, dtype=np.int64)
    for fold_index in range(n_splits):
        test_groups_array = unique_groups[fold_index::n_splits]
        test_mask = np.isin(groups, test_groups_array)
        train_rows = _readonly(all_rows[~test_mask])
        test_rows = _readonly(all_rows[test_mask])
        folds.append(
            GroupFold(
                index=fold_index,
                train_rows=train_rows,
                test_rows=test_rows,
                test_groups=tuple(int(value) for value in test_groups_array),
            )
        )
    return tuple(folds)


def circular_targets(
    theta_rel: ArrayLike, *, symmetry_order: int
) -> NDArray[np.float64]:
    theta = np.asarray(theta_rel, dtype=np.float64)
    if theta.ndim != 1 or not np.isfinite(theta).all():
        raise ProbeError("theta_rel must be a finite one-dimensional array")
    if (
        isinstance(symmetry_order, bool)
        or not isinstance(symmetry_order, Integral)
        or symmetry_order < 1
    ):
        raise ProbeError("symmetry_order must be a positive integer")
    scaled = int(symmetry_order) * theta
    return np.column_stack((np.cos(scaled), np.sin(scaled)))


def symmetry_aware_circular_error(
    predicted_angle: ArrayLike,
    true_angle: ArrayLike,
    *,
    symmetry_order: int,
) -> NDArray[np.float64]:
    """Return absolute angular error modulo ``2*pi/s`` in radians."""

    predicted = np.asarray(predicted_angle, dtype=np.float64)
    truth = np.asarray(true_angle, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ProbeError("predicted and true angles must be equal-length vectors")
    if not (np.isfinite(predicted).all() and np.isfinite(truth).all()):
        raise ProbeError("angles must contain only finite values")
    if (
        isinstance(symmetry_order, bool)
        or not isinstance(symmetry_order, Integral)
        or symmetry_order < 1
    ):
        raise ProbeError("symmetry_order must be a positive integer")
    scaled_delta = int(symmetry_order) * (predicted - truth)
    wrapped = np.arctan2(np.sin(scaled_delta), np.cos(scaled_delta))
    return np.abs(wrapped) / int(symmetry_order)


def fit_centered_ridge(
    features: ArrayLike,
    samples: ProbeSamples,
    *,
    alpha: float,
) -> CenteredCircularRidge:
    """Fit one episode-weighted centered ridge probe."""

    matrix = _feature_matrix(features, expected_rows=samples.n_rows)
    path = _RidgePath(matrix, samples)
    return path.fit(alpha)


def cross_validate_circular_probe(
    candidate: str,
    features: ArrayLike,
    samples: ProbeSamples,
    *,
    alpha_grid: Sequence[float],
    folds: Sequence[GroupFold] | None = None,
) -> CandidateCVResult:
    """Evaluate every frozen alpha with shared grouped folds."""

    if not candidate:
        raise ProbeError("candidate name must not be empty")
    matrix = _feature_matrix(features, expected_rows=samples.n_rows)
    alphas = _validate_alpha_grid(alpha_grid)
    selected_folds = tuple(folds or make_group_folds(samples.base_init_state_id))
    _validate_folds(selected_folds, samples.n_rows)

    fold_scores: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for fold in selected_folds:
        train_samples = _subset_samples(samples, fold.train_rows)
        path = _RidgePath(matrix[fold.train_rows], train_samples)
        heldout = matrix[fold.test_rows]
        heldout_theta = samples.theta_rel[fold.test_rows]
        heldout_episodes = samples.episode_id[fold.test_rows]
        for alpha in alphas:
            model = path.fit(alpha)
            errors = symmetry_aware_circular_error(
                model.predict_angle(heldout),
                heldout_theta,
                symmetry_order=samples.symmetry_order,
            )
            fold_scores[alpha].append(_episode_equal_mean(errors, heldout_episodes))

    alpha_results = tuple(_alpha_result(alpha, fold_scores[alpha]) for alpha in alphas)
    selected = min(
        enumerate(alpha_results), key=lambda item: (item[1].mean_mae_rad, item[0])
    )[1]
    return CandidateCVResult(
        candidate=candidate,
        alpha_results=alpha_results,
        selected_alpha=selected.alpha,
        mean_mae_rad=selected.mean_mae_rad,
        standard_error_rad=selected.standard_error_rad,
    )


def select_candidate_one_standard_error(
    candidate_results: Sequence[CandidateCVResult],
    *,
    candidate_preference: Sequence[str] = DEFAULT_CANDIDATE_PREFERENCE,
) -> tuple[CandidateCVResult, float, tuple[str, ...]]:
    """Select the first preferred candidate within one SE of the empirical best."""

    results = tuple(candidate_results)
    preference = tuple(candidate_preference)
    if not results:
        raise ProbeError("candidate_results must not be empty")
    if len(preference) != len(set(preference)):
        raise ProbeError("candidate preference order must not contain duplicates")
    by_name = {result.candidate: result for result in results}
    if len(by_name) != len(results) or set(by_name) != set(preference):
        raise ProbeError(
            "candidate results must contain each preferred candidate exactly once"
        )
    best = min(enumerate(results), key=lambda item: (item[1].mean_mae_rad, item[0]))[1]
    threshold = best.mean_mae_rad + best.standard_error_rad
    eligible = tuple(
        name for name in preference if by_name[name].mean_mae_rad <= threshold
    )
    return by_name[eligible[0]], float(threshold), eligible


def select_and_fit_circular_probe(
    candidate_features: Mapping[str, ArrayLike],
    samples: ProbeSamples,
    *,
    selection_config: CalibrationSelectionConfig,
) -> ProbeSelection:
    """Run the exact frozen five-candidate pipeline and fit the selected probe."""

    preference = tuple(selection_config.representation_candidates)
    if preference != DEFAULT_CANDIDATE_PREFERENCE:
        raise ProbeError("representation candidate order differs from frozen protocol")
    if set(candidate_features) != set(preference):
        raise ProbeError("feature matrices must match the five frozen candidates")
    alphas = _validate_alpha_grid(selection_config.ridge_alpha_candidates)
    folds = make_group_folds(samples.base_init_state_id, n_splits=5)
    matrices = {
        candidate: _feature_matrix(
            candidate_features[candidate], expected_rows=samples.n_rows
        )
        for candidate in preference
    }
    results = tuple(
        cross_validate_circular_probe(
            candidate,
            matrices[candidate],
            samples,
            alpha_grid=alphas,
            folds=folds,
        )
        for candidate in preference
    )
    selected, threshold, eligible = select_candidate_one_standard_error(
        results, candidate_preference=preference
    )
    model = fit_centered_ridge(
        matrices[selected.candidate], samples, alpha=selected.selected_alpha
    )
    artifact = ProbeArtifact(
        model=model,
        candidate=selected.candidate,
        alpha_grid=alphas,
        candidate_preference=preference,
        candidate_results=results,
        one_standard_error_threshold_rad=threshold,
        fold_test_groups=tuple(fold.test_groups for fold in folds),
        training_rows=samples.n_rows,
        training_episodes=samples.n_episodes,
        training_base_init_state_ids=tuple(
            int(value) for value in np.unique(samples.base_init_state_id)
        ),
    )
    return ProbeSelection(artifact=artifact, eligible_candidates=eligible)


def evaluate_feature_baseline(
    name: str,
    features: ArrayLike,
    samples: ProbeSamples,
    *,
    selection_config: CalibrationSelectionConfig,
) -> CandidateCVResult:
    """Evaluate any prespecified baseline features with the frozen alpha grid."""

    return cross_validate_circular_probe(
        name,
        features,
        samples,
        alpha_grid=selection_config.ridge_alpha_candidates,
    )


def evaluate_time_only_baseline(
    time: ArrayLike,
    samples: ProbeSamples,
    *,
    selection_config: CalibrationSelectionConfig,
) -> CandidateCVResult:
    values = np.asarray(time, dtype=np.float64)
    if values.ndim != 1:
        raise ProbeError("time-only baseline expects one scalar per row")
    return evaluate_feature_baseline(
        "time_only", values[:, None], samples, selection_config=selection_config
    )


def evaluate_proprioception_only_baseline(
    proprioception: ArrayLike,
    samples: ProbeSamples,
    *,
    selection_config: CalibrationSelectionConfig,
) -> CandidateCVResult:
    return evaluate_feature_baseline(
        "proprioception_only",
        proprioception,
        samples,
        selection_config=selection_config,
    )


def evaluate_random_label_baseline(
    features: ArrayLike,
    samples: ProbeSamples,
    *,
    randomized_theta_rel: ArrayLike,
    selection_config: CalibrationSelectionConfig,
) -> CandidateCVResult:
    """Evaluate caller-prespecified randomized labels without choosing a shuffle.

    The randomization design remains explicit in the calling analysis rather than
    silently introducing an outcome-dependent or row-vs-episode shuffle choice.
    """

    randomized = ProbeSamples.from_arrays(
        theta_rel=randomized_theta_rel,
        base_init_state_id=samples.base_init_state_id,
        episode_id=samples.episode_id,
        symmetry_order=samples.symmetry_order,
    )
    return evaluate_feature_baseline(
        "random_label", features, randomized, selection_config=selection_config
    )


def evaluate_mean_prediction_baseline(
    samples: ProbeSamples,
    *,
    folds: Sequence[GroupFold] | None = None,
) -> MeanPredictionCVResult:
    """Evaluate the episode-weighted circular mean learned in each train fold."""

    selected_folds = tuple(folds or make_group_folds(samples.base_init_state_id))
    _validate_folds(selected_folds, samples.n_rows)
    targets = circular_targets(samples.theta_rel, symmetry_order=samples.symmetry_order)
    fold_scores: list[float] = []
    for fold in selected_folds:
        train_weights = episode_equal_weights(samples.episode_id[fold.train_rows])
        mean_vector = np.average(
            targets[fold.train_rows], axis=0, weights=train_weights
        )
        mean_angle = math.atan2(mean_vector[1], mean_vector[0]) / samples.symmetry_order
        errors = symmetry_aware_circular_error(
            np.full(fold.test_rows.size, mean_angle),
            samples.theta_rel[fold.test_rows],
            symmetry_order=samples.symmetry_order,
        )
        fold_scores.append(
            _episode_equal_mean(errors, samples.episode_id[fold.test_rows])
        )
    mean, standard_error = _mean_and_standard_error(fold_scores)
    return MeanPredictionCVResult(tuple(fold_scores), mean, standard_error)


class _RidgePath:
    """One SVD per train fold, then cheap exact fits for every ridge alpha."""

    def __init__(self, features: NDArray[np.float64], samples: ProbeSamples) -> None:
        weights = episode_equal_weights(samples.episode_id)
        targets = circular_targets(
            samples.theta_rel, symmetry_order=samples.symmetry_order
        )
        weight_sum = float(weights.sum())
        self.feature_center = np.sum(features * weights[:, None], axis=0) / weight_sum
        self.target_center = np.sum(targets * weights[:, None], axis=0) / weight_sum
        root_weights = np.sqrt(weights)[:, None]
        centered_features = (features - self.feature_center) * root_weights
        centered_targets = (targets - self.target_center) * root_weights
        self.u, self.singular_values, self.vt = np.linalg.svd(
            centered_features, full_matrices=False
        )
        self.projected_targets = self.u.T @ centered_targets
        self.symmetry_order = samples.symmetry_order

    def fit(self, alpha: float) -> CenteredCircularRidge:
        alpha_value = _validate_alpha(alpha)
        factors = self.singular_values / (self.singular_values**2 + alpha_value)
        coefficient = (self.vt.T @ (factors[:, None] * self.projected_targets)).T
        return CenteredCircularRidge(
            alpha=alpha_value,
            symmetry_order=self.symmetry_order,
            feature_center=_readonly(self.feature_center.copy()),
            target_center=_readonly(self.target_center.copy()),
            coefficient=_readonly(coefficient),
        )


def _feature_matrix(
    values: ArrayLike,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ProbeError("features must be a nonempty rank-2 matrix")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ProbeError(
            f"features have {matrix.shape[0]} rows; expected {expected_rows}"
        )
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ProbeError(
            f"features have dimension {matrix.shape[1]}; expected {expected_dim}"
        )
    if not np.isfinite(matrix).all():
        raise ProbeError("features must contain only finite values")
    return matrix


def _readonly_vector(values: ArrayLike, dtype: Any, name: str) -> NDArray[Any]:
    vector = np.array(values, dtype=dtype, copy=True)
    if vector.ndim != 1:
        raise ProbeError(f"{name} must be a one-dimensional array")
    return _readonly(vector)


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool):
        raise ProbeError("ridge alpha must be a finite positive number")
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0:
        raise ProbeError("ridge alpha must be a finite positive number")
    return value


def _validate_alpha_grid(alpha_grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(_validate_alpha(alpha) for alpha in alpha_grid)
    if not values or len(values) != len(set(values)):
        raise ProbeError("ridge alpha grid must be nonempty and unique")
    return values


def _subset_samples(samples: ProbeSamples, rows: NDArray[np.int64]) -> ProbeSamples:
    return ProbeSamples.from_arrays(
        theta_rel=samples.theta_rel[rows],
        base_init_state_id=samples.base_init_state_id[rows],
        episode_id=samples.episode_id[rows],
        symmetry_order=samples.symmetry_order,
    )


def _validate_folds(folds: Sequence[GroupFold], n_rows: int) -> None:
    if len(folds) < 2:
        raise ProbeError("cross-validation needs at least two folds")
    heldout = np.zeros(n_rows, dtype=np.int64)
    for fold in folds:
        if fold.train_rows.size == 0 or fold.test_rows.size == 0:
            raise ProbeError("every fold needs nonempty train and test rows")
        if np.intersect1d(fold.train_rows, fold.test_rows).size:
            raise ProbeError("fold train and test rows overlap")
        if np.any(fold.test_rows < 0) or np.any(fold.test_rows >= n_rows):
            raise ProbeError("fold test row is out of bounds")
        heldout[fold.test_rows] += 1
    if not np.all(heldout == 1):
        raise ProbeError("each row must be held out exactly once")


def _episode_equal_mean(errors: ArrayLike, episode_id: ArrayLike) -> float:
    values = np.asarray(errors, dtype=np.float64)
    weights = episode_equal_weights(episode_id)
    if values.ndim != 1 or values.size != weights.size:
        raise ProbeError("errors and episode IDs must be equal-length vectors")
    return float(np.sum(values * weights) / np.sum(weights))


def _mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ProbeError("standard error needs at least two finite fold scores")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def _alpha_result(alpha: float, fold_scores: Sequence[float]) -> AlphaCVResult:
    mean, standard_error = _mean_and_standard_error(fold_scores)
    return AlphaCVResult(alpha, tuple(fold_scores), mean, standard_error)


def _candidate_result_metadata(result: CandidateCVResult) -> dict[str, Any]:
    return {
        "candidate": result.candidate,
        "selected_alpha": result.selected_alpha,
        "mean_mae_rad": result.mean_mae_rad,
        "standard_error_rad": result.standard_error_rad,
        "alpha_results": [
            {
                "alpha": alpha.alpha,
                "fold_mae_rad": list(alpha.fold_mae_rad),
                "mean_mae_rad": alpha.mean_mae_rad,
                "standard_error_rad": alpha.standard_error_rad,
            }
            for alpha in result.alpha_results
        ],
    }


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ProbeError(f"{name} must be a positive integer")
    return int(value)


def _integer_tuple(
    values: Any, name: str, *, nonempty: bool = False, minimum: int | None = 0
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ProbeError(f"{name} must be a sequence of integers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ProbeError(f"{name} must be a sequence of integers") from exc
    if nonempty and not raw:
        raise ProbeError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise ProbeError(f"{name} must contain only integers")
    result = tuple(int(value) for value in raw)
    if minimum is not None and any(value < minimum for value in result):
        raise ProbeError(f"{name} must contain integers >= {minimum}")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProbeError(f"{name} must be a nonempty string")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ProbeError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ProbeError(f"{name} must be a finite nonnegative number")
    return result


def _finite_nonnegative_tuple(
    values: Any, name: str, *, minimum_length: int
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ProbeError(f"{name} must be a sequence")
    try:
        result = tuple(_finite_nonnegative(value, name) for value in tuple(values))
    except TypeError as exc:
        raise ProbeError(f"{name} must be a sequence") from exc
    if len(result) < minimum_length:
        raise ProbeError(f"{name} must contain at least {minimum_length} folds")
    return result


def _readonly_float_array(
    values: Any,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProbeError(f"{name} must be a finite numeric array") from exc
    if ndim is not None and array.ndim != ndim:
        raise ProbeError(f"{name} must be rank {ndim}")
    if shape is not None and array.shape != shape:
        raise ProbeError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ProbeError(f"{name} must contain only finite values")
    return _readonly(array)


def _floats_match(left: float, right: float) -> bool:
    return left == right


def _normalized_absolute_path(path: str | Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    except (TypeError, ValueError, OSError) as exc:
        raise ProbeError("artifact path is invalid") from exc


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ProbeError(
                f"could not inspect path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            return True
    return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError(f"probe.json contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ProbeError(f"probe.json contains non-finite JSON constant {value!r}")


def _json_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeError(f"{where} must be a JSON object")
    return value


def _json_exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProbeError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _json_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProbeError(f"{where} must be a JSON array")
    return value


def _json_string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ProbeError(f"{where} must be a string")
    return value


def _json_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProbeError(f"{where} must be a finite number")
    return result


def _json_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProbeError(f"{where} must be an integer")
    return value


def _json_float_list(value: Any, where: str) -> tuple[float, ...]:
    return tuple(
        _json_float(item, f"{where}[{index}]")
        for index, item in enumerate(_json_list(value, where))
    )


def _json_integer_list(value: Any, where: str) -> tuple[int, ...]:
    return tuple(
        _json_integer(item, f"{where}[{index}]")
        for index, item in enumerate(_json_list(value, where))
    )


def _alpha_result_from_metadata(value: Any, where: str) -> AlphaCVResult:
    metadata = _json_mapping(value, where)
    _json_exact_keys(
        metadata,
        {
            "alpha",
            "fold_mae_rad",
            "mean_mae_rad",
            "standard_error_rad",
        },
        where,
    )
    return AlphaCVResult(
        alpha=_json_float(metadata["alpha"], f"{where}.alpha"),
        fold_mae_rad=_json_float_list(
            metadata["fold_mae_rad"], f"{where}.fold_mae_rad"
        ),
        mean_mae_rad=_json_float(metadata["mean_mae_rad"], f"{where}.mean_mae_rad"),
        standard_error_rad=_json_float(
            metadata["standard_error_rad"], f"{where}.standard_error_rad"
        ),
    )


def _candidate_result_from_metadata(value: Any, where: str) -> CandidateCVResult:
    metadata = _json_mapping(value, where)
    _json_exact_keys(
        metadata,
        {
            "candidate",
            "selected_alpha",
            "mean_mae_rad",
            "standard_error_rad",
            "alpha_results",
        },
        where,
    )
    raw_alpha_results = _json_list(metadata["alpha_results"], f"{where}.alpha_results")
    return CandidateCVResult(
        candidate=_json_string(metadata["candidate"], f"{where}.candidate"),
        alpha_results=tuple(
            _alpha_result_from_metadata(item, f"{where}.alpha_results[{index}]")
            for index, item in enumerate(raw_alpha_results)
        ),
        selected_alpha=_json_float(
            metadata["selected_alpha"], f"{where}.selected_alpha"
        ),
        mean_mae_rad=_json_float(metadata["mean_mae_rad"], f"{where}.mean_mae_rad"),
        standard_error_rad=_json_float(
            metadata["standard_error_rad"], f"{where}.standard_error_rad"
        ),
    )


def _probe_artifact_from_metadata(value: Any) -> ProbeArtifact:
    metadata = _json_mapping(value, "probe.json")
    _json_exact_keys(
        metadata,
        {
            "schema_version",
            "estimator",
            "target",
            "prediction_vector_normalization",
            "metric",
            "cv",
            "selection",
            "training",
            "parameters",
            "candidate_results",
        },
        "probe.json",
    )
    constants = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "estimator": "episode_weighted_centered_ridge",
        "target": "[cos(s*theta_rel),sin(s*theta_rel)]",
        "prediction_vector_normalization": "prediction_time_only",
        "metric": "episode_equal_symmetry_aware_circular_mae_rad",
    }
    for key, expected in constants.items():
        if metadata[key] != expected or type(metadata[key]) is not type(expected):
            raise ProbeError(f"probe.json.{key} differs from the artifact schema")

    cv = _json_mapping(metadata["cv"], "probe.json.cv")
    _json_exact_keys(cv, {"kind", "group", "fold_test_groups"}, "probe.json.cv")
    if cv["kind"] != "group_5_fold" or cv["group"] != "base_init_state_id":
        raise ProbeError("probe.json.cv constants differ from the frozen protocol")
    raw_folds = _json_list(cv["fold_test_groups"], "probe.json.cv.fold_test_groups")
    fold_test_groups = tuple(
        _json_integer_list(item, f"probe.json.cv.fold_test_groups[{index}]")
        for index, item in enumerate(raw_folds)
    )

    selection = _json_mapping(metadata["selection"], "probe.json.selection")
    _json_exact_keys(
        selection,
        {
            "candidate",
            "ridge_alpha",
            "alpha_grid",
            "candidate_preference",
            "rule",
            "one_standard_error_threshold_rad",
        },
        "probe.json.selection",
    )
    if selection["rule"] != "lowest_mean_then_candidate_one_standard_error":
        raise ProbeError("probe.json.selection.rule differs from the frozen protocol")
    candidate_preference = tuple(
        _json_string(item, f"probe.json.selection.candidate_preference[{index}]")
        for index, item in enumerate(
            _json_list(
                selection["candidate_preference"],
                "probe.json.selection.candidate_preference",
            )
        )
    )

    training = _json_mapping(metadata["training"], "probe.json.training")
    _json_exact_keys(
        training,
        {
            "rows",
            "episodes",
            "base_init_state_ids",
            "symmetry_order",
            "feature_dim",
        },
        "probe.json.training",
    )
    symmetry_order = _json_integer(
        training["symmetry_order"], "probe.json.training.symmetry_order"
    )
    feature_dim = _json_integer(
        training["feature_dim"], "probe.json.training.feature_dim"
    )

    parameters = _json_mapping(metadata["parameters"], "probe.json.parameters")
    _json_exact_keys(
        parameters,
        {"feature_center", "target_center", "coefficient"},
        "probe.json.parameters",
    )
    raw_coefficient = _json_list(
        parameters["coefficient"], "probe.json.parameters.coefficient"
    )
    model = CenteredCircularRidge(
        alpha=_json_float(selection["ridge_alpha"], "probe.json.selection.ridge_alpha"),
        symmetry_order=symmetry_order,
        feature_center=_json_float_list(
            parameters["feature_center"], "probe.json.parameters.feature_center"
        ),
        target_center=_json_float_list(
            parameters["target_center"], "probe.json.parameters.target_center"
        ),
        coefficient=tuple(
            _json_float_list(row, f"probe.json.parameters.coefficient[{index}]")
            for index, row in enumerate(raw_coefficient)
        ),
    )
    if feature_dim != model.feature_center.size:
        raise ProbeError("probe.json.training.feature_dim is inconsistent")

    raw_candidate_results = _json_list(
        metadata["candidate_results"], "probe.json.candidate_results"
    )
    return ProbeArtifact(
        model=model,
        candidate=_json_string(
            selection["candidate"], "probe.json.selection.candidate"
        ),
        alpha_grid=_json_float_list(
            selection["alpha_grid"], "probe.json.selection.alpha_grid"
        ),
        candidate_preference=candidate_preference,
        candidate_results=tuple(
            _candidate_result_from_metadata(
                item, f"probe.json.candidate_results[{index}]"
            )
            for index, item in enumerate(raw_candidate_results)
        ),
        one_standard_error_threshold_rad=_json_float(
            selection["one_standard_error_threshold_rad"],
            "probe.json.selection.one_standard_error_threshold_rad",
        ),
        fold_test_groups=fold_test_groups,
        training_rows=_json_integer(training["rows"], "probe.json.training.rows"),
        training_episodes=_json_integer(
            training["episodes"], "probe.json.training.episodes"
        ),
        training_base_init_state_ids=_json_integer_list(
            training["base_init_state_ids"],
            "probe.json.training.base_init_state_ids",
        ),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_CANDIDATE_PREFERENCE",
    "FROZEN_RIDGE_ALPHA_GRID",
    "AlphaCVResult",
    "CandidateCVResult",
    "CenteredCircularRidge",
    "GroupFold",
    "MeanPredictionCVResult",
    "ProbeArtifact",
    "ProbeError",
    "ProbeSamples",
    "ProbeSelection",
    "circular_targets",
    "cross_validate_circular_probe",
    "episode_equal_weights",
    "evaluate_feature_baseline",
    "evaluate_mean_prediction_baseline",
    "evaluate_proprioception_only_baseline",
    "evaluate_random_label_baseline",
    "evaluate_time_only_baseline",
    "fit_centered_ridge",
    "load_probe_artifact",
    "make_group_folds",
    "probe_artifact_from_metadata",
    "select_and_fit_circular_probe",
    "select_candidate_one_standard_error",
    "symmetry_aware_circular_error",
    "write_probe_artifact",
]
