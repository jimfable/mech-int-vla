"""Symmetry-aware circular ridge probes for the frozen Calibration protocol.

The implementation is deliberately NumPy-only so probe fitting can run without
the policy stack.  Rows may contain multiple states from an episode; fitting and
evaluation give every episode total weight one regardless of its row count.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
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


@dataclass(frozen=True)
class CandidateCVResult:
    candidate: str
    alpha_results: tuple[AlphaCVResult, ...]
    selected_alpha: float
    mean_mae_rad: float
    standard_error_rad: float


@dataclass(frozen=True)
class MeanPredictionCVResult:
    fold_mae_rad: tuple[float, ...]
    mean_mae_rad: float
    standard_error_rad: float


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


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_CANDIDATE_PREFERENCE",
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
    "make_group_folds",
    "select_and_fit_circular_probe",
    "select_candidate_one_standard_error",
    "symmetry_aware_circular_error",
]
