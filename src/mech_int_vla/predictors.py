"""Preregistered Calibration-stage failure-predictor fitting.

This module deliberately owns only the final M0/M1/M2 predictor fit.  In
particular, callers must construct M1's coverage features out of fold before
calling :func:`fit_failure_predictors`; recomputing those features here would
mix representation construction with predictor selection.

The implementation follows ``PREREG.md`` section 9:

* five folds are grouped by base initialization ID;
* every episode has total sample weight one, irrespective of its state count;
* every original feature gets an explicit missingness indicator;
* the shared family and hyperparameters are selected using raw M1 OOF log loss;
* each of M0/M1/M2 gets its own OOF Platt calibrator; and
* each base predictor is finally refit on all Calibration rows.

Scikit-learn is an optional project dependency so rollout-only processes do not
need to import it.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CalibrationSelectionConfig

MODEL_NAMES: Final[tuple[str, ...]] = ("M0", "M1", "M2")
N_FOLDS: Final[int] = 5
LOG_LOSS_EPSILON: Final[float] = 1e-6
DEFAULT_RANDOM_STATE: Final[int] = 260803
_LOGISTIC_C: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)
_BOOSTING_GRID: Final[dict[str, tuple[float | int, ...]]] = {
    "learning_rate": (0.03, 0.1),
    "max_leaf_nodes": (7, 15),
    "min_samples_leaf": (10, 20),
    "l2_regularization": (0.0, 1.0),
}
_BOOSTING_MAX_ITER: Final[int] = 200


class PredictorFitError(ValueError):
    """Raised when Calibration data cannot support the frozen fit."""


@dataclass(frozen=True)
class FeatureSet:
    """Named state-level features for one preregistered predictor.

    ``values`` is copied and made read-only.  Missing values must be represented
    by NaN; infinities are rejected rather than silently treated as missing.
    """

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise PredictorFitError("feature values must be a two-dimensional array")
        if values.shape[1] == 0:
            raise PredictorFitError("a feature set must contain at least one feature")
        if np.isinf(values).any():
            raise PredictorFitError("feature values may contain NaN but not infinity")
        names = tuple(str(name) for name in self.names)
        if len(names) != values.shape[1]:
            raise PredictorFitError("feature-name count does not match matrix width")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise PredictorFitError("feature names must be nonempty and unique")
        frozen = np.array(values, dtype=np.float64, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "names", names)


class MissingIndicatorImputer(TransformerMixin, BaseEstimator):
    """Weighted-median imputation plus one indicator for every input feature.

    Always allocating all indicators makes the transform schema independent of
    which fold happened to contain a missing value.  The weighted median also
    extends the episode-total-weight-one rule to fitted preprocessing.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> MissingIndicatorImputer:
        del y
        matrix = _feature_matrix(X)
        weights = _validated_weights(sample_weight, matrix.shape[0])
        statistics = np.empty(matrix.shape[1], dtype=np.float64)
        for column_index in range(matrix.shape[1]):
            column = matrix[:, column_index]
            present = ~np.isnan(column)
            if not present.any():
                statistics[column_index] = 0.0
                continue
            statistics[column_index] = _weighted_median(
                column[present], weights[present]
            )
        self.n_features_in_ = matrix.shape[1]
        self.statistics_ = statistics
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "statistics_"):
            raise PredictorFitError("missing-value transformer is not fitted")
        matrix = _feature_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise PredictorFitError(
                "feature width differs from fitted missing-value transformer"
            )
        missing = np.isnan(matrix)
        imputed = np.where(missing, self.statistics_[None, :], matrix)
        return np.concatenate((imputed, missing.astype(np.float64)), axis=1)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise PredictorFitError("missing-value transformer is not fitted")
        if input_features is None:
            names = tuple(f"x{i}" for i in range(self.n_features_in_))
        else:
            names = tuple(str(name) for name in input_features)
            if len(names) != self.n_features_in_:
                raise PredictorFitError("input feature-name count has changed")
        return np.asarray(
            (*names, *(f"{name}__missing" for name in names)), dtype=object
        )


@dataclass(frozen=True)
class CandidateResult:
    """One M1-only family/hyperparameter OOF selection result."""

    family: str
    hyperparameters: tuple[tuple[str, float | int], ...]
    oof_log_loss: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "hyperparameters": dict(self.hyperparameters),
            "m1_raw_oof_log_loss": self.oof_log_loss,
        }


@dataclass(frozen=True)
class FoldAssignment:
    """Serializable assignment of whole base-init groups to one OOF fold."""

    fold_index: int
    validation_groups: tuple[str, ...]
    validation_row_count: int
    validation_episode_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "validation_groups": list(self.validation_groups),
            "validation_row_count": self.validation_row_count,
            "validation_episode_count": self.validation_episode_count,
        }


@dataclass(frozen=True)
class FittedFailurePredictor:
    """A full-Calibration base predictor and its OOF-fitted Platt map."""

    name: str
    feature_names: tuple[str, ...]
    estimator: Pipeline
    platt_slope: float
    platt_intercept: float
    raw_oof_log_loss: float
    calibrated_oof_log_loss: float
    raw_oof_auroc: float
    calibrated_oof_auroc: float
    n_rows: int
    n_episodes: int

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        matrix = _feature_matrix(values)
        if matrix.shape[1] != len(self.feature_names):
            raise PredictorFitError(
                f"{self.name} expected {len(self.feature_names)} features, "
                f"received {matrix.shape[1]}"
            )
        return np.asarray(self.estimator.decision_function(matrix), dtype=np.float64)

    def predict_raw_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = _feature_matrix(values)
        if matrix.shape[1] != len(self.feature_names):
            raise PredictorFitError(
                f"{self.name} expected {len(self.feature_names)} features, "
                f"received {matrix.shape[1]}"
            )
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        score = self.decision_function(values)
        return _sigmoid(self.platt_slope * score + self.platt_intercept)

    def to_metadata(self) -> dict[str, Any]:
        imputer: MissingIndicatorImputer = self.estimator.named_steps["missing"]
        classifier = self.estimator.named_steps["classifier"]
        estimator_metadata: dict[str, Any]
        if isinstance(classifier, LogisticRegression):
            scaler: StandardScaler = self.estimator.named_steps["standardize"]
            estimator_metadata = {
                "class": "LogisticRegression",
                "coefficient": classifier.coef_.tolist(),
                "intercept": classifier.intercept_.tolist(),
                "n_iter": classifier.n_iter_.astype(int).tolist(),
                "standardizer_mean": scaler.mean_.tolist(),
                "standardizer_scale": scaler.scale_.tolist(),
            }
        elif isinstance(classifier, HistGradientBoostingClassifier):
            estimator_metadata = {
                "class": "HistGradientBoostingClassifier",
                "n_iter": int(classifier.n_iter_),
            }
        else:  # pragma: no cover - construction is private and exhaustive
            raise TypeError(f"unsupported fitted classifier {type(classifier)!r}")
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "transformed_feature_names": imputer.get_feature_names_out(
                self.feature_names
            ).tolist(),
            "weighted_median_imputation": imputer.statistics_.tolist(),
            "platt": {
                "input": "group_oof_base_decision_function",
                "slope": self.platt_slope,
                "intercept": self.platt_intercept,
            },
            "oof_metrics": {
                "raw_log_loss": self.raw_oof_log_loss,
                "calibrated_log_loss": self.calibrated_oof_log_loss,
                "raw_auroc": self.raw_oof_auroc,
                "calibrated_auroc": self.calibrated_oof_auroc,
            },
            "fit_counts": {"rows": self.n_rows, "episodes": self.n_episodes},
            "estimator": estimator_metadata,
        }


@dataclass(frozen=True)
class FrozenPredictorBundle:
    """The three frozen predictors sharing one selected base specification."""

    family: str
    hyperparameters: tuple[tuple[str, float | int], ...]
    predictors: tuple[FittedFailurePredictor, ...]
    candidates: tuple[CandidateResult, ...]
    folds: tuple[FoldAssignment, ...]
    random_state: int
    calibration_data_sha256: str
    n_rows: int
    n_episodes: int
    n_groups: int
    kill_switch_triggered: bool

    def model(self, name: str) -> FittedFailurePredictor:
        for predictor in self.predictors:
            if predictor.name == name:
                return predictor
        raise KeyError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")

    def predict_proba(self, name: str, values: np.ndarray) -> np.ndarray:
        return self.model(name).predict_proba(values)

    def to_bytes(self) -> bytes:
        """Serialize the executable frozen artifact.

        Pickle payloads must only be loaded from a trusted research artifact.
        The accompanying SHA-256 in :meth:`to_metadata` is intended for the
        calibration lock and detects accidental or malicious replacement.
        """

        return pickle.dumps(self, protocol=5)

    @classmethod
    def from_bytes(cls, payload: bytes) -> FrozenPredictorBundle:
        loaded = pickle.loads(payload)
        if not isinstance(loaded, cls):
            raise PredictorFitError("serialized payload is not a predictor bundle")
        return loaded

    def to_metadata(self) -> dict[str, Any]:
        payload = self.to_bytes()
        metadata = {
            "schema_version": 1,
            "protocol": {
                "selection_input": "M1 group-5-fold raw OOF log loss only",
                "shared_family_and_hyperparameters": True,
                "episode_total_sample_weight": 1.0,
                "missing_indicator_for_every_feature": True,
                "platt_source": "model-specific group-OOF decision scores",
                "base_refit": "all Calibration rows",
                "log_loss_clip": LOG_LOSS_EPSILON,
            },
            "selected": {
                "family": self.family,
                "hyperparameters": dict(self.hyperparameters),
            },
            "counts": {
                "rows": self.n_rows,
                "episodes": self.n_episodes,
                "base_init_groups": self.n_groups,
                "folds": N_FOLDS,
            },
            "random_state": self.random_state,
            "calibration_data_sha256": self.calibration_data_sha256,
            "artifact": {
                "format": "python-pickle-protocol-5",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": _sklearn_version(),
            },
            "fold_assignments": [fold.to_metadata() for fold in self.folds],
            "m1_selection_candidates": [
                candidate.to_metadata() for candidate in self.candidates
            ],
            "models": {
                predictor.name: predictor.to_metadata() for predictor in self.predictors
            },
            "kill_switch_1": {
                "criterion": "M0 or M1 calibrated group-OOF AUROC >= 0.95",
                "triggered": self.kill_switch_triggered,
            },
        }
        # A hard assertion here prevents numpy scalar types or NaN from leaking
        # into a calibration lock that only appears serializable in Python.
        json.dumps(metadata, sort_keys=True, allow_nan=False)
        return metadata


@dataclass(frozen=True)
class _Fold:
    index: int
    train: np.ndarray
    validation: np.ndarray
    validation_groups: tuple[str, ...]


@dataclass(frozen=True)
class _OOFResult:
    raw_probability: np.ndarray
    decision_score: np.ndarray


def fit_failure_predictors(
    *,
    m0: FeatureSet,
    m1: FeatureSet,
    m2: FeatureSet,
    labels: Sequence[int | bool],
    episode_ids: Sequence[Hashable],
    base_init_ids: Sequence[Hashable],
    selection_config: CalibrationSelectionConfig | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FrozenPredictorBundle:
    """Select, calibrate, and refit the preregistered M0/M1/M2 predictors.

    The three matrices must describe the same state rows.  ``labels`` is the
    episode failure indicator repeated for each state row.  All rows sharing an
    ``episode_id`` must therefore have one label and one ``base_init_id``.

    M1 coverage features must already have been generated out of fold with both
    the episode and its base initialization ID excluded.  This function cannot
    reconstruct or safely verify that upstream reference-set operation.
    """

    feature_sets = {"M0": m0, "M1": m1, "M2": m2}
    n_rows = m0.values.shape[0]
    if any(features.values.shape[0] != n_rows for features in feature_sets.values()):
        raise PredictorFitError("M0, M1, and M2 must contain the same state rows")
    _validate_feature_hierarchy(m0, m1, m2)
    y = _binary_labels(labels, n_rows)
    episodes = _identifiers(episode_ids, n_rows, "episode_ids")
    groups = _identifiers(base_init_ids, n_rows, "base_init_ids")
    _validate_episode_invariants(y, episodes, groups)
    if len(set(groups)) < N_FOLDS:
        raise PredictorFitError(
            "grouped 5-fold CV requires at least five base init IDs"
        )
    if np.unique(y).size != 2:
        raise PredictorFitError(
            "Calibration labels must contain successes and failures"
        )

    weights = _episode_total_one_weights(episodes)
    folds = _make_group_folds(groups, episodes)
    for fold in folds:
        if np.unique(y[fold.train]).size != 2:
            raise PredictorFitError(
                f"fold {fold.index} training labels contain only one class"
            )

    candidates = _candidate_specifications(selection_config)
    candidate_results: list[CandidateResult] = []
    best: tuple[str, tuple[tuple[str, float | int], ...]] | None = None
    best_loss = math.inf
    for family, hyperparameters in candidates:
        oof = _fit_oof(
            m1.values,
            y,
            weights,
            folds,
            family=family,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )
        loss = _weighted_log_loss(y, oof.raw_probability, weights)
        result = CandidateResult(family, hyperparameters, loss)
        candidate_results.append(result)
        if loss < best_loss:
            best_loss = loss
            best = (family, hyperparameters)
    if best is None:  # pragma: no cover - the frozen grids are never empty
        raise PredictorFitError("predictor candidate grid is empty")
    selected_family, selected_hyperparameters = best

    fitted: list[FittedFailurePredictor] = []
    for model_name in MODEL_NAMES:
        features = feature_sets[model_name]
        oof = _fit_oof(
            features.values,
            y,
            weights,
            folds,
            family=selected_family,
            hyperparameters=selected_hyperparameters,
            random_state=random_state,
        )
        platt_slope, platt_intercept = _fit_platt(
            oof.decision_score, y, weights, random_state=random_state
        )
        calibrated = _sigmoid(platt_slope * oof.decision_score + platt_intercept)
        final_estimator = _fit_estimator(
            features.values,
            y,
            weights,
            family=selected_family,
            hyperparameters=selected_hyperparameters,
            random_state=random_state,
        )
        fitted.append(
            FittedFailurePredictor(
                name=model_name,
                feature_names=features.names,
                estimator=final_estimator,
                platt_slope=platt_slope,
                platt_intercept=platt_intercept,
                raw_oof_log_loss=_weighted_log_loss(y, oof.raw_probability, weights),
                calibrated_oof_log_loss=_weighted_log_loss(y, calibrated, weights),
                raw_oof_auroc=float(
                    roc_auc_score(y, oof.raw_probability, sample_weight=weights)
                ),
                calibrated_oof_auroc=float(
                    roc_auc_score(y, calibrated, sample_weight=weights)
                ),
                n_rows=n_rows,
                n_episodes=len(set(episodes)),
            )
        )

    fold_metadata = tuple(
        FoldAssignment(
            fold_index=fold.index,
            validation_groups=fold.validation_groups,
            validation_row_count=int(fold.validation.size),
            validation_episode_count=len({episodes[i] for i in fold.validation}),
        )
        for fold in folds
    )
    kill_switch = any(
        predictor.calibrated_oof_auroc >= 0.95
        for predictor in fitted
        if predictor.name in {"M0", "M1"}
    )
    return FrozenPredictorBundle(
        family=selected_family,
        hyperparameters=selected_hyperparameters,
        predictors=tuple(fitted),
        candidates=tuple(candidate_results),
        folds=fold_metadata,
        random_state=int(random_state),
        calibration_data_sha256=_calibration_data_hash(
            feature_sets, y, episodes, groups
        ),
        n_rows=n_rows,
        n_episodes=len(set(episodes)),
        n_groups=len(set(groups)),
        kill_switch_triggered=kill_switch,
    )


def _candidate_specifications(
    config: CalibrationSelectionConfig | None,
) -> tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...]:
    if config is None:
        logistic_values = _LOGISTIC_C
        boosting_values = _BOOSTING_GRID
        max_iter = _BOOSTING_MAX_ITER
    else:
        logistic = config.predictor_candidates["logistic_regression"]
        boosting = config.predictor_candidates["histogram_gradient_boosting"]
        logistic_values = tuple(
            float(value) for value in logistic.hyperparameter_grid["C"]
        )
        boosting_values = {
            name: tuple(values) for name, values in boosting.hyperparameter_grid.items()
        }
        if boosting.max_iter_min != 1 or boosting.max_iter_max != _BOOSTING_MAX_ITER:
            raise PredictorFitError(
                "boosting iterations must remain bounded to [1, 200]"
            )
        max_iter = int(boosting.max_iter_max)
    if set(logistic_values) != set(_LOGISTIC_C):
        raise PredictorFitError("logistic C grid differs from the preregistration")
    if set(boosting_values) != set(_BOOSTING_GRID) or any(
        {float(value) for value in boosting_values[name]}
        != {float(value) for value in expected}
        for name, expected in _BOOSTING_GRID.items()
    ):
        raise PredictorFitError(
            "histogram boosting grid differs from the preregistration"
        )

    candidates: list[tuple[str, tuple[tuple[str, float | int], ...]]] = []
    for c_value in logistic_values:
        candidates.append(("logistic_regression", (("C", float(c_value)),)))
    boosting_names = tuple(_BOOSTING_GRID)
    for values in product(*(boosting_values[name] for name in boosting_names)):
        hyperparameters: list[tuple[str, float | int]] = []
        for name, value in zip(boosting_names, values, strict=True):
            if name in {"max_leaf_nodes", "min_samples_leaf"}:
                hyperparameters.append((name, int(value)))
            else:
                hyperparameters.append((name, float(value)))
        hyperparameters.append(("max_iter", max_iter))
        candidates.append(("histogram_gradient_boosting", tuple(hyperparameters)))
    return tuple(candidates)


def _fit_oof(
    values: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    folds: tuple[_Fold, ...],
    *,
    family: str,
    hyperparameters: tuple[tuple[str, float | int], ...],
    random_state: int,
) -> _OOFResult:
    raw_probability = np.full(y.shape, np.nan, dtype=np.float64)
    decision_score = np.full(y.shape, np.nan, dtype=np.float64)
    for fold in folds:
        estimator = _fit_estimator(
            values[fold.train],
            y[fold.train],
            weights[fold.train],
            family=family,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )
        raw_probability[fold.validation] = estimator.predict_proba(
            values[fold.validation]
        )[:, 1]
        decision_score[fold.validation] = estimator.decision_function(
            values[fold.validation]
        )
    if not np.isfinite(raw_probability).all() or not np.isfinite(decision_score).all():
        raise PredictorFitError(
            "OOF fitting did not produce one finite prediction per row"
        )
    return _OOFResult(raw_probability, decision_score)


def _fit_estimator(
    values: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    family: str,
    hyperparameters: tuple[tuple[str, float | int], ...],
    random_state: int,
) -> Pipeline:
    params = dict(hyperparameters)
    missing = MissingIndicatorImputer()
    if family == "logistic_regression":
        classifier = LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=2_000,
            random_state=random_state,
        )
        estimator = Pipeline(
            (
                ("missing", missing),
                ("standardize", StandardScaler()),
                ("classifier", classifier),
            )
        )
        estimator.fit(
            values,
            y,
            missing__sample_weight=weights,
            standardize__sample_weight=weights,
            classifier__sample_weight=weights,
        )
    elif family == "histogram_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_iter=int(params["max_iter"]),
            early_stopping=False,
            random_state=random_state,
        )
        estimator = Pipeline((("missing", missing), ("classifier", classifier)))
        estimator.fit(
            values,
            y,
            missing__sample_weight=weights,
            classifier__sample_weight=weights,
        )
    else:  # pragma: no cover - private candidate construction is exhaustive
        raise PredictorFitError(f"unknown predictor family {family!r}")
    return estimator


def _fit_platt(
    decision_score: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    random_state: int,
) -> tuple[float, float]:
    # C=1e12 is the stable cross-version approximation to unregularized
    # one-dimensional logistic regression; unlike using clipped probabilities,
    # decision scores preserve the preregistration's "clipping only for log
    # loss" rule.
    calibrator = LogisticRegression(
        C=1e12,
        solver="lbfgs",
        max_iter=2_000,
        random_state=random_state,
    )
    calibrator.fit(decision_score[:, None], y, sample_weight=weights)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _make_group_folds(
    groups: tuple[Hashable, ...], episodes: tuple[Hashable, ...]
) -> tuple[_Fold, ...]:
    unique_groups = set(groups)
    episodes_by_group = {
        group: len(
            {episodes[i] for i, observed in enumerate(groups) if observed == group}
        )
        for group in unique_groups
    }
    # Outcome-independent greedy balancing by episode count.  Canonical group
    # strings make the assignment independent of input row order.
    ordered_groups = sorted(
        unique_groups,
        key=lambda group: (-episodes_by_group[group], _canonical_identifier(group)),
    )
    assignments: list[list[Hashable]] = [[] for _ in range(N_FOLDS)]
    loads = [0] * N_FOLDS
    for group in ordered_groups:
        fold_index = min(range(N_FOLDS), key=lambda index: (loads[index], index))
        assignments[fold_index].append(group)
        loads[fold_index] += episodes_by_group[group]

    group_array = np.empty(len(groups), dtype=object)
    group_array[:] = groups
    folds: list[_Fold] = []
    for fold_index, validation_groups in enumerate(assignments):
        validation_set = set(validation_groups)
        validation = np.asarray(
            [i for i, group in enumerate(group_array) if group in validation_set],
            dtype=np.int64,
        )
        train = np.asarray(
            [i for i, group in enumerate(group_array) if group not in validation_set],
            dtype=np.int64,
        )
        folds.append(
            _Fold(
                fold_index,
                train,
                validation,
                tuple(
                    sorted(_canonical_identifier(group) for group in validation_groups)
                ),
            )
        )
    return tuple(folds)


def _episode_total_one_weights(episodes: tuple[Hashable, ...]) -> np.ndarray:
    counts: dict[Hashable, int] = {}
    for episode in episodes:
        counts[episode] = counts.get(episode, 0) + 1
    return np.asarray([1.0 / counts[episode] for episode in episodes], dtype=np.float64)


def _validate_episode_invariants(
    y: np.ndarray,
    episodes: tuple[Hashable, ...],
    groups: tuple[Hashable, ...],
) -> None:
    observed: dict[Hashable, tuple[int, Hashable]] = {}
    for label, episode, group in zip(y, episodes, groups, strict=True):
        current = (int(label), group)
        previous = observed.setdefault(episode, current)
        if current != previous:
            raise PredictorFitError(
                "each episode_id must have one failure label and one base_init_id"
            )


def _validate_feature_hierarchy(m0: FeatureSet, m1: FeatureSet, m2: FeatureSet) -> None:
    _validate_feature_subset("M0", m0, "M1", m1)
    _validate_feature_subset("M1", m1, "M2", m2)


def _validate_feature_subset(
    subset_name: str,
    subset: FeatureSet,
    superset_name: str,
    superset: FeatureSet,
) -> None:
    index = {name: i for i, name in enumerate(superset.names)}
    missing = [name for name in subset.names if name not in index]
    if missing:
        raise PredictorFitError(
            f"{superset_name} must contain every {subset_name} feature; missing {missing}"
        )
    for subset_index, name in enumerate(subset.names):
        left = subset.values[:, subset_index]
        right = superset.values[:, index[name]]
        if not np.array_equal(left, right, equal_nan=True):
            raise PredictorFitError(
                f"shared feature {name!r} differs between {subset_name} and {superset_name}"
            )


def _binary_labels(labels: Sequence[int | bool], n_rows: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.shape[0] != n_rows:
        raise PredictorFitError("labels must be a one-dimensional value per state row")
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictorFitError("labels must be binary numeric values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise PredictorFitError(
            "labels must contain only zero (success) or one (failure)"
        )
    return numeric.astype(np.int8)


def _identifiers(
    values: Sequence[Hashable], n_rows: int, name: str
) -> tuple[Hashable, ...]:
    identifiers = tuple(values)
    if len(identifiers) != n_rows:
        raise PredictorFitError(f"{name} must contain one value per state row")
    for value in identifiers:
        try:
            hash(value)
        except TypeError as exc:
            raise PredictorFitError(f"{name} values must be hashable") from exc
        _canonical_identifier(value)
    return identifiers


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PredictorFitError("features must be numeric") from exc
    if matrix.ndim != 2:
        raise PredictorFitError("features must be a two-dimensional array")
    if np.isinf(matrix).any():
        raise PredictorFitError("features may contain NaN but not infinity")
    return matrix


def _validated_weights(sample_weight: np.ndarray | None, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_rows,) or not np.isfinite(weights).all():
        raise PredictorFitError("sample weights must be one finite value per row")
    if (weights <= 0).any():
        raise PredictorFitError("sample weights must be strictly positive")
    return weights


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _weighted_log_loss(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> float:
    clipped = np.clip(
        np.asarray(probability, dtype=np.float64),
        LOG_LOSS_EPSILON,
        1.0 - LOG_LOSS_EPSILON,
    )
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    return float(np.average(losses, weights=weights))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    nonnegative = value >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    output[~nonnegative] = exp_value / (1.0 + exp_value)
    return output


def _canonical_identifier(value: Hashable) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, bool)) or value is None:
        serializable: Any = value
    elif isinstance(value, float) and math.isfinite(value):
        serializable = value
    elif isinstance(value, tuple):
        serializable = [json.loads(_canonical_identifier(item)) for item in value]
    else:
        raise PredictorFitError(
            "episode/base-init identifiers must be JSON-stable scalars or tuples"
        )
    return json.dumps(serializable, sort_keys=True, separators=(",", ":"))


def _calibration_data_hash(
    feature_sets: Mapping[str, FeatureSet],
    y: np.ndarray,
    episodes: tuple[Hashable, ...],
    groups: tuple[Hashable, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"mech-int-vla-calibration-predictors-v1\0")
    for model_name in MODEL_NAMES:
        features = feature_sets[model_name]
        digest.update(model_name.encode("utf-8") + b"\0")
        digest.update(json.dumps(features.names, separators=(",", ":")).encode("utf-8"))
        digest.update(np.asarray(features.values.shape, dtype="<i8").tobytes())
        # Canonicalize NaN payload bits before hashing.
        values = np.array(features.values, dtype="<f8", copy=True, order="C")
        values[np.isnan(values)] = np.nan
        digest.update(values.tobytes(order="C"))
    digest.update(np.asarray(y, dtype=np.uint8).tobytes())
    for episode, group in zip(episodes, groups, strict=True):
        digest.update(_canonical_identifier(episode).encode("utf-8") + b"\0")
        digest.update(_canonical_identifier(group).encode("utf-8") + b"\0")
    return digest.hexdigest()


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


__all__ = [
    "CandidateResult",
    "FeatureSet",
    "FittedFailurePredictor",
    "FoldAssignment",
    "FrozenPredictorBundle",
    "MissingIndicatorImputer",
    "PredictorFitError",
    "fit_failure_predictors",
]
