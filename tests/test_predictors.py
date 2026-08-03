from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

import mech_int_vla.predictors as predictor_module
from mech_int_vla.config import load_protocol_config
from mech_int_vla.predictors import (
    FeatureSet,
    FrozenPredictorBundle,
    MissingIndicatorImputer,
    PredictorFitError,
    fit_failure_predictors,
)

ROOT = Path(__file__).parents[1]


def synthetic_calibration(
    *, seed: int = 7
) -> tuple[FeatureSet, FeatureSet, FeatureSet, list[int], list[str], list[int]]:
    rng = np.random.default_rng(seed)
    m0_rows: list[list[float]] = []
    m1_rows: list[list[float]] = []
    m2_rows: list[list[float]] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    base_init_ids: list[int] = []
    for group in range(10, 20):
        # Both classes occur in every group, making every group-held-out training
        # set valid without outcome-aware fold construction.
        for episode_in_group in range(2):
            failure = episode_in_group
            state_count = 1 + ((group + episode_in_group) % 3)
            episode_noise = rng.normal(scale=0.15)
            for state in range(state_count):
                output_only = rng.normal()
                privileged = 1.6 * failure + episode_noise + 0.03 * state
                internal = 0.8 * failure + rng.normal(scale=0.2)
                if group == 10 and state == 0:
                    output_only = np.nan
                m0_rows.append([output_only])
                m1_rows.append([output_only, privileged])
                m2_rows.append([output_only, privileged, internal])
                labels.append(failure)
                episode_ids.append(f"g{group}-e{episode_in_group}")
                base_init_ids.append(group)
    return (
        FeatureSet(np.asarray(m0_rows), ("output",)),
        FeatureSet(np.asarray(m1_rows), ("output", "privileged")),
        FeatureSet(np.asarray(m2_rows), ("output", "privileged", "internal")),
        labels,
        episode_ids,
        base_init_ids,
    )


def small_candidate_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = (
        ("logistic_regression", (("C", 0.1),)),
        ("logistic_regression", (("C", 1.0),)),
    )
    monkeypatch.setattr(
        predictor_module,
        "_candidate_specifications",
        lambda selection_config: candidates,
    )


def test_frozen_config_expands_to_exact_preregistered_candidate_grid() -> None:
    config = load_protocol_config(ROOT / "configs")
    candidates = predictor_module._candidate_specifications(
        config.split.calibration_selection
    )

    assert len(candidates) == 21
    assert [dict(params)["C"] for family, params in candidates[:5]] == [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]
    boosting = [
        dict(params) for family, params in candidates if family.endswith("boosting")
    ]
    assert len(boosting) == 16
    assert {params["max_iter"] for params in boosting} == {200}
    assert {params["learning_rate"] for params in boosting} == {0.03, 0.1}
    assert {params["max_leaf_nodes"] for params in boosting} == {7, 15}
    assert {params["min_samples_leaf"] for params in boosting} == {10, 20}
    assert {params["l2_regularization"] for params in boosting} == {0.0, 1.0}


def test_missing_transform_uses_weights_and_always_appends_all_indicators() -> None:
    values = np.asarray([[0.0, np.nan], [0.0, 5.0], [100.0, 7.0]])
    # The final observation outweighs the two zeros, so the weighted median is
    # 100 rather than the unweighted median 0.
    transformer = MissingIndicatorImputer().fit(
        values, sample_weight=np.asarray([0.1, 0.1, 0.8])
    )
    transformed = transformer.transform(values)

    np.testing.assert_allclose(transformer.statistics_, [100.0, 7.0])
    assert transformed.shape == (3, 4)
    np.testing.assert_array_equal(transformed[:, 2:], np.isnan(values))
    assert transformer.get_feature_names_out(("a", "b")).tolist() == [
        "a",
        "b",
        "a__missing",
        "b__missing",
    ]


def test_fit_is_grouped_shared_calibrated_serializable_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_candidate_grid(monkeypatch)
    m0, m1, m2, labels, episodes, groups = synthetic_calibration()
    kwargs = {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "labels": labels,
        "episode_ids": episodes,
        "base_init_ids": groups,
        "random_state": 123,
    }
    first = fit_failure_predictors(**kwargs)
    second = fit_failure_predictors(**kwargs)

    assert first.family == "logistic_regression"
    assert first.hyperparameters in ((("C", 0.1),), (("C", 1.0),))
    assert len(first.predictors) == 3
    assert len(first.candidates) == 2
    assert first.n_episodes == 20
    assert first.n_groups == 10

    observed_groups: list[str] = []
    for fold in first.folds:
        observed_groups.extend(fold.validation_groups)
        assert fold.validation_episode_count == 4
    assert sorted(observed_groups) == [str(group) for group in range(10, 20)]
    assert len(observed_groups) == len(set(observed_groups))

    for name, features in (("M0", m0), ("M1", m1), ("M2", m2)):
        predictor = first.model(name)
        probability = predictor.predict_proba(features.values)
        assert probability.shape == (len(labels),)
        assert np.all((probability > 0.0) & (probability < 1.0))
        assert np.isfinite(predictor.platt_slope)
        assert np.isfinite(predictor.platt_intercept)
        np.testing.assert_array_equal(
            probability, second.model(name).predict_proba(features.values)
        )

    metadata = first.to_metadata()
    json.dumps(metadata, sort_keys=True, allow_nan=False)
    assert metadata == second.to_metadata()
    assert metadata["selected"]["family"] == "logistic_regression"
    assert metadata["protocol"]["episode_total_sample_weight"] == 1.0
    assert metadata["models"]["M2"]["platt"]["input"].startswith("group_oof")
    assert len(metadata["models"]["M2"]["transformed_feature_names"]) == 6

    payload = first.to_bytes()
    restored = FrozenPredictorBundle.from_bytes(payload)
    np.testing.assert_array_equal(
        first.predict_proba("M2", m2.values), restored.predict_proba("M2", m2.values)
    )
    assert (
        metadata["artifact"]["sha256"]
        == predictor_module.hashlib.sha256(payload).hexdigest()
    )


def test_family_selection_depends_on_m1_only(monkeypatch: pytest.MonkeyPatch) -> None:
    small_candidate_grid(monkeypatch)
    m0, m1, m2, labels, episodes, groups = synthetic_calibration()
    original = fit_failure_predictors(
        m0=m0,
        m1=m1,
        m2=m2,
        labels=labels,
        episode_ids=episodes,
        base_init_ids=groups,
    )
    rng = np.random.default_rng(99)
    altered_m2_values = np.column_stack(
        (m1.values, rng.normal(size=(m1.values.shape[0], 1)))
    )
    altered = fit_failure_predictors(
        m0=FeatureSet(m1.values[:, 1:2], ("privileged",)),
        m1=m1,
        m2=FeatureSet(altered_m2_values, m2.names),
        labels=labels,
        episode_ids=episodes,
        base_init_ids=groups,
    )

    assert original.family == altered.family
    assert original.hyperparameters == altered.hyperparameters
    assert original.candidates == altered.candidates


def test_histogram_path_freezes_no_early_stopping_and_respects_tree_cap() -> None:
    m0, _, _, labels, _, _ = synthetic_calibration()
    estimator = predictor_module._fit_estimator(
        m0.values,
        np.asarray(labels),
        np.ones(len(labels)),
        family="histogram_gradient_boosting",
        hyperparameters=(
            ("learning_rate", 0.1),
            ("max_leaf_nodes", 7),
            ("min_samples_leaf", 10),
            ("l2_regularization", 1.0),
            ("max_iter", 200),
        ),
        random_state=17,
    )
    classifier = estimator.named_steps["classifier"]

    assert isinstance(classifier, HistGradientBoostingClassifier)
    assert classifier.max_iter == 200
    assert classifier.n_iter_ <= 200
    assert classifier.early_stopping is False


def test_rejects_episode_leakage_and_inconsistent_feature_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_candidate_grid(monkeypatch)
    m0, m1, m2, labels, episodes, groups = synthetic_calibration()
    bad_groups = groups.copy()
    duplicate_row = episodes.index(episodes[0], 1)
    bad_groups[duplicate_row] = 999
    with pytest.raises(
        PredictorFitError, match="one failure label and one base_init_id"
    ):
        fit_failure_predictors(
            m0=m0,
            m1=m1,
            m2=m2,
            labels=labels,
            episode_ids=episodes,
            base_init_ids=bad_groups,
        )

    changed = np.array(m1.values, copy=True)
    changed[0, 0] = 42.0
    with pytest.raises(PredictorFitError, match="shared feature 'output' differs"):
        fit_failure_predictors(
            m0=m0,
            m1=FeatureSet(changed, m1.names),
            m2=m2,
            labels=labels,
            episode_ids=episodes,
            base_init_ids=groups,
        )
