from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.config import load_protocol_config
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    AlphaCVResult,
    CandidateCVResult,
    ProbeError,
    ProbeSamples,
    circular_targets,
    cross_validate_circular_probe,
    episode_equal_weights,
    evaluate_mean_prediction_baseline,
    evaluate_proprioception_only_baseline,
    evaluate_random_label_baseline,
    evaluate_time_only_baseline,
    fit_centered_ridge,
    make_group_folds,
    select_and_fit_circular_probe,
    select_candidate_one_standard_error,
    symmetry_aware_circular_error,
)

ROOT = Path(__file__).parents[1]


def _samples(*, symmetry_order: int = 2) -> ProbeSamples:
    # Two episodes and three rows per base-init ID.
    groups = np.repeat(np.arange(10, 20), 6)
    episodes = np.array(
        [
            f"init{group}-episode{episode}"
            for group in range(10, 20)
            for episode in range(2)
            for _ in range(3)
        ]
    )
    theta = np.linspace(-1.2, 1.2, groups.size)
    return ProbeSamples.from_arrays(
        theta_rel=theta,
        base_init_state_id=groups,
        episode_id=episodes,
        symmetry_order=symmetry_order,
    )


def _candidate_result(name: str, mean: float, se: float) -> CandidateCVResult:
    alpha = AlphaCVResult(
        alpha=0.1,
        fold_mae_rad=(mean,) * 5,
        mean_mae_rad=mean,
        standard_error_rad=se,
    )
    return CandidateCVResult(
        candidate=name,
        alpha_results=(alpha,),
        selected_alpha=0.1,
        mean_mae_rad=mean,
        standard_error_rad=se,
    )


def test_episode_equal_weights_ignore_row_count() -> None:
    weights = episode_equal_weights(["a", "b", "b", "b"])

    assert weights.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])
    assert weights[0] == pytest.approx(weights[1:].sum())


def test_group_folds_hold_out_every_group_and_row_once() -> None:
    samples = _samples()
    folds = make_group_folds(samples.base_init_state_id)

    assert len(folds) == 5
    assert [fold.test_groups for fold in folds] == [
        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),
    ]
    heldout_rows = np.concatenate([fold.test_rows for fold in folds])
    assert sorted(heldout_rows.tolist()) == list(range(samples.n_rows))
    for fold in folds:
        train_groups = set(samples.base_init_state_id[fold.train_rows])
        assert train_groups.isdisjoint(fold.test_groups)


def test_symmetry_aware_error_wraps_and_respects_object_symmetry() -> None:
    error = symmetry_aware_circular_error(
        np.array([math.pi - 0.1, math.pi / 2, 0.0]),
        np.array([-math.pi + 0.1, 0.0, math.pi]),
        symmetry_order=2,
    )

    assert error == pytest.approx([0.2, math.pi / 2, 0.0])


def test_centered_ridge_predicts_with_unit_normalization_only_at_prediction() -> None:
    samples = _samples(symmetry_order=1)
    target = circular_targets(samples.theta_rel, symmetry_order=1)
    features = np.column_stack((target, np.linspace(0.0, 1.0, samples.n_rows)))
    features += np.array([7.0, -4.0, 2.0])

    model = fit_centered_ridge(features, samples, alpha=100.0)
    raw = model.predict_raw(features)
    unit = model.predict_unit(features)

    assert model.coefficient.shape == (2, 3)
    assert np.linalg.norm(raw, axis=1) != pytest.approx(np.ones(samples.n_rows))
    assert np.linalg.norm(unit, axis=1) == pytest.approx(np.ones(samples.n_rows))
    assert np.isfinite(model.predict_angle(features)).all()


def test_cross_validation_uses_complete_alpha_grid_and_reports_fold_se() -> None:
    samples = _samples(symmetry_order=1)
    features = circular_targets(samples.theta_rel, symmetry_order=1)
    result = cross_validate_circular_probe(
        "synthetic",
        features,
        samples,
        alpha_grid=(0.0001, 0.1, 100.0),
    )

    assert [entry.alpha for entry in result.alpha_results] == [0.0001, 0.1, 100.0]
    assert all(len(entry.fold_mae_rad) == 5 for entry in result.alpha_results)
    assert all(entry.standard_error_rad >= 0.0 for entry in result.alpha_results)
    assert result.selected_alpha == 0.0001
    assert result.mean_mae_rad < 0.01


def test_one_standard_error_selection_uses_exact_frozen_preference() -> None:
    means = {
        "vlm_context": (0.205, 0.01),
        "early_expert_t1_0": (0.202, 0.01),
        "early_expert_t0_5": (0.201, 0.01),
        "late_expert_t1_0": (0.200, 0.01),
        "late_expert_t0_5": (0.199, 0.01),
    }
    results = tuple(
        _candidate_result(name, *means[name])
        for name in reversed(DEFAULT_CANDIDATE_PREFERENCE)
    )

    selected, threshold, eligible = select_candidate_one_standard_error(results)

    assert threshold == pytest.approx(0.209)
    assert eligible == DEFAULT_CANDIDATE_PREFERENCE
    assert selected.candidate == "vlm_context"


def test_full_selection_fits_final_probe_and_has_stable_hash_ready_metadata() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples(symmetry_order=2)
    features = circular_targets(samples.theta_rel, symmetry_order=2)
    candidates = {name: features.copy() for name in DEFAULT_CANDIDATE_PREFERENCE}

    first = select_and_fit_circular_probe(
        candidates,
        samples,
        selection_config=protocol.split.calibration_selection,
    )
    second = select_and_fit_circular_probe(
        candidates,
        samples,
        selection_config=protocol.split.calibration_selection,
    )

    assert first.artifact.candidate == "vlm_context"
    assert first.eligible_candidates == DEFAULT_CANDIDATE_PREFERENCE
    assert first.artifact.model.alpha in {
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    }
    assert first.artifact.model.coefficient.shape == (2, 2)
    assert first.artifact.sha256() == second.artifact.sha256()
    assert len(first.artifact.sha256()) == 64
    metadata = json.loads(first.artifact.canonical_json())
    assert metadata["cv"]["group"] == "base_init_state_id"
    assert metadata["training"]["episodes"] == 20
    assert metadata["selection"]["candidate_preference"] == list(
        DEFAULT_CANDIDATE_PREFERENCE
    )
    assert np.allclose(
        metadata["parameters"]["coefficient"], first.artifact.model.coefficient
    )


def test_required_baseline_interfaces_share_grouped_circular_evaluation() -> None:
    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples(symmetry_order=1)
    target = circular_targets(samples.theta_rel, symmetry_order=1)
    time = np.linspace(0.0, 1.0, samples.n_rows)

    time_result = evaluate_time_only_baseline(
        time, samples, selection_config=protocol.split.calibration_selection
    )
    proprio_result = evaluate_proprioception_only_baseline(
        target, samples, selection_config=protocol.split.calibration_selection
    )
    random_result = evaluate_random_label_baseline(
        target,
        samples,
        randomized_theta_rel=samples.theta_rel[::-1],
        selection_config=protocol.split.calibration_selection,
    )
    mean_result = evaluate_mean_prediction_baseline(samples)

    assert time_result.candidate == "time_only"
    assert proprio_result.candidate == "proprioception_only"
    assert random_result.candidate == "random_label"
    assert len(mean_result.fold_mae_rad) == 5
    assert all(
        np.isfinite(result.mean_mae_rad)
        for result in (time_result, proprio_result, random_result, mean_result)
    )


def test_rejects_leaky_episode_groups_and_non_frozen_candidate_order() -> None:
    with pytest.raises(ProbeError, match="positive integer"):
        ProbeSamples.from_arrays(
            theta_rel=[0.0],
            base_init_state_id=[10],
            episode_id=["episode"],
            symmetry_order=1.5,
        )

    with pytest.raises(ProbeError, match="more than one base init"):
        ProbeSamples.from_arrays(
            theta_rel=[0.0, 0.1],
            base_init_state_id=[10, 11],
            episode_id=["same", "same"],
            symmetry_order=1,
        )

    protocol = load_protocol_config(ROOT / "configs")
    samples = _samples()
    features = circular_targets(samples.theta_rel, symmetry_order=2)
    with pytest.raises(ProbeError, match="five frozen candidates"):
        select_and_fit_circular_probe(
            {"vlm_context": features},
            samples,
            selection_config=protocol.split.calibration_selection,
        )
