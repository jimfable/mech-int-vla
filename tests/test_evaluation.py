from __future__ import annotations

import math

import numpy as np
import pytest

from mech_int_vla.evaluation import (
    EpisodePredictions,
    EvaluationError,
    binary_auroc,
    binary_log_loss,
    black_box_ceiling_triggered,
    brier_score,
    calibrate_alarm_threshold,
    cluster_bootstrap_ci,
    dynamic_range_gate_decision,
    episode_primary_log_loss,
    first_alarm_time,
    paired_lead_time_summary,
    paired_prediction_comparison,
    reproduction_gate_decision,
    wilson_interval,
)


def episode(
    episode_id: str,
    init_id: int,
    label: int,
    scores: list[float],
    *,
    stride: int = 5,
    failure_step: int | None = None,
) -> EpisodePredictions:
    return EpisodePredictions(
        episode_id=episode_id,
        init_id=init_id,
        label=label,
        scores={index * stride: score for index, score in enumerate(scores)},
        failure_step=failure_step,
    )


def test_log_loss_clips_extreme_probabilities_and_brier_is_unclipped() -> None:
    loss = binary_log_loss([0.0, 1.0], [1, 0])
    assert loss == pytest.approx(-math.log(1e-6))
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    with pytest.raises(EvaluationError, match=r"\[0, 1\]"):
        binary_log_loss([1.1], [1])


def test_episode_loss_averages_only_available_primary_steps() -> None:
    record = EpisodePredictions(
        episode_id="early-stop",
        init_id=30,
        label=1,
        scores={0: 0.5, 5: 0.9, 50: 0.75, 100: 1.0},
    )
    assert episode_primary_log_loss(record) == pytest.approx(
        np.mean([-math.log(0.5), -math.log(0.75), -math.log(1 - 1e-6)])
    )
    with pytest.raises(EvaluationError, match="no available primary"):
        episode_primary_log_loss(
            EpisodePredictions("missing", 30, 0, {5: 0.1, 10: 0.2})
        )


def test_auroc_uses_half_credit_for_ties_and_rejects_one_class() -> None:
    assert binary_auroc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]) == pytest.approx(0.875)
    with pytest.raises(EvaluationError, match="positive and one negative"):
        binary_auroc([0.2, 0.3], [0, 0])


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_clusters() -> None:
    values = [0.0, 2.0, 10.0, 12.0]
    clusters = ["a", "a", "b", "b"]
    first = cluster_bootstrap_ci(values, clusters, replicates=250, seed=91)
    second = cluster_bootstrap_ci(values, clusters, replicates=250, seed=91)
    assert first == second
    assert first.estimate == pytest.approx(6.0)
    # Whole-cluster draws can produce only means 1, 6, or 11.
    assert first.interval.lower == pytest.approx(1.0)
    assert first.interval.upper == pytest.approx(11.0)
    with pytest.raises(EvaluationError, match="at least two clusters"):
        cluster_bootstrap_ci([1.0, 2.0], ["a", "a"], replicates=2)


def test_paired_prediction_estimand_and_decision_flag() -> None:
    model1 = [
        EpisodePredictions("success-a", 30, 0, {0: 0.4, 50: 0.4}),
        EpisodePredictions("failure-b", 31, 1, {0: 0.6, 50: 0.6}),
    ]
    model2 = [
        EpisodePredictions("failure-b", 31, 1, {0: 0.9, 50: 0.9}),
        EpisodePredictions("success-a", 30, 0, {0: 0.1, 50: 0.1}),
    ]
    result = paired_prediction_comparison(
        model1,
        model2,
        bootstrap_replicates=500,
        bootstrap_seed=7,
    )
    assert result.delta_log_loss < 0
    assert result.relative_lift > 0.03
    assert result.delta_interval.upper < 0
    assert result.primary_claim_succeeds
    assert result.model2_brier < result.model1_brier
    assert result.model2_auroc == 1.0


@pytest.mark.parametrize(
    "second, message",
    [
        ([EpisodePredictions("other", 30, 0, {0: 0.1})], "unpaired"),
        ([EpisodePredictions("same", 31, 0, {0: 0.1})], "metadata mismatch"),
        ([EpisodePredictions("same", 30, 0, {50: 0.1})], "score-step mismatch"),
    ],
)
def test_prediction_comparison_fails_closed_on_unpaired_data(
    second: list[EpisodePredictions], message: str
) -> None:
    first = [EpisodePredictions("same", 30, 0, {0: 0.2})]
    with pytest.raises(EvaluationError, match=message):
        paired_prediction_comparison(first, second, bootstrap_replicates=2)


def test_alarm_requires_three_contiguous_exceedances_and_equality_counts() -> None:
    scores = {0: 0.7, 5: 0.7, 10: 0.69, 15: 0.7, 20: 0.7, 25: 0.7}
    assert first_alarm_time(scores, 0.7) == 25
    with pytest.raises(EvaluationError, match="gap"):
        first_alarm_time({0: 0.8, 5: 0.8, 15: 0.8}, 0.7)


def test_alarm_threshold_is_lowest_candidate_under_episode_fpr_cap() -> None:
    episodes = [
        episode("s0", 10, 0, [0.1, 0.2, 0.3]),
        episode("s1", 11, 0, [0.1, 0.4, 0.4]),
        episode("s2", 12, 0, [0.6, 0.6, 0.6]),
        episode("s3", 13, 0, [0.7, 0.7, 0.7]),
        episode("f0", 14, 1, [0.9, 0.9, 0.9], failure_step=20),
    ]
    result = calibrate_alarm_threshold(episodes, max_episode_fpr=0.25)
    # >= tie handling means threshold .6 alarms for two successes; .7 alarms one.
    assert result.threshold == pytest.approx(0.7)
    assert result.false_positive_episodes == 1
    assert result.episode_false_positive_rate == 0.25
    assert result.comparison_rule == "score >= threshold"


def test_threshold_has_conservative_above_maximum_fallback() -> None:
    result = calibrate_alarm_threshold(
        [episode("s", 10, 0, [1.0, 1.0, 1.0])], max_episode_fpr=0.0
    )
    assert result.threshold > 1.0
    assert result.episode_false_positive_rate == 0.0


def test_paired_lead_time_summary_counts_misses_as_zero() -> None:
    model1 = [
        episode("a", 30, 1, [0.1] * 11, failure_step=50),
        episode("b", 31, 1, [0.1] * 11, failure_step=50),
    ]
    model2 = [
        episode("a", 30, 1, [0.9, 0.9, 0.9] + [0.1] * 8, failure_step=50),
        episode("b", 31, 1, [0.1, 0.9, 0.9, 0.9] + [0.1] * 7, failure_step=50),
    ]
    result = paired_lead_time_summary(
        model1,
        model2,
        model1_threshold=0.8,
        model2_threshold=0.8,
        bootstrap_replicates=200,
        bootstrap_seed=11,
    )
    assert result.model1_median_lead == 0.0
    assert result.model2_median_lead == pytest.approx(37.5)
    assert result.model1_detection_rate == 0.0
    assert result.model1_conditional_median_lead is None
    assert result.model2_detection_rate == 1.0
    assert result.paired_difference_interval.lower > 0
    assert result.lead_time_claim_succeeds


def test_lead_time_fails_closed_without_failure_event() -> None:
    malformed = [episode("a", 30, 1, [0.9, 0.9, 0.9])]
    with pytest.raises(EvaluationError, match="missing failure_step"):
        paired_lead_time_summary(
            malformed,
            malformed,
            model1_threshold=0.5,
            model2_threshold=0.5,
            bootstrap_replicates=2,
        )


def test_wilson_intervals_and_gate_flags_use_point_estimates() -> None:
    interval = wilson_interval(0, 10)
    assert interval.rate == 0.0
    assert interval.lower == 0.0
    assert 0.0 < interval.upper < 0.5

    reproduction = reproduction_gate_decision(6, 10)
    assert reproduction.passes
    dynamic = dynamic_range_gate_decision(27, 30, 6)
    assert dynamic.passes_validity
    assert dynamic.passes_failure_range
    assert dynamic.passes
    assert dynamic.validity_interval.lower < 0.9  # CI is descriptive, not decisive.


def test_decision_boundary_and_black_box_ceiling_flags_are_explicit() -> None:
    assert black_box_ceiling_triggered(0.94, 0.95)
    assert not black_box_ceiling_triggered(0.949, 0.949)
    with pytest.raises(EvaluationError, match="finite"):
        black_box_ceiling_triggered(float("nan"), 0.8)
