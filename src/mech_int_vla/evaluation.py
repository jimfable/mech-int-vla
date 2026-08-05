"""Fail-closed statistical evaluation for the preregistered VLA study.

This module deliberately has no dependency on simulator or model code.  It accepts
already-frozen episode predictions, validates pairing and clustering metadata, and
implements only the estimands fixed in :mod:`PREREG.md`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from statistics import NormalDist
from typing import Any

import numpy as np

from .config import SplitName
from .manifest import Manifest

PRIMARY_STEPS = (0, 50, 100, 150, 200)
PROBABILITY_CLIP = 1e-6


class EvaluationError(ValueError):
    """Raised when an evaluation input is malformed or cannot support a metric."""


@dataclass(frozen=True)
class EpisodePredictions:
    """Predicted failure probabilities for one closed-loop episode."""

    episode_id: str
    init_id: Hashable
    label: int
    scores: Mapping[int, float]
    failure_step: int | None = None
    manifest_metadata: Mapping[str, Any] | None = None
    valid_reset: bool | None = None


@dataclass(frozen=True)
class PercentileInterval:
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    interval: PercentileInterval
    replicates: int
    seed: int
    clusters: int


@dataclass(frozen=True)
class PairedPredictionResult:
    model1_log_loss: float
    model2_log_loss: float
    delta_log_loss: float
    relative_lift: float
    delta_interval: PercentileInterval
    model1_brier: float
    model2_brier: float
    model1_auroc: float
    model2_auroc: float
    episodes: int
    clusters: int
    primary_claim_succeeds: bool


@dataclass(frozen=True)
class AlarmThresholdResult:
    threshold: float
    episode_false_positive_rate: float
    false_positive_episodes: int
    successful_episodes: int
    candidates_evaluated: int
    comparison_rule: str = "score >= threshold"


@dataclass(frozen=True)
class LeadTimeResult:
    model1_median_lead: float
    model2_median_lead: float
    median_paired_difference: float
    paired_difference_interval: PercentileInterval
    model1_detection_rate: float
    model2_detection_rate: float
    model1_conditional_median_lead: float | None
    model2_conditional_median_lead: float | None
    failed_episodes: int
    clusters: int
    lead_time_claim_succeeds: bool


@dataclass(frozen=True)
class RateInterval:
    rate: float
    lower: float
    upper: float
    confidence: float
    successes: int
    total: int


@dataclass(frozen=True)
class ReproductionGateResult:
    success_interval: RateInterval
    minimum_successes: int
    passes: bool


@dataclass(frozen=True)
class DynamicRangeGateResult:
    validity_interval: RateInterval
    failure_interval: RateInterval
    minimum_valid_fraction: float
    failure_rate_bounds: tuple[float, float]
    passes_validity: bool
    passes_failure_range: bool
    passes: bool


def _as_1d_finite(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise EvaluationError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise EvaluationError(f"{name} must contain only finite values")
    return array


def _as_binary_labels(labels: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.size != size:
        raise EvaluationError("labels must be one-dimensional and match probabilities")
    if not np.all((array == 0) | (array == 1)):
        raise EvaluationError("labels must contain only 0 and 1")
    return array.astype(np.float64, copy=False)


def _probabilities(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = _as_1d_finite(values, "probabilities")
    if np.any((array < 0.0) | (array > 1.0)):
        raise EvaluationError("probabilities must lie in [0, 1]")
    return array


def binary_log_loss(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    clip: float = PROBABILITY_CLIP,
) -> float:
    """Return mean binary log loss after preregistered probability clipping."""

    probabilities_array = _probabilities(probabilities)
    labels_array = _as_binary_labels(labels, probabilities_array.size)
    if not math.isfinite(clip) or not 0.0 < clip < 0.5:
        raise EvaluationError("clip must be finite and strictly between 0 and 0.5")
    clipped = np.clip(probabilities_array, clip, 1.0 - clip)
    losses = -labels_array * np.log(clipped) - (1.0 - labels_array) * np.log1p(-clipped)
    return float(np.mean(losses))


def brier_score(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Return the mean squared probability error."""

    probabilities_array = _probabilities(probabilities)
    labels_array = _as_binary_labels(labels, probabilities_array.size)
    return float(np.mean(np.square(probabilities_array - labels_array)))


def binary_auroc(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Compute binary AUROC with exact half credit for tied scores.

    This is the Mann--Whitney definition and avoids an sklearn/scipy dependency.
    """

    scores = _probabilities(probabilities)
    binary = _as_binary_labels(labels, scores.size).astype(np.int8)
    positives = scores[binary == 1]
    negatives = scores[binary == 0]
    if positives.size == 0 or negatives.size == 0:
        raise EvaluationError("AUROC requires at least one positive and one negative")
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (
            np.count_nonzero(comparisons > 0.0)
            + 0.5 * np.count_nonzero(comparisons == 0.0)
        )
        / comparisons.size
    )


def _validate_episode(
    episode: EpisodePredictions,
    *,
    cadence: int | None = None,
    require_failure_step: bool = False,
) -> tuple[tuple[int, float], ...]:
    if not isinstance(episode.episode_id, str) or not episode.episode_id:
        raise EvaluationError("episode_id must be a non-empty string")
    try:
        hash(episode.init_id)
    except TypeError as exc:
        raise EvaluationError("init_id must be hashable") from exc
    if isinstance(episode.init_id, bool) or episode.init_id is None:
        raise EvaluationError("init_id must identify a real initialization cluster")
    if episode.label not in (0, 1) or isinstance(episode.label, bool):
        raise EvaluationError(f"episode {episode.episode_id!r} has a non-binary label")
    if not isinstance(episode.scores, Mapping) or not episode.scores:
        raise EvaluationError(f"episode {episode.episode_id!r} has no scores")

    ordered: list[tuple[int, float]] = []
    for raw_step, raw_score in episode.scores.items():
        if not isinstance(raw_step, Integral) or isinstance(raw_step, bool):
            raise EvaluationError("score steps must be integer control steps")
        step = int(raw_step)
        if step < 0:
            raise EvaluationError("score steps must be nonnegative")
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise EvaluationError(
                "episode scores must be finite probabilities in [0, 1]"
            )
        ordered.append((step, score))
    ordered.sort()

    if cadence is not None:
        if cadence <= 0:
            raise EvaluationError("cadence must be positive")
        steps = [step for step, _ in ordered]
        if steps[0] != 0 or any(step % cadence for step in steps):
            raise EvaluationError(
                f"episode {episode.episode_id!r} scores must start at zero and use "
                f"the {cadence}-step cadence"
            )
        if any(right - left != cadence for left, right in pairwise(steps)):
            raise EvaluationError(
                f"episode {episode.episode_id!r} has a gap in its score cadence"
            )

    if episode.failure_step is not None and (
        not isinstance(episode.failure_step, Integral)
        or isinstance(episode.failure_step, bool)
        or episode.failure_step < 0
    ):
        raise EvaluationError("failure_step must be a nonnegative integer")
    if require_failure_step and episode.label == 1 and episode.failure_step is None:
        raise EvaluationError(
            f"failed episode {episode.episode_id!r} is missing failure_step"
        )
    if episode.label == 0 and episode.failure_step is not None:
        raise EvaluationError("successful episodes must not have a failure_step")
    return tuple(ordered)


def episode_primary_log_loss(
    episode: EpisodePredictions,
    *,
    primary_steps: Sequence[int] = PRIMARY_STEPS,
    clip: float = PROBABILITY_CLIP,
) -> float:
    """Average loss over primary steps available before episode termination."""

    ordered = dict(_validate_episode(episode))
    selected = _primary_step_scores(episode.episode_id, ordered, primary_steps)
    return binary_log_loss(
        [score for _, score in selected],
        [episode.label] * len(selected),
        clip=clip,
    )


def _primary_step_scores(
    episode_id: str,
    scores: Mapping[int, float],
    primary_steps: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    if not primary_steps:
        raise EvaluationError("primary_steps must not be empty")
    normalized: list[int] = []
    for step in primary_steps:
        if not isinstance(step, Integral) or isinstance(step, bool) or step < 0:
            raise EvaluationError("primary_steps must be nonnegative integers")
        normalized.append(int(step))
    if len(set(normalized)) != len(normalized):
        raise EvaluationError("primary_steps must not contain duplicates")
    selected = tuple((step, scores[step]) for step in normalized if step in scores)
    if not selected:
        raise EvaluationError(f"episode {episode_id!r} has no available primary step")
    return selected


def cluster_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[Hashable] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    replicates: int = 10_000,
    confidence: float = 0.90,
    seed: int = 260_803,
) -> BootstrapResult:
    """Percentile CI from deterministic resampling of whole init-ID clusters."""

    sample = _as_1d_finite(values, "values")
    clusters_array = np.asarray(cluster_ids, dtype=object)
    if clusters_array.ndim != 1 or clusters_array.size != sample.size:
        raise EvaluationError("cluster_ids must be one-dimensional and match values")
    if (
        not isinstance(replicates, Integral)
        or isinstance(replicates, bool)
        or replicates < 1
    ):
        raise EvaluationError("replicates must be a positive integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise EvaluationError("confidence must be strictly between 0 and 1")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise EvaluationError("seed must be a nonnegative integer")

    unique: list[Hashable] = []
    rows_by_cluster: list[np.ndarray] = []
    positions: dict[Hashable, int] = {}
    for row, cluster in enumerate(clusters_array.tolist()):
        try:
            hash(cluster)
        except TypeError as exc:
            raise EvaluationError("every cluster ID must be hashable") from exc
        if cluster not in positions:
            positions[cluster] = len(unique)
            unique.append(cluster)
            rows_by_cluster.append(np.asarray([row], dtype=np.int64))
        else:
            index = positions[cluster]
            rows_by_cluster[index] = np.append(rows_by_cluster[index], row)
    if len(unique) < 2:
        raise EvaluationError("cluster bootstrap requires at least two clusters")

    estimate = float(statistic(sample))
    if not math.isfinite(estimate):
        raise EvaluationError("statistic returned a non-finite estimate")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    draws = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        selected_clusters = generator.integers(0, len(unique), size=len(unique))
        indices = np.concatenate(
            [rows_by_cluster[index] for index in selected_clusters]
        )
        value = float(statistic(sample[indices]))
        if not math.isfinite(value):
            raise EvaluationError("statistic returned a non-finite bootstrap value")
        draws[replicate] = value
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapResult(
        estimate=estimate,
        interval=PercentileInterval(float(lower), float(upper), confidence),
        replicates=int(replicates),
        seed=int(seed),
        clusters=len(unique),
    )


def _episodes_by_id(
    episodes: Sequence[EpisodePredictions], model_name: str
) -> dict[str, EpisodePredictions]:
    if not episodes:
        raise EvaluationError(f"{model_name} episodes must not be empty")
    indexed: dict[str, EpisodePredictions] = {}
    for episode in episodes:
        _validate_episode(episode)
        if episode.episode_id in indexed:
            raise EvaluationError(
                f"{model_name} contains duplicate episode {episode.episode_id!r}"
            )
        indexed[episode.episode_id] = episode
    return indexed


def _paired_episodes(
    model1: Sequence[EpisodePredictions],
    model2: Sequence[EpisodePredictions],
    *,
    require_failure_step: bool = False,
    cadence: int | None = None,
) -> tuple[tuple[EpisodePredictions, EpisodePredictions], ...]:
    left = _episodes_by_id(model1, "model1")
    right = _episodes_by_id(model2, "model2")
    if set(left) != set(right):
        only_left = sorted(set(left) - set(right))
        only_right = sorted(set(right) - set(left))
        raise EvaluationError(
            f"models are unpaired; only model1={only_left}, only model2={only_right}"
        )
    pairs: list[tuple[EpisodePredictions, EpisodePredictions]] = []
    for episode_id in sorted(left):
        first, second = left[episode_id], right[episode_id]
        first_scores = _validate_episode(
            first, cadence=cadence, require_failure_step=require_failure_step
        )
        second_scores = _validate_episode(
            second, cadence=cadence, require_failure_step=require_failure_step
        )
        if (
            first.init_id != second.init_id
            or first.label != second.label
            or first.failure_step != second.failure_step
        ):
            raise EvaluationError(f"metadata mismatch for episode {episode_id!r}")
        if tuple(step for step, _ in first_scores) != tuple(
            step for step, _ in second_scores
        ):
            raise EvaluationError(f"score-step mismatch for episode {episode_id!r}")
        pairs.append((first, second))
    return tuple(pairs)


def paired_prediction_comparison(
    model1: Sequence[EpisodePredictions],
    model2: Sequence[EpisodePredictions],
    *,
    primary_steps: Sequence[int] = PRIMARY_STEPS,
    clip: float = PROBABILITY_CLIP,
    bootstrap_replicates: int = 10_000,
    bootstrap_confidence: float = 0.90,
    bootstrap_seed: int = 260_803,
    minimum_relative_lift: float = 0.03,
) -> PairedPredictionResult:
    """Evaluate the preregistered paired M2-minus-M1 prediction estimand."""

    pairs = _paired_episodes(model1, model2)
    model1_losses: list[float] = []
    model2_losses: list[float] = []
    model1_briers: list[float] = []
    model2_briers: list[float] = []
    flat_model1: list[float] = []
    flat_model2: list[float] = []
    flat_labels: list[int] = []
    clusters: list[Hashable] = []

    for first, second in pairs:
        first_selected = _primary_step_scores(
            first.episode_id, dict(_validate_episode(first)), primary_steps
        )
        second_selected = _primary_step_scores(
            second.episode_id, dict(_validate_episode(second)), primary_steps
        )
        if tuple(step for step, _ in first_selected) != tuple(
            step for step, _ in second_selected
        ):
            raise EvaluationError(
                f"available primary steps differ for episode {first.episode_id!r}"
            )
        first_scores = [score for _, score in first_selected]
        second_scores = [score for _, score in second_selected]
        labels = [first.label] * len(first_scores)
        model1_losses.append(binary_log_loss(first_scores, labels, clip=clip))
        model2_losses.append(binary_log_loss(second_scores, labels, clip=clip))
        model1_briers.append(brier_score(first_scores, labels))
        model2_briers.append(brier_score(second_scores, labels))
        flat_model1.extend(first_scores)
        flat_model2.extend(second_scores)
        flat_labels.extend(labels)
        clusters.append(first.init_id)

    first_losses = np.asarray(model1_losses)
    second_losses = np.asarray(model2_losses)
    delta = second_losses - first_losses
    bootstrap = cluster_bootstrap_ci(
        delta,
        clusters,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    model1_mean = float(np.mean(first_losses))
    model2_mean = float(np.mean(second_losses))
    relative_lift = -bootstrap.estimate / model1_mean
    if not math.isfinite(minimum_relative_lift) or minimum_relative_lift < 0.0:
        raise EvaluationError("minimum_relative_lift must be finite and nonnegative")
    claim = bool(
        relative_lift >= minimum_relative_lift and bootstrap.interval.upper < 0.0
    )
    return PairedPredictionResult(
        model1_log_loss=model1_mean,
        model2_log_loss=model2_mean,
        delta_log_loss=bootstrap.estimate,
        relative_lift=relative_lift,
        delta_interval=bootstrap.interval,
        model1_brier=float(np.mean(model1_briers)),
        model2_brier=float(np.mean(model2_briers)),
        model1_auroc=binary_auroc(flat_model1, flat_labels),
        model2_auroc=binary_auroc(flat_model2, flat_labels),
        episodes=len(pairs),
        clusters=bootstrap.clusters,
        primary_claim_succeeds=claim,
    )


def _canonical_metadata(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"manifest metadata is not finite JSON: {exc}") from exc


def validate_locked_test_prediction_coverage(
    expected_manifest: Manifest,
    model1: Sequence[EpisodePredictions],
    model2: Sequence[EpisodePredictions],
) -> None:
    """Require the complete preregistered 20-by-8 Locked Test grid.

    This is intentionally separate from the generic paired estimator: exploratory
    comparisons remain reusable, while any confirmatory claim must pass exact
    manifest, validity, and pairing checks first.
    """

    if not isinstance(expected_manifest, Manifest):
        raise EvaluationError("expected_manifest must be a Manifest")
    if expected_manifest.schema_version != 1:
        raise EvaluationError("Locked Test requires manifest schema_version 1")
    if expected_manifest.split is not SplitName.LOCKED_TEST:
        raise EvaluationError("confirmatory evaluation requires a locked_test manifest")

    expected_by_id: dict[str, Any] = {}
    cells_by_init: dict[Hashable, set[int]] = {}
    for specification in expected_manifest.episodes:
        if specification.split is not SplitName.LOCKED_TEST:
            raise EvaluationError("manifest contains a non-locked_test episode")
        if (
            specification.suite != expected_manifest.task.suite
            or specification.task_id != expected_manifest.task.task_id
            or specification.task_rank != expected_manifest.task.rank
        ):
            raise EvaluationError("manifest episode metadata does not match its task")
        if specification.episode_id in expected_by_id:
            raise EvaluationError("locked_test manifest contains duplicate episode IDs")
        expected_by_id[specification.episode_id] = specification
        cells_by_init.setdefault(specification.base_init_state_id, set()).add(
            specification.condition_index
        )
    if set(cells_by_init) != set(range(30, 50)):
        raise EvaluationError(
            "locked_test manifest init clusters must be exactly IDs 30 through 49"
        )
    expected_cells = set(range(8))
    if any(cells != expected_cells for cells in cells_by_init.values()):
        raise EvaluationError(
            "locked_test manifest cells must be exactly condition indices 0 through 7 "
            "for every init cluster"
        )
    if len(expected_by_id) != 160:
        raise EvaluationError("locked_test manifest must contain exactly 160 episodes")

    indexed_models = (
        _episodes_by_id(model1, "model1"),
        _episodes_by_id(model2, "model2"),
    )
    expected_ids = set(expected_by_id)
    for model_name, indexed in zip(("model1", "model2"), indexed_models, strict=True):
        actual_ids = set(indexed)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise EvaluationError(
                f"{model_name} locked_test coverage mismatch; "
                f"missing={missing}, extra={extra}"
            )
        for episode_id, prediction in indexed.items():
            specification = expected_by_id[episode_id]
            if prediction.valid_reset is not True:
                raise EvaluationError(
                    f"{model_name} episode {episode_id!r} is invalid or lacks validity metadata"
                )
            if prediction.init_id != specification.base_init_state_id:
                raise EvaluationError(
                    f"{model_name} init metadata mismatch for episode {episode_id!r}"
                )
            if not isinstance(prediction.manifest_metadata, Mapping):
                raise EvaluationError(
                    f"{model_name} episode {episode_id!r} lacks manifest metadata"
                )
            if _canonical_metadata(prediction.manifest_metadata) != _canonical_metadata(
                specification.to_dict()
            ):
                raise EvaluationError(
                    f"{model_name} manifest metadata mismatch for episode {episode_id!r}"
                )

    _paired_episodes(model1, model2)


def locked_test_paired_prediction_comparison(
    expected_manifest: Manifest,
    model1: Sequence[EpisodePredictions],
    model2: Sequence[EpisodePredictions],
    **comparison_options: Any,
) -> PairedPredictionResult:
    """Run the primary comparison only after exact Locked Test coverage validation."""

    validate_locked_test_prediction_coverage(expected_manifest, model1, model2)
    return paired_prediction_comparison(model1, model2, **comparison_options)


def first_alarm_time(
    scores: Mapping[int, float],
    threshold: float,
    *,
    consecutive: int = 3,
    cadence: int = 5,
) -> int | None:
    """Return the confirmation step of the first consecutive alarm run.

    Equality counts as an exceedance.  Missing cadence positions are rejected,
    rather than silently treating sparsely sampled predictions as consecutive.
    """

    if not math.isfinite(threshold):
        raise EvaluationError("threshold must be finite")
    if (
        not isinstance(consecutive, Integral)
        or isinstance(consecutive, bool)
        or consecutive < 1
    ):
        raise EvaluationError("consecutive must be a positive integer")
    synthetic = EpisodePredictions("alarm-series", "alarm-cluster", 0, scores)
    ordered = _validate_episode(synthetic, cadence=cadence)
    run = 0
    for step, score in ordered:
        run = run + 1 if score >= threshold else 0
        if run >= consecutive:
            return step
    return None


def calibrate_alarm_threshold(
    episodes: Sequence[EpisodePredictions],
    *,
    max_episode_fpr: float = 0.10,
    consecutive: int = 3,
    cadence: int = 5,
) -> AlarmThresholdResult:
    """Choose the lowest observed threshold satisfying the successful-episode FPR.

    Candidate thresholds are distinct observed scores plus one representable value
    above the maximum.  Thus the rule always has a conservative zero-alarm
    fallback.  Scores tied at the chosen threshold count as exceedances.
    """

    if not math.isfinite(max_episode_fpr) or not 0.0 <= max_episode_fpr <= 1.0:
        raise EvaluationError("max_episode_fpr must lie in [0, 1]")
    if not episodes:
        raise EvaluationError("calibration episodes must not be empty")
    seen: set[str] = set()
    successes: list[EpisodePredictions] = []
    all_scores: list[float] = []
    for episode in episodes:
        ordered = _validate_episode(episode, cadence=cadence)
        if episode.episode_id in seen:
            raise EvaluationError(f"duplicate episode {episode.episode_id!r}")
        seen.add(episode.episode_id)
        all_scores.extend(score for _, score in ordered)
        if episode.label == 0:
            successes.append(episode)
    if not successes:
        raise EvaluationError(
            "alarm calibration requires at least one successful episode"
        )

    observed = np.unique(np.asarray(all_scores, dtype=np.float64))
    sentinel = float(np.nextafter(observed[-1], math.inf))
    candidates = np.concatenate((observed, [sentinel]))
    chosen: tuple[float, int] | None = None
    for threshold in candidates:
        false_positives = sum(
            first_alarm_time(
                episode.scores,
                float(threshold),
                consecutive=consecutive,
                cadence=cadence,
            )
            is not None
            for episode in successes
        )
        fpr = false_positives / len(successes)
        if fpr <= max_episode_fpr:
            chosen = (float(threshold), false_positives)
            break
    if chosen is None:  # pragma: no cover - protected by the above-maximum sentinel
        raise EvaluationError("no threshold satisfies the false-positive-rate cap")
    threshold, false_positives = chosen
    return AlarmThresholdResult(
        threshold=threshold,
        episode_false_positive_rate=false_positives / len(successes),
        false_positive_episodes=false_positives,
        successful_episodes=len(successes),
        candidates_evaluated=len(candidates),
    )


def _lead_for_episode(
    episode: EpisodePredictions,
    threshold: float,
    *,
    consecutive: int,
    cadence: int,
) -> tuple[float, bool]:
    if episode.label != 1 or episode.failure_step is None:
        raise EvaluationError("lead time is defined only for failed episodes")
    alarm = first_alarm_time(
        episode.scores, threshold, consecutive=consecutive, cadence=cadence
    )
    if alarm is None or alarm >= episode.failure_step:
        return 0.0, False
    return float(episode.failure_step - alarm), True


def paired_lead_time_summary(
    model1: Sequence[EpisodePredictions],
    model2: Sequence[EpisodePredictions],
    *,
    model1_threshold: float,
    model2_threshold: float,
    consecutive: int = 3,
    cadence: int = 5,
    bootstrap_replicates: int = 10_000,
    bootstrap_confidence: float = 0.90,
    bootstrap_seed: int = 260_803,
    minimum_median_gain_steps: float = 5.0,
) -> LeadTimeResult:
    """Summarize paired failed-episode lead time and its init-ID cluster CI."""

    pairs = _paired_episodes(
        model1,
        model2,
        require_failure_step=True,
        cadence=cadence,
    )
    failed = [(first, second) for first, second in pairs if first.label == 1]
    if not failed:
        raise EvaluationError(
            "lead-time evaluation requires at least one failed episode"
        )
    leads1: list[float] = []
    leads2: list[float] = []
    detections1: list[bool] = []
    detections2: list[bool] = []
    clusters: list[Hashable] = []
    for first, second in failed:
        lead1, detected1 = _lead_for_episode(
            first, model1_threshold, consecutive=consecutive, cadence=cadence
        )
        lead2, detected2 = _lead_for_episode(
            second, model2_threshold, consecutive=consecutive, cadence=cadence
        )
        leads1.append(lead1)
        leads2.append(lead2)
        detections1.append(detected1)
        detections2.append(detected2)
        clusters.append(first.init_id)
    array1 = np.asarray(leads1)
    array2 = np.asarray(leads2)
    differences = array2 - array1
    bootstrap = cluster_bootstrap_ci(
        differences,
        clusters,
        statistic=np.median,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    if not math.isfinite(minimum_median_gain_steps) or minimum_median_gain_steps < 0:
        raise EvaluationError(
            "minimum_median_gain_steps must be finite and nonnegative"
        )

    positive1 = array1[np.asarray(detections1)]
    positive2 = array2[np.asarray(detections2)]
    claim = bool(
        bootstrap.estimate >= minimum_median_gain_steps
        and bootstrap.interval.lower > 0.0
    )
    return LeadTimeResult(
        model1_median_lead=float(np.median(array1)),
        model2_median_lead=float(np.median(array2)),
        median_paired_difference=bootstrap.estimate,
        paired_difference_interval=bootstrap.interval,
        model1_detection_rate=float(np.mean(detections1)),
        model2_detection_rate=float(np.mean(detections2)),
        model1_conditional_median_lead=(
            float(np.median(positive1)) if positive1.size else None
        ),
        model2_conditional_median_lead=(
            float(np.median(positive2)) if positive2.size else None
        ),
        failed_episodes=len(failed),
        clusters=bootstrap.clusters,
        lead_time_claim_succeeds=claim,
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> RateInterval:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if (
        not isinstance(successes, Integral)
        or isinstance(successes, bool)
        or not isinstance(total, Integral)
        or isinstance(total, bool)
        or total <= 0
        or not 0 <= successes <= total
    ):
        raise EvaluationError(
            "successes and total must satisfy 0 <= successes <= total"
        )
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise EvaluationError("confidence must be strictly between 0 and 1")
    successes = int(successes)
    total = int(total)
    proportion = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    # At the boundaries the Wilson centre and radius are algebraically equal, so
    # the endpoint is exactly 0 or 1.  Computing them separately leaves a
    # rounding residue (~1e-17) whose sign depends on the platform's floating
    # point, so pin the exact values rather than relying on cancellation.
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return RateInterval(
        rate=proportion,
        lower=lower,
        upper=upper,
        confidence=confidence,
        successes=successes,
        total=total,
    )


def reproduction_gate_decision(
    successes: int,
    total: int,
    *,
    minimum_successes: int = 6,
    interval_confidence: float = 0.95,
) -> ReproductionGateResult:
    """Apply the fixed IID reproduction count gate; interval is descriptive."""

    interval = wilson_interval(successes, total, confidence=interval_confidence)
    if (
        not isinstance(minimum_successes, Integral)
        or isinstance(minimum_successes, bool)
        or not 0 <= minimum_successes <= total
    ):
        raise EvaluationError("minimum_successes must lie between zero and total")
    return ReproductionGateResult(
        success_interval=interval,
        minimum_successes=int(minimum_successes),
        passes=successes >= minimum_successes,
    )


def dynamic_range_gate_decision(
    valid_episodes: int,
    total_episodes: int,
    failures_among_valid: int,
    *,
    minimum_valid_fraction: float = 0.90,
    minimum_failure_rate: float = 0.20,
    maximum_failure_rate: float = 0.80,
    interval_confidence: float = 0.95,
) -> DynamicRangeGateResult:
    """Apply fixed perturbation validity/failure point-estimate gates."""

    validity = wilson_interval(
        valid_episodes, total_episodes, confidence=interval_confidence
    )
    failure = wilson_interval(
        failures_among_valid, valid_episodes, confidence=interval_confidence
    )
    rates = (minimum_valid_fraction, minimum_failure_rate, maximum_failure_rate)
    if not all(math.isfinite(value) for value in rates):
        raise EvaluationError("gate-rate bounds must be finite")
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise EvaluationError("minimum_valid_fraction must lie in [0, 1]")
    if not 0.0 <= minimum_failure_rate <= maximum_failure_rate <= 1.0:
        raise EvaluationError("failure-rate bounds must be ordered within [0, 1]")
    passes_validity = validity.rate >= minimum_valid_fraction
    passes_failure = minimum_failure_rate <= failure.rate <= maximum_failure_rate
    return DynamicRangeGateResult(
        validity_interval=validity,
        failure_interval=failure,
        minimum_valid_fraction=minimum_valid_fraction,
        failure_rate_bounds=(minimum_failure_rate, maximum_failure_rate),
        passes_validity=passes_validity,
        passes_failure_range=passes_failure,
        passes=passes_validity and passes_failure,
    )


def black_box_ceiling_triggered(
    model0_auroc: float,
    model1_auroc: float,
    *,
    threshold: float = 0.95,
) -> bool:
    """Return the explicit preregistered Calibration kill-switch flag."""

    values = (model0_auroc, model1_auroc, threshold)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise EvaluationError("AUROCs and threshold must be finite values in [0, 1]")
    return model0_auroc >= threshold or model1_auroc >= threshold
