#!/usr/bin/env python3
"""Calibration alarm-threshold calibration and the Lead-Time secondary question.

The script consumes only the published Calibration feature cohort, the frozen
predictor metadata, and the Calibration failure annotations.  It never opens
Locked Test and never writes into an existing directory.

Per ``AMENDMENTS.md`` (2026-08-05, "Fix the score source and failure step for
alarm calibration") the alarm threshold is calibrated on **group-out-of-fold
calibrated probabilities**, never on in-sample scores from the base model refit
on all Calibration rows, and ``t_failure`` maps to ``onset_step``.

The out-of-fold vectors are not retained by the frozen bundle, so they are
reconstructed deterministically and bound to frozen anchors before use: the
cohort publish digest, an independently pinned ``calibration_data_sha256``, the
recorded ``fold_assignments``, the recorded per-model ``oof_metrics``, and the
frozen Platt map.  Any mismatch aborts before a threshold is computed.

The cohort is loaded through the frozen validated loader so that rows, dtypes,
identifier types, and matrix hashes are exactly the fit-time inputs used by
``calibration_score_18d6494.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from mech_int_vla.evaluation import (
    EpisodePredictions,
    calibrate_alarm_threshold,
    paired_lead_time_summary,
)
from mech_int_vla.feature_artifacts import load_feature_cohort
from mech_int_vla.predictors import (
    FeatureSet,
    _calibration_data_hash,
    _episode_total_one_weights,
    _fit_oof,
    _make_group_folds,
    _sigmoid,
    _weighted_log_loss,
)

SCHEMA_VERSION = 1
MODEL_NAMES = ("M0", "M1", "M2")
CADENCE = 5
CONSECUTIVE = 3
MAX_EPISODE_FPR = 0.10
RELATIVE_TOLERANCE = 1e-9

# Independently pinned so that a regenerated or altered predictors.json cannot
# supply self-consistent expectations for its own verification.
EXPECTED_CALIBRATION_DATA_SHA256 = (
    "215e52cf3c6d50627bdd72834a95610d4b3d9b78118fc4cebc187d57f3488986"
)
EXPECTED_COHORT_SHA256 = (
    "03c37788c579e746ea9c9bd039a3636ca7f851c7694900b53f95e5861b5343ed"
)
EXPECTED_PREDICTOR_METADATA_SHA256 = (
    "acecac7b14856682e6d4b79a6cdba0815c50b76555b7863530351895db3adda7"
)

# PREREG.md:50 defines the internal-signal lead-time claim as M2 versus M1.
PREREGISTERED_CLAIM_PAIR = "M2_vs_M1"


class AlarmCalibrationError(RuntimeError):
    """Raised when a frozen anchor does not reproduce."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlarmCalibrationError(message)


def _close(observed: float, expected: float, label: str) -> None:
    _require(
        bool(np.isclose(observed, expected, rtol=RELATIVE_TOLERANCE, atol=0.0)),
        f"{label} did not reproduce: observed {observed!r}, frozen {expected!r}",
    )


def _verify_folds(folds, frozen_assignments: list[dict[str, Any]]) -> None:
    _require(
        len(folds) == len(frozen_assignments),
        "reconstructed fold count differs from the frozen assignment",
    )
    by_index = {int(entry["fold_index"]): entry for entry in frozen_assignments}
    for fold in folds:
        _require(
            int(fold.index) in by_index,
            f"fold {fold.index} is absent from the frozen assignment",
        )
        frozen = by_index[int(fold.index)]
        _require(
            tuple(fold.validation_groups) == tuple(frozen["validation_groups"]),
            f"fold {fold.index} validation groups differ from the frozen assignment",
        )
        _require(
            int(fold.validation.size) == int(frozen["validation_row_count"]),
            f"fold {fold.index} validation row count differs from the frozen assignment",
        )


def _episode_predictions(
    probabilities: np.ndarray,
    records: tuple,
    failure_steps: dict[str, int],
) -> list[EpisodePredictions]:
    """Group per-row probabilities into per-episode step->score maps."""

    scores: dict[str, dict[int, float]] = {}
    labels: dict[str, int] = {}
    clusters: dict[str, Any] = {}
    for index, record in enumerate(records):
        episode_id = record.episode_id
        step = int(record.control_step)
        bucket = scores.setdefault(episode_id, {})
        _require(step not in bucket, f"duplicate control step {step} in {episode_id}")
        bucket[step] = float(probabilities[index])
        label = int(record.terminal_failure_label)
        if episode_id in labels:
            _require(labels[episode_id] == label, f"inconsistent label in {episode_id}")
        labels[episode_id] = label
        clusters[episode_id] = record.base_init_state_id
    result = []
    for episode_id in sorted(scores):
        label = labels[episode_id]
        failure_step = failure_steps.get(episode_id) if label == 1 else None
        if label == 1:
            _require(
                failure_step is not None,
                f"failed episode {episode_id} has no annotated onset step",
            )
        result.append(
            EpisodePredictions(
                episode_id=episode_id,
                init_id=clusters[episode_id],
                label=label,
                scores=scores[episode_id],
                failure_step=failure_step,
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--failure-annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    feature_root = args.feature_root.resolve()
    output_root = args.output_root.resolve()
    _require(not output_root.exists() and not output_root.is_symlink(),
             "output root must be absent")

    predictor_bytes = (feature_root / "predictors.json").read_bytes()
    _require(
        hashlib.sha256(predictor_bytes).hexdigest()
        == EXPECTED_PREDICTOR_METADATA_SHA256,
        "predictors.json does not match the frozen predictor metadata digest",
    )
    frozen = json.loads(predictor_bytes.decode("utf-8"))

    # Load through the frozen validated loader: this reproduces the exact
    # fit-time inputs, including int base-init identifiers and matrix hashes.
    cohort = load_feature_cohort(
        feature_root / "cohort" / EXPECTED_COHORT_SHA256,
        EXPECTED_COHORT_SHA256,
    )
    records = tuple(cohort.records)
    feature_sets = {
        "M0": FeatureSet(cohort.m0_matrix, cohort.m0_names),
        "M1": FeatureSet(cohort.m1_matrix, cohort.m1_names),
        "M2": FeatureSet(cohort.m2_matrix, cohort.m2_names),
    }
    labels = np.asarray(
        [record.terminal_failure_label for record in records], dtype=np.int8
    )
    episodes = tuple(record.episode_id for record in records)
    groups = tuple(record.base_init_state_id for record in records)

    # --- Anchor: the exact Calibration matrices, labels and identifiers. ---
    observed_hash = _calibration_data_hash(feature_sets, labels, episodes, groups)
    _require(
        observed_hash == EXPECTED_CALIBRATION_DATA_SHA256,
        "calibration data hash differs from the independently pinned digest",
    )
    _require(
        frozen["calibration_data_sha256"] == EXPECTED_CALIBRATION_DATA_SHA256,
        "predictors.json records a different calibration data hash",
    )

    # --- Anchor: the recorded group-fold partition (no RNG involved). ---
    folds = _make_group_folds(groups, episodes)
    _verify_folds(folds, frozen["fold_assignments"])

    weights = _episode_total_one_weights(episodes)
    selected = frozen["selected"]
    family = str(selected["family"])
    hyperparameters = tuple(
        (str(name), value) for name, value in sorted(selected["hyperparameters"].items())
    )
    random_state = int(frozen["random_state"])

    calibrated_probabilities: dict[str, np.ndarray] = {}
    reproduction: dict[str, Any] = {}
    for name in MODEL_NAMES:
        oof = _fit_oof(
            feature_sets[name].values,
            labels,
            weights,
            folds,
            family=family,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )
        model = frozen["models"][name]
        metrics = model["oof_metrics"]

        # --- Anchor: the recorded out-of-fold metrics. ---
        raw_log_loss = _weighted_log_loss(labels, oof.raw_probability, weights)
        raw_auroc = float(
            roc_auc_score(labels, oof.raw_probability, sample_weight=weights)
        )
        _close(raw_log_loss, metrics["raw_log_loss"], f"{name} raw OOF log loss")
        _close(raw_auroc, metrics["raw_auroc"], f"{name} raw OOF AUROC")

        # --- Anchor: the frozen Platt map, applied rather than refitted. ---
        slope = float(model["platt"]["slope"])
        intercept = float(model["platt"]["intercept"])
        _require(slope > 0.0, f"{name} frozen Platt slope is not positive")
        calibrated = _sigmoid(slope * oof.decision_score + intercept)
        calibrated_log_loss = _weighted_log_loss(labels, calibrated, weights)
        _close(
            calibrated_log_loss,
            metrics["calibrated_log_loss"],
            f"{name} calibrated OOF log loss",
        )

        calibrated_probabilities[name] = calibrated
        reproduction[name] = {
            "raw_oof_log_loss": raw_log_loss,
            "raw_oof_auroc": raw_auroc,
            "calibrated_oof_log_loss": calibrated_log_loss,
            "platt_slope": slope,
            "platt_intercept": intercept,
        }

    # --- Failure steps: onset, never confirmation. ---
    annotations = json.loads(args.failure_annotations.read_text(encoding="utf-8"))
    failure_steps: dict[str, int] = {}
    annotated_success: dict[str, bool] = {}
    onset_equals_confirmation = 0
    for event in annotations["events"]:
        annotation = event["annotation"]
        episode_id = str(annotation["episode_id"])
        annotated_success[episode_id] = bool(annotation["success"])
        if not annotation["success"]:
            onset = annotation["onset_step"]
            _require(
                onset is not None, f"failed episode {episode_id} lacks an onset step"
            )
            failure_steps[episode_id] = int(onset)
            if annotation["confirmation_step"] == onset:
                onset_equals_confirmation += 1

    per_model_episodes = {
        name: _episode_predictions(
            calibrated_probabilities[name], records, failure_steps
        )
        for name in MODEL_NAMES
    }

    # Cross-check the annotation source against the cohort labels.
    for episode in per_model_episodes["M1"]:
        _require(
            episode.episode_id in annotated_success,
            f"{episode.episode_id} is missing from the failure annotations",
        )
        _require(
            annotated_success[episode.episode_id] == (episode.label == 0),
            f"annotation success disagrees with the cohort label for {episode.episode_id}",
        )

    thresholds = {
        name: calibrate_alarm_threshold(
            per_model_episodes[name],
            max_episode_fpr=MAX_EPISODE_FPR,
            consecutive=CONSECUTIVE,
            cadence=CADENCE,
        )
        for name in MODEL_NAMES
    }

    lead_times = {}
    for baseline in ("M0", "M1"):
        result = paired_lead_time_summary(
            per_model_episodes[baseline],
            per_model_episodes["M2"],
            model1_threshold=thresholds[baseline].threshold,
            model2_threshold=thresholds["M2"].threshold,
            consecutive=CONSECUTIVE,
            cadence=CADENCE,
        )
        lead_times[f"M2_vs_{baseline}"] = {
            "preregistered_internal_claim": f"M2_vs_{baseline}"
            == PREREGISTERED_CLAIM_PAIR,
            "model1_median_lead": result.model1_median_lead,
            "model2_median_lead": result.model2_median_lead,
            "median_paired_difference": result.median_paired_difference,
            "paired_difference_interval": {
                "lower": result.paired_difference_interval.lower,
                "upper": result.paired_difference_interval.upper,
            },
            "model1_detection_rate": result.model1_detection_rate,
            "model2_detection_rate": result.model2_detection_rate,
            "model1_conditional_median_lead": result.model1_conditional_median_lead,
            "model2_conditional_median_lead": result.model2_conditional_median_lead,
            "failed_episodes": result.failed_episodes,
            "clusters": result.clusters,
            "lead_time_claim_succeeds": result.lead_time_claim_succeeds,
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration_alarm_and_lead_time_receipt",
        "split": "calibration",
        "locked_test_accessed": False,
        "score_source": "group_out_of_fold_calibrated_probabilities",
        "failure_step_source": "annotation.onset_step",
        "preregistered_claim_pair": PREREGISTERED_CLAIM_PAIR,
        "calibration_data_sha256": EXPECTED_CALIBRATION_DATA_SHA256,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "predictor_metadata_sha256": EXPECTED_PREDICTOR_METADATA_SHA256,
        "selected_family": family,
        "selected_hyperparameters": dict(selected["hyperparameters"]),
        "alarm_protocol": {
            "max_episode_fpr": MAX_EPISODE_FPR,
            "consecutive": CONSECUTIVE,
            "cadence": CADENCE,
        },
        "oof_reproduction": reproduction,
        "thresholds": {
            name: {
                "threshold": thresholds[name].threshold,
                "episode_false_positive_rate": thresholds[
                    name
                ].episode_false_positive_rate,
                "false_positive_episodes": thresholds[name].false_positive_episodes,
                "successful_episodes": thresholds[name].successful_episodes,
                "candidates_evaluated": thresholds[name].candidates_evaluated,
            }
            for name in MODEL_NAMES
        },
        "lead_time": lead_times,
        "counts": {
            "episodes": len(per_model_episodes["M1"]),
            "rows": int(labels.size),
            "failed_episodes": sum(
                1 for episode in per_model_episodes["M1"] if episode.label == 1
            ),
            "successful_episodes": sum(
                1 for episode in per_model_episodes["M1"] if episode.label == 0
            ),
            "failures_with_onset_equal_confirmation": onset_equals_confirmation,
        },
    }

    output_root.mkdir(parents=True, exist_ok=False)
    digest = _write_exclusive(
        output_root / "alarm-lead-time-summary.json", _canonical(summary)
    )
    print(json.dumps({"receipt_sha256": digest, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
