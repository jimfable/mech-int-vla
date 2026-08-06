#!/usr/bin/env python3
"""Preregistered sensitivity refit for the coverage-feature fold coupling.

``AMENDMENTS.md`` (2026-08-05, "Record the coverage-feature fold coupling as a
known limitation") commits to reporting how much the Calibration comparison
depends on the features whose reference set uses other episodes' outcome labels.
Those references exclude the query's own episode and base-init group, exactly as
``PREREG.md`` §7 requires, but not the remaining groups inside its
cross-validation fold, so a training row can see outcomes from its own
validation fold.

Three such features sit identically in M1 and M2 and therefore cancel in the
preregistered M2-versus-M1 contrast; one sits only in the M2 increment and does
not cancel, biasing that contrast in M2's favour.  This script refits all three
models with those columns removed and reports the change.

It is a *sensitivity report*, not a replacement estimand: the frozen models,
thresholds and decision rules are untouched.
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

from mech_int_vla.feature_artifacts import load_feature_cohort
from mech_int_vla.predictors import (
    _episode_total_one_weights,
    _fit_estimator,
    _fit_oof,
    _fit_platt,
    _make_group_folds,
    _sigmoid,
    _weighted_log_loss,
)

SCHEMA_VERSION = 1
MODEL_NAMES = ("M0", "M1", "M2")

# Label-informed columns.  The first three are shared by M1 and M2; the fourth is
# M2-only and is the one that can bias the preregistered contrast.
SHARED_COVERAGE = (
    "m1_success_same_phase_five_nn_median_distance",
    "m1_all_phase_nearest_distance",
    "m1_all_phase_25nn_failure_fraction",
)
M2_ONLY_COVERAGE = ("m2_probe_norm_success_same_phase_robust_z_abs",)
DROPPED = SHARED_COVERAGE + M2_ONLY_COVERAGE


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _oof_metrics(values, labels, weights, folds, family, hyperparameters, seed):
    oof = _fit_oof(
        values, labels, weights, folds,
        family=family, hyperparameters=hyperparameters, random_state=seed,
    )
    slope, intercept = _fit_platt(oof.decision_score, labels, weights, random_state=seed)
    calibrated = _sigmoid(slope * oof.decision_score + intercept)
    return {
        "raw_log_loss": _weighted_log_loss(labels, oof.raw_probability, weights),
        "raw_auroc": float(roc_auc_score(labels, oof.raw_probability, sample_weight=weights)),
        "calibrated_log_loss": _weighted_log_loss(labels, calibrated, weights),
        "calibrated_auroc": float(roc_auc_score(labels, calibrated, sample_weight=weights)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    feature_root = args.feature_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError("output root must be absent")

    frozen = json.loads((feature_root / "predictors.json").read_text(encoding="utf-8"))
    selected = frozen["selected"]
    family = str(selected["family"])
    hyperparameters = tuple(
        (str(k), v) for k, v in sorted(selected["hyperparameters"].items())
    )
    seed = int(frozen["random_state"])

    cohort = load_feature_cohort(
        feature_root / "cohort" / args.cohort_sha256, args.cohort_sha256
    )
    records = tuple(cohort.records)
    labels = np.asarray([r.terminal_failure_label for r in records], dtype=np.int8)
    episodes = tuple(r.episode_id for r in records)
    groups = tuple(r.base_init_state_id for r in records)
    weights = _episode_total_one_weights(episodes)
    folds = _make_group_folds(groups, episodes)

    matrices = {"M0": cohort.m0_matrix, "M1": cohort.m1_matrix, "M2": cohort.m2_matrix}
    all_names = {"M0": cohort.m0_names, "M1": cohort.m1_names, "M2": cohort.m2_names}

    full: dict[str, Any] = {}
    reduced: dict[str, Any] = {}
    dropped_per_model: dict[str, list[str]] = {}
    for name in MODEL_NAMES:
        names = list(all_names[name])
        values = np.asarray(matrices[name], dtype=np.float64)
        full[name] = _oof_metrics(
            values, labels, weights, folds, family, hyperparameters, seed
        )
        keep = [i for i, n in enumerate(names) if n not in DROPPED]
        dropped_per_model[name] = [n for n in names if n in DROPPED]
        reduced[name] = _oof_metrics(
            values[:, keep], labels, weights, folds, family, hyperparameters, seed
        )

    def lift(a: str, b: str, table: dict[str, Any]) -> float:
        la = table[a]["calibrated_log_loss"]
        lb = table[b]["calibrated_log_loss"]
        return 100.0 * (la - lb) / la

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration_leakage_sensitivity_report",
        "split": "calibration",
        "locked_test_accessed": False,
        "purpose": "sensitivity only; the frozen models and thresholds are unchanged",
        "cohort_sha256": args.cohort_sha256,
        "selected_family": family,
        "dropped_columns": {"requested": list(DROPPED), "per_model": dropped_per_model},
        "full": full,
        "reduced": reduced,
        "lift_percent": {
            "full": {
                "M2_over_M1": lift("M1", "M2", full),
                "M2_over_M0": lift("M0", "M2", full),
                "M1_over_M0": lift("M0", "M1", full),
            },
            "reduced": {
                "M2_over_M1": lift("M1", "M2", reduced),
                "M2_over_M0": lift("M0", "M2", reduced),
                "M1_over_M0": lift("M0", "M1", reduced),
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=False)
    digest = _write_exclusive(
        output_root / "leakage-sensitivity.json", _canonical(summary)
    )
    print(json.dumps({"receipt_sha256": digest, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
