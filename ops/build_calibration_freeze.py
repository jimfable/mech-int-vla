#!/usr/bin/env python3
"""Assemble the Calibration freeze that authorizes Locked Test.

``guard.assert_locked_test_ready`` accepts Locked Test only from a committed
freeze file naming the frozen task, variable, policy revision, representation
probe, predictor, alarm thresholds, patch strength, Calibration metrics, and the
byte hashes of four git-tracked artifacts.

Everything written here is *read* from artifacts that already exist; nothing is
chosen at this point.  The out-of-fold probabilities needed for the Brier score
are reconstructed deterministically and bound to the same frozen anchors the
alarm calibration used, so a mismatch aborts rather than silently reporting a
different number than the one the predictors were selected on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from mech_int_vla.evaluation import brier_score
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

MODEL_NAMES = ("M0", "M1", "M2")
RELATIVE_TOLERANCE = 1e-9
EXPECTED_CALIBRATION_DATA_SHA256 = (
    "8713c4cba62b2e7b6dab8c088f1ec8085fcd6ca5ddfab8eef3e407b864d0a4a1"
)
EXPECTED_COHORT_SHA256 = (
    "989f67f8b18dbc7349dc85bc4552cfc0d4c0bbf379b0d2fd5e33ce0ef82446e0"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--alarm", type=Path, required=True)
    parser.add_argument("--alpha-sign", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reality-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    _require(not args.output.exists(), "freeze file already exists")

    frozen = json.loads((args.feature_root / "predictors.json").read_text())
    alarm = json.loads(args.alarm.read_text())
    sign = json.loads(args.alpha_sign.read_text())
    probe = json.loads(args.probe.read_text())
    gate = json.loads(args.reality_gate.read_text())

    # --- reconstruct the OOF probabilities, bound to the frozen anchors ---
    cohort = load_feature_cohort(
        args.feature_root / "cohort" / EXPECTED_COHORT_SHA256, EXPECTED_COHORT_SHA256
    )
    records = tuple(cohort.records)
    feature_sets = {
        "M0": FeatureSet(cohort.m0_matrix, cohort.m0_names),
        "M1": FeatureSet(cohort.m1_matrix, cohort.m1_names),
        "M2": FeatureSet(cohort.m2_matrix, cohort.m2_names),
    }
    labels = np.asarray([r.terminal_failure_label for r in records], dtype=np.int8)
    episodes = tuple(r.episode_id for r in records)
    groups = tuple(r.base_init_state_id for r in records)
    _require(
        _calibration_data_hash(feature_sets, labels, episodes, groups)
        == EXPECTED_CALIBRATION_DATA_SHA256,
        "calibration data hash differs from the frozen digest",
    )
    folds = _make_group_folds(groups, episodes)
    weights = _episode_total_one_weights(episodes)
    selected = frozen["selected"]
    family = str(selected["family"])
    hyperparameters = tuple(
        (str(k), v) for k, v in sorted(selected["hyperparameters"].items())
    )
    seed = int(frozen["random_state"])

    metrics: dict[str, Any] = {}
    for name in MODEL_NAMES:
        oof = _fit_oof(
            feature_sets[name].values, labels, weights, folds,
            family=family, hyperparameters=hyperparameters, random_state=seed,
        )
        recorded = frozen["models"][name]["oof_metrics"]
        raw_ll = _weighted_log_loss(labels, oof.raw_probability, weights)
        _require(
            bool(np.isclose(raw_ll, recorded["raw_log_loss"], rtol=RELATIVE_TOLERANCE)),
            f"{name} raw OOF log loss did not reproduce",
        )
        slope = float(frozen["models"][name]["platt"]["slope"])
        intercept = float(frozen["models"][name]["platt"]["intercept"])
        calibrated = _sigmoid(slope * oof.decision_score + intercept)
        calibrated_ll = _weighted_log_loss(labels, calibrated, weights)
        _require(
            bool(np.isclose(calibrated_ll, recorded["calibrated_log_loss"],
                            rtol=RELATIVE_TOLERANCE)),
            f"{name} calibrated OOF log loss did not reproduce",
        )
        metrics[name.lower()] = {
            "log_loss": float(calibrated_ll),
            "brier": float(brier_score(calibrated, labels)),
            "auroc": float(roc_auc_score(labels, calibrated, sample_weight=weights)),
        }

    coefficient = np.asarray(probe["parameters"]["coefficient"], dtype=np.float64)
    probe_coefficient_hash = hashlib.sha256(
        np.ascontiguousarray(coefficient, dtype="<f8").tobytes()
    ).hexdigest()

    def tracked(path: Path) -> dict[str, str]:
        relative = path.resolve().relative_to(root)
        return {"path": str(relative), "sha256": _sha256_file(path)}

    payload = {
        "schema_version": 1,
        "protocol": "preregistered-calibration-freeze-v1",
        "selected_task": gate["selected_task"],
        "selected_variable": gate["selected_variable"],
        "policy_revision": gate["policy_revision"],
        "representation_probe": {
            "candidate": probe["selection"]["candidate"],
            "ridge_alpha": float(probe["selection"]["ridge_alpha"]),
            "coefficient_hash": probe_coefficient_hash,
        },
        "predictor": {
            "family": family,
            "hyperparameters": dict(selected["hyperparameters"]),
            # Histogram gradient boosting has no coefficient vector, so the
            # frozen predictor metadata digest identifies the fitted model.
            "coefficient_hash": hashlib.sha256(
                (args.feature_root / "predictors.json").read_bytes()
            ).hexdigest(),
        },
        "artifact_hashes": {
            "predictor_bundle": tracked(args.feature_root / "predictors.pkl"),
            "probe": tracked(args.probe),
            "reality_gate_manifest": tracked(args.reality_gate),
            "calibration_manifest": tracked(args.manifest),
        },
        "alarm_thresholds": {
            model.lower(): float(alarm["thresholds"][model]["threshold"])
            for model in MODEL_NAMES
        },
        "patch_strength": float(sign["frozen_alpha"]),
        "calibration_metrics": metrics,
    }
    args.output.write_bytes(_canonical(payload))
    print(json.dumps({"freeze_sha256": _sha256_file(args.output), **payload},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
