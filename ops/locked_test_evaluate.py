#!/usr/bin/env python3
"""Fail-closed, preregistered Locked Test evaluation.

Every numerical input is content addressed and all required post-scoring
receipts are validated before the first outcome metric is computed.  Frozen
M0/M1/M2 probabilities are consumed as data; this module deliberately imports
no predictor fitting code and never fits, selects, or calibrates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

# The runbook invokes this file by absolute path from a locked checkout.  Make
# that mode independent of an editable installation or caller PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mech_int_vla.artifacts import RolloutArtifact, load_rollout_artifact
from mech_int_vla.config import SplitName, TaskSpec
from mech_int_vla.evaluation import (
    PRIMARY_STEPS,
    EpisodePredictions,
    EvaluationError,
    binary_log_loss,
    brier_score,
    cluster_bootstrap_ci,
    locked_test_paired_prediction_comparison,
    paired_lead_time_summary,
    validate_locked_test_prediction_coverage,
)
from mech_int_vla.failure_artifacts import failure_event_freeze_from_metadata
from mech_int_vla.failure_events import (
    AnnotationStatus,
    ReachableBounds,
    annotate_failure_event,
    failure_event_trace_from_artifact,
)
from mech_int_vla.manifest import EpisodeSpec, Manifest

SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 260_803
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.90
MODELS = ("M0", "M1", "M2")
PAIRING_SEEDS = (260_803, 260_804, 260_805)
VALIDITY_CATEGORIES = (
    "reject_nan",
    "workspace",
    "speed",
    "penetration",
    "phase_displacement",
)
SECTION_TITLES = (
    "Data-integrity checks",
    "Primary estimand — paired log loss",
    "Brier / AUROC",
    "M2 vs M0",
    "Lead time",
    "Condition rankings",
    "Causal patching",
    "Cost accounting",
    "Sensitivity",
    "Decision-table mapping",
)


class LockedTestEvaluationError(RuntimeError):
    """Raised before publishing when any frozen contract is incomplete."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LockedTestEvaluationError(f"value is not finite canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise LockedTestEvaluationError(f"required JSON input is absent or unsafe: {path}")
    payload = path.read_bytes()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LockedTestEvaluationError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                LockedTestEvaluationError(f"non-finite JSON constant in {path}: {token}")
            ),
        )
    except LockedTestEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedTestEvaluationError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockedTestEvaluationError(f"{path} must contain one JSON object")
    if payload != _canonical(value):
        raise LockedTestEvaluationError(f"{path} is not exact canonical JSON")
    return value, payload


def _load_addressed_json(
    path: Path,
    expected_sha256: str,
    *,
    directory_addressed: bool = False,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise LockedTestEvaluationError("expected input digest must be a lowercase SHA-256")
    value, payload = _strict_json_bytes(path)
    observed = _sha256(payload)
    if observed != expected_sha256:
        raise LockedTestEvaluationError(
            f"content digest mismatch for {path}: expected {expected_sha256}, got {observed}"
        )
    if directory_addressed and path.parent.name != expected_sha256:
        raise LockedTestEvaluationError(
            f"content-addressed directory for {path} is not named {expected_sha256}"
        )
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedTestEvaluationError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise LockedTestEvaluationError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedTestEvaluationError(f"{where} must be an object")
    return value


def _finite(value: Any, where: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool):
        raise LockedTestEvaluationError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LockedTestEvaluationError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (lower is not None and result < lower):
        raise LockedTestEvaluationError(f"{where} is outside its allowed range")
    return result


def _integer(value: Any, where: str, *, lower: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise LockedTestEvaluationError(f"{where} must be an integer >= {lower}")
    return int(value)


def _rate(value: Any, where: str) -> float:
    result = _finite(value, where)
    if not 0.0 <= result <= 1.0:
        raise LockedTestEvaluationError(f"{where} must lie in [0, 1]")
    return result


def _manifest_from_payload(payload: Mapping[str, Any]) -> Manifest:
    _exact_keys(payload, {"schema_version", "split", "task", "episodes"}, "manifest")
    task_payload = _exact_keys(
        payload["task"],
        {"rank", "suite", "task_id", "language", "primary_object", "planar_symmetry_order"},
        "manifest.task",
    )
    try:
        task = TaskSpec(
            rank=task_payload["rank"],
            suite=task_payload["suite"],
            task_id=task_payload["task_id"],
            language=task_payload["language"],
            primary_object=task_payload["primary_object"],
            planar_symmetry_order=task_payload["planar_symmetry_order"],
        )
        raw_episodes = payload["episodes"]
        if not isinstance(raw_episodes, list):
            raise LockedTestEvaluationError("manifest.episodes must be an array")
        episodes: list[EpisodeSpec] = []
        episode_keys = {
            "episode_id", "suite", "task_id", "task_rank", "split",
            "base_init_state_id", "condition_index", "condition_name",
            "condition_family", "condition_parameters", "reset_seed",
            "inference_seed", "policy_revision", "code_commit",
        }
        for index, raw in enumerate(raw_episodes):
            item = _exact_keys(raw, episode_keys, f"manifest.episodes[{index}]")
            episode = EpisodeSpec(
                suite=item["suite"],
                task_id=item["task_id"],
                task_rank=item["task_rank"],
                split=SplitName(item["split"]),
                base_init_state_id=item["base_init_state_id"],
                condition_index=item["condition_index"],
                condition_name=item["condition_name"],
                condition_family=item["condition_family"],
                condition_parameters=item["condition_parameters"],
                reset_seed=item["reset_seed"],
                inference_seed=item["inference_seed"],
                policy_revision=item["policy_revision"],
                code_commit=item["code_commit"],
            )
            if item["episode_id"] != episode.episode_id:
                raise LockedTestEvaluationError(
                    f"manifest episode ID is not derivable at index {index}"
                )
            episodes.append(episode)
        manifest = Manifest(
            schema_version=payload["schema_version"],
            split=SplitName(payload["split"]),
            task=task,
            episodes=tuple(episodes),
        )
    except LockedTestEvaluationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LockedTestEvaluationError(f"malformed Locked Test manifest: {exc}") from exc
    if _canonical(manifest.to_dict()) != _canonical(payload):
        raise LockedTestEvaluationError("manifest differs from its validated reconstruction")
    # Reuse the library's complete 20 x 8 topology checks.  Empty prediction
    # validation is not useful here, so enforce the manifest half directly.
    if (
        manifest.schema_version != 1
        or manifest.split is not SplitName.LOCKED_TEST
        or len(manifest.episodes) != 160
    ):
        raise LockedTestEvaluationError("manifest is not the schema-1 160-episode Locked Test")
    grid = {(item.base_init_state_id, item.condition_index) for item in manifest.episodes}
    expected_grid = {(init_id, cell) for init_id in range(30, 50) for cell in range(8)}
    if grid != expected_grid:
        raise LockedTestEvaluationError("manifest is not the exact 20 x 8 Locked Test grid")
    return manifest


def _validate_prediction_receipt(
    payload: Mapping[str, Any], manifest_sha256: str, receipt_sha256: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_frozen_predictions":
        raise LockedTestEvaluationError("prediction receipt has the wrong schema or kind")
    source = _mapping(payload.get("source"), "predictions.source")
    required_hashes = {
        "manifest_sha256", "bound_probe_sha256", "score_allocation_sha256",
        "feature_cohort_sha256", "reference_bundle_sha256",
        "predictor_bundle_sha256", "predictor_metadata_sha256",
        "calibration_data_sha256", "calibration_freeze_sha256",
    }
    for key in required_hashes:
        if not _is_sha256(source.get(key)):
            raise LockedTestEvaluationError(f"predictions.source.{key} is not a SHA-256")
    if source["manifest_sha256"] != manifest_sha256:
        raise LockedTestEvaluationError("prediction receipt is bound to another manifest")
    if source.get("label_source") != (
        "feature_cohort_terminal_outcome_joined_only_during_prediction_serialization"
    ):
        raise LockedTestEvaluationError(
            "prediction receipt does not prove serialization-only label join"
        )
    if source.get("prediction_rule") != (
        "frozen_all_calibration_predictor_applied_without_label_argument"
    ):
        raise LockedTestEvaluationError(
            "prediction receipt does not prove label-free frozen predictor invocation"
        )
    records = payload.get("records")
    invalid = payload.get("invalid_resets")
    counts = _mapping(payload.get("counts"), "predictions.counts")
    if not isinstance(records, list) or not isinstance(invalid, list):
        raise LockedTestEvaluationError("prediction records and invalid resets must be arrays")
    if _integer(counts.get("attempted_episodes"), "counts.attempted_episodes") != 160:
        raise LockedTestEvaluationError("prediction receipt does not cover 160 manifest episodes")
    if _integer(counts.get("valid_episodes"), "counts.valid_episodes") != 160 - len(invalid):
        raise LockedTestEvaluationError("valid episode count disagrees with invalid resets")
    if _integer(counts.get("invalid_resets"), "counts.invalid_resets") != len(invalid):
        raise LockedTestEvaluationError("invalid reset count disagrees with its records")
    if _integer(counts.get("state_rows"), "counts.state_rows") != len(records):
        raise LockedTestEvaluationError("prediction row count disagrees with records")
    identities: list[tuple[str, int]] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, f"predictions.records[{index}]")
        if set(record) != {
            "episode_id", "base_init_state_id", "control_step",
            "terminal_failure_label", "source_hashes", "probabilities",
        }:
            raise LockedTestEvaluationError(f"prediction record {index} has a non-frozen schema")
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise LockedTestEvaluationError("prediction episode_id must be nonempty")
        step = _integer(record.get("control_step"), "prediction.control_step")
        if step % 5:
            raise LockedTestEvaluationError("prediction steps must use the 5-step cadence")
        if type(record.get("terminal_failure_label")) is not bool:
            raise LockedTestEvaluationError("prediction labels must be boolean")
        hashes = _exact_keys(
            record.get("source_hashes"),
            {"episode_id", "raw_metadata_sha256", "raw_trajectory_sha256",
             "score_metadata_sha256", "score_primitives_sha256"},
            "prediction.source_hashes",
        )
        if hashes["episode_id"] != episode_id or any(
            not _is_sha256(hashes[key]) for key in hashes if key != "episode_id"
        ):
            raise LockedTestEvaluationError("prediction source hashes are malformed")
        probabilities = _exact_keys(
            record.get("probabilities"), set(MODELS), "prediction.probabilities"
        )
        for name in MODELS:
            probability = _finite(probabilities[name], f"prediction.{name}")
            if not 0.0 <= probability <= 1.0:
                raise LockedTestEvaluationError("predictions must lie in [0, 1]")
        identities.append((episode_id, step))
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise LockedTestEvaluationError("prediction rows must be unique and sorted")
    invalid_ids: list[str] = []
    for index, raw in enumerate(invalid):
        item = _exact_keys(
            raw,
            {"episode_id", "base_init_state_id", "condition_index"},
            f"predictions.invalid_resets[{index}]",
        )
        if not isinstance(item["episode_id"], str):
            raise LockedTestEvaluationError("invalid reset episode ID must be a string")
        _integer(item["base_init_state_id"], "invalid base_init_state_id")
        _integer(item["condition_index"], "invalid condition_index")
        invalid_ids.append(item["episode_id"])
    if invalid_ids != sorted(invalid_ids) or len(invalid_ids) != len(set(invalid_ids)):
        raise LockedTestEvaluationError("invalid reset records must be unique and sorted")
    if any(episode_id in set(invalid_ids) for episode_id, _ in identities):
        raise LockedTestEvaluationError("invalid resets must not have prediction rows")
    if not _is_sha256(receipt_sha256):  # documents the binding used below
        raise LockedTestEvaluationError("prediction receipt digest is malformed")


def _validate_calibration_freeze(
    payload: Mapping[str, Any],
    prediction_source: Mapping[str, Any],
    *,
    freeze_sha256: str,
    reality_gate_sha256: str,
) -> dict[str, float]:
    if payload.get("schema_version") != 1 or payload.get("protocol") != "preregistered-calibration-freeze-v1":
        raise LockedTestEvaluationError("calibration freeze has the wrong schema or protocol")
    hashes = _mapping(payload.get("artifact_hashes"), "calibration_freeze.artifact_hashes")
    if prediction_source.get("calibration_freeze_sha256") != freeze_sha256:
        raise LockedTestEvaluationError("predictions are bound to another Calibration freeze")
    frozen_bindings = {
        "bound_probe": "bound_probe_sha256",
        "predictor_bundle": "predictor_bundle_sha256",
        "predictor_metadata": "predictor_metadata_sha256",
    }
    for artifact_name, source_name in frozen_bindings.items():
        artifact = _mapping(hashes.get(artifact_name), artifact_name)
        if artifact.get("sha256") != prediction_source.get(source_name):
            raise LockedTestEvaluationError(
                f"predictions do not use frozen artifact {artifact_name}"
            )
    reference = _mapping(
        hashes.get("feature_reference_metadata"), "feature_reference_metadata"
    )
    reference_path = reference.get("path")
    if (
        not isinstance(reference_path, str)
        or Path(reference_path).parent.name
        != prediction_source.get("reference_bundle_sha256")
    ):
        raise LockedTestEvaluationError(
            "predictions do not use the frozen Calibration reference bundle"
        )
    reality = _mapping(hashes.get("reality_gate_manifest"), "reality_gate_manifest")
    if reality.get("sha256") != reality_gate_sha256:
        raise LockedTestEvaluationError("supplied Reality-Gate lock differs from Calibration freeze")
    thresholds = _exact_keys(payload.get("alarm_thresholds"), {"m0", "m1", "m2"}, "alarm_thresholds")
    result: dict[str, float] = {}
    for lower_name, model in zip(("m0", "m1", "m2"), MODELS, strict=True):
        value = _finite(thresholds[lower_name], f"alarm_thresholds.{lower_name}")
        if not 0.0 <= value <= 1.0:
            raise LockedTestEvaluationError("frozen alarm threshold must lie in [0, 1]")
        result[model] = value
    if _finite(payload.get("patch_strength"), "patch_strength") != 0.25:
        raise LockedTestEvaluationError("calibration freeze does not contain alpha 0.25")
    return result


def _validate_causal_receipt(
    payload: Mapping[str, Any], manifest_sha: str, prediction_sha: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_causal_patching_receipt":
        raise LockedTestEvaluationError("causal patching receipt has the wrong schema or kind")
    source = _mapping(payload.get("source"), "causal.source")
    if source.get("manifest_sha256") != manifest_sha or source.get("prediction_receipt_sha256") != prediction_sha:
        raise LockedTestEvaluationError("causal patching receipt is bound to different inputs")
    if _finite(source.get("alpha"), "causal.source.alpha") != 0.25:
        raise LockedTestEvaluationError("causal receipt did not use frozen alpha 0.25")
    if source.get("pairing_seeds") != list(PAIRING_SEEDS):
        raise LockedTestEvaluationError("causal receipt did not use all three pairing seeds")
    if _integer(source.get("random_subspaces_per_pair"), "random_subspaces_per_pair", lower=1) != 1000:
        raise LockedTestEvaluationError("causal receipt did not use 1000 random subspaces")
    if _finite(source.get("matched_donor_max_degrees"), "matched_donor_max_degrees") != 5.0:
        raise LockedTestEvaluationError("causal receipt did not use the <5 degree control")
    if source.get("matched_donor_rule") != "orientation_difference_degrees < 5":
        raise LockedTestEvaluationError("causal receipt did not use a strict <5 degree rule")
    result = _mapping(payload.get("confirmatory"), "causal.confirmatory")
    attempted = _integer(result.get("attempted_pairs"), "attempted_pairs")
    valid = _integer(result.get("valid_pairs"), "valid_pairs")
    if attempted != 60 or valid > attempted:
        raise LockedTestEvaluationError("causal receipt must account for the fixed 60-pair set")
    expected_status = "inconclusive" if valid < 30 else "complete"
    if result.get("status") != expected_status:
        raise LockedTestEvaluationError("causal status disagrees with the 30-pair minimum")
    if valid:
        sign_count = _integer(result.get("sign_correct_count"), "sign_correct_count")
        sign_rate = _rate(result.get("sign_correct_rate"), "sign_correct_rate")
        if sign_count > valid or not math.isclose(sign_rate, sign_count / valid, abs_tol=1e-12):
            raise LockedTestEvaluationError("causal sign rate disagrees with pair counts")
        interval = _mapping(result.get("sign_interval"), "causal.sign_interval")
        if _finite(interval.get("confidence"), "sign_interval.confidence") != 0.90:
            raise LockedTestEvaluationError("causal sign interval is not 90%")
        lower = _finite(interval.get("lower"), "sign_interval.lower")
        upper = _finite(interval.get("upper"), "sign_interval.upper")
        if not 0 <= lower <= upper <= 1:
            raise LockedTestEvaluationError("causal sign interval is malformed")
        ratio = _finite(result.get("median_off_target_ratio"), "median_off_target_ratio", lower=0)
        if type(result.get("random_control_passes")) is not bool:
            raise LockedTestEvaluationError("causal random-control result must be boolean")
        _rate(result.get("off_manifold_rate"), "off_manifold_rate")
        _rate(result.get("matched_donor_control_rate"), "matched_donor_control_rate")
        if type(result.get("specificity_passes")) is not bool or result["specificity_passes"] != (ratio <= 0.25):
            raise LockedTestEvaluationError("causal specificity result is inconsistent")
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, list) or [item.get("seed") for item in per_seed if isinstance(item, Mapping)] != list(PAIRING_SEEDS):
        raise LockedTestEvaluationError("causal receipt lacks ordered three-seed results")


def _validate_sensitivity_receipt(
    payload: Mapping[str, Any], manifest_sha: str, prediction_sha: str, causal_sha: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_sensitivity_receipt":
        raise LockedTestEvaluationError("sensitivity receipt has the wrong schema or kind")
    source = _mapping(payload.get("source"), "sensitivity.source")
    expected = {
        "manifest_sha256": manifest_sha,
        "prediction_receipt_sha256": prediction_sha,
        "causal_receipt_sha256": causal_sha,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise LockedTestEvaluationError("sensitivity receipt is bound to different inputs")
    diagnostics = _mapping(payload.get("rollout_diagnostics"), "rollout_diagnostics")
    if _integer(diagnostics.get("episode_count"), "diagnostics.episode_count") != 160:
        raise LockedTestEvaluationError("rollout diagnostics do not cover 160 episodes")
    by_cell = diagnostics.get("by_cell")
    if not isinstance(by_cell, list) or [item.get("condition_index") for item in by_cell if isinstance(item, Mapping)] != list(range(8)):
        raise LockedTestEvaluationError("rollout diagnostics must cover cells 0 through 7")
    _rate(
        diagnostics.get("nearest_object_identity_accuracy"),
        "rollout_diagnostics.nearest_object_identity_accuracy",
    )
    for name in ("mean_identity_error", "mean_identity_distance"):
        _finite(diagnostics.get(name), f"rollout_diagnostics.{name}", lower=0)
    dose = payload.get("dose_by_difficulty")
    if not isinstance(dose, list):
        raise LockedTestEvaluationError("dose_by_difficulty must be an array")
    dose_grid = []
    for index, raw in enumerate(dose):
        item = _mapping(raw, f"dose_by_difficulty[{index}]")
        alpha = _finite(item.get("alpha"), "dose alpha")
        cell = _integer(item.get("condition_index"), "dose condition_index")
        _rate(item.get("sign_correct_rate"), "dose sign_correct_rate")
        _finite(item.get("median_off_target_ratio"), "dose median_off_target_ratio", lower=0)
        dose_grid.append((alpha, cell))
    if dose_grid != [(alpha, cell) for alpha in (0.5, 1.0) for cell in range(8)]:
        raise LockedTestEvaluationError("dose sensitivity is not the ordered 2 x 8 grid")
    ledger = _mapping(payload.get("broken_successes"), "broken_successes")
    if _integer(ledger.get("total_pairs"), "broken_successes.total_pairs") != 60:
        raise LockedTestEvaluationError("broken-success ledger is not bound to all 60 pairs")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != 60:
        raise LockedTestEvaluationError("broken-success ledger must contain exactly 60 rows")


def _validate_cost_receipt(
    payload: Mapping[str, Any], manifest_sha: str, prediction_sha: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_cost_receipt":
        raise LockedTestEvaluationError("cost receipt has the wrong schema or kind")
    source = _mapping(payload.get("source"), "cost.source")
    if source.get("manifest_sha256") != manifest_sha or source.get("prediction_receipt_sha256") != prediction_sha:
        raise LockedTestEvaluationError("cost receipt is bound to different inputs")
    stages = payload.get("stages")
    expected_names = ["collection", "scoring", "evaluation", "causal_patching", "sensitivity"]
    if not isinstance(stages, list) or [item.get("name") for item in stages if isinstance(item, Mapping)] != expected_names:
        raise LockedTestEvaluationError("cost stages are absent or out of frozen order")
    for index, stage in enumerate(stages):
        for field in ("wall_seconds", "gpu_hours", "instance_charges"):
            _finite(stage.get(field), f"cost.stages[{index}].{field}", lower=0)
    stops = payload.get("budget_gate_stops")
    if not isinstance(stops, list) or any(not isinstance(item, str) for item in stops):
        raise LockedTestEvaluationError("budget_gate_stops must be a string array")


def _failure_bounds(reality_gate: Mapping[str, Any], task: TaskSpec) -> ReachableBounds:
    freeze_payload = reality_gate.get("failure_event_freeze")
    if not isinstance(freeze_payload, Mapping):
        raise LockedTestEvaluationError("Reality-Gate lock lacks failure_event_freeze")
    try:
        freeze = failure_event_freeze_from_metadata(freeze_payload)
    except ValueError as exc:
        raise LockedTestEvaluationError(f"invalid frozen failure-event definition: {exc}") from exc
    if (
        freeze.task.suite != task.suite
        or freeze.task.task_id != task.task_id
        or freeze.task.task_rank != task.rank
    ):
        raise LockedTestEvaluationError("failure-event freeze is for another task")
    return freeze.bounds.expanded_bounds


def _raw_validity_categories(artifact: RolloutArtifact) -> dict[str, bool]:
    validity = artifact.metadata["validity"]
    reasons = tuple(str(value).lower() for value in validity["reasons"])
    joined = " ".join(reasons)
    return {
        "reject_nan": not bool(validity["finite"]) or "finite" in joined or "nan" in joined,
        "workspace": not bool(validity["in_workspace"]) or "workspace" in joined,
        "speed": "speed" in joined or "linear" in joined or "angular" in joined,
        "penetration": "penetration" in joined,
        "phase_displacement": "phase" in joined or "displacement" in joined,
    }


def _load_raw_inputs(
    manifest: Manifest, raw_root: Path, bounds: ReachableBounds
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    split_root = raw_root / SplitName.LOCKED_TEST.value
    if not split_root.is_dir() or split_root.is_symlink():
        raise LockedTestEvaluationError("Locked Test raw root is absent or unsafe")
    expected = {item.episode_id: item for item in manifest.episodes}
    observed = {item.name for item in split_root.iterdir() if item.is_dir() and not item.is_symlink()}
    if observed != set(expected):
        raise LockedTestEvaluationError(
            f"raw artifact count/topology mismatch: missing={sorted(set(expected)-observed)}, "
            f"extra={sorted(observed-set(expected))}"
        )
    info: dict[str, dict[str, Any]] = {}
    cell_integrity = {
        cell: {name: 0 for name in VALIDITY_CATEGORIES} | {"invalid_resets": 0}
        for cell in range(8)
    }
    for episode_id in sorted(expected):
        specification = expected[episode_id]
        artifact = load_rollout_artifact(split_root / episode_id, expected_task=manifest.task)
        if _canonical(artifact.metadata["episode"]) != _canonical(specification.to_dict()):
            raise LockedTestEvaluationError(f"raw manifest provenance mismatch for {episode_id}")
        categories = _raw_validity_categories(artifact)
        if not artifact.valid_reset:
            cell_integrity[specification.condition_index]["invalid_resets"] += 1
        for name, violated in categories.items():
            cell_integrity[specification.condition_index][name] += int(violated)
        # Invalid resets remain part of the exact 160-artifact integrity audit,
        # but never enter outcome annotation or any estimand.
        annotation = None
        if artifact.valid_reset:
            annotation = annotate_failure_event(
                failure_event_trace_from_artifact(artifact), bounds
            )
            if annotation.status is AnnotationStatus.UNANNOTATABLE_EARLY_TERMINAL:
                raise LockedTestEvaluationError(
                    f"failed episode {episode_id} has no frozen failure onset"
                )
        info[episode_id] = {
            "valid_reset": artifact.valid_reset,
            "label": int(not artifact.success),
            "failure_step": None if annotation is None else annotation.onset_step,
            "metadata_sha256": artifact.hashes.metadata_sha256,
            "trajectory_sha256": artifact.hashes.trajectory_sha256,
            "categories": categories,
        }
        del artifact
    cells: list[dict[str, Any]] = []
    for cell in range(8):
        counts = cell_integrity[cell]
        rates = {name: counts[name] / 20.0 for name in VALIDITY_CATEGORIES}
        invalid_rate = counts["invalid_resets"] / 20.0
        if invalid_rate > 0.10 or any(rate > 0.10 for rate in rates.values()):
            raise LockedTestEvaluationError(
                f"condition cell {cell} exceeds the 10% validity envelope"
            )
        cells.append(
            {
                "condition_index": cell,
                "episodes": 20,
                "invalid_resets": counts["invalid_resets"],
                "invalid_reset_rate": invalid_rate,
                "envelope_violation_counts": {name: counts[name] for name in VALIDITY_CATEGORIES},
                "envelope_violation_rates": rates,
                "valid": True,
            }
        )
    return info, cells


def _prediction_episodes(
    payload: Mapping[str, Any], manifest: Manifest, raw: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, list[EpisodePredictions]], tuple[str, ...]]:
    expected = {item.episode_id: item for item in manifest.episodes}
    invalid_records = payload["invalid_resets"]
    invalid_ids = tuple(item["episode_id"] for item in invalid_records)
    actual_invalid = tuple(sorted(episode_id for episode_id, item in raw.items() if not item["valid_reset"]))
    if invalid_ids != actual_invalid:
        raise LockedTestEvaluationError("prediction invalid-reset inventory differs from raw artifacts")
    for item in invalid_records:
        specification = expected[item["episode_id"]]
        if item["base_init_state_id"] != specification.base_init_state_id or item["condition_index"] != specification.condition_index:
            raise LockedTestEvaluationError("invalid-reset manifest metadata mismatch")
    grouped: dict[str, dict[str, Any]] = {}
    invalid_set = set(invalid_ids)
    for record in payload["records"]:
        episode_id = record["episode_id"]
        if episode_id not in expected or episode_id in invalid_set:
            raise LockedTestEvaluationError("prediction record is outside valid manifest membership")
        specification = expected[episode_id]
        source = record["source_hashes"]
        raw_item = raw[episode_id]
        if (
            record["base_init_state_id"] != specification.base_init_state_id
            or record["terminal_failure_label"] != bool(raw_item["label"])
            or source["raw_metadata_sha256"] != raw_item["metadata_sha256"]
            or source["raw_trajectory_sha256"] != raw_item["trajectory_sha256"]
        ):
            raise LockedTestEvaluationError(f"prediction/raw identity mismatch for {episode_id}")
        group = grouped.setdefault(
            episode_id,
            {"label": raw_item["label"], "init": specification.base_init_state_id, "scores": {name: {} for name in MODELS}},
        )
        for name in MODELS:
            group["scores"][name][record["control_step"]] = float(record["probabilities"][name])
    valid_ids = set(expected) - set(invalid_ids)
    if set(grouped) != valid_ids:
        raise LockedTestEvaluationError("prediction episode coverage differs from valid raw set")
    result = {name: [] for name in MODELS}
    for episode_id in sorted(grouped):
        group = grouped[episode_id]
        steps = sorted(group["scores"]["M0"])
        if not steps or steps[0] != 0 or any(
            right - left != 5 for left, right in pairwise(steps)
        ):
            raise LockedTestEvaluationError(f"prediction cadence is incomplete for {episode_id}")
        if any(sorted(group["scores"][name]) != steps for name in MODELS):
            raise LockedTestEvaluationError(f"model score steps differ for {episode_id}")
        metadata = expected[episode_id].to_dict()
        for name in MODELS:
            result[name].append(
                EpisodePredictions(
                    episode_id=episode_id,
                    init_id=group["init"],
                    label=group["label"],
                    scores=group["scores"][name],
                    failure_step=raw[episode_id]["failure_step"],
                    manifest_metadata=metadata,
                    valid_reset=True,
                )
            )
    validate_locked_test_prediction_coverage(
        manifest, result["M1"], result["M2"], invalid_episode_ids=invalid_ids
    )
    return result, invalid_ids


def _primary_rows(episodes: Sequence[EpisodePredictions]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    labels: list[int] = []
    weights: list[float] = []
    clusters: list[int] = []
    for episode in episodes:
        selected = [episode.scores[step] for step in PRIMARY_STEPS if step in episode.scores]
        if not selected:
            raise LockedTestEvaluationError(f"episode {episode.episode_id} lacks primary states")
        probabilities.extend(selected)
        labels.extend([episode.label] * len(selected))
        weights.extend([1.0 / len(selected)] * len(selected))
        clusters.extend([int(episode.init_id)] * len(selected))
    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
        np.asarray(clusters, dtype=np.int64),
    )


def _weighted_auroc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    positive_weight = float(np.sum(weights[positives]))
    negative_weight = float(np.sum(weights[negatives]))
    if positive_weight <= 0 or negative_weight <= 0:
        raise EvaluationError("AUROC requires both outcome classes")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    concordance = 0.0
    negative_below = 0.0
    index = 0
    while index < len(order):
        stop = index + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[index]:
            stop += 1
        tied_positive = float(np.sum(ordered_weights[index:stop][ordered_labels[index:stop] == 1]))
        tied_negative = float(np.sum(ordered_weights[index:stop][ordered_labels[index:stop] == 0]))
        concordance += tied_positive * (negative_below + 0.5 * tied_negative)
        negative_below += tied_negative
        index = stop
    return concordance / (positive_weight * negative_weight)


def _model_metric_summary(
    episodes: Sequence[EpisodePredictions], *, replicates: int
) -> dict[str, Any]:
    scores, labels, weights, clusters = _primary_rows(episodes)
    episode_ids = [episode.episode_id for episode in episodes]
    losses = np.asarray([
        binary_log_loss(
            [episode.scores[step] for step in PRIMARY_STEPS if step in episode.scores],
            [episode.label] * sum(step in episode.scores for step in PRIMARY_STEPS),
        )
        for episode in episodes
    ])
    briers = np.asarray([
        brier_score(
            [episode.scores[step] for step in PRIMARY_STEPS if step in episode.scores],
            [episode.label] * sum(step in episode.scores for step in PRIMARY_STEPS),
        )
        for episode in episodes
    ])
    episode_clusters = np.asarray([episode.init_id for episode in episodes], dtype=object)
    log_boot = cluster_bootstrap_ci(losses, episode_clusters, replicates=replicates, confidence=0.90, seed=BOOTSTRAP_SEED)
    brier_boot = cluster_bootstrap_ci(briers, episode_clusters, replicates=replicates, confidence=0.90, seed=BOOTSTRAP_SEED)

    unique_clusters = list(dict.fromkeys(clusters.tolist()))
    rows_by_cluster = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    auroc_draws = np.empty(replicates, dtype=np.float64)
    completed = 0
    attempted = 0
    # Whole-cluster draws can occasionally contain just one outcome class,
    # for which AUROC is undefined.  Keep drawing from the same deterministic
    # stream until the preregistered number of *defined* cluster replicates is
    # reached, and report the attempt count for auditability.
    while completed < replicates:
        attempted += 1
        if attempted > replicates * 100:
            raise LockedTestEvaluationError(
                "cluster bootstrap could not produce enough two-class AUROC draws"
            )
        selected = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        indices = np.concatenate([rows_by_cluster[index] for index in selected])
        if np.unique(labels[indices]).size < 2:
            continue
        auroc_draws[completed] = _weighted_auroc(
            scores[indices], labels[indices], weights[indices]
        )
        completed += 1
    lower, upper = np.quantile(auroc_draws, [0.05, 0.95])
    return {
        "episodes": len(episode_ids),
        "episode_total_sample_weight": 1.0,
        "log_loss": log_boot.estimate,
        "log_loss_interval": {"lower": log_boot.interval.lower, "upper": log_boot.interval.upper, "confidence": 0.90},
        "brier": brier_boot.estimate,
        "brier_interval": {"lower": brier_boot.interval.lower, "upper": brier_boot.interval.upper, "confidence": 0.90},
        "auroc": _weighted_auroc(scores, labels, weights),
        "auroc_interval": {"lower": float(lower), "upper": float(upper), "confidence": 0.90},
        "bootstrap": {
            "clusters": len(unique_clusters), "replicates": replicates,
            "auroc_draws_attempted": attempted, "seed": BOOTSTRAP_SEED,
        },
    }


def _condition_rankings(
    manifest: Manifest, predictions: Mapping[str, Sequence[EpisodePredictions]]
) -> list[dict[str, Any]]:
    cell_by_episode = {item.episode_id: item.condition_index for item in manifest.episodes}
    name_by_cell = {item.condition_index: item.condition_name for item in manifest.episodes}
    rows: list[dict[str, Any]] = []
    for cell in range(8):
        model_metrics: dict[str, Any] = {}
        for name in MODELS:
            subset = [episode for episode in predictions[name] if cell_by_episode[episode.episode_id] == cell]
            flat_scores, flat_labels, flat_weights, _ = _primary_rows(subset)
            losses = [
                binary_log_loss(
                    [episode.scores[step] for step in PRIMARY_STEPS if step in episode.scores],
                    [episode.label] * sum(step in episode.scores for step in PRIMARY_STEPS),
                )
                for episode in subset
            ]
            try:
                auroc: float | None = _weighted_auroc(flat_scores, flat_labels, flat_weights)
                status = "estimated"
            except EvaluationError:
                auroc = None
                status = "single_outcome_class"
            model_metrics[name] = {
                "episodes": len(subset), "log_loss": float(np.mean(losses)),
                "auroc": auroc, "auroc_status": status,
            }
        rows.append({"condition_index": cell, "condition_name": name_by_cell[cell], "models": model_metrics})
    for name in MODELS:
        log_order = sorted(range(8), key=lambda cell: (rows[cell]["models"][name]["log_loss"], cell))
        auroc_order = sorted(
            range(8),
            key=lambda cell: (
                rows[cell]["models"][name]["auroc"] is None,
                -(rows[cell]["models"][name]["auroc"] or 0.0),
                cell,
            ),
        )
        for rank, cell in enumerate(log_order, 1):
            rows[cell]["models"][name]["log_loss_rank"] = rank
        for rank, cell in enumerate(auroc_order, 1):
            rows[cell]["models"][name]["auroc_rank"] = rank
    return rows


def _plain_dataclass(value: Any) -> Any:
    return asdict(value)


def _cost_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    stages = payload["stages"]
    return {
        "wall_seconds": float(sum(float(item["wall_seconds"]) for item in stages)),
        "gpu_hours": float(sum(float(item["gpu_hours"]) for item in stages)),
        "instance_charges": float(
            sum(float(item["instance_charges"]) for item in stages)
        ),
        "budget_gate_stops": list(payload["budget_gate_stops"]),
    }


def build_report(
    *,
    manifest: Manifest,
    manifest_sha256: str,
    raw: Mapping[str, Mapping[str, Any]],
    integrity_cells: list[dict[str, Any]],
    prediction_payload: Mapping[str, Any],
    prediction_sha256: str,
    causal_payload: Mapping[str, Any],
    causal_sha256: str,
    sensitivity_payload: Mapping[str, Any],
    sensitivity_sha256: str,
    cost_payload: Mapping[str, Any],
    cost_sha256: str,
    thresholds: Mapping[str, float],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    predictions, invalid_ids = _prediction_episodes(prediction_payload, manifest, raw)
    primary = locked_test_paired_prediction_comparison(
        manifest,
        predictions["M1"],
        predictions["M2"],
        invalid_episode_ids=invalid_ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
    )
    model_metrics = {
        name: _model_metric_summary(predictions[name], replicates=bootstrap_replicates)
        for name in MODELS
    }
    m2_m0 = locked_test_paired_prediction_comparison(
        manifest,
        predictions["M0"],
        predictions["M2"],
        invalid_episode_ids=invalid_ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
    )
    lead = paired_lead_time_summary(
        predictions["M1"], predictions["M2"],
        model1_threshold=thresholds["M1"], model2_threshold=thresholds["M2"],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
    )
    sections = [
        {
            "number": 1, "title": SECTION_TITLES[0],
            "artifact_count": 160, "manifest_episodes": 160,
            "valid_episodes": 160 - len(invalid_ids), "invalid_episode_ids": list(invalid_ids),
            "invalid_semantics": "excluded from all estimands; retained in 160-artifact integrity accounting",
            "condition_cells": integrity_cells,
            "provenance": {
                "split": "locked_test", "suite": manifest.task.suite,
                "task_id": manifest.task.task_id,
                "policy_revision": manifest.episodes[0].policy_revision,
                "code_commit": manifest.episodes[0].code_commit,
            },
        },
        {
            "number": 2, "title": SECTION_TITLES[1], "comparison": "M2_vs_M1",
            "episode_weighting": "each episode has total sample weight one",
            "result": _plain_dataclass(primary),
            "bar": {"minimum_relative_lift": 0.03, "succeeds": primary.primary_claim_succeeds},
        },
        {"number": 3, "title": SECTION_TITLES[2], "models": model_metrics},
        {"number": 4, "title": SECTION_TITLES[3], "comparison": "M2_vs_M0", "result": _plain_dataclass(m2_m0)},
        {
            "number": 5, "title": SECTION_TITLES[4], "comparison": "M2_vs_M1",
            "threshold_source": "Calibration-only frozen 10% episode-FPR thresholds",
            "thresholds": dict(thresholds), "failure_step_source": "frozen annotation.onset_step",
            "result": _plain_dataclass(lead), "bar_steps": 5.0,
        },
        {"number": 6, "title": SECTION_TITLES[5], "cells": _condition_rankings(manifest, predictions)},
        {"number": 7, "title": SECTION_TITLES[6], "receipt_sha256": causal_sha256, "result": causal_payload},
        {
            "number": 8, "title": SECTION_TITLES[7],
            "receipt_sha256": cost_sha256, "summary": _cost_summary(cost_payload),
            "result": cost_payload,
        },
        {
            "number": 9, "title": SECTION_TITLES[8], "receipt_sha256": sensitivity_sha256,
            "subsections": [
                {"number": "9a", "title": "rollout diagnostics", "result": sensitivity_payload["rollout_diagnostics"]},
                {"number": "9b", "title": "patching dose × difficulty", "result": sensitivity_payload["dose_by_difficulty"]},
                {"number": "9c", "title": "broken-successes ledger", "result": sensitivity_payload["broken_successes"]},
            ],
        },
        {
            "number": 10, "title": SECTION_TITLES[9],
            "pre_stated_expectation": "toy evidence predicts M2 approximately M1 with M2 much better than M0",
            "reporting_rule": "report exactly as measured; neither lift nor specificity routes to the negative-result publication path",
            "measured": {
                "m2_vs_m1_relative_lift": primary.relative_lift,
                "m2_vs_m0_relative_lift": m2_m0.relative_lift,
                "m2_vs_m1_primary_succeeds": primary.primary_claim_succeeds,
                "causal_status": causal_payload["confirmatory"]["status"],
                "causal_specificity_passes": causal_payload["confirmatory"].get("specificity_passes"),
            },
        },
    ]
    if tuple(section["title"] for section in sections) != SECTION_TITLES:
        raise AssertionError("report section order changed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_final_report",
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "prediction_receipt_sha256": prediction_sha256,
            "causal_receipt_sha256": causal_sha256,
            "sensitivity_receipt_sha256": sensitivity_sha256,
            "cost_receipt_sha256": cost_sha256,
        },
        "evaluation_protocol": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "primary_steps": list(PRIMARY_STEPS),
            "model_training_or_selection_performed": False,
        },
        "sections": sections,
    }
    _canonical(report)
    return report


def evaluate(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    raw_root: Path,
    predictions_path: Path,
    predictions_sha256: str,
    calibration_freeze_path: Path,
    calibration_freeze_sha256: str,
    reality_gate_lock_path: Path,
    reality_gate_lock_sha256: str,
    causal_receipt_path: Path,
    causal_receipt_sha256: str,
    sensitivity_receipt_path: Path,
    sensitivity_receipt_sha256: str,
    cost_receipt_path: Path,
    cost_receipt_sha256: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    # Complete file-level preflight first.  A missing causal/sensitivity/cost
    # input stops here, before raw outcomes or primary metrics are opened.
    manifest_payload = _load_addressed_json(manifest_path, manifest_sha256)
    prediction_payload = _load_addressed_json(
        predictions_path, predictions_sha256, directory_addressed=True
    )
    calibration_freeze = _load_addressed_json(calibration_freeze_path, calibration_freeze_sha256)
    reality_gate = _load_addressed_json(reality_gate_lock_path, reality_gate_lock_sha256)
    causal_payload = _load_addressed_json(causal_receipt_path, causal_receipt_sha256)
    sensitivity_payload = _load_addressed_json(sensitivity_receipt_path, sensitivity_receipt_sha256)
    cost_payload = _load_addressed_json(cost_receipt_path, cost_receipt_sha256)
    manifest = _manifest_from_payload(manifest_payload)
    if manifest.sha256 != manifest_sha256:
        raise LockedTestEvaluationError("manifest logical digest differs from its file digest")
    _validate_prediction_receipt(prediction_payload, manifest_sha256, predictions_sha256)
    prediction_source = _mapping(prediction_payload["source"], "predictions.source")
    thresholds = _validate_calibration_freeze(
        calibration_freeze,
        prediction_source,
        freeze_sha256=calibration_freeze_sha256,
        reality_gate_sha256=reality_gate_lock_sha256,
    )
    _validate_causal_receipt(causal_payload, manifest_sha256, predictions_sha256)
    _validate_sensitivity_receipt(
        sensitivity_payload, manifest_sha256, predictions_sha256, causal_receipt_sha256
    )
    _validate_cost_receipt(cost_payload, manifest_sha256, predictions_sha256)
    bounds = _failure_bounds(reality_gate, manifest.task)
    if not isinstance(bootstrap_replicates, int) or isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 1:
        raise LockedTestEvaluationError("bootstrap_replicates must be positive")

    raw, cells = _load_raw_inputs(manifest, raw_root, bounds)
    return build_report(
        manifest=manifest, manifest_sha256=manifest_sha256, raw=raw,
        integrity_cells=cells, prediction_payload=prediction_payload,
        prediction_sha256=predictions_sha256, causal_payload=causal_payload,
        causal_sha256=causal_receipt_sha256, sensitivity_payload=sensitivity_payload,
        sensitivity_sha256=sensitivity_receipt_sha256, cost_payload=cost_payload,
        cost_sha256=cost_receipt_sha256, thresholds=thresholds,
        bootstrap_replicates=bootstrap_replicates,
    )


def _publish(report: Mapping[str, Any], output_root: Path) -> tuple[Path, str]:
    payload = _canonical(report)
    digest = _sha256(payload)
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise LockedTestEvaluationError("output root is not a safe directory")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / digest
    if destination.exists():
        path = destination / "report.json"
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or {item.name for item in destination.iterdir()} != {"report.json"}
            or path.is_symlink()
            or path.read_bytes() != payload
        ):
            raise LockedTestEvaluationError(
                f"existing evaluation report differs from content address {destination}"
            )
        return path, digest
    if destination.is_symlink():
        raise LockedTestEvaluationError(f"unsafe evaluation destination {destination}")
    destination.mkdir()
    path = destination / "report.json"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--calibration-freeze-sha256", required=True)
    parser.add_argument("--reality-gate-lock", type=Path, required=True)
    parser.add_argument("--reality-gate-lock-sha256", required=True)
    parser.add_argument("--causal-receipt", type=Path, required=True)
    parser.add_argument("--causal-receipt-sha256", required=True)
    parser.add_argument("--sensitivity-receipt", type=Path, required=True)
    parser.add_argument("--sensitivity-receipt-sha256", required=True)
    parser.add_argument("--cost-receipt", type=Path, required=True)
    parser.add_argument("--cost-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        manifest_path=args.manifest.resolve(), manifest_sha256=args.manifest_sha256,
        raw_root=args.raw_root.resolve(), predictions_path=args.predictions.resolve(),
        predictions_sha256=args.predictions_sha256,
        calibration_freeze_path=args.calibration_freeze.resolve(),
        calibration_freeze_sha256=args.calibration_freeze_sha256,
        reality_gate_lock_path=args.reality_gate_lock.resolve(),
        reality_gate_lock_sha256=args.reality_gate_lock_sha256,
        causal_receipt_path=args.causal_receipt.resolve(),
        causal_receipt_sha256=args.causal_receipt_sha256,
        sensitivity_receipt_path=args.sensitivity_receipt.resolve(),
        sensitivity_receipt_sha256=args.sensitivity_receipt_sha256,
        cost_receipt_path=args.cost_receipt.resolve(),
        cost_receipt_sha256=args.cost_receipt_sha256,
    )
    path, digest = _publish(report, args.output_root.resolve())
    print(_canonical({"kind": "locked_test_evaluation_complete", "report_path": str(path), "report_sha256": digest}).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
