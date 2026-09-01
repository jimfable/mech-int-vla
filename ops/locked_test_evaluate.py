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
from mech_int_vla.causal import cluster_bootstrap_rate_interval
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
POSITION_DIAGNOSTIC_STATUS = "unavailable_preaccess_missing_position_trace"
POSITION_DIAGNOSTIC_REASON = "frozen_position_decoder_and_all_object_trace_absent"
SUPPORTING_LAYER_REASON = "frozen_supporting_layer_coefficients_absent"
BROKEN_SUCCESS_REASON = "patched_closed_loop_outcome_not_defined"
CALIBRATION_ACTIVATION_REFERENCE_SHA256 = (
    "cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c"
)
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
    if directory_addressed and (
        path.parent.is_symlink() or not path.parent.is_dir()
    ):
        raise LockedTestEvaluationError(
            f"content-addressed directory for {path} is absent or unsafe"
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


def _relative_evidence(
    receipt_path: Path, relative_path: Any, expected_sha256: str, where: str
) -> Mapping[str, Any]:
    if not isinstance(relative_path, str) or not relative_path:
        raise LockedTestEvaluationError(f"{where} evidence path must be nonempty")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise LockedTestEvaluationError(f"{where} evidence path escapes receipt directory")
    expected = Path("evidence") / f"{expected_sha256}.json"
    if relative != expected:
        raise LockedTestEvaluationError(
            f"{where} evidence must use evidence/<content-sha256>.json"
        )
    evidence_root = receipt_path.parent / "evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise LockedTestEvaluationError(f"{where} evidence directory is absent or unsafe")
    path = receipt_path.parent / relative
    value, raw = _strict_json_bytes(path)
    if _sha256(raw) != expected_sha256 or raw != _canonical(value):
        raise LockedTestEvaluationError(f"{where} evidence digest/canonical bytes differ")
    return value


def _validated_ratio(value: Any, status: Any, where: str) -> float:
    if status == "finite":
        if value is None:
            raise LockedTestEvaluationError(f"{where} finite ratio is absent")
        return _finite(value, where, lower=0)
    if status == "infinite_zero_yaw_effect" and value is None:
        return math.inf
    raise LockedTestEvaluationError(f"{where} ratio/status encoding is invalid")


def _validate_causal_receipt(
    payload: Mapping[str, Any],
    receipt_path: Path,
    manifest: Manifest,
    manifest_sha: str,
    prediction_sha: str,
    prediction_payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_causal_patching_receipt":
        raise LockedTestEvaluationError("causal patching receipt has the wrong schema or kind")
    _exact_keys(
        payload,
        {
            "schema_version", "kind", "source", "evidence_hashes", "pairs",
            "selected_layer_summary", "supporting_layers", "confirmatory",
        },
        "causal",
    )
    source = _exact_keys(
        payload.get("source"),
        {
            "manifest_sha256", "prediction_receipt_sha256", "raw_inventory_sha256",
            "score_allocation_sha256", "bound_probe_sha256",
            "calibration_reference_sha256", "calibration_activation_reference_sha256",
            "alpha", "pairing_seeds", "random_subspaces_per_pair",
            "matched_donor_rule", "pair_selection_rule",
        },
        "causal.source",
    )
    if source.get("manifest_sha256") != manifest_sha or source.get("prediction_receipt_sha256") != prediction_sha:
        raise LockedTestEvaluationError("causal patching receipt is bound to different inputs")
    if _finite(source.get("alpha"), "causal.source.alpha") != 0.25:
        raise LockedTestEvaluationError("causal receipt did not use frozen alpha 0.25")
    if source.get("pairing_seeds") != list(PAIRING_SEEDS):
        raise LockedTestEvaluationError("causal receipt did not use all three pairing seeds")
    if _integer(source.get("random_subspaces_per_pair"), "random_subspaces_per_pair", lower=1) != 1000:
        raise LockedTestEvaluationError("causal receipt did not use 1000 indexed controls per pair")
    if source.get("matched_donor_rule") != "orientation_difference_degrees < 5":
        raise LockedTestEvaluationError("causal receipt did not use a strict <5 degree rule")
    if source.get("pair_selection_rule") != "outcome_blind_frozen_state_matching":
        raise LockedTestEvaluationError("causal pair selection is not explicitly outcome blind")
    for name in (
        "raw_inventory_sha256", "calibration_activation_reference_sha256",
    ):
        if not _is_sha256(source.get(name)):
            raise LockedTestEvaluationError(f"causal.source.{name} is not a digest")
    if (
        source["calibration_activation_reference_sha256"]
        != CALIBRATION_ACTIVATION_REFERENCE_SHA256
    ):
        raise LockedTestEvaluationError(
            "causal receipt does not use the final frozen Calibration activation reference"
        )
    prediction_source = _mapping(prediction_payload.get("source"), "predictions.source")
    expected_prediction_bindings = {
        "score_allocation_sha256": "score_allocation_sha256",
        "bound_probe_sha256": "bound_probe_sha256",
        "calibration_reference_sha256": "reference_bundle_sha256",
    }
    for causal_name, prediction_name in expected_prediction_bindings.items():
        if source.get(causal_name) != prediction_source.get(prediction_name):
            raise LockedTestEvaluationError(
                f"causal receipt differs from prediction source {prediction_name}"
            )
    raw_by_episode: dict[str, dict[str, str]] = {}
    predicted_state_ids: set[tuple[str, int]] = set()
    records = prediction_payload.get("records")
    if not isinstance(records, list):
        raise LockedTestEvaluationError("prediction records are absent during causal binding")
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"predictions.records[{index}]")
        hashes = _mapping(record.get("source_hashes"), "prediction source_hashes")
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise LockedTestEvaluationError("prediction record episode ID is invalid")
        identity = {
            "episode_id": episode_id,
            "raw_metadata_sha256": hashes.get("raw_metadata_sha256"),
            "raw_trajectory_sha256": hashes.get("raw_trajectory_sha256"),
        }
        if not all(_is_sha256(identity[name]) for name in (
            "raw_metadata_sha256", "raw_trajectory_sha256"
        )):
            raise LockedTestEvaluationError("prediction raw source hash is invalid")
        previous = raw_by_episode.setdefault(episode_id, identity)
        if previous != identity:
            raise LockedTestEvaluationError("prediction raw source hashes drift within episode")
        predicted_state_ids.add(
            (episode_id, _integer(record.get("control_step"), "prediction control step"))
        )
    raw_inventory = [raw_by_episode[episode_id] for episode_id in sorted(raw_by_episode)]
    if source["raw_inventory_sha256"] != _sha256(_canonical(raw_inventory)):
        raise LockedTestEvaluationError("causal raw inventory is not prediction-evidence-derived")

    pairs = payload.get("pairs")
    evidence_hashes = payload.get("evidence_hashes")
    if not isinstance(pairs, list) or len(pairs) != 60:
        raise LockedTestEvaluationError("causal receipt must contain exactly 60 pair rows")
    if not isinstance(evidence_hashes, list) or len(evidence_hashes) != 60:
        raise LockedTestEvaluationError("causal receipt must contain 60 evidence hashes")
    source_sha = _sha256(_canonical(source))
    manifest_by_episode = {episode.episode_id: episode for episode in manifest.episodes}
    valid_rows: list[dict[str, Any]] = []
    all_control_effects: list[list[float]] = []
    for index, raw_pair in enumerate(pairs):
        pair = _exact_keys(
            raw_pair,
            {
                "pair_index", "seed", "condition_index", "base_init_state_id",
                "valid", "evidence_sha256", "evidence_path",
            },
            f"causal.pairs[{index}]",
        )
        if pair["pair_index"] != index or pair["seed"] != PAIRING_SEEDS[index // 20]:
            raise LockedTestEvaluationError("causal pair indices/seeds are not frozen order")
        cell = _integer(pair["condition_index"], "pair.condition_index")
        _integer(pair["base_init_state_id"], "pair.base_init_state_id")
        if cell >= 8 or type(pair["valid"]) is not bool:
            raise LockedTestEvaluationError("causal pair metadata is invalid")
        evidence_sha = pair["evidence_sha256"]
        if not _is_sha256(evidence_sha) or evidence_hashes[index] != evidence_sha:
            raise LockedTestEvaluationError("causal pair/evidence hash inventory differs")
        evidence = _relative_evidence(
            receipt_path, pair["evidence_path"], evidence_sha, f"causal pair {index}"
        )
        _exact_keys(
            evidence,
            {
                "schema_version", "kind", "source_sha256", "pair",
                "selected_patch", "off_manifold", "matched_control", "random_controls",
            },
            f"causal evidence {index}",
        )
        if (
            evidence.get("schema_version") != 1
            or evidence.get("kind") != "locked_test_causal_pair_evidence"
            or evidence.get("source_sha256") != source_sha
        ):
            raise LockedTestEvaluationError("causal pair evidence binding differs")
        evidence_pair_raw = _mapping(evidence["pair"], f"causal evidence {index}.pair")
        pair_valid = evidence_pair_raw.get("valid")
        expected_pair_keys = {
            "pair_index", "seed", "condition_index", "base_init_state_id",
            "recipient_id", "donor_id", "orientation_difference_degrees", "valid",
        } | ({"invalid_reason"} if pair_valid is False else set())
        evidence_pair = _exact_keys(
            evidence_pair_raw, expected_pair_keys, f"causal evidence {index}.pair"
        )
        if any(evidence_pair.get(name) != pair.get(name) for name in (
            "pair_index", "seed", "condition_index", "base_init_state_id"
        )) or evidence_pair.get("valid") is not pair["valid"]:
            raise LockedTestEvaluationError("causal evidence pair identity differs")
        if not isinstance(evidence_pair["recipient_id"], str) or not evidence_pair["recipient_id"]:
            raise LockedTestEvaluationError("causal evidence recipient ID is absent")
        candidate_roles = ["recipient_id"]
        invalid_reason = evidence_pair.get("invalid_reason")
        if pair["valid"] or invalid_reason == "no_eligible_matched_donor":
            if not isinstance(evidence_pair["donor_id"], str) or not evidence_pair["donor_id"]:
                raise LockedTestEvaluationError("causal evidence donor ID is absent")
            candidate_roles.append("donor_id")
        elif invalid_reason == "no_eligible_confirmatory_donor":
            if evidence_pair["donor_id"] is not None:
                raise LockedTestEvaluationError(
                    "unmatched confirmatory slot must use donor_id=null"
                )
        elif not pair["valid"]:
            raise LockedTestEvaluationError("invalid causal pair reason is not frozen")
        for candidate_role in candidate_roles:
            candidate_id = evidence_pair[candidate_role]
            episode_id, separator, step_text = candidate_id.rpartition("@")
            try:
                candidate_step = int(step_text)
            except ValueError as exc:
                raise LockedTestEvaluationError("causal candidate ID has invalid step") from exc
            if separator != "@" or (episode_id, candidate_step) not in predicted_state_ids:
                raise LockedTestEvaluationError(
                    "causal candidate is not an exact frozen prediction state"
                )
        recipient_episode = evidence_pair["recipient_id"].rpartition("@")[0]
        specification = manifest_by_episode.get(recipient_episode)
        if (
            specification is None
            or specification.base_init_state_id != pair["base_init_state_id"]
            or specification.condition_index != pair["condition_index"]
        ):
            raise LockedTestEvaluationError(
                "causal pair condition/base-init identity differs from manifest"
            )
        if invalid_reason == "no_eligible_confirmatory_donor":
            if evidence_pair["orientation_difference_degrees"] is not None:
                raise LockedTestEvaluationError(
                    "unmatched confirmatory slot must use orientation_difference_degrees=null"
                )
        else:
            pair_angle = _finite(
                evidence_pair["orientation_difference_degrees"],
                "confirmatory orientation difference", lower=0,
            )
            if pair_angle < 30.0 or pair_angle > 90.0:
                raise LockedTestEvaluationError("confirmatory pair is outside [30, 90] degrees")
        if not pair["valid"]:
            if (
                any(evidence[name] is not None for name in (
                    "selected_patch", "off_manifold", "matched_control"
                ))
                or evidence["random_controls"] != []
            ):
                raise LockedTestEvaluationError("invalid causal pair contains scientific results")
            continue
        selected = _exact_keys(
            evidence["selected_patch"],
            {"alpha", "sign_correct", "donor_aligned_target_effect", "off_target_ratio", "off_target_ratio_status"},
            f"causal evidence {index}.selected_patch",
        )
        if _finite(selected["alpha"], "selected alpha") != 0.25 or type(selected["sign_correct"]) is not bool:
            raise LockedTestEvaluationError("causal selected patch differs from frozen alpha")
        effect = _finite(selected["donor_aligned_target_effect"], "selected effect")
        ratio = _validated_ratio(
            selected["off_target_ratio"], selected["off_target_ratio_status"],
            "selected off-target ratio",
        )
        off = _exact_keys(
            evidence["off_manifold"],
            {"patched_five_nn_distance", "natural_95th_percentile", "off_manifold"},
            f"causal evidence {index}.off_manifold",
        )
        distance = _finite(off["patched_five_nn_distance"], "patched 5NN", lower=0)
        threshold = _finite(off["natural_95th_percentile"], "natural 95th", lower=0)
        if type(off["off_manifold"]) is not bool or off["off_manifold"] != (distance > threshold):
            raise LockedTestEvaluationError("off-manifold flag is not derived from 5-NN evidence")
        matched = _exact_keys(
            evidence["matched_control"],
            {"donor_id", "orientation_difference_degrees", "sign_correct", "donor_aligned_target_effect"},
            f"causal evidence {index}.matched_control",
        )
        angle = _finite(matched["orientation_difference_degrees"], "matched angle", lower=0)
        if (
            not isinstance(matched["donor_id"], str) or not matched["donor_id"]
            or not angle < 5.0 or type(matched["sign_correct"]) is not bool
        ):
            raise LockedTestEvaluationError("matched control violates the strict <5 degree rule")
        matched_episode, separator, matched_step_text = matched["donor_id"].rpartition("@")
        try:
            matched_step = int(matched_step_text)
        except ValueError as exc:
            raise LockedTestEvaluationError("matched-control candidate ID has invalid step") from exc
        if separator != "@" or (matched_episode, matched_step) not in predicted_state_ids:
            raise LockedTestEvaluationError(
                "matched-control donor is not an exact frozen prediction state"
            )
        _finite(matched["donor_aligned_target_effect"], "matched effect")
        controls = evidence["random_controls"]
        if not isinstance(controls, list) or len(controls) != 1000:
            raise LockedTestEvaluationError("each valid pair needs 1000 random controls")
        control_effects: list[float] = []
        for control_index, raw_control in enumerate(controls):
            control = _exact_keys(
                raw_control, {"control_index", "donor_aligned_target_effect"},
                f"causal evidence {index}.random_controls[{control_index}]",
            )
            if control["control_index"] != control_index:
                raise LockedTestEvaluationError("random control indices must be exactly 0..999 per pair")
            control_effects.append(
                _finite(control["donor_aligned_target_effect"], "random-control effect")
            )
        all_control_effects.append(control_effects)
        valid_rows.append(
            {
                "sign": selected["sign_correct"], "effect": effect, "ratio": ratio,
                "off": off["off_manifold"], "matched": matched["sign_correct"],
                "base_init": pair["base_init_state_id"], "seed": pair["seed"],
            }
        )

    summary = _exact_keys(
        payload.get("selected_layer_summary"),
        {
            "status", "attempted_pairs", "valid_pairs", "sign_correct_count",
            "sign_correct_rate", "sign_interval", "median_donor_aligned_target_effect",
            "median_off_target_ratio", "median_off_target_ratio_status",
            "off_manifold_rate", "matched_control_sign_rate",
            "random_control_95th_percentile", "random_control_passes",
            "specificity_passes", "sign_passes", "positive_seed_count",
            "seed_stability_passes",
        },
        "causal.selected_layer_summary",
    )
    valid = len(valid_rows)
    expected_status = "inconclusive_insufficient_valid_pairs" if valid < 30 else "complete"
    if summary["status"] != expected_status or summary["attempted_pairs"] != 60 or summary["valid_pairs"] != valid:
        raise LockedTestEvaluationError("selected-layer status/counts are not evidence-derived")
    signs = [bool(row["sign"]) for row in valid_rows]
    sign_count = sum(signs)
    sign_rate = sign_count / valid if valid else None
    if summary["sign_correct_count"] != sign_count or (valid and not math.isclose(_rate(summary["sign_correct_rate"], "summary sign rate"), sign_rate, abs_tol=1e-12)):
        raise LockedTestEvaluationError("selected-layer sign summary differs from evidence")
    if not valid and summary["sign_correct_rate"] is not None:
        raise LockedTestEvaluationError("empty selected-layer summary contains a sign rate")
    if valid:
        cluster_ids = [row["base_init"] for row in valid_rows]
        expected_interval = None
        if len(set(cluster_ids)) >= 2:
            interval = _exact_keys(
                summary["sign_interval"],
                {"estimate", "lower", "upper", "confidence", "replicates", "clusters", "seed"},
                "causal.selected_layer_summary.sign_interval",
            )
            expected_interval = cluster_bootstrap_rate_interval(
                signs, cluster_ids, seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES, confidence=BOOTSTRAP_CONFIDENCE,
            )
            if any(not math.isclose(_finite(interval[name], f"sign_interval.{name}"), float(getattr(expected_interval, name)), abs_tol=1e-12) for name in ("estimate", "lower", "upper", "confidence")) or any(interval[name] != getattr(expected_interval, name) for name in ("replicates", "clusters", "seed")):
                raise LockedTestEvaluationError("selected-layer sign interval is not evidence-derived")
        elif summary["sign_interval"] is not None:
            raise LockedTestEvaluationError("single-cluster sign interval must be unavailable")
        expected_effect = float(np.median([row["effect"] for row in valid_rows]))
        if not math.isclose(_finite(summary["median_donor_aligned_target_effect"], "median effect"), expected_effect, abs_tol=1e-12):
            raise LockedTestEvaluationError("selected-layer median effect differs from evidence")
        expected_ratio = float(np.median([row["ratio"] for row in valid_rows]))
        observed_ratio = _validated_ratio(summary["median_off_target_ratio"], summary["median_off_target_ratio_status"], "summary median ratio")
        if observed_ratio != expected_ratio and not math.isclose(observed_ratio, expected_ratio, abs_tol=1e-12):
            raise LockedTestEvaluationError("selected-layer specificity differs from evidence")
        pooled_controls = np.median(np.asarray(all_control_effects, dtype=np.float64), axis=0)
        control_95 = float(np.percentile(pooled_controls, 95.0))
        observed_median = expected_effect
        expected_random_pass = observed_median > control_95
        expected_off_rate = sum(row["off"] for row in valid_rows) / valid
        expected_matched_rate = sum(row["matched"] for row in valid_rows) / valid
        positive_seeds = sum(
            bool(
                np.median(
                    [row["effect"] for row in valid_rows if row["seed"] == seed]
                ) > 0.0
            )
            for seed in PAIRING_SEEDS
            if any(row["seed"] == seed for row in valid_rows)
        )
        checks = {
            "off_manifold_rate": expected_off_rate,
            "matched_control_sign_rate": expected_matched_rate,
            "random_control_95th_percentile": control_95,
        }
        for name, expected_value in checks.items():
            if not math.isclose(_finite(summary[name], f"summary.{name}"), expected_value, abs_tol=1e-12):
                raise LockedTestEvaluationError(f"{name} differs from pair evidence")
        expected_specificity = expected_ratio <= 0.25
        expected_sign_pass = bool(
            expected_interval is not None
            and expected_interval.estimate > 0.5
            and expected_interval.lower > 0.5
        )
        expected_seed_pass = bool(positive_seeds >= 2)
        boolean_checks = {
            "random_control_passes": expected_random_pass,
            "specificity_passes": expected_specificity,
            "sign_passes": expected_sign_pass,
            "seed_stability_passes": expected_seed_pass,
        }
        for name, expected_value in boolean_checks.items():
            if summary[name] is not expected_value:
                raise LockedTestEvaluationError(f"{name} differs from pair evidence")
        if summary["positive_seed_count"] != positive_seeds:
            raise LockedTestEvaluationError("positive seed count differs from pair evidence")
    else:
        empty_expected = {
            "sign_interval": None,
            "median_donor_aligned_target_effect": None,
            "median_off_target_ratio": None,
            "median_off_target_ratio_status": "unavailable_no_valid_pairs",
            "off_manifold_rate": None,
            "matched_control_sign_rate": None,
            "random_control_95th_percentile": None,
            "random_control_passes": False,
            "specificity_passes": False,
            "sign_passes": False,
            "positive_seed_count": 0,
            "seed_stability_passes": False,
        }
        if any(summary[name] != expected_value for name, expected_value in empty_expected.items()):
            raise LockedTestEvaluationError("empty selected-layer summary contains invented results")
    supporting = _exact_keys(
        payload.get("supporting_layers"),
        {"status", "reason", "multi_layer_support_available", "layer_support_passes"},
        "causal.supporting_layers",
    )
    if supporting != {
        "status": "unavailable", "reason": SUPPORTING_LAYER_REASON,
        "multi_layer_support_available": False, "layer_support_passes": False,
    }:
        raise LockedTestEvaluationError("supporting-layer limitation marker differs")
    confirmatory = _exact_keys(
        payload.get("confirmatory"), {"status", "succeeds", "reason"},
        "causal.confirmatory",
    )
    if confirmatory != {
        "status": "unsupported", "succeeds": False,
        "reason": SUPPORTING_LAYER_REASON,
    }:
        raise LockedTestEvaluationError(
            "positive confirmatory claim must be deterministically unsupported/false"
        )


def _validate_sensitivity_receipt(
    payload: Mapping[str, Any],
    receipt_path: Path,
    manifest_sha: str,
    prediction_sha: str,
    causal_sha: str,
) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "locked_test_sensitivity_receipt":
        raise LockedTestEvaluationError("sensitivity receipt has the wrong schema or kind")
    _exact_keys(
        payload,
        {
            "schema_version", "kind", "source", "evidence_hashes",
            "dose_evidence", "dose_by_difficulty", "rollout_diagnostics",
            "broken_successes",
        },
        "sensitivity",
    )
    source = _exact_keys(
        payload.get("source"),
        {
            "manifest_sha256", "prediction_receipt_sha256",
            "causal_receipt_sha256", "alphas", "pairing_seeds",
            "patch_rule",
        },
        "sensitivity.source",
    )
    expected = {
        "manifest_sha256": manifest_sha,
        "prediction_receipt_sha256": prediction_sha,
        "causal_receipt_sha256": causal_sha,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise LockedTestEvaluationError("sensitivity receipt is bound to different inputs")
    if source.get("alphas") != [0.5, 1.0] or source.get("pairing_seeds") != list(PAIRING_SEEDS):
        raise LockedTestEvaluationError("sensitivity source does not use the frozen dose/seeds")
    if source.get("patch_rule") != "same_selected_layer_pair_plan_no_refit":
        raise LockedTestEvaluationError("sensitivity receipt used another patching rule")
    diagnostics = _exact_keys(
        payload.get("rollout_diagnostics"), {"status", "reason"},
        "rollout_diagnostics",
    )
    if diagnostics != {
        "status": POSITION_DIAGNOSTIC_STATUS,
        "reason": POSITION_DIAGNOSTIC_REASON,
    }:
        raise LockedTestEvaluationError(
            "9a must use the exact pre-access missing-position-trace marker"
        )
    broken = _exact_keys(
        payload.get("broken_successes"), {"status", "reason"}, "broken_successes"
    )
    if broken != {"status": "unavailable", "reason": BROKEN_SUCCESS_REASON}:
        raise LockedTestEvaluationError("broken-success limitation marker differs")

    dose_evidence = payload.get("dose_evidence")
    evidence_hashes = payload.get("evidence_hashes")
    if not isinstance(dose_evidence, list) or len(dose_evidence) != 60:
        raise LockedTestEvaluationError("sensitivity must contain 60 dose-evidence rows")
    if not isinstance(evidence_hashes, list) or len(evidence_hashes) != 60:
        raise LockedTestEvaluationError("sensitivity evidence-hash inventory must contain 60 rows")
    source_sha = _sha256(_canonical(source))
    by_alpha_cell: dict[tuple[float, int], list[dict[str, Any]]] = {
        (alpha, cell): [] for alpha in (0.5, 1.0) for cell in range(8)
    }
    attempted_indices_by_cell: dict[int, list[int]] = {cell: [] for cell in range(8)}
    for index, raw_row in enumerate(dose_evidence):
        row = _exact_keys(
            raw_row, {"pair_index", "evidence_sha256", "evidence_path"},
            f"dose_evidence[{index}]",
        )
        if row["pair_index"] != index:
            raise LockedTestEvaluationError("dose-evidence pair indices are not 0..59")
        digest = row["evidence_sha256"]
        if not _is_sha256(digest) or evidence_hashes[index] != digest:
            raise LockedTestEvaluationError("dose evidence hash inventory differs")
        evidence = _relative_evidence(
            receipt_path, row["evidence_path"], digest, f"dose pair {index}"
        )
        _exact_keys(
            evidence,
            {"schema_version", "kind", "source_sha256", "pair", "alphas"},
            f"dose evidence {index}",
        )
        if (
            evidence.get("schema_version") != 1
            or evidence.get("kind") != "locked_test_sensitivity_pair_evidence"
            or evidence.get("source_sha256") != source_sha
        ):
            raise LockedTestEvaluationError("dose evidence source binding differs")
        pair_raw = _mapping(evidence["pair"], f"dose evidence {index}.pair")
        valid = pair_raw.get("valid")
        pair = _exact_keys(
            pair_raw,
            {
                "pair_index", "seed", "condition_index", "base_init_state_id", "valid",
            } | ({"invalid_reason"} if valid is False else set()),
            f"dose evidence {index}.pair",
        )
        if (
            pair["pair_index"] != index
            or pair["seed"] != PAIRING_SEEDS[index // 20]
            or type(valid) is not bool
        ):
            raise LockedTestEvaluationError("dose evidence pair binding differs")
        cell = _integer(pair["condition_index"], "dose condition_index")
        _integer(pair["base_init_state_id"], "dose base-init ID")
        if cell >= 8:
            raise LockedTestEvaluationError("dose condition index is outside 0..7")
        attempted_indices_by_cell[cell].append(index)
        if not valid:
            if pair["invalid_reason"] not in {
                "no_eligible_confirmatory_donor", "no_eligible_matched_donor"
            }:
                raise LockedTestEvaluationError("invalid dose pair reason is not frozen")
            if evidence["alphas"] is not None:
                raise LockedTestEvaluationError("invalid dose pair contains scientific results")
            continue
        alphas = evidence["alphas"]
        if not isinstance(alphas, list) or len(alphas) != 2:
            raise LockedTestEvaluationError("valid dose pair must contain alpha 0.5 and 1.0")
        for alpha_index, raw_alpha in enumerate(alphas):
            alpha_row = _exact_keys(
                raw_alpha,
                {
                    "alpha", "sign_correct", "donor_aligned_target_effect",
                    "off_target_ratio", "off_target_ratio_status",
                },
                f"dose evidence {index}.alphas[{alpha_index}]",
            )
            alpha = (0.5, 1.0)[alpha_index]
            if _finite(alpha_row["alpha"], "dose alpha") != alpha or type(alpha_row["sign_correct"]) is not bool:
                raise LockedTestEvaluationError("dose evidence alpha/sign differs")
            effect = _finite(alpha_row["donor_aligned_target_effect"], "dose effect")
            ratio = _validated_ratio(
                alpha_row["off_target_ratio"], alpha_row["off_target_ratio_status"],
                "dose off-target ratio",
            )
            by_alpha_cell[(alpha, cell)].append(
                {"pair_index": index, "sign": alpha_row["sign_correct"], "effect": effect, "ratio": ratio}
            )
    dose = payload.get("dose_by_difficulty")
    if not isinstance(dose, list):
        raise LockedTestEvaluationError("dose_by_difficulty must be an array")
    dose_grid = []
    for index, raw in enumerate(dose):
        item = _exact_keys(
            raw,
            {
                "alpha", "condition_index", "pair_indices", "valid_pairs",
                "sign_correct_count", "sign_correct_rate", "median_donor_aligned_target_effect",
                "median_off_target_ratio", "median_off_target_ratio_status",
                "specificity_passes",
            },
            f"dose_by_difficulty[{index}]",
        )
        alpha = _finite(item.get("alpha"), "dose alpha")
        cell = _integer(item.get("condition_index"), "dose condition_index")
        evidence_rows = by_alpha_cell.get((alpha, cell))
        indices = attempted_indices_by_cell[cell]
        if not evidence_rows:
            expected_empty = {
                "alpha": alpha,
                "condition_index": cell,
                "pair_indices": indices,
                "valid_pairs": 0,
                "sign_correct_count": 0,
                "sign_correct_rate": None,
                "median_donor_aligned_target_effect": None,
                "median_off_target_ratio": None,
                "median_off_target_ratio_status": "unavailable_no_valid_pairs",
                "specificity_passes": False,
            }
            if dict(item) != expected_empty:
                raise LockedTestEvaluationError(
                    "empty dose cell must use the exact evidence-derived unavailable form"
                )
            dose_grid.append((alpha, cell))
            continue
        sign_count = sum(row["sign"] for row in evidence_rows)
        sign_rate = sign_count / len(evidence_rows)
        median_effect = float(np.median([row["effect"] for row in evidence_rows]))
        median_ratio = float(np.median([row["ratio"] for row in evidence_rows]))
        observed_ratio = _validated_ratio(
            item["median_off_target_ratio"], item["median_off_target_ratio_status"],
            "dose aggregate ratio",
        )
        if (
            item["pair_indices"] != indices
            or item["valid_pairs"] != len(evidence_rows)
            or item["sign_correct_count"] != sign_count
            or not math.isclose(_rate(item["sign_correct_rate"], "dose sign rate"), sign_rate, abs_tol=1e-12)
            or not math.isclose(_finite(item["median_donor_aligned_target_effect"], "dose median effect"), median_effect, abs_tol=1e-12)
            or (observed_ratio != median_ratio and not math.isclose(observed_ratio, median_ratio, abs_tol=1e-12))
            or item["specificity_passes"] is not (median_ratio <= 0.25)
        ):
            raise LockedTestEvaluationError("dose aggregate differs from pair evidence")
        dose_grid.append((alpha, cell))
    if dose_grid != [(alpha, cell) for alpha in (0.5, 1.0) for cell in range(8)]:
        raise LockedTestEvaluationError("dose sensitivity is not the ordered 2 x 8 grid")


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
                "selected_layer_specificity_passes": causal_payload[
                    "selected_layer_summary"
                ].get("specificity_passes"),
                "multi_layer_support_available": causal_payload[
                    "supporting_layers"
                ]["multi_layer_support_available"],
                "positive_confirmatory_causal_claim_succeeds": causal_payload[
                    "confirmatory"
                ]["succeeds"],
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
    causal_payload = _load_addressed_json(
        causal_receipt_path, causal_receipt_sha256, directory_addressed=True
    )
    sensitivity_payload = _load_addressed_json(
        sensitivity_receipt_path, sensitivity_receipt_sha256, directory_addressed=True
    )
    cost_payload = _load_addressed_json(
        cost_receipt_path, cost_receipt_sha256, directory_addressed=True
    )
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
    _validate_causal_receipt(
        causal_payload, causal_receipt_path, manifest, manifest_sha256,
        predictions_sha256, prediction_payload,
    )
    _validate_sensitivity_receipt(
        sensitivity_payload, sensitivity_receipt_path, manifest_sha256,
        predictions_sha256, causal_receipt_sha256,
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
