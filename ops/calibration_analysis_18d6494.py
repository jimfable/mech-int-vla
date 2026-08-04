#!/usr/bin/env python3
"""Read-only Calibration failure/probe analysis on the immutable raw set.

This runner is intentionally separate from the collection Supervisor.  It
validates the guarded Calibration manifest, receipt, Reality-Gate lock, and
all 160 raw artifacts, then writes only a new content-addressed analysis
staging directory.  It never opens a Locked-Test path and never writes raw or
backup targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mech_int_vla.allocation import (
    audit_rollout_allocation,
    write_allocation_receipt,
)
from mech_int_vla.artifacts import assemble_probe_cohort, load_rollout_artifact
from mech_int_vla.config import SplitName, TaskSpec, load_protocol_config
from mech_int_vla.failure_events import (
    ReachableBounds,
    annotate_failure_event,
    artifact_identity_from_rollout,
    failure_event_protocol_metadata,
    failure_event_trace_from_artifact,
)
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.probe_artifacts import bind_probe_artifact, write_bound_probe_artifact
from mech_int_vla.probes import (
    evaluate_mean_prediction_baseline,
    evaluate_proprioception_only_baseline,
    evaluate_random_label_baseline,
    evaluate_time_only_baseline,
    select_and_fit_circular_probe,
    write_probe_artifact,
)
from mech_int_vla.provenance import frozen_config_sha256, scoring_source_sha256


LOCK_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
LOCK_TAG = "prereg-locked-v1"
MANIFEST_SHA256 = "6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f"
LOCK_PAYLOAD_SHA256 = "64524c974e62c2ff500c385f049ce0589ca83c220caabc396358a9053051893c"
FAILURE_FREEZE_SHA256 = "dd42e46b055163ca7b8ca777e0bc1a04b9907eab265f87f7234e502a19839328"
RECEIPT_INVENTORY_SHA256 = (
    "2040ad899b95cc691b23afd58cfb9f03de63f5573c035ff8a77216c382840ca8"
)
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
FAILURE_IMPLEMENTATION_COMMIT = "b41867e01ba50e6eec7fd869b4b18c0b8ea46a01"
ANALYSIS_SCHEMA_VERSION = 1
RANDOM_LABEL_SEED = 260803


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, *, require_canonical: bool = False) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    if require_canonical and path.read_bytes() not in {
        _canonical(value),
        _canonical(value) + b"\n",
    }:
        raise RuntimeError(f"{path} is not canonical JSON")
    return value


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite analysis artifact {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(payload)


def _validate_lock_and_manifest(
    root: Path,
    manifest_path: Path,
    receipt_path: Path,
    authority_path: Path,
    task: TaskSpec,
    protocol: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _git(root, "rev-parse", "HEAD") != LOCK_COMMIT:
        raise RuntimeError("locked checkout HEAD drifted")
    if _git(root, "rev-parse", f"refs/tags/{LOCK_TAG}^{{commit}}") != LOCK_COMMIT:
        raise RuntimeError("locked Calibration tag drifted")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("locked checkout is dirty")
    authority = _strict_json(authority_path)
    for key, expected in {
        "head_commit": LOCK_COMMIT,
        "tag_commit": LOCK_COMMIT,
        "calibration_manifest_sha256": MANIFEST_SHA256,
        "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
        "locked_test_accessed": False,
    }.items():
        if authority.get(key) != expected:
            raise RuntimeError(f"authority mismatch at {key}")
    manifest_payload = _strict_json(manifest_path, require_canonical=True)
    manifest = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=POLICY_REVISION,
        code_commit=LOCK_COMMIT,
    )
    if _canonical(manifest.to_dict()) != _canonical(manifest_payload):
        raise RuntimeError("reconstructed Calibration manifest differs from authority")
    if manifest.split is not SplitName.CALIBRATION or len(manifest.episodes) != 160:
        raise RuntimeError("manifest is not the exact 160-cell Calibration set")
    by_init: dict[int, list[int]] = {}
    for episode in manifest.episodes:
        by_init.setdefault(episode.base_init_state_id, []).append(
            episode.condition_index
        )
    if set(by_init) != set(range(10, 30)) or any(
        sorted(indices) != list(range(8)) for indices in by_init.values()
    ):
        raise RuntimeError("manifest does not contain exactly eight cells for init IDs 10..29")
    receipt = _strict_json(receipt_path)
    if (
        receipt.get("kind") != "calibration_collection_receipt"
        or receipt.get("code_commit") != LOCK_COMMIT
        or receipt.get("required_tag") != LOCK_TAG
        or receipt.get("manifest_sha256") != MANIFEST_SHA256
        or receipt.get("locked_test_accessed") is not False
        or len(receipt.get("episodes", [])) != 160
    ):
        raise RuntimeError("completion receipt is not the guarded Calibration receipt")
    receipt_by_id = {item["episode_id"]: item for item in receipt["episodes"]}
    if set(receipt_by_id) != {item.episode_id for item in manifest.episodes}:
        raise RuntimeError("completion receipt episode set differs from manifest")

    lock_path = root / "locks/reality_gate_frozen.json"
    lock_payload = _strict_json(lock_path, require_canonical=True)
    if _sha256_bytes(_canonical(lock_payload)) != LOCK_PAYLOAD_SHA256:
        raise RuntimeError("Reality-Gate lock payload changed")
    freeze = lock_payload.get("failure_event_freeze")
    if not isinstance(freeze, dict) or _sha256_bytes(_canonical(freeze)) != FAILURE_FREEZE_SHA256:
        raise RuntimeError("embedded failure-event freeze hash mismatch")
    if freeze.get("implementation_commit") != FAILURE_IMPLEMENTATION_COMMIT:
        raise RuntimeError("failure-event implementation provenance drifted")
    if freeze.get("task") != {
        "suite": task.suite,
        "task_id": task.task_id,
        "task_rank": task.rank,
        "language": task.language,
        "primary_object": task.primary_object,
        "planar_symmetry_order": task.planar_symmetry_order,
    }:
        raise RuntimeError("failure-event freeze task differs from Calibration task")
    if freeze.get("protocol") != failure_event_protocol_metadata():
        raise RuntimeError("failure-event numerical protocol differs from freeze")
    expanded = freeze.get("bounds", {}).get("expanded_bounds")
    if not isinstance(expanded, dict):
        raise RuntimeError("failure-event freeze lacks expanded bounds")
    bounds = ReachableBounds(
        tuple(float(value) for value in expanded["lower_xyz"]),
        tuple(float(value) for value in expanded["upper_xyz"]),
    )
    return manifest, receipt, authority, {"payload": lock_payload, "freeze": freeze, "bounds": bounds}


def _episode_result_metadata(result: Any) -> dict[str, Any]:
    return {
        "candidate": result.candidate,
        "selected_alpha": result.selected_alpha,
        "mean_mae_rad": result.mean_mae_rad,
        "standard_error_rad": result.standard_error_rad,
        "alpha_results": [
            {
                "alpha": item.alpha,
                "fold_mae_rad": list(item.fold_mae_rad),
                "mean_mae_rad": item.mean_mae_rad,
                "standard_error_rad": item.standard_error_rad,
            }
            for item in result.alpha_results
        ],
    }


def _mean_result_metadata(result: Any) -> dict[str, Any]:
    return {
        "candidate": "mean_prediction",
        "fold_mae_rad": list(result.fold_mae_rad),
        "mean_mae_rad": result.mean_mae_rad,
        "standard_error_rad": result.standard_error_rad,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    manifest, receipt, authority, freeze_info = _validate_lock_and_manifest(
        root,
        args.manifest,
        args.receipt,
        args.authority,
        task,
        protocol,
    )
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"analysis output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    expected_by_id = {episode.episode_id: episode for episode in manifest.episodes}
    receipt_by_id = {item["episode_id"]: item for item in receipt["episodes"]}
    paths = [
        args.raw_root / SplitName.CALIBRATION.value / episode.episode_id
        for episode in sorted(manifest.episodes, key=lambda value: value.episode_id)
    ]
    raw_artifacts = []
    failure_rows = []
    for path in paths:
        artifact = load_rollout_artifact(path, expected_task=task)
        episode = expected_by_id[artifact.episode_id]
        if _canonical(artifact.metadata["episode"]) != _canonical(episode.to_dict()):
            raise RuntimeError(f"manifest provenance mismatch for {artifact.episode_id}")
        declared = receipt_by_id[artifact.episode_id]
        if (
            declared.get("metadata_sha256") != artifact.hashes.metadata_sha256
            or declared.get("trajectory_sha256") != artifact.hashes.trajectory_sha256
            or bool(declared.get("valid_reset")) != artifact.valid_reset
            or bool(declared.get("success")) != artifact.success
        ):
            raise RuntimeError(f"completion receipt hash/outcome mismatch for {artifact.episode_id}")
        raw_artifacts.append(artifact)
        trace = failure_event_trace_from_artifact(artifact)
        annotation = annotate_failure_event(trace, freeze_info["bounds"])
        failure_rows.append(
            {
                "source_artifact": artifact_identity_from_rollout(artifact).to_dict(),
                "annotation": annotation.to_dict(),
            }
        )
    if len(raw_artifacts) != 160:
        raise RuntimeError("independent raw load did not cover 160 cells")

    failure_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "calibration_failure_event_annotations",
        "source": {
            "split": "calibration",
            "code_commit": LOCK_COMMIT,
            "manifest_sha256": MANIFEST_SHA256,
            "completion_receipt_sha256": _sha256_file(args.receipt),
            "raw_inventory_sha256": RECEIPT_INVENTORY_SHA256,
            "reality_gate_lock_payload_sha256": LOCK_PAYLOAD_SHA256,
            "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
            "failure_event_implementation_commit": FAILURE_IMPLEMENTATION_COMMIT,
            "bounds": freeze_info["bounds"].to_dict(),
            "outcome_conditioned": False,
        },
        "counts": {
            "attempted": len(failure_rows),
            "valid_reset": sum(row["annotation"]["valid_reset"] for row in failure_rows),
            "success": sum(row["annotation"]["success"] for row in failure_rows),
            "failed": sum(not row["annotation"]["success"] for row in failure_rows),
            "annotated": sum(row["annotation"]["status"] == "annotated" for row in failure_rows),
            "unannotatable_early_terminal": sum(
                row["annotation"]["status"] == "unannotatable_early_terminal"
                for row in failure_rows
            ),
        },
        "events": failure_rows,
    }
    failure_digest = _write_exclusive(
        output / "failure-annotations.json", _canonical(failure_payload)
    )

    allocation = audit_rollout_allocation(
        manifest,
        raw_artifacts,
        protocol=protocol,
        repo_root=root,
    )
    allocation_path = write_allocation_receipt(
        allocation,
        output / "allocation",
        protocol=protocol,
        repo_root=root,
    )
    del raw_artifacts

    cohort = assemble_probe_cohort(
        paths,
        requested_episode_ids=tuple(expected_by_id),
        expected_task=task,
        stride=5,
    )
    selection = select_and_fit_circular_probe(
        cohort.activation_features,
        cohort.samples,
        selection_config=protocol.split.calibration_selection,
    )
    probe_path = write_probe_artifact(selection.artifact, output / "probe")
    bound = bind_probe_artifact(
        selection.artifact,
        allocation,
        cohort,
        protocol=protocol,
        repo_root=root,
    )
    bound_path = write_bound_probe_artifact(
        bound,
        output / "bound-probe",
        protocol=protocol,
        repo_root=root,
    )
    _write_exclusive(output / "cohort-manifest.json", cohort.manifest.canonical_json().encode())

    # The fixed policy-state vector is the recorded eight-dimensional
    # proprioceptive state.  It is assembled in the same sorted artifact/step
    # order as assemble_probe_cohort, with no outcome-dependent filtering.
    proprio_parts: list[np.ndarray] = []
    for path in paths:
        artifact = load_rollout_artifact(path, expected_task=task)
        if not artifact.valid_reset:
            continue
        steps = artifact.arrays["activation_control_step"]
        selected = np.flatnonzero(steps % 5 == 0)
        proprio_parts.append(
            np.asarray(artifact.arrays["frame_policy_state"][steps[selected]], dtype=np.float64)
        )
    proprio = np.concatenate(proprio_parts, axis=0)
    if proprio.shape[0] != cohort.samples.n_rows or proprio.shape[1] != 8:
        raise RuntimeError("proprioceptive rows do not align with the probe cohort")
    selection_config = protocol.split.calibration_selection
    controls = {
        "time_only": _episode_result_metadata(
            evaluate_time_only_baseline(
                cohort.control_step / float(protocol.split.policy_execution.max_steps),
                cohort.samples,
                selection_config=selection_config,
            )
        ),
        "proprioception_only": _episode_result_metadata(
            evaluate_proprioception_only_baseline(
                proprio, cohort.samples, selection_config=selection_config
            )
        ),
        "random_label": _episode_result_metadata(
            evaluate_random_label_baseline(
                cohort.activation_features[selection.artifact.candidate],
                cohort.samples,
                randomized_theta_rel=cohort.theta_rel[::-1],
                selection_config=selection_config,
            )
        ),
        "mean_prediction": _mean_result_metadata(
            evaluate_mean_prediction_baseline(cohort.samples)
        ),
    }
    probe_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "calibration_probe_selection",
        "source": {
            "split": "calibration",
            "code_commit": LOCK_COMMIT,
            "manifest_sha256": MANIFEST_SHA256,
            "raw_inventory_sha256": RECEIPT_INVENTORY_SHA256,
            "allocation_receipt_sha256": allocation.sha256,
            "cohort_manifest_sha256": cohort.manifest_sha256,
            "cohort_rows": cohort.samples.n_rows,
            "cohort_episodes": cohort.samples.n_episodes,
            "invalid_reset_episode_ids": list(cohort.invalid_reset_episode_ids),
            "configuration_sha256": frozen_config_sha256(root),
            "scoring_source_sha256": scoring_source_sha256(root),
        },
        "probe": {
            "numerical_sha256": selection.artifact.sha256(),
            "bound_sha256": bound.sha256,
            "selected_candidate": selection.artifact.candidate,
            "eligible_candidates": list(selection.eligible_candidates),
            "metadata": json.loads(selection.artifact.canonical_json()),
            "probe_path": str(probe_path),
            "bound_probe_path": str(bound_path),
        },
        "controls": controls,
        "random_label_design": {
            "kind": "fixed_row_reverse",
            "seed": RANDOM_LABEL_SEED,
            "outcome_independent": True,
        },
        "analysis_status": {
            "failure_events": "complete",
            "probe_cv": "complete",
            "m0_m1_m2_scores": "blocked_until_calibration_score_sidecars_exist",
            "locked_test_accessed": False,
        },
    }
    probe_digest = _write_exclusive(output / "probe-analysis.json", _canonical(probe_payload))
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "calibration_analysis_receipt",
        "analysis_root": str(output),
        "failure_annotations_sha256": failure_digest,
        "allocation_receipt_sha256": allocation.sha256,
        "probe_analysis_sha256": probe_digest,
        "numerical_probe_sha256": selection.artifact.sha256(),
        "bound_probe_sha256": bound.sha256,
        "selected_candidate": selection.artifact.candidate,
        "probe_rows": cohort.samples.n_rows,
        "probe_episodes": cohort.samples.n_episodes,
        "events": failure_payload["counts"],
        "m0_m1_m2_scores": "not_started_missing_score_sidecars",
        "locked_test_accessed": False,
        "allocation_receipt_path": str(allocation_path),
    }
    _write_exclusive(output / "analysis-summary.json", _canonical(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
