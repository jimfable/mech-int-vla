#!/usr/bin/env python3
"""Serial, resumable Calibration counterfactual scoring.

The script consumes only the immutable Calibration raw set and the bound probe
published by ``calibration_analysis_18d6494.py``.  It never instantiates or
opens Locked Test, never writes raw/backup paths, and refuses to overwrite an
existing score sidecar.  A shell ``flock`` around this script is recommended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mech_int_vla.allocation import (
    audit_rollout_allocation,
    audit_score_allocation,
    write_allocation_receipt,
)
from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import ConditionSpec, SplitName, load_protocol_config
from mech_int_vla.feature_artifacts import (
    write_feature_cohort,
    write_feature_reference_bundle,
)
from mech_int_vla.feature_pipeline import build_calibration_features
from mech_int_vla.failure_events import artifact_identity_from_rollout
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.predictors import FeatureSet, fit_failure_predictors
from mech_int_vla.probe_artifacts import load_bound_probe_artifact
from mech_int_vla.provenance import content_links_for
from mech_int_vla.scoring import FROZEN_TRANSFORMS, load_scoring_sidecar, score_replay_to_sidecar
from mech_int_vla.scoring_runtime import SmolVLAScoringAdapter, factual_replay_from_artifact
from mech_int_vla.snapshots import load_locked_smolvla, resolve_snapshot_paths
from mech_int_vla.instrumentation import SmolVLAInstrumentation
from mech_int_vla.libero_runtime import RawLiberoEpisode


# Two distinct commits, deliberately separated.
#
# COLLECTION_* identifies the commit the raw rollouts were collected at.  The
# raw set, its authority receipt and the episode manifest are all bound to it and
# must never move: re-scoring does not re-collect anything, and the manifest must
# still reconstruct to the same digest.
#
# SCORING_LOCK_TAG identifies the commit the *scoring code* runs at.  It changes
# whenever a file in SCORING_SOURCE_FILES changes, as it did for the
# counterfactual re-render fix.  It is expressed as a tag rather than a hardcoded
# hash because the constant would otherwise have to name the very commit that
# contains it; requiring HEAD to equal the tag is self-consistent and equally
# strict, since the guard also demands a clean, fully tracked worktree.
COLLECTION_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
COLLECTION_TAG = "prereg-locked-v1"
SCORING_LOCK_TAG = "calibration-rescore-v1"
MANIFEST_SHA256 = "6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
# The bound probe is re-published whenever the scoring source digest changes.
# The probe itself is unchanged — it was fitted on factual rollouts only — but
# its binding records the repository digests and must therefore be re-issued.
# Its digest is taken from the artifact directory name and verified on load,
# which also re-checks the recorded config/code digests against this repository;
# hardcoding it here would require the constant to name the very commit it lives
# in.
ANALYSIS_SCHEMA_VERSION = 1


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


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(payload)


def _require_authority(root: Path, manifest_path: Path, authority_path: Path) -> Any:
    head = _git(root, "rev-parse", "HEAD")
    scoring_lock = _git(root, "rev-parse", f"refs/tags/{SCORING_LOCK_TAG}^{{commit}}")
    if head != scoring_lock:
        raise RuntimeError("scoring checkout is not at the scoring lock tag")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("locked checkout is dirty")
    # The collection tag must still exist and still point at the collection
    # commit: re-scoring may advance the scoring code but must never move the
    # commit the raw set was collected at.
    if (
        _git(root, "rev-parse", f"refs/tags/{COLLECTION_TAG}^{{commit}}")
        != COLLECTION_COMMIT
    ):
        raise RuntimeError("collection tag drifted away from the collection commit")
    authority = _strict_json(authority_path)
    if (
        authority.get("head_commit") != COLLECTION_COMMIT
        or authority.get("tag_commit") != COLLECTION_COMMIT
        or authority.get("calibration_manifest_sha256") != MANIFEST_SHA256
        or authority.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("Calibration authority is not the guarded immutable receipt")
    payload = _strict_json(manifest_path)
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    manifest = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=POLICY_REVISION,
        code_commit=COLLECTION_COMMIT,
    )
    if _sha256_bytes(_canonical(payload)) != MANIFEST_SHA256:
        raise RuntimeError("Calibration manifest hash differs from authority")
    if _canonical(payload) != _canonical(manifest.to_dict()):
        raise RuntimeError("Calibration manifest topology differs from authority")
    return protocol, task, manifest


def _save_pickle(path: Path, value: Any) -> tuple[Path, str]:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.tmp-", dir=path.parent, delete=False) as stream:
        stream.write(pickle.dumps(value, protocol=5))
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.rename(temporary, path)
    payload = path.read_bytes()
    return path, _sha256_bytes(payload)


def _bound_probe_digest(path: Path) -> str:
    """The artifact directory name IS its content address; verify the shape."""

    digest = path.resolve().name
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("bound probe path does not end in a SHA-256 digest")
    return digest


def _sidecar_path(score_root: Path, episode_id: str) -> Path:
    return score_root / "calibration" / episode_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bound-probe", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    protocol, task, manifest = _require_authority(
        root, args.manifest.resolve(), args.authority.resolve()
    )
    bound = load_bound_probe_artifact(
        args.bound_probe.resolve(),
        protocol=protocol,
        repo_root=root,
        expected_sha256=_bound_probe_digest(args.bound_probe),
    )
    if bound.rollout.source.manifest_sha256 != MANIFEST_SHA256:
        raise RuntimeError("bound probe manifest hash differs from authority")

    score_root = args.score_root.resolve()
    feature_root = args.feature_root.resolve()
    if score_root.exists() and not score_root.is_dir():
        raise RuntimeError("score root is not a directory")
    score_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    expected = {episode.episode_id: episode for episode in manifest.episodes}
    raw_paths = {
        episode.episode_id: args.raw_root / "calibration" / episode.episode_id
        for episode in manifest.episodes
    }

    valid_ids: list[str] = []
    invalid_ids: list[str] = []
    # Read one artifact at a time so this preflight does not retain image arrays.
    for episode_id in sorted(expected):
        artifact = load_rollout_artifact(raw_paths[episode_id], expected_task=task)
        if artifact.valid_reset:
            valid_ids.append(episode_id)
        else:
            invalid_ids.append(episode_id)
    if invalid_ids:
        raise RuntimeError(f"Calibration scoring unexpectedly saw invalid resets: {invalid_ids}")
    if tuple(valid_ids) != bound.rollout.valid_episode_ids:
        raise RuntimeError("valid raw episode set differs from bound probe allocation")

    snapshots = None
    policy_runtime = None
    pending = [episode_id for episode_id in valid_ids if not _sidecar_path(score_root, episode_id).exists()]
    if pending:
        snapshots = resolve_snapshot_paths(
            args.environment_lock.resolve(), cache_dir=args.cache_dir.resolve(), local_files_only=True
        )
        policy_runtime = load_locked_smolvla(snapshots, device="cuda")

    completed = 0
    for episode_id in valid_ids:
        destination = _sidecar_path(score_root, episode_id)
        if destination.exists():
            # Full loader verification is the resume guard; later allocation
            # audit rechecks every raw/probe/source link.
            load_scoring_sidecar(destination, expected_episode_id=episode_id)
            completed += 1
            print(json.dumps({"kind": "score_resume_validated", "episode_id": episode_id}), flush=True)
            continue
        if policy_runtime is None:
            raise RuntimeError("policy runtime was not loaded for a missing sidecar")
        spec = expected[episode_id]
        condition = ConditionSpec(
            spec.condition_name,
            spec.condition_family,
            spec.condition_index,
            spec.condition_parameters,
        )
        artifact = load_rollout_artifact(raw_paths[episode_id], expected_task=task)
        episode = RawLiberoEpisode.create(
            task,
            base_init_state_id=spec.base_init_state_id,
            execution=protocol.split.policy_execution,
            validity=protocol.perturbations.validity,
        )
        instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
        try:
            adapter = SmolVLAScoringAdapter(
                episode,
                policy_runtime,
                artifact,
                bound,
                instrumentation,
                reset_seed=spec.reset_seed,
                original_condition=condition,
                protocol=protocol,
                repo_root=root,
            )
            replay = factual_replay_from_artifact(artifact)
            links = content_links_for(
                artifact_identity_from_rollout(artifact),
                bound,
                root,
                protocol=protocol,
            )
            result = score_replay_to_sidecar(
                adapter,
                replay,
                links,
                transforms=FROZEN_TRANSFORMS,
                output_root=score_root,
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "kind": "score_completed",
                        "episode_id": episode_id,
                        "sha256": result.sha256,
                        "completed": completed,
                        "total": len(valid_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            instrumentation.remove()
            episode.close()

    # Build the exact score allocation and downstream M0/M1/M2 feature cohort
    # only after every valid raw has a fully validated sidecar.
    raw_artifacts = [
        load_rollout_artifact(raw_paths[episode_id], expected_task=task)
        for episode_id in valid_ids
    ]
    sidecars = [
        load_scoring_sidecar(
            _sidecar_path(score_root, episode_id),
            expected_episode_id=episode_id,
        )
        for episode_id in valid_ids
    ]
    allocation = audit_rollout_allocation(
        manifest,
        raw_artifacts + [
            load_rollout_artifact(raw_paths[episode_id], expected_task=task)
            for episode_id in invalid_ids
        ],
        protocol=protocol,
        repo_root=root,
    )
    score_allocation = audit_score_allocation(
        allocation,
        sidecars,
        bound,
        protocol=protocol,
        repo_root=root,
    )
    score_receipt_path = write_allocation_receipt(
        score_allocation,
        feature_root / "score-allocation",
        protocol=protocol,
        repo_root=root,
    )
    reference_bundle, feature_cohort = build_calibration_features(
        raw_artifacts,
        sidecars,
        bound,
        score_allocation,
        protocol=protocol,
        repo_root=root,
    )
    reference_path = write_feature_reference_bundle(reference_bundle, feature_root / "reference")
    cohort_path = write_feature_cohort(feature_cohort, feature_root / "cohort")
    labels = np.asarray(
        [record.terminal_failure_label for record in feature_cohort.records], dtype=np.int8
    )
    episodes = [record.episode_id for record in feature_cohort.records]
    groups = [record.base_init_state_id for record in feature_cohort.records]
    predictors = fit_failure_predictors(
        m0=FeatureSet(feature_cohort.m0_matrix, feature_cohort.m0_names),
        m1=FeatureSet(feature_cohort.m1_matrix, feature_cohort.m1_names),
        m2=FeatureSet(feature_cohort.m2_matrix, feature_cohort.m2_names),
        labels=labels,
        episode_ids=episodes,
        base_init_ids=groups,
        selection_config=protocol.split.calibration_selection,
    )
    predictor_payload = _canonical(predictors.to_metadata())
    predictor_metadata_digest = _write_exclusive(feature_root / "predictors.json", predictor_payload)
    predictor_pickle_path, predictor_pickle_digest = _save_pickle(
        feature_root / "predictors.pkl", predictors
    )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": "calibration_score_feature_receipt",
        "manifest_sha256": MANIFEST_SHA256,
        "bound_probe_sha256": bound.sha256,
        "score_allocation_sha256": score_allocation.sha256,
        "score_allocation_path": str(score_receipt_path),
        "sidecar_count": len(sidecars),
        "reference_bundle_path": str(reference_path),
        "feature_cohort_path": str(cohort_path),
        "feature_cohort_sha256": feature_cohort.provenance_sha256,
        "predictor_metadata_sha256": predictor_metadata_digest,
        "predictor_pickle_path": str(predictor_pickle_path),
        "predictor_pickle_sha256": predictor_pickle_digest,
        "predictor_selected_family": predictors.family,
        "predictor_selected_hyperparameters": dict(predictors.hyperparameters),
        "kill_switch_1_triggered": predictors.kill_switch_triggered,
        "models": {
            predictor.name: {
                "raw_oof_auroc": predictor.raw_oof_auroc,
                "calibrated_oof_auroc": predictor.calibrated_oof_auroc,
                "raw_oof_log_loss": predictor.raw_oof_log_loss,
                "calibrated_oof_log_loss": predictor.calibrated_oof_log_loss,
            }
            for predictor in predictors.predictors
        },
        "locked_test_accessed": False,
    }
    _write_exclusive(feature_root / "score-feature-summary.json", _canonical(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
