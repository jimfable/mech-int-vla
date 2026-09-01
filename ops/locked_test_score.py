#!/usr/bin/env python3
"""Score the immutable Locked Test with frozen Calibration artifacts.

The only learned objects accepted here are the Calibration-bound probe, the
full-Calibration feature-reference bundle, and the all-Calibration predictor
bundle named by ``locks/calibration_frozen.json``.  Locked Test outcomes are
never passed to fitting, selection, or probability-calibration code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mech_int_vla.allocation import (
    audit_rollout_allocation,
    audit_score_allocation,
    load_allocation_receipt,
    write_allocation_receipt,
)
from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import ConditionSpec, SplitName, load_protocol_config
from mech_int_vla.failure_events import artifact_identity_from_rollout
from mech_int_vla.feature_artifacts import (
    load_feature_cohort,
    load_feature_reference_bundle,
    write_feature_cohort,
)
from mech_int_vla.feature_pipeline import FeatureCohort, build_locked_test_features
from mech_int_vla.instrumentation import SmolVLAInstrumentation
from mech_int_vla.libero_runtime import RawLiberoEpisode
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.predictors import FrozenPredictorBundle
from mech_int_vla.probe_artifacts import load_bound_probe_artifact
from mech_int_vla.provenance import content_links_for
from mech_int_vla.scoring import (
    FROZEN_TRANSFORMS,
    load_scoring_sidecar,
    score_replay_to_sidecar,
)
from mech_int_vla.scoring_runtime import (
    SmolVLAScoringAdapter,
    factual_replay_from_artifact,
)
from mech_int_vla.snapshots import load_locked_smolvla, resolve_snapshot_paths

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
SCORING_LOCK_TAG = "locked-test-score-v1"
MANIFEST_SHA256 = "1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
# The bound probe is re-published whenever the scoring source digest changes.
# The probe itself is unchanged — it was fitted on factual rollouts only — but
# its binding records the repository digests and must therefore be re-issued.
# Its digest is taken from the artifact directory name and verified on load,
# which also re-checks the recorded config/code digests against this repository;
# hardcoding it here would require the constant to name the very commit it lives
# in.
ANALYSIS_SCHEMA_VERSION = 1
PREDICTION_KIND = "locked_test_frozen_predictions"
SUMMARY_KIND = "locked_test_score_feature_receipt"
_PREDICTOR_MODELS = ("M0", "M1", "M2")


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
        raise TypeError(f"{path} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"frozen artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_artifact(
    freeze: Mapping[str, Any],
    name: str,
    *,
    root: Path,
) -> tuple[Path, str]:
    artifacts = freeze.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(name), Mapping
    ):
        raise TypeError(f"Calibration freeze is missing artifact_hashes.{name}")
    item = artifacts[name]
    if set(item) != {"path", "sha256"}:
        raise RuntimeError(f"Calibration freeze artifact {name} has unexpected keys")
    declared_path = item["path"]
    declared_sha = item["sha256"]
    if not isinstance(declared_path, str) or not declared_path:
        raise RuntimeError(f"Calibration freeze artifact {name} has an invalid path")
    if (
        not isinstance(declared_sha, str)
        or len(declared_sha) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha)
    ):
        raise RuntimeError(f"Calibration freeze artifact {name} has an invalid SHA-256")
    path = (root / declared_path).resolve()
    if _sha256_file(path) != declared_sha:
        raise RuntimeError(
            f"Calibration freeze artifact {name} differs from its SHA-256"
        )
    return path, declared_sha


def _require_frozen_path(actual: Path, expected: Path, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise RuntimeError(f"{name} path differs from locks/calibration_frozen.json")


def _load_calibration_inputs(
    *,
    root: Path,
    freeze_path: Path,
    bound_probe_path: Path,
    reference_path: Path,
    predictor_metadata_path: Path,
    predictor_bundle_path: Path,
    protocol: Any,
) -> tuple[Any, Any, FrozenPredictorBundle, dict[str, str]]:
    """Load Calibration artifacts only after checking the repository freeze."""

    expected_freeze = (root / "locks" / "calibration_frozen.json").resolve()
    _require_frozen_path(freeze_path, expected_freeze, "Calibration freeze")
    freeze = _strict_json(expected_freeze)
    bound_file, bound_sha = _frozen_artifact(freeze, "bound_probe", root=root)
    reference_metadata, reference_metadata_sha = _frozen_artifact(
        freeze, "feature_reference_metadata", root=root
    )
    reference_arrays, reference_arrays_sha = _frozen_artifact(
        freeze, "feature_reference_arrays", root=root
    )
    predictor_metadata_file, predictor_metadata_sha = _frozen_artifact(
        freeze, "predictor_metadata", root=root
    )
    predictor_file, predictor_sha = _frozen_artifact(
        freeze, "predictor_bundle", root=root
    )
    _require_frozen_path(
        bound_probe_path / "bound_probe.json", bound_file, "bound probe"
    )
    _require_frozen_path(
        reference_path / "metadata.json",
        reference_metadata,
        "feature reference metadata",
    )
    _require_frozen_path(
        reference_path / "arrays.npz", reference_arrays, "feature reference arrays"
    )
    _require_frozen_path(
        predictor_metadata_path, predictor_metadata_file, "predictor metadata"
    )
    _require_frozen_path(predictor_bundle_path, predictor_file, "predictor bundle")

    reference_digest = reference_path.resolve().name
    if len(reference_digest) != 64 or any(
        character not in "0123456789abcdef" for character in reference_digest
    ):
        raise RuntimeError("feature reference directory is not content-addressed")
    reference = load_feature_reference_bundle(
        reference_path.resolve(), expected_sha256=reference_digest
    )
    bound = load_bound_probe_artifact(
        bound_probe_path.resolve(),
        protocol=protocol,
        repo_root=root,
        expected_sha256=bound_sha,
    )

    predictor_bytes = predictor_bundle_path.resolve().read_bytes()
    if _sha256_bytes(predictor_bytes) != predictor_sha:
        raise RuntimeError("predictor bundle changed while loading")
    predictor = FrozenPredictorBundle.from_bytes(predictor_bytes)
    predictor_metadata_bytes = predictor_metadata_path.resolve().read_bytes()
    if _sha256_bytes(predictor_metadata_bytes) != predictor_metadata_sha:
        raise RuntimeError("predictor metadata changed while loading")
    predictor_metadata = _strict_json(predictor_metadata_path.resolve())
    if predictor_metadata_bytes != _canonical(predictor_metadata):
        raise RuntimeError("predictor metadata is not canonical")
    expected_predictor_metadata = predictor.to_metadata()
    # Re-pickling and runtime-version fields legitimately differ after loading
    # the exact frozen bytes in another compatible Python runtime. Bind those
    # two fields to the freeze/file itself, then require every learned parameter,
    # fold, metric, feature name, and Calibration data hash to reconstruct.
    expected_predictor_metadata["artifact"] = {
        "format": "python-pickle-protocol-5",
        "sha256": predictor_sha,
        "size_bytes": len(predictor_bytes),
    }
    expected_predictor_metadata["software"] = predictor_metadata.get("software")
    if _canonical(predictor_metadata) != _canonical(expected_predictor_metadata):
        raise RuntimeError("predictor metadata differs from the executable bundle")

    return (
        bound,
        reference,
        predictor,
        {
            "calibration_freeze_sha256": _sha256_file(expected_freeze),
            "bound_probe_sha256": bound_sha,
            "reference_bundle_sha256": reference_digest,
            "reference_metadata_sha256": reference_metadata_sha,
            "reference_arrays_sha256": reference_arrays_sha,
            "predictor_metadata_sha256": predictor_metadata_sha,
            "predictor_bundle_sha256": predictor_sha,
        },
    )


def _require_authority(root: Path, manifest_path: Path, authority_path: Path) -> Any:
    head = _git(root, "rev-parse", "HEAD")
    scoring_lock = _git(root, "rev-parse", f"refs/tags/{SCORING_LOCK_TAG}^{{commit}}")
    if head != scoring_lock:
        raise RuntimeError("scoring checkout is not at the scoring lock tag")
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)mein_verständnis.md",
    ):
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
        or authority.get("locked_test_accessed") is not True
    ):
        raise RuntimeError("Locked Test authority is not the guarded immutable receipt")
    payload = _strict_json(manifest_path)
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    manifest = reconstruct_episode_manifest(
        SplitName.LOCKED_TEST,
        task,
        protocol,
        policy_revision=POLICY_REVISION,
        code_commit=COLLECTION_COMMIT,
    )
    if _sha256_bytes(_canonical(payload)) != MANIFEST_SHA256:
        raise RuntimeError("Locked Test manifest hash differs from authority")
    if _canonical(payload) != _canonical(manifest.to_dict()):
        raise RuntimeError("Locked Test manifest topology differs from authority")
    return protocol, task, manifest


def _validate_locked_probe_compatibility(bound: Any, allocation: Any) -> None:
    """Accept a Calibration probe across splits, but not across model/task state."""

    calibration_source = bound.rollout.source
    locked_source = allocation.source
    if calibration_source.split is not SplitName.CALIBRATION:
        raise RuntimeError("bound probe is not bound to Calibration")
    if locked_source.split is not SplitName.LOCKED_TEST:
        raise RuntimeError("raw allocation is not Locked Test")
    if (
        calibration_source.task != locked_source.task
        or calibration_source.policy_revision != locked_source.policy_revision
        or calibration_source.base_vlm_revision != locked_source.base_vlm_revision
    ):
        raise RuntimeError("Locked Test inputs differ from bound Calibration inputs")
    # Deliberately do not compare manifest hashes or episode IDs here: a probe
    # fitted on Calibration must have a different manifest and membership.


def _validate_invalid_allocation(
    allocation: Any,
    *,
    max_invalid_fraction: float,
) -> None:
    """Enforce the frozen invalid-reset cap globally and within every cell."""

    if not 0.0 <= max_invalid_fraction < 1.0:
        raise RuntimeError("invalid-reset fraction in protocol is invalid")
    if allocation.invalid_fraction > max_invalid_fraction:
        raise RuntimeError(
            "Locked Test invalid-reset fraction exceeds the frozen allocation cap"
        )
    invalid = set(allocation.invalid_episode_ids)
    cells: dict[tuple[str, int | None], list[str]] = {}
    for episode in allocation.manifest.episodes:
        key = (episode.condition_name, episode.condition_index)
        cells.setdefault(key, []).append(episode.episode_id)
    for key, episode_ids in sorted(cells.items(), key=lambda item: repr(item[0])):
        invalid_count = sum(episode_id in invalid for episode_id in episode_ids)
        fraction = invalid_count / len(episode_ids)
        if fraction > max_invalid_fraction:
            raise RuntimeError(
                f"Locked Test invalid-reset fraction exceeds the frozen cap in cell {key}: "
                f"{invalid_count}/{len(episode_ids)}"
            )


def _validate_resumed_sidecar(
    sidecar: Any,
    *,
    episode_id: str,
    artifact: Any,
    bound: Any,
    root: Path,
    protocol: Any,
) -> None:
    links = content_links_for(
        artifact_identity_from_rollout(artifact),
        bound,
        root,
        protocol=protocol,
    )
    expected_links = {
        "raw_metadata_sha256": links.raw_metadata_sha256,
        "raw_trajectory_sha256": links.raw_trajectory_sha256,
        "probe_sha256": links.probe_sha256,
        "config_sha256": links.config_sha256,
        "code_sha256": links.code_sha256,
    }
    if sidecar.metadata.get("split") != SplitName.LOCKED_TEST.value:
        raise RuntimeError(f"{episode_id}: resumed score is not Locked Test")
    if sidecar.metadata.get("links") != expected_links:
        raise RuntimeError(f"{episode_id}: resumed score has stale source links")


def _wait_for_sidecar(
    path: Path,
    *,
    episode_id: str,
    artifact: Any,
    bound: Any,
    root: Path,
    protocol: Any,
    timeout_seconds: float = 30.0,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if path.exists():
            sidecar = load_scoring_sidecar(path, expected_episode_id=episode_id)
            _validate_resumed_sidecar(
                sidecar,
                episode_id=episode_id,
                artifact=artifact,
                bound=bound,
                root=root,
                protocol=protocol,
            )
            return sidecar
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for concurrent score {episode_id}")
        time.sleep(0.1)


def _apply_frozen_predictors(
    *,
    m0_matrix: np.ndarray,
    m1_matrix: np.ndarray,
    m2_matrix: np.ndarray,
    m0_names: tuple[str, ...],
    m1_names: tuple[str, ...],
    m2_names: tuple[str, ...],
    predictors: FrozenPredictorBundle,
) -> dict[str, np.ndarray]:
    """Apply all-Calibration estimators without accepting any label input."""

    matrices = {
        "M0": m0_matrix,
        "M1": m1_matrix,
        "M2": m2_matrix,
    }
    names = {
        "M0": m0_names,
        "M1": m1_names,
        "M2": m2_names,
    }
    row_counts = {np.asarray(matrix).shape[0] for matrix in matrices.values()}
    if len(row_counts) != 1:
        raise RuntimeError("Locked Test feature matrices have different row counts")
    row_count = next(iter(row_counts))
    predictions: dict[str, np.ndarray] = {}
    for model_name in _PREDICTOR_MODELS:
        model = predictors.model(model_name)
        if tuple(model.feature_names) != tuple(names[model_name]):
            raise RuntimeError(
                f"frozen {model_name} feature names differ from Locked Test features"
            )
        values = np.asarray(
            predictors.predict_proba(model_name, matrices[model_name]),
            dtype=np.float64,
        )
        if (
            values.shape != (row_count,)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise RuntimeError(f"frozen {model_name} returned invalid probabilities")
        predictions[model_name] = values
    return predictions


def _prediction_payload(
    cohort: FeatureCohort,
    probabilities: Mapping[str, np.ndarray],
    *,
    manifest_sha256: str,
    score_allocation_sha256: str,
    feature_cohort_sha256: str,
    frozen_hashes: Mapping[str, str],
    invalid_records: list[dict[str, Any]],
    calibration_data_sha256: str,
) -> dict[str, Any]:
    records = []
    for index, record in enumerate(cohort.records):
        records.append(
            {
                "episode_id": record.episode_id,
                "base_init_state_id": record.base_init_state_id,
                "control_step": record.control_step,
                "terminal_failure_label": record.terminal_failure_label,
                "source_hashes": record.source_hashes.to_metadata(),
                "probabilities": {
                    model: float(probabilities[model][index])
                    for model in _PREDICTOR_MODELS
                },
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": PREDICTION_KIND,
        "source": {
            "manifest_sha256": manifest_sha256,
            "calibration_freeze_sha256": frozen_hashes["calibration_freeze_sha256"],
            "bound_probe_sha256": frozen_hashes["bound_probe_sha256"],
            "score_allocation_sha256": score_allocation_sha256,
            "feature_cohort_sha256": feature_cohort_sha256,
            "reference_bundle_sha256": frozen_hashes["reference_bundle_sha256"],
            "predictor_bundle_sha256": frozen_hashes["predictor_bundle_sha256"],
            "predictor_metadata_sha256": frozen_hashes["predictor_metadata_sha256"],
            "calibration_data_sha256": calibration_data_sha256,
            "label_source": (
                "feature_cohort_terminal_outcome_joined_only_during_prediction_serialization"
            ),
            "prediction_rule": (
                "frozen_all_calibration_predictor_applied_without_label_argument"
            ),
        },
        "counts": {
            "attempted_episodes": len({record.episode_id for record in cohort.records})
            + len(invalid_records),
            "valid_episodes": len({record.episode_id for record in cohort.records}),
            "invalid_resets": len(invalid_records),
            "state_rows": len(records),
        },
        "invalid_resets": invalid_records,
        "records": records,
    }


def _validate_content_root(root: Path, digest: str) -> None:
    """Reject ambiguous old artifacts while allowing one cooperating publisher."""

    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"content-addressed root is unsafe: {root}")
    lock_name = f".{digest}.publish.lock"
    temporary_prefix = f".{digest}.tmp-"
    names = {entry.name for entry in root.iterdir()}
    for entry in root.iterdir():
        if entry.name == digest:
            if entry.is_symlink() or not entry.is_dir():
                raise RuntimeError(f"content-addressed artifact is unsafe: {entry}")
        elif entry.name == lock_name:
            if entry.is_symlink() or not entry.is_file():
                raise RuntimeError(f"publication lock is unsafe: {entry}")
        elif entry.name.startswith(temporary_prefix):
            if lock_name not in names or entry.is_symlink() or not entry.is_dir():
                raise RuntimeError(
                    f"orphan or unsafe publication staging path: {entry}"
                )
        else:
            raise RuntimeError(f"ambiguous artifact in content-addressed root: {entry}")


def _publish_canonical_json(
    output_root: Path, filename: str, value: Mapping[str, Any]
) -> tuple[Path, str]:
    payload = _canonical(value)
    digest = _sha256_bytes(payload)
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _validate_content_root(root, digest)
    destination = root / digest
    target = destination / filename
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError(
                f"content-addressed destination is unsafe: {destination}"
            )
        if {path.name for path in destination.iterdir()} != {filename}:
            raise RuntimeError(
                f"content-addressed destination layout differs: {destination}"
            )
        if target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError(
                f"content-addressed destination bytes differ: {destination}"
            )
        return target, digest
    lock = root / f".{digest}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if destination.exists():
                return _publish_canonical_json(root, filename, value)
            time.sleep(0.1)
        raise RuntimeError(f"timed out waiting for publication {destination}")
    os.close(descriptor)
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{digest}.tmp-", dir=root))
        with (staging / filename).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise RuntimeError(
                f"destination appeared without publication lock: {destination}"
            )
        os.rename(staging, destination)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)
    return destination / filename, digest


def _write_score_allocation_resumable(
    receipt: Any,
    output_root: Path,
    *,
    protocol: Any,
    root: Path,
) -> Path:
    destination = output_root.resolve() / receipt.sha256
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_content_root(output_root.resolve(), receipt.sha256)
    if destination.exists():
        loaded = load_allocation_receipt(
            destination,
            protocol=protocol,
            repo_root=root,
            expected_sha256=receipt.sha256,
        )
        if loaded.sha256 != receipt.sha256:
            raise RuntimeError("resumed score allocation differs")
        return destination
    try:
        return write_allocation_receipt(
            receipt, output_root, protocol=protocol, repo_root=root
        )
    except FileExistsError:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if destination.exists():
                return _write_score_allocation_resumable(
                    receipt, output_root, protocol=protocol, root=root
                )
            time.sleep(0.1)
        raise RuntimeError("timed out waiting for score-allocation publication")


def _write_feature_cohort_resumable(cohort: FeatureCohort, output_root: Path) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cohort-build-", dir=output_root))
    try:
        built = write_feature_cohort(cohort, staging)
        digest = built.name
        destination = output_root / digest
        existing = [entry for entry in output_root.iterdir() if entry != staging]
        for entry in existing:
            if entry.name != digest:
                raise RuntimeError(
                    f"ambiguous artifact in feature cohort root: {entry}"
                )
        if destination.exists():
            load_feature_cohort(destination, expected_sha256=digest)
            return destination
        lock = output_root / f".{digest}.publish.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if destination.exists():
                    load_feature_cohort(destination, expected_sha256=digest)
                    return destination
                time.sleep(0.1)
            raise RuntimeError(f"timed out waiting for feature cohort {digest}")
        os.close(descriptor)
        try:
            if destination.exists():
                load_feature_cohort(destination, expected_sha256=digest)
            else:
                os.rename(built, destination)
        finally:
            lock.unlink(missing_ok=True)
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _sidecar_path(score_root: Path, episode_id: str) -> Path:
    return score_root / "locked_test" / episode_id


def _validate_raw_split_layout(raw_root: Path, expected_episode_ids: set[str]) -> None:
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("raw root must be a real directory")
    split_root = raw_root / SplitName.LOCKED_TEST.value
    if split_root.is_symlink() or not split_root.is_dir():
        raise RuntimeError("Locked Test raw split root must be a real directory")
    entries = {entry.name: entry for entry in split_root.iterdir()}
    missing = sorted(expected_episode_ids - set(entries))
    extra = sorted(set(entries) - expected_episode_ids)
    if missing or extra:
        raise RuntimeError(
            f"Locked Test raw split topology differs: missing={missing}, extra={extra}"
        )
    for episode_id, entry in entries.items():
        if entry.is_symlink() or not entry.is_dir():
            raise RuntimeError(f"Locked Test raw episode path is unsafe: {episode_id}")


def _validate_score_root_layout(
    score_root: Path,
    expected_episode_ids: set[str],
    *,
    allow_active_publication: bool,
) -> None:
    if not score_root.exists():
        return
    if score_root.is_symlink() or not score_root.is_dir():
        raise RuntimeError("score root is unsafe")
    root_entries = list(score_root.iterdir())
    if any(entry.name != SplitName.LOCKED_TEST.value for entry in root_entries):
        raise RuntimeError("score root contains an unexpected split or file")
    split_root = score_root / SplitName.LOCKED_TEST.value
    if not split_root.exists():
        return
    if split_root.is_symlink() or not split_root.is_dir():
        raise RuntimeError("Locked Test score split root is unsafe")
    names = {entry.name for entry in split_root.iterdir()}
    for entry in split_root.iterdir():
        if entry.name in expected_episode_ids:
            if entry.is_symlink() or not entry.is_dir():
                raise RuntimeError(f"score episode path is unsafe: {entry}")
            continue
        matched_episode = next(
            (
                episode_id
                for episode_id in expected_episode_ids
                if entry.name == f".{episode_id}.publish.lock"
                or entry.name.startswith(f".{episode_id}.tmp-")
            ),
            None,
        )
        if matched_episode is None:
            raise RuntimeError(f"unexpected Locked Test score artifact: {entry}")
        if not allow_active_publication:
            raise RuntimeError(f"unfinished Locked Test score publication: {entry}")
        lock_name = f".{matched_episode}.publish.lock"
        if entry.name.startswith(f".{matched_episode}.tmp-") and lock_name not in names:
            raise RuntimeError(f"orphan Locked Test score staging directory: {entry}")
        if (
            entry.is_symlink()
            or (entry.name == lock_name and not entry.is_file())
            or (entry.name != lock_name and not entry.is_dir())
        ):
            raise RuntimeError(f"unsafe Locked Test score publication path: {entry}")


def _validate_feature_root_layout(feature_root: Path) -> None:
    if not feature_root.exists():
        return
    if feature_root.is_symlink() or not feature_root.is_dir():
        raise RuntimeError("feature root is unsafe")
    filenames = {
        "score-allocation": "receipt.json",
        "predictions": "predictions.json",
        "summary": "summary.json",
    }
    allowed = {*filenames, "cohort"}
    for entry in feature_root.iterdir():
        if entry.name not in allowed or entry.is_symlink() or not entry.is_dir():
            raise RuntimeError(f"unexpected or unsafe feature-root artifact: {entry}")
        artifacts = list(entry.iterdir())
        if len(artifacts) > 1:
            raise RuntimeError(f"ambiguous prior finalization artifacts: {entry}")
        if not artifacts:
            continue
        artifact = artifacts[0]
        digest = artifact.name
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or artifact.is_symlink()
            or not artifact.is_dir()
        ):
            raise RuntimeError(f"unsafe prior finalization artifact: {artifact}")
        if entry.name == "cohort":
            load_feature_cohort(artifact, expected_sha256=digest)
            continue
        filename = filenames[entry.name]
        contents = list(artifact.iterdir())
        if len(contents) != 1 or contents[0].name != filename:
            raise RuntimeError(
                f"prior finalization artifact layout differs: {artifact}"
            )
        payload_path = contents[0]
        if payload_path.is_symlink() or not payload_path.is_file():
            raise RuntimeError(f"prior finalization payload is unsafe: {payload_path}")
        payload = payload_path.read_bytes()
        parsed = _strict_json(payload_path)
        if _sha256_bytes(payload) != digest or payload != _canonical(parsed):
            raise RuntimeError(
                f"prior finalization payload hash differs: {payload_path}"
            )


def _sidecar_sha256(path: Path) -> str:
    return _sha256_bytes(
        (path / "metadata.json").read_bytes() + (path / "primitives.npz").read_bytes()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="raw artifact root before the split directory (not .../locked_test)",
    )
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument(
        "--bound-probe",
        type=Path,
        required=True,
        help="content-addressed BoundProbe directory containing bound_probe.json",
    )
    parser.add_argument("--calibration-feature-reference", type=Path, required=True)
    parser.add_argument("--calibration-predictor-metadata", type=Path, required=True)
    parser.add_argument("--calibration-predictor-bundle", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index/count must satisfy 0 <= index < count")
    if args.shard_count > 1 and not args.skip_finalize:
        parser.error(
            "multi-worker shards must use --skip-finalize; finalize in one resume run"
        )

    root = args.repo_root.resolve()
    protocol, task, manifest = _require_authority(
        root, args.manifest.resolve(), args.authority.resolve()
    )
    bound, reference_bundle, predictors, frozen_hashes = _load_calibration_inputs(
        root=root,
        freeze_path=args.calibration_freeze.resolve(),
        bound_probe_path=args.bound_probe.resolve(),
        reference_path=args.calibration_feature_reference.resolve(),
        predictor_metadata_path=args.calibration_predictor_metadata.resolve(),
        predictor_bundle_path=args.calibration_predictor_bundle.resolve(),
        protocol=protocol,
    )

    score_root = args.score_root.resolve()
    feature_root = args.feature_root.resolve()
    if score_root.exists() and not score_root.is_dir():
        raise RuntimeError("score root is not a directory")
    score_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    expected = {episode.episode_id: episode for episode in manifest.episodes}
    _validate_score_root_layout(
        score_root, set(expected), allow_active_publication=True
    )
    _validate_feature_root_layout(feature_root)
    raw_root = args.raw_root.resolve()
    if raw_root.name == SplitName.LOCKED_TEST.value:
        raise RuntimeError("--raw-root must name the root before the locked_test split")
    _validate_raw_split_layout(raw_root, set(expected))
    raw_paths = {
        episode.episode_id: raw_root / SplitName.LOCKED_TEST.value / episode.episode_id
        for episode in manifest.episodes
    }

    raw_by_id = {
        episode_id: load_rollout_artifact(raw_paths[episode_id], expected_task=task)
        for episode_id in sorted(expected)
    }
    allocation = audit_rollout_allocation(
        manifest,
        list(raw_by_id.values()),
        protocol=protocol,
        repo_root=root,
    )
    _validate_invalid_allocation(
        allocation,
        max_invalid_fraction=protocol.perturbations.validity.max_invalid_fraction,
    )
    _validate_locked_probe_compatibility(bound, allocation)
    valid_ids = list(allocation.valid_episode_ids)
    invalid_ids = list(allocation.invalid_episode_ids)
    shard_ids = valid_ids[args.shard_index :: args.shard_count]
    if args.reverse:
        shard_ids.reverse()

    snapshots = None
    policy_runtime = None
    pending = [
        episode_id
        for episode_id in shard_ids
        if not _sidecar_path(score_root, episode_id).exists()
    ]
    if pending:
        snapshots = resolve_snapshot_paths(
            args.environment_lock.resolve(),
            cache_dir=args.cache_dir.resolve(),
            local_files_only=True,
        )
        policy_runtime = load_locked_smolvla(snapshots, device="cuda")

    completed = 0
    for episode_id in shard_ids:
        destination = _sidecar_path(score_root, episode_id)
        artifact = raw_by_id[episode_id]
        if destination.exists():
            sidecar = load_scoring_sidecar(destination, expected_episode_id=episode_id)
            _validate_resumed_sidecar(
                sidecar,
                episode_id=episode_id,
                artifact=artifact,
                bound=bound,
                root=root,
                protocol=protocol,
            )
            completed += 1
            print(
                json.dumps(
                    {"kind": "score_resume_validated", "episode_id": episode_id}
                ),
                flush=True,
            )
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
            try:
                result = score_replay_to_sidecar(
                    adapter,
                    replay,
                    links,
                    transforms=FROZEN_TRANSFORMS,
                    output_root=score_root,
                )
                result_sha = result.sha256
            except FileExistsError:
                concurrent = _wait_for_sidecar(
                    destination,
                    episode_id=episode_id,
                    artifact=artifact,
                    bound=bound,
                    root=root,
                    protocol=protocol,
                )
                result_sha = _sidecar_sha256(concurrent.path)
            completed += 1
            print(
                json.dumps(
                    {
                        "kind": "score_completed",
                        "episode_id": episode_id,
                        "sha256": result_sha,
                        "completed": completed,
                        "total": len(shard_ids),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            instrumentation.remove()
            episode.close()

    if args.skip_finalize:
        print(
            json.dumps(
                {
                    "kind": "score_shard_complete",
                    "completed": completed,
                    "total": len(shard_ids),
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "reverse": bool(args.reverse),
                    "invalid_resets_excluded": len(invalid_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    _validate_score_root_layout(
        score_root, set(expected), allow_active_publication=False
    )
    raw_artifacts = [raw_by_id[episode_id] for episode_id in valid_ids]
    sidecars = []
    for episode_id in valid_ids:
        sidecar = load_scoring_sidecar(
            _sidecar_path(score_root, episode_id), expected_episode_id=episode_id
        )
        _validate_resumed_sidecar(
            sidecar,
            episode_id=episode_id,
            artifact=raw_by_id[episode_id],
            bound=bound,
            root=root,
            protocol=protocol,
        )
        sidecars.append(sidecar)
    score_allocation = audit_score_allocation(
        allocation,
        sidecars,
        bound,
        protocol=protocol,
        repo_root=root,
    )
    score_receipt_path = _write_score_allocation_resumable(
        score_allocation,
        feature_root / "score-allocation",
        protocol=protocol,
        root=root,
    )
    feature_cohort = build_locked_test_features(
        raw_artifacts,
        sidecars,
        bound,
        score_allocation,
        reference_bundle,
        protocol=protocol,
        repo_root=root,
    )
    cohort_path = _write_feature_cohort_resumable(
        feature_cohort, feature_root / "cohort"
    )
    cohort_artifact_sha = cohort_path.name

    # This call has no label argument.  Only after the frozen probabilities
    # exist do we join terminal outcomes into the immutable prediction receipt.
    probabilities = _apply_frozen_predictors(
        m0_matrix=feature_cohort.m0_matrix,
        m1_matrix=feature_cohort.m1_matrix,
        m2_matrix=feature_cohort.m2_matrix,
        m0_names=feature_cohort.m0_names,
        m1_names=feature_cohort.m1_names,
        m2_names=feature_cohort.m2_names,
        predictors=predictors,
    )
    invalid_records = [
        {
            "episode_id": episode_id,
            "base_init_state_id": expected[episode_id].base_init_state_id,
            "condition_index": expected[episode_id].condition_index,
        }
        for episode_id in invalid_ids
    ]
    prediction_payload = _prediction_payload(
        feature_cohort,
        probabilities,
        manifest_sha256=MANIFEST_SHA256,
        score_allocation_sha256=score_allocation.sha256,
        feature_cohort_sha256=cohort_artifact_sha,
        frozen_hashes=frozen_hashes,
        invalid_records=invalid_records,
        calibration_data_sha256=predictors.calibration_data_sha256,
    )
    prediction_path, prediction_sha = _publish_canonical_json(
        feature_root / "predictions", "predictions.json", prediction_payload
    )
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "manifest_sha256": MANIFEST_SHA256,
        "calibration_freeze_sha256": frozen_hashes["calibration_freeze_sha256"],
        "bound_probe_sha256": bound.sha256,
        "score_allocation_sha256": score_allocation.sha256,
        "score_allocation_path": str(score_receipt_path),
        "attempted_episode_count": len(expected),
        "sidecar_count": len(sidecars),
        "invalid_reset_count": len(invalid_ids),
        "calibration_feature_reference_path": str(
            args.calibration_feature_reference.resolve()
        ),
        "calibration_feature_reference_sha256": frozen_hashes[
            "reference_bundle_sha256"
        ],
        "feature_cohort_path": str(cohort_path),
        "feature_cohort_sha256": cohort_artifact_sha,
        "feature_cohort_provenance_sha256": feature_cohort.provenance_sha256,
        "calibration_predictor_bundle_path": str(
            args.calibration_predictor_bundle.resolve()
        ),
        "calibration_predictor_bundle_sha256": frozen_hashes["predictor_bundle_sha256"],
        "calibration_predictor_metadata_sha256": frozen_hashes[
            "predictor_metadata_sha256"
        ],
        "calibration_data_sha256": predictors.calibration_data_sha256,
        "prediction_receipt_path": str(prediction_path),
        "prediction_receipt_sha256": prediction_sha,
        "predictor_selected_family": predictors.family,
        "predictor_selected_hyperparameters": dict(predictors.hyperparameters),
        "kill_switch_1_triggered": predictors.kill_switch_triggered,
        "prediction_rule": (
            "frozen_all_calibration_predictor_applied_without_label_argument"
        ),
        "locked_test_accessed": True,
    }
    summary_path, summary_sha = _publish_canonical_json(
        feature_root / "summary", "summary.json", summary
    )
    print(
        json.dumps(
            {
                **summary,
                "summary_path": str(summary_path),
                "summary_sha256": summary_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
