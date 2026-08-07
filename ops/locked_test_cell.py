#!/venv/main/bin/python
"""Run exactly one preregistered Calibration cell from the locked checkout.

This helper intentionally uses ``reconstruct_episode_manifest`` rather than
``generate_episode_manifest``.  The latter rehydrates Wilson intervals with
strict float equality and is not portable between the local macOS/libm lock
host and the Linux GPU host.  The parent runner verifies the local
guard-authority receipt, exact tag/tree, lock payload, and canonical manifest
before this helper is allowed to execute any protected episode.
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

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import ConditionSpec, SplitName, load_protocol_config
from mech_int_vla.libero_runtime import RawLiberoEpisode
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.rollout import run_single_episode
from mech_int_vla.snapshots import (
    load_locked_smolvla,
    load_model_input_lock,
    resolve_snapshot_paths,
)
from mech_int_vla.instrumentation import SmolVLAInstrumentation


# Collection identity: the raw set and the manifest stay bound to the commit the
# study was locked at, exactly as for Calibration.
COLLECTION_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
COLLECTION_TAG = "prereg-locked-v1"
# Authorization identity: Locked Test may only run from the committed Calibration
# freeze, which the guard verifies field by field before a single episode runs.
FREEZE_FILE = "locks/calibration_frozen.json"
FREEZE_TAG = "calibration-locked-v1"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
MANIFEST_SHA256 = "1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    return value


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(payload, dict) or _canonical_sha(payload) != MANIFEST_SHA256:
        raise RuntimeError("Calibration manifest is not the canonical guarded payload")
    return payload


def _require_locked_checkout(root: Path) -> None:
    """Authorize Locked Test from the committed Calibration freeze.

    This is the only gate that may open Locked Test.  It checks far more than a
    checkout hash: the freeze file must be tracked at the tagged commit, the tag
    must sit exactly at HEAD, the worktree must be clean including untracked
    files, and every frozen field — task, variable, policy revision, probe,
    predictor, alarm thresholds, patch strength, Calibration metrics — must be
    present and consistent, with the four referenced artifacts hashing to their
    recorded bytes.
    """

    from mech_int_vla.guard import (
        LockedTestGuardConfig,
        assert_locked_test_ready,
    )

    protocol = load_protocol_config(root / "configs")
    receipt = assert_locked_test_ready(
        root,
        LockedTestGuardConfig(
            required_file=Path(FREEZE_FILE),
            required_tag=FREEZE_TAG,
            require_clean_worktree=True,
        ),
        task=protocol.task_order.tasks[0],
        policy_revision=POLICY_REVISION,
        selection=protocol.split.calibration_selection,
    )
    print(
        json.dumps(
            {
                "kind": "locked_test_authorized",
                "head_commit": receipt.head_commit,
                "required_tag": receipt.tag,
                "collection_tag": COLLECTION_TAG,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    _require_locked_checkout(root)
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    lock = load_model_input_lock(args.environment_lock)
    if lock.policy_revision != POLICY_REVISION:
        raise RuntimeError("policy revision differs from the frozen lock")
    manifest_payload = _load_manifest(args.manifest)
    manifest = reconstruct_episode_manifest(
        SplitName.LOCKED_TEST,
        task,
        protocol,
        policy_revision=lock.policy_revision,
        code_commit=COLLECTION_COMMIT,
    )
    if (
        manifest.sha256 != MANIFEST_SHA256
        or _canonical_sha(manifest.to_dict()) != _canonical_sha(manifest_payload)
    ):
        raise RuntimeError("reconstructed Locked Test manifest differs from authority")
    if not 0 <= args.index < len(manifest.episodes):
        raise ValueError("Locked Test episode index is outside the manifest")
    episode_spec = manifest.episodes[args.index]
    destination = args.artifact_root / SplitName.LOCKED_TEST.value / episode_spec.episode_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact {destination}")
    staging = list(destination.parent.glob(f".{episode_spec.episode_id}.tmp-*"))
    if staging:
        raise RuntimeError(f"unexplained staging directories for {episode_spec.episode_id}")

    condition = ConditionSpec(
        episode_spec.condition_name,
        episode_spec.condition_family,
        episode_spec.condition_index,
        episode_spec.condition_parameters,
    )
    snapshots = resolve_snapshot_paths(
        args.environment_lock, cache_dir=args.cache_dir, local_files_only=True
    )
    policy_runtime = load_locked_smolvla(snapshots, device="cuda")
    episode = RawLiberoEpisode.create(
        task,
        base_init_state_id=episode_spec.base_init_state_id,
        execution=protocol.split.policy_execution,
        validity=protocol.perturbations.validity,
    )
    instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
    try:
        result = run_single_episode(
            policy_runtime,
            episode,
            instrumentation,
            episode_spec,
            condition,
            validity_retry_factory=lambda: RawLiberoEpisode.create(
                task,
                base_init_state_id=episode_spec.base_init_state_id,
                execution=protocol.split.policy_execution,
                validity=protocol.perturbations.validity,
            ),
            artifact_root=args.artifact_root,
        )
    finally:
        instrumentation.remove()
        episode.close()

    artifact = load_rollout_artifact(result.artifact_path, expected_task=task)
    if _canonical_sha(artifact.metadata["episode"]) != _canonical_sha(episode_spec.to_dict()):
        raise RuntimeError("published Calibration artifact provenance differs")
    print(
        json.dumps(
            {
                "episode_id": episode_spec.episode_id,
                "artifact_path": str(result.artifact_path),
                "status": result.status,
                "valid_reset": result.valid_reset,
                "success": result.success,
                "control_steps": result.control_steps,
                "metadata_sha256": artifact.hashes.metadata_sha256,
                "trajectory_sha256": artifact.hashes.trajectory_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
