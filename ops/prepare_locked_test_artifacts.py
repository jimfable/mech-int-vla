#!/usr/bin/env python3
"""Prepare (never run) Locked Test bookkeeping artifacts, CPU-only.

This is the PRE-INSTANCE gate of the Locked Test runbook. It performs the
freeze verification and materializes the two bookkeeping files the locked test
runner requires, WITHOUT instantiating any episode:

1. verify the calibration freeze: tag `calibration-locked-v1` at HEAD, clean
   worktree, and every frozen artifact hash from
   `locks/calibration_frozen.json`;
2. reconstruct the Locked Test manifest (160 episodes = 20 init states x 8
   cells) with `reconstruct_episode_manifest` and verify its canonical digest
   against the runner's constant 1fd8c818...; ANY mismatch raises and the
   runbook says stop + report (no instance start);
3. write `artifacts/manifests/locked-test-manifest-<digest>.json` and
   `artifacts/locked-test-authority.json` (guard bookkeeping copies of the
   frozen authority fields, with locked_test_accessed=True and the Locked Test
   manifest digest bound in).

Protocol: PREREG.md §11. Nothing here reads, writes or validates any Locked
Test rollout; the raw Locked Test set does not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.snapshots import load_model_input_lock

COLLECTION_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
FREEZE_TAG = "calibration-locked-v1"
MANIFEST_SHA256 = "1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49"
LOCK_PAYLOAD_SHA256 = "64524c974e62c2ff500c385f049ce0589ca83c220caabc396358a9053051893c"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()

    # ---------- 1. freeze verification ----------
    head = _git(root, "rev-parse", "HEAD")
    tag_commit = _git(root, "rev-list", "-n", "1", FREEZE_TAG)
    if tag_commit != head:
        raise RuntimeError(
            f"freeze tag {FREEZE_TAG} is not at HEAD ({tag_commit} != {head})"
        )
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("worktree is dirty: refuse to prepare bookkeeping")
    frozen = json.loads((root / "locks/calibration_frozen.json").read_text())
    for name, meta in frozen["artifact_hashes"].items():
        p = root / meta["path"]
        if not p.exists():
            raise RuntimeError(f"frozen artifact missing: {p}")
        if sha256_file(p) != meta["sha256"]:
            raise RuntimeError(f"frozen artifact hash mismatch: {name}")
    print("freeze verified:", FREEZE_TAG, "at", head[:12])

    # ---------- 2. reconstruct the Locked Test manifest ----------
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    lock = load_model_input_lock(args.environment_lock)
    if lock.policy_revision != POLICY_REVISION:
        raise RuntimeError("policy revision differs from the frozen lock")
    manifest = reconstruct_episode_manifest(
        SplitName.LOCKED_TEST,
        task,
        protocol,
        policy_revision=lock.policy_revision,
        code_commit=COLLECTION_COMMIT,
    )
    episodes = list(manifest.episodes)
    if len(episodes) != 160:
        raise RuntimeError(f"expected 160 episodes, got {len(episodes)}")
    canonical = json.dumps(
        manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    print(f"locked-test manifest: {len(episodes)} episodes, sha256 {digest[:12]}...")
    if digest != MANIFEST_SHA256:
        raise RuntimeError(
            "LOCKED TEST MANIFEST DIGEST DOES NOT MATCH THE FROZEN CONSTANT "
            f"({digest} != {MANIFEST_SHA256}); freeze deviation -- stop and report"
        )

    # ---------- 3. write bookkeeping artifacts (committed only on match) ----------
    manifest_path = root / f"artifacts/manifests/locked-test-manifest-{digest}.json"
    manifest_path.write_bytes(canonical)
    print("wrote", manifest_path.name)

    authority_path = root / "artifacts/locked-test-authority.json"
    authority = {
        "schema_version": 1,
        "kind": "locked_test_authority",
        "head_commit": COLLECTION_COMMIT,
        "tag": "prereg-locked-v1",
        "tag_commit": COLLECTION_COMMIT,
        "lock_payload_sha256": LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": (
            "fd82aae6dd90462820a90448d3d75b649578f58ce898e94b31f4a23bfb6e2566"
        ),
        "orientation_eligibility_sha256": (
            "3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb"
        ),
        "reality_gate_lock_receipt_sha256": (
            "17f033b935ea3f600373b5953cdc5bad5c0fd9dfd3dc1260d022acf3355f36f4"
        ),
        "failure_event_freeze_sha256": (
            "dd42e46b055163ca7b8ca777e0bc1a04b9907eab265f87f7234e502a19839328"
        ),
        "calibration_manifest_sha256": MANIFEST_SHA256,
        "locked_test_accessed": True,
        "guard_runtime": "local CPython / macOS prep script (prepare_locked_test_artifacts)",
        "remote_guard_note": (
            "the remote locked-test runner independently verifies the exact tag, "
            "clean tree, lock payload, all authority fields and the manifest digest "
            "before any episode runs; this file is guard bookkeeping only."
        ),
    }
    authority_path.write_text(json.dumps(authority, indent=2) + "\n", encoding="utf-8")
    print("wrote", authority_path.name)
    print("prepared; digest", MANIFEST_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
