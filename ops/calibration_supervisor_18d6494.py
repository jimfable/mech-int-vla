#!/venv/main/bin/python
"""Resume/execute the 160-cell Calibration collection under Supervisor.

The process is fail-closed and resumable: an existing cell is only accepted
after full artifact/provenance validation, staging directories abort the run,
and the atomic rollout writer refuses overwrite.  Locked Test is never
instantiated or inspected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.snapshots import load_model_input_lock


LOCK_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
LOCK_TAG = "prereg-locked-v1"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
MANIFEST_SHA256 = "6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f"
LOCK_PAYLOAD_SHA256 = "64524c974e62c2ff500c385f049ce0589ca83c220caabc396358a9053051893c"
REALITY_GATE_RECEIPT_SHA256 = "fd82aae6dd90462820a90448d3d75b649578f58ce898e94b31f4a23bfb6e2566"
ORIENTATION_SHA256 = "3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb"
LOCK_RECEIPT_SHA256 = "17f033b935ea3f600373b5953cdc5bad5c0fd9dfd3dc1260d022acf3355f36f4"
FAILURE_FREEZE_SHA256 = "dd42e46b055163ca7b8ca777e0bc1a04b9907eab265f87f7234e502a19839328"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha_file(path: Path) -> str:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    return _canonical_sha(payload)


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
    if _git(root, "rev-parse", "HEAD") != LOCK_COMMIT:
        raise RuntimeError("locked Calibration checkout drifted")
    if _git(root, "rev-parse", f"refs/tags/{LOCK_TAG}^{{commit}}") != LOCK_COMMIT:
        raise RuntimeError("locked Calibration tag drifted")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("locked Calibration checkout is dirty")


def _validate_artifact(destination: Path, task: Any, expected: Any) -> dict[str, Any]:
    artifact = load_rollout_artifact(destination, expected_task=task)
    if _canonical_sha(artifact.metadata["episode"]) != _canonical_sha(expected.to_dict()):
        raise RuntimeError(f"artifact provenance mismatch: {expected.episode_id}")
    return {
        "episode_id": artifact.episode_id,
        "metadata_sha256": artifact.hashes.metadata_sha256,
        "trajectory_sha256": artifact.hashes.trajectory_sha256,
        "valid_reset": artifact.valid_reset,
        "success": artifact.success,
        "control_steps": artifact.action_count,
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing Calibration completion receipt differs")
        return
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True).encode()
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.tmp-", dir=path.parent, delete=False
    ) as stream:
        stream.write(encoded)
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cell-script", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    _require_locked_checkout(root)
    if _canonical_json_sha_file(root / "locks/reality_gate_frozen.json") != LOCK_PAYLOAD_SHA256:
        raise RuntimeError("Reality-Gate lock payload changed")
    authority = json.loads(
        args.authority.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if authority.get("head_commit") != LOCK_COMMIT or authority.get("tag_commit") != LOCK_COMMIT:
        raise RuntimeError("local guard authority is not bound to the locked tag")
    expected_authority = {
        "lock_payload_sha256": LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": REALITY_GATE_RECEIPT_SHA256,
        "orientation_eligibility_sha256": ORIENTATION_SHA256,
        "reality_gate_lock_receipt_sha256": LOCK_RECEIPT_SHA256,
        "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
        "calibration_manifest_sha256": MANIFEST_SHA256,
        "locked_test_accessed": False,
    }
    for key, expected in expected_authority.items():
        if authority.get(key) != expected:
            raise RuntimeError(f"guard authority mismatch at {key}")

    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    lock = load_model_input_lock(args.environment_lock)
    if lock.policy_revision != POLICY_REVISION:
        raise RuntimeError("policy revision differs from the frozen lock")
    manifest_payload = _load_manifest(args.manifest)
    manifest = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=lock.policy_revision,
        code_commit=LOCK_COMMIT,
    )
    if (
        manifest.sha256 != MANIFEST_SHA256
        or _canonical_sha(manifest.to_dict()) != _canonical_sha(manifest_payload)
    ):
        raise RuntimeError("reconstructed Calibration manifest differs from authority")
    if args.plan_only:
        print(
            json.dumps(
                {
                    "kind": "calibration_plan_validated",
                    "episodes": len(manifest.episodes),
                    "manifest_sha256": MANIFEST_SHA256,
                    "code_commit": LOCK_COMMIT,
                    "required_tag": LOCK_TAG,
                    "locked_test_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0

    calibration_root = args.artifact_root / SplitName.CALIBRATION.value
    calibration_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(root / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env.update(
        {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    for index, spec in enumerate(manifest.episodes):
        destination = calibration_root / spec.episode_id
        if destination.exists():
            record = _validate_artifact(destination, task, spec)
            records.append(record)
            print(json.dumps({"kind": "resume_validated", **record}, sort_keys=True), flush=True)
            continue
        staging = list(calibration_root.glob(f".{spec.episode_id}.tmp-*"))
        if staging:
            raise RuntimeError(f"unexplained staging directories for {spec.episode_id}")
        log_path = args.log_dir / f"{spec.episode_id}.log"
        print(
            json.dumps(
                {"kind": "starting", "index": index, "episode_id": spec.episode_id, "log": str(log_path)},
                sort_keys=True,
            ),
            flush=True,
        )
        command = [
            sys.executable,
            str(args.cell_script),
            "--index",
            str(index),
            "--repo-root",
            str(root),
            "--environment-lock",
            str(args.environment_lock),
            "--manifest",
            str(args.manifest),
            "--cache-dir",
            str(args.cache_dir),
            "--artifact-root",
            str(args.artifact_root),
        ]
        with log_path.open("ab", buffering=0) as stream:
            stream.write(
                (f"\n=== launch {spec.episode_id} commit={LOCK_COMMIT} index={index} ===\n").encode()
            )
            subprocess.run(
                command,
                cwd=root,
                env=child_env,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record = _validate_artifact(destination, task, spec)
        records.append(record)
        print(json.dumps({"kind": "completed", **record}, sort_keys=True), flush=True)

    payload = {
        "schema_version": 1,
        "kind": "calibration_collection_receipt",
        "code_commit": LOCK_COMMIT,
        "required_tag": LOCK_TAG,
        "policy_revision": POLICY_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "lock_payload_sha256": LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": REALITY_GATE_RECEIPT_SHA256,
        "orientation_eligibility_sha256": ORIENTATION_SHA256,
        "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
        "locked_test_accessed": False,
        "episodes": records,
    }
    _write_receipt(args.completion_receipt, payload)
    print(
        json.dumps(
            {
                "kind": "calibration_complete",
                "episodes": len(records),
                "successes": sum(bool(r["success"]) for r in records),
                "valid_resets": sum(bool(r["valid_reset"]) for r in records),
                "manifest_sha256": MANIFEST_SHA256,
                "completion_receipt": str(args.completion_receipt),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
