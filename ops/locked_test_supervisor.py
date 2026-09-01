#!/venv/main/bin/python
"""Resume/execute the 160-cell Locked Test collection under one Supervisor.

The process is fail-closed and resumable: an existing cell is only accepted
after full artifact/provenance validation, staging directories abort the run,
and the atomic rollout writer refuses overwrite.  ``--plan-only`` performs all
CPU-safe path, cache, manifest, resume, and disk checks without loading a GPU
model or instantiating a simulator.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Make the checkout self-contained when this file is launched by absolute path.
# This must precede project imports; no external PYTHONPATH is required.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _SCRIPT_REPO_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import SplitName, load_protocol_config
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.snapshots import load_model_input_lock, resolve_snapshot_paths

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
LOCK_PAYLOAD_SHA256 = "64524c974e62c2ff500c385f049ce0589ca83c220caabc396358a9053051893c"
REALITY_GATE_RECEIPT_SHA256 = "fd82aae6dd90462820a90448d3d75b649578f58ce898e94b31f4a23bfb6e2566"
ORIENTATION_SHA256 = "3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb"
LOCK_RECEIPT_SHA256 = "17f033b935ea3f600373b5953cdc5bad5c0fd9dfd3dc1260d022acf3355f36f4"
FAILURE_FREEZE_SHA256 = "dd42e46b055163ca7b8ca777e0bc1a04b9907eab265f87f7234e502a19839328"
ENVIRONMENT_LOCK_RELPATH = Path("environment.lock")
CELL_SCRIPT_RELPATH = Path("ops/locked_test_cell.py")
GPU_FREEZE_RELPATH = Path("environment-gpu.freeze")
GPU_FREEZE_SHA256 = "d738fb679db3682292481dfb74154b2d1d22da37630fd0156092c281ff31f821"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
EXPECTED_GPU_COMPUTE_CAPABILITY = "12.0"
EXPECTED_CUDA_VERSION = "13.0"
REQUIRED_DISTRIBUTIONS = {
    "lerobot": "0.6.0",
    "hf-libero": "0.1.4",
    "mujoco": "3.8.1",
    "torch": "2.11.0",
    "torchvision": "0.26.0+cu130",
    "transformers": "5.5.4",
    "huggingface-hub": "1.18.0",
    "egl-probe": "1.0.2",
    "hf-egl-probe": "1.0.2",
}
REQUIRED_COLLECTION_ENVIRONMENT = {
    "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_GIB = 1024**3
# Calibration used ~14.6 GB for 160 cells.  This deliberately conservative
# bound leaves room for atomic staging plus filesystem/runtime headroom.
MIN_FREE_RESERVE_BYTES = 5 * _GIB
MAX_PLANNED_BYTES_PER_MISSING_EPISODE = 256 * 1024**2


def _configure_collection_environment() -> None:
    for name, expected in REQUIRED_COLLECTION_ENVIRONMENT.items():
        present = os.environ.get(name)
        if present is not None and present != expected:
            raise RuntimeError(
                f"unsafe Locked Test supervisor environment: {name}={present!r}, "
                f"required {expected!r}"
            )
        os.environ[name] = expected


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _require_exact_tracked_repo_file(
    root: Path, supplied: Path, expected_relative: Path
) -> Path:
    """Bind a security-sensitive CLI path to the exact blob at ``HEAD``."""

    if supplied.is_symlink():
        raise RuntimeError(f"{expected_relative.as_posix()} argument may not be a symlink")
    expected = (root / expected_relative).resolve(strict=True)
    actual = supplied.resolve(strict=True)
    if actual != expected:
        raise RuntimeError(
            f"{expected_relative.as_posix()} must be the exact file in --repo-root"
        )
    relative = expected_relative.as_posix()
    try:
        _git(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"required repository file is not tracked: {relative}") from exc
    if actual.read_bytes() != committed:
        raise RuntimeError(f"required repository file bytes differ from HEAD: {relative}")
    return actual


def _require_input_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} may not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{label} must be a regular, non-symlink file: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise RuntimeError(f"{label} is not readable: {resolved}")
    return resolved


def _require_output_parent(path: Path, label: str) -> Path:
    """Require an existing writable parent and prove a small atomic write works."""

    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(f"{label} parent must be a real directory: {parent}")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"{label} parent is not writable/searchable: {parent}")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".locked-test-preflight-", dir=parent
        )
        try:
            os.write(descriptor, b"locked-test-preflight\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        Path(temporary_name).unlink()
    except OSError as exc:
        raise RuntimeError(f"cannot write {label} parent {parent}: {exc}") from exc
    return parent


def _require_output_directory(path: Path, label: str) -> Path:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"{label} must be a real directory: {path}")
        probe = path / ".locked-test-output-probe"
        return _require_output_parent(probe, label)
    return _require_output_parent(path, label)


def _supervisor_lock_path(artifact_root: Path) -> Path:
    """Use one lock per canonical raw-artifact root, independent of log/receipt."""

    return artifact_root.parent / f".{artifact_root.name}.locked-test-supervisor.lock"


@contextlib.contextmanager
def _exclusive_supervisor_lock(path: Path):
    """Hold a non-blocking process lock for the entire plan or collection."""

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Locked Test supervisor holds the global lock: {path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_output_layout(args: argparse.Namespace) -> None:
    """Validate output parents before freeze/manifest work opens protected data."""

    _require_output_directory(args.artifact_root, "artifact root")
    _require_output_directory(args.log_dir, "log directory")
    _require_output_parent(args.completion_receipt, "completion receipt")
    if args.completion_receipt.exists() and not args.completion_receipt.is_file():
        raise RuntimeError("completion receipt path exists but is not a regular file")


def _validate_collection_tree(
    collection_root: Path, task: Any, episodes: tuple[Any, ...] | list[Any]
) -> list[dict[str, Any]]:
    """Validate all published cells and reject any non-manifest or staging entry."""

    expected = {spec.episode_id: spec for spec in episodes}
    if not collection_root.exists():
        return []
    if not collection_root.is_dir() or collection_root.is_symlink():
        raise RuntimeError(f"Locked Test collection root is not a real directory: {collection_root}")
    actual_entries = list(collection_root.iterdir())
    staging = sorted(
        entry.name
        for entry in actual_entries
        if entry.name.startswith(".") and ".tmp-" in entry.name
    )
    if staging:
        raise RuntimeError(
            "unexplained staging directories in Locked Test collection root: "
            + ", ".join(staging)
        )
    extras = sorted(entry.name for entry in actual_entries if entry.name not in expected)
    if extras:
        raise RuntimeError(
            "non-manifest entries in Locked Test collection root: " + ", ".join(extras)
        )
    records: list[dict[str, Any]] = []
    present_names = {entry.name for entry in actual_entries}
    for spec in episodes:
        destination = collection_root / spec.episode_id
        if spec.episode_id in present_names:
            if not destination.is_dir() or destination.is_symlink():
                raise RuntimeError(f"episode artifact is not a real directory: {destination}")
            records.append(_validate_artifact(destination, task, spec))
    return records


def _required_collection_bytes(missing_episode_count: int) -> int:
    return MIN_FREE_RESERVE_BYTES + (
        missing_episode_count * MAX_PLANNED_BYTES_PER_MISSING_EPISODE
    )


def _validate_disk_space(artifact_root: Path, missing_episode_count: int) -> int:
    free = shutil.disk_usage(artifact_root.parent).free
    required = _required_collection_bytes(missing_episode_count)
    if free < required:
        raise RuntimeError(
            "insufficient disk for Locked Test collection: "
            f"{free} bytes free, {required} required for {missing_episode_count} missing episodes"
        )
    return free


def _validate_remote_runtime(root: Path) -> dict[str, Any]:
    """Validate the frozen GPU host without loading a model or simulator."""

    gpu_freeze = _require_exact_tracked_repo_file(
        root, root / GPU_FREEZE_RELPATH, GPU_FREEZE_RELPATH
    )
    actual_freeze_sha = _sha256_file(gpu_freeze)
    if actual_freeze_sha != GPU_FREEZE_SHA256:
        raise RuntimeError(
            "environment-gpu.freeze SHA-256 mismatch: "
            f"{actual_freeze_sha} != {GPU_FREEZE_SHA256}"
        )
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Python runtime mismatch: {platform.python_version()} != {EXPECTED_PYTHON_VERSION}"
        )
    if Path(sys.prefix) != Path("/venv/main"):
        raise RuntimeError(f"Locked Test must run from /venv/main, got {sys.prefix}")

    versions: dict[str, str] = {}
    for distribution, expected in REQUIRED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required distribution is missing: {distribution}") from exc
        if actual != expected:
            raise RuntimeError(
                f"distribution mismatch: {distribution}=={actual}, expected {expected}"
            )
        versions[distribution] = actual

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - rollout-host integration
        raise RuntimeError("PyTorch is missing from the Locked Test runtime") from exc
    if torch.__version__ != "2.11.0+cu130":
        raise RuntimeError(
            f"PyTorch CUDA build mismatch: {torch.__version__} != 2.11.0+cu130"
        )
    if torch.version.cuda != EXPECTED_CUDA_VERSION:
        raise RuntimeError(
            f"PyTorch CUDA runtime mismatch: {torch.version.cuda} != {EXPECTED_CUDA_VERSION}"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != EXPECTED_GPU_NAME
        or torch.cuda.get_device_capability(0) != (12, 0)
    ):
        raise RuntimeError("PyTorch does not expose the exact single RTX 5090 runtime")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi GPU preflight failed") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one physical GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi query output: {rows[0]!r}")
    index, name, driver, compute_capability = fields
    if (index, name, compute_capability) != (
        "0",
        EXPECTED_GPU_NAME,
        EXPECTED_GPU_COMPUTE_CAPABILITY,
    ):
        raise RuntimeError(
            "GPU freeze mismatch: expected index 0, one RTX 5090, compute capability 12.0"
        )
    for env_name, expected in REQUIRED_COLLECTION_ENVIRONMENT.items():
        if os.environ.get(env_name) != expected:
            raise RuntimeError(f"Locked Test environment changed during preflight: {env_name}")
    return {
        "environment_gpu_freeze_sha256": actual_freeze_sha,
        "python_version": platform.python_version(),
        "python_prefix": sys.prefix,
        "gpu_index": index,
        "gpu_name": name,
        "gpu_driver_version": driver,
        "gpu_compute_capability": compute_capability,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "distributions": versions,
        "runtime_environment": dict(REQUIRED_COLLECTION_ENVIRONMENT),
    }


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
        raise RuntimeError("Locked Test manifest is not the canonical guarded payload")
    return payload


def _require_locked_checkout(root: Path) -> None:
    """Authorize Locked Test from the committed Calibration freeze.

    This is the only gate that may open Locked Test.  It checks far more than a
    checkout hash: the freeze file must be tracked at the tagged commit, the tag
    must sit exactly at HEAD, the worktree must be clean including untracked
    files, and every frozen field — task, variable, policy revision, probe,
    predictor, alarm thresholds, patch strength, Calibration metrics — must be
    present and consistent, with every referenced artifact hashing to its
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
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Locked Test completion receipt is not a regular file")
        existing = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        if existing != payload:
            raise RuntimeError("existing Locked Test completion receipt differs")
        return
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True).encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.tmp-", dir=path.parent, delete=False
        ) as stream:
            stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                "Locked Test completion receipt appeared during atomic publication"
            ) from exc
        temporary.unlink()
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _completion_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "locked_test_collection_receipt",
        "code_commit": COLLECTION_COMMIT,
        "required_tag": FREEZE_TAG,
        "collection_tag": COLLECTION_TAG,
        "policy_revision": POLICY_REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "lock_payload_sha256": LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": REALITY_GATE_RECEIPT_SHA256,
        "orientation_eligibility_sha256": ORIENTATION_SHA256,
        "reality_gate_lock_receipt_sha256": LOCK_RECEIPT_SHA256,
        "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
        "locked_test_accessed": True,
        "episodes": records,
    }


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

    _configure_collection_environment()
    root = args.repo_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"--repo-root is not a Git checkout: {root}")
    try:
        if _git(root, "rev-parse", "--show-toplevel") != str(root):
            raise RuntimeError(f"--repo-root is not the Git top-level: {root}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"--repo-root is not a Git checkout: {root}") from exc
    args.environment_lock = _require_exact_tracked_repo_file(
        root, args.environment_lock, ENVIRONMENT_LOCK_RELPATH
    )
    args.cell_script = _require_exact_tracked_repo_file(
        root, args.cell_script, CELL_SCRIPT_RELPATH
    )
    args.manifest = _require_input_file(args.manifest, "manifest")
    args.authority = _require_input_file(args.authority, "authority")
    args.cache_dir = args.cache_dir.resolve(strict=True)
    if not args.cache_dir.is_dir() or not os.access(args.cache_dir, os.R_OK | os.X_OK):
        raise RuntimeError(f"cache directory is not readable/searchable: {args.cache_dir}")
    for output, label in (
        (args.artifact_root, "artifact root"),
        (args.log_dir, "log directory"),
        (args.completion_receipt, "completion receipt"),
    ):
        if output.is_symlink():
            raise RuntimeError(f"{label} may not be a symlink: {output}")
    args.artifact_root = args.artifact_root.resolve(strict=False)
    args.log_dir = args.log_dir.resolve(strict=False)
    args.completion_receipt = args.completion_receipt.resolve(strict=False)
    if not os.access(sys.executable, os.X_OK):
        raise RuntimeError(f"Python interpreter is not executable: {sys.executable}")
    if not os.access(args.cell_script, os.X_OK):
        raise RuntimeError(f"cell script is not executable: {args.cell_script}")
    try:
        compile(args.cell_script.read_bytes(), str(args.cell_script), "exec")
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"cell script is not executable Python: {exc}") from exc
    _validate_output_layout(args)

    lock_path = _supervisor_lock_path(args.artifact_root)
    with _exclusive_supervisor_lock(lock_path):
        return _run_locked(args, root, lock_path)


def _run_locked(args: argparse.Namespace, root: Path, lock_path: Path) -> int:
    _require_locked_checkout(root)
    runtime = _validate_remote_runtime(root)
    if _canonical_json_sha_file(root / "locks/reality_gate_frozen.json") != LOCK_PAYLOAD_SHA256:
        raise RuntimeError("Reality-Gate lock payload changed")
    authority = json.loads(
        args.authority.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if authority.get("head_commit") != COLLECTION_COMMIT or authority.get("tag_commit") != COLLECTION_COMMIT:
        raise RuntimeError("local guard authority is not bound to the locked tag")
    expected_authority = {
        "lock_payload_sha256": LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": REALITY_GATE_RECEIPT_SHA256,
        "orientation_eligibility_sha256": ORIENTATION_SHA256,
        "reality_gate_lock_receipt_sha256": LOCK_RECEIPT_SHA256,
        "failure_event_freeze_sha256": FAILURE_FREEZE_SHA256,
        "calibration_manifest_sha256": MANIFEST_SHA256,
        "locked_test_accessed": True,
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

    collection_root = args.artifact_root / SplitName.LOCKED_TEST.value
    existing_records = _validate_collection_tree(collection_root, task, manifest.episodes)
    completed_ids = {record["episode_id"] for record in existing_records}
    missing_count = len(manifest.episodes) - len(completed_ids)
    receipt_validated = False
    if args.completion_receipt.exists():
        if missing_count:
            raise RuntimeError(
                "Locked Test completion receipt exists before all manifest episodes"
            )
        _write_receipt(args.completion_receipt, _completion_payload(existing_records))
        receipt_validated = True
    free_bytes = _validate_disk_space(args.artifact_root, missing_count)

    # Resolve and hash the exact cached snapshots.  This deliberately does not
    # call load_locked_smolvla or create a simulator/model instance.
    snapshots = resolve_snapshot_paths(
        args.environment_lock, cache_dir=args.cache_dir, local_files_only=True
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "kind": "locked_test_plan_validated",
                    "episodes": len(manifest.episodes),
                    "resume_episodes": len(existing_records),
                    "missing_episodes": missing_count,
                    "manifest_sha256": MANIFEST_SHA256,
                    "code_commit": COLLECTION_COMMIT,
                    "required_tag": FREEZE_TAG,
                    "policy_snapshot": str(snapshots.policy),
                    "base_vlm_snapshot": str(snapshots.base_vlm),
                    "free_disk_bytes": free_bytes,
                    "global_lock": str(lock_path),
                    "remote_runtime": runtime,
                    "completion_receipt_validated": receipt_validated,
                    "locked_test_accessed": True,
                },
                sort_keys=True,
            )
        )
        return 0

    collection_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    child_env = os.environ.copy()
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
        destination = collection_root / spec.episode_id
        if destination.exists():
            record = _validate_artifact(destination, task, spec)
            records.append(record)
            print(
                json.dumps(
                    {"kind": "locked_test_resume_validated", **record}, sort_keys=True
                ),
                flush=True,
            )
            continue
        staging = list(collection_root.glob(f".{spec.episode_id}.tmp-*"))
        if staging:
            raise RuntimeError(f"unexplained staging directories for {spec.episode_id}")
        log_path = args.log_dir / f"{spec.episode_id}.log"
        if log_path.exists() and (log_path.is_symlink() or not log_path.is_file()):
            raise RuntimeError(f"episode log is not a regular file: {log_path}")
        print(
            json.dumps(
                {
                    "kind": "locked_test_episode_starting",
                    "index": index,
                    "episode_id": spec.episode_id,
                    "log": str(log_path),
                },
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
                (f"\n=== launch {spec.episode_id} commit={COLLECTION_COMMIT} index={index} ===\n").encode()
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
        print(
            json.dumps(
                {"kind": "locked_test_episode_completed", **record}, sort_keys=True
            ),
            flush=True,
        )

    payload = _completion_payload(records)
    _write_receipt(args.completion_receipt, payload)
    print(
        json.dumps(
            {
                "kind": "locked_test_collection_complete",
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
