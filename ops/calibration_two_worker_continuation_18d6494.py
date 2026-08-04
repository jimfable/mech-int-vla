#!/usr/bin/env python3
"""Fail-closed serial-to-two-worker Calibration scoring continuation.

This control-plane program never opens Locked Test.  ``cutover`` observes the
authoritative serial runner's flushed ``score_completed`` record, interrupts the
exact process identity once, freezes every published serial sidecar, acquires the
same global flock, and starts two deterministic disjoint workers.  Workers score
only the frozen complement, validate in private same-filesystem staging paths,
and publish with Linux ``RENAME_NOREPLACE``.  The original locked serial runner is
then invoked with zero pending episodes to run its unchanged allocation, feature,
and predictor finalizer.

The coordinator may be restarted with ``resume``.  The narrowly scoped
``recover-ignored-sigint`` mode appends a truthful SIGTERM recovery chain when
the original detached scorer inherited SIGINT as ignored; it never rewrites the
original attempt and never repeats a durably dispatched signal.  Existing
authoritative sidecars are never replaced, and promotion reconciliation requires
a hash-bound prepared receipt.  Physical costs are mapped to an execution mode
in a separate receipt that is not an input to M0/M1/M2.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


# The control process imports scientific modules from the immutable locked
# checkout during preflight.  Disable bytecode writes before any such dynamic
# import, independent of how the coordinator itself was launched.
sys.dont_write_bytecode = True


SCHEMA_VERSION = 1
KIND = "calibration_two_worker_scoring_continuation"
AMENDMENT_COMMIT = "6ad024c44165ea94d23dffaaab593b5478bd441a"
RECOVERY_AMENDMENT_COMMIT = "a19a50eff6aa3e0f886e7597f2e0a80e3e38735a"
ORIGINAL_IMPLEMENTATION_COMMIT = "6cb3733a197f1374025fd08fee44d065e3350c04"
ORIGINAL_CONTROL_HEAD = "7fe9ebce54926bbb9b8ae3a47161f2fb478b5cab"
ORIGINAL_SCRIPT_SHA256 = (
    "e0f96f9818d45611e737962772ba9302447a1c747a5d556843c3686ccee7e654"
)
ORIGINAL_EXIT_TIMEOUT_SECONDS = 120.0
EQUIVALENCE_AUDIT_SHA256 = (
    "68904e5285b029f7330cdeb43de85d35396f8e10b10f898298744ab086dc6d85"
)
BENCHMARK_SUMMARY_SHA256 = (
    "0970accf13dca1c9cca7eb6c0381976c85d600a75bd11fb4cb7a688348adf0f3"
)
BENCHMARK_RESOURCE_SAMPLES_SHA256 = (
    "7b3771906049bce5d73d020b37cab7e708400fee8f069714266fbd8874738a61"
)
COST_ARRAY_NAMES = (
    "original_cost",
    "transformed_cost",
    "intervention_minus_cost",
    "intervention_plus_cost",
)
COST_FIELDS = (
    "cuda_event_ms",
    "wall_time_ns",
    "forward_count",
    "intervention_count",
    "peak_allocated_bytes",
    "incremental_peak_allocated_bytes",
    "logical_activation_bytes",
    "compressed_activation_bytes",
)
EXECUTION_MODES = (
    "serial",
    "serial_with_equivalence_benchmark_contention",
    "two_worker",
)
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class CoordinatorInterrupted(BaseException):
    """Raised to enter deterministic child cleanup on control-plane signals."""


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # The boundary preamble needs only the standard library.
        np = None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, (np.integer, np.floating, np.bool_)):
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


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"JSON receipt must be a regular non-symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON receipt must contain an object: {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"hash target must be a regular non-symlink: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected regular non-symlink file: {path}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"expected regular file: {path}")
    return {
        "sha256": _sha256_file(path),
        "size_bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite control artifact {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return _sha256_bytes(payload)


def _write_json_exclusive(path: Path, value: Any) -> str:
    return _write_exclusive(path, _canonical(value))


def _append_jsonl(path: Path, value: Any) -> None:
    payload = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str, binary: bool = False) -> Any:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _paths_overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(), second.resolve()
    return left == right or left in right.parents or right in left.parents


def _require_disjoint_path(
    candidate: Path, protected: Sequence[tuple[str, Path]], *, name: str
) -> None:
    resolved = candidate.resolve()
    for protected_name, protected_path in protected:
        if _paths_overlap(resolved, protected_path):
            raise RuntimeError(
                f"{name} overlaps protected {protected_name}: "
                f"{resolved} versus {protected_path.resolve()}"
            )


def _require_control_checkout(
    root: Path, script: Path, expected_implementation_commit: str
) -> dict[str, str]:
    root = root.resolve()
    script = script.resolve()
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("control checkout is dirty")
    for ancestor in (
        AMENDMENT_COMMIT,
        RECOVERY_AMENDMENT_COMMIT,
        expected_implementation_commit,
    ):
        result = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, head],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"control checkout does not contain required commit {ancestor}")
    relative = script.relative_to(root).as_posix()
    committed = _git(root, "show", f"{expected_implementation_commit}:{relative}", binary=True)
    if not isinstance(committed, bytes):
        raise RuntimeError("binary git read unexpectedly returned text")
    script_sha = _sha256_file(script)
    if _sha256_bytes(committed) != script_sha:
        raise RuntimeError("running coordinator differs from implementing commit")
    return {
        "control_checkout": str(root),
        "control_head": head,
        "implementation_commit": expected_implementation_commit,
        "amendment_commit": AMENDMENT_COMMIT,
        "recovery_amendment_commit": RECOVERY_AMENDMENT_COMMIT,
        "script_sha256": script_sha,
    }


def _load_serial_runner(path: Path) -> ModuleType:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        "calibration_serial_runner_18d6494", resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import serial runner {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "_require_authority",
        "BOUND_PROBE_SHA256",
        "MANIFEST_SHA256",
        "LOCK_COMMIT",
        "LOCK_TAG",
    ):
        if not hasattr(module, name):
            raise RuntimeError(f"serial runner lacks required symbol {name}")
    return module


def _require_locked_python_imports(repo_root: Path) -> None:
    source_root = (repo_root.resolve() / "src").resolve()
    package = importlib.util.find_spec("mech_int_vla")
    if package is None or package.origin is None:
        raise RuntimeError("mech_int_vla cannot be resolved from locked checkout")
    origin = Path(package.origin).resolve()
    if source_root not in origin.parents:
        raise RuntimeError(
            f"mech_int_vla resolves outside locked source: {origin} versus {source_root}"
        )
    for name, module in tuple(sys.modules.items()):
        if name != "mech_int_vla" and not name.startswith("mech_int_vla."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or source_root not in Path(module_file).resolve().parents:
            raise RuntimeError(f"loaded scientific module is outside locked source: {name}")


def _locked_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(args.repo_root.resolve() / "src"),
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
    return environment


def _locked_context(args: argparse.Namespace, *, validate_all_raw: bool) -> dict[str, Any]:
    _require_locked_python_imports(args.repo_root.resolve())
    from mech_int_vla.artifacts import load_rollout_artifact
    from mech_int_vla.failure_events import artifact_identity_from_rollout
    from mech_int_vla.probe_artifacts import load_bound_probe_artifact

    _require_locked_python_imports(args.repo_root.resolve())

    serial = _load_serial_runner(args.serial_runner)
    root = args.repo_root.resolve()
    protocol, task, manifest = serial._require_authority(
        root, args.manifest.resolve(), args.authority.resolve()
    )
    bound = load_bound_probe_artifact(
        args.bound_probe.resolve(),
        protocol=protocol,
        repo_root=root,
        expected_sha256=serial.BOUND_PROBE_SHA256,
    )
    if bound.rollout.source.manifest_sha256 != serial.MANIFEST_SHA256:
        raise RuntimeError("bound probe manifest differs from Calibration authority")
    ordered_ids = tuple(episode.episode_id for episode in manifest.episodes)
    if tuple(bound.rollout.valid_episode_ids) != tuple(sorted(ordered_ids)):
        raise RuntimeError("bound probe valid episode set differs from manifest")
    expected = {episode.episode_id: episode for episode in manifest.episodes}
    expected_raw_identities = {
        identity.episode_id: identity for identity in bound.rollout.raw_artifacts
    }
    if set(expected_raw_identities) != set(ordered_ids):
        raise RuntimeError("bound probe raw identities do not exactly cover manifest")
    if validate_all_raw:
        observed: list[str] = []
        for episode_id in ordered_ids:
            artifact = load_rollout_artifact(
                args.raw_root.resolve() / "calibration" / episode_id,
                expected_task=task,
            )
            if not artifact.valid_reset:
                raise RuntimeError(f"unexpected invalid Calibration reset: {episode_id}")
            if artifact_identity_from_rollout(artifact) != expected_raw_identities[episode_id]:
                raise RuntimeError(f"Calibration raw artifact drifted: {episode_id}")
            observed.append(episode_id)
        if tuple(sorted(observed)) != tuple(bound.rollout.valid_episode_ids):
            raise RuntimeError("raw Calibration allocation differs from bound probe")
    return {
        "serial": serial,
        "root": root,
        "protocol": protocol,
        "task": task,
        "manifest": manifest,
        "bound": bound,
        "ordered_ids": ordered_ids,
        "expected": expected,
        "expected_raw_identities": expected_raw_identities,
    }


def _require_frozen_raw_identity(context: Mapping[str, Any], artifact: Any) -> None:
    from mech_int_vla.failure_events import artifact_identity_from_rollout

    identity = artifact_identity_from_rollout(artifact)
    expected = context["expected_raw_identities"].get(identity.episode_id)
    if expected is None or identity != expected:
        raise RuntimeError(f"raw artifact differs from bound probe: {identity.episode_id}")


def _cost_summary(sidecar: Any) -> dict[str, float | int]:
    import numpy as np

    sums = {name: 0.0 for name in COST_FIELDS}
    maxima = {
        "peak_allocated_bytes": 0.0,
        "incremental_peak_allocated_bytes": 0.0,
    }
    for array_name in COST_ARRAY_NAMES:
        array = np.asarray(sidecar.arrays[array_name], dtype=np.float64)
        if array.shape[-1] != len(COST_FIELDS):
            raise RuntimeError(f"unexpected cost field width in {array_name}")
        for index, field in enumerate(COST_FIELDS):
            values = array[..., index]
            sums[field] += float(np.nansum(values))
            if field in maxima and np.isfinite(values).any():
                maxima[field] = max(maxima[field], float(np.nanmax(values)))
    return {
        "cuda_event_ms_sum": sums["cuda_event_ms"],
        "wall_time_ns_sum": int(round(sums["wall_time_ns"])),
        "forward_count": int(round(sums["forward_count"])),
        "intervention_count": int(round(sums["intervention_count"])),
        "peak_allocated_bytes_max": int(round(maxima["peak_allocated_bytes"])),
        "incremental_peak_allocated_bytes_max": int(
            round(maxima["incremental_peak_allocated_bytes"])
        ),
        "logical_activation_bytes": int(round(sums["logical_activation_bytes"])),
        "compressed_activation_bytes": int(round(sums["compressed_activation_bytes"])),
    }


def _sidecar_record(
    path: Path,
    episode_id: str,
    *,
    expected_links: Any,
) -> dict[str, Any]:
    from mech_int_vla.scoring import load_scoring_sidecar

    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"sidecar must be a real directory: {path}")
    if {entry.name for entry in path.iterdir()} != {"metadata.json", "primitives.npz"}:
        raise RuntimeError(f"sidecar contains unexpected entries: {path}")
    sidecar = load_scoring_sidecar(
        path, expected_links=expected_links, expected_episode_id=episode_id
    )
    metadata = _file_record(path / "metadata.json")
    primitives = _file_record(path / "primitives.npz")
    metadata_bytes = (path / "metadata.json").read_bytes()
    primitive_bytes = (path / "primitives.npz").read_bytes()
    if _sha256_bytes(metadata_bytes) != metadata["sha256"]:
        raise RuntimeError("metadata changed between validation and combined hash")
    if _sha256_bytes(primitive_bytes) != primitives["sha256"]:
        raise RuntimeError("primitives changed between validation and combined hash")
    combined = hashlib.sha256()
    combined.update(metadata_bytes)
    combined.update(primitive_bytes)
    if metadata["sha256"] != sidecar.metadata_sha256:
        raise RuntimeError("loader/file metadata digest mismatch")
    if primitives["sha256"] != sidecar.primitives_sha256:
        raise RuntimeError("loader/file primitives digest mismatch")
    return {
        "episode_id": episode_id,
        "path": str(path),
        "metadata": metadata,
        "primitives": primitives,
        "combined_sha256": combined.hexdigest(),
        "state_count": int(sidecar.metadata["capture"]["state_count"]),
        "links": dict(sidecar.metadata["links"]),
        "cost": _cost_summary(sidecar),
    }


def _expected_links_for_artifact(context: Mapping[str, Any], artifact: Any) -> Any:
    from mech_int_vla.failure_events import artifact_identity_from_rollout
    from mech_int_vla.provenance import content_links_for

    return content_links_for(
        artifact_identity_from_rollout(artifact),
        context["bound"],
        context["root"],
        protocol=context["protocol"],
    )


def _inventory(context: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from mech_int_vla.artifacts import load_rollout_artifact

    split_root = args.score_root.resolve() / "calibration"
    if split_root.is_symlink() or not split_root.is_dir():
        raise RuntimeError("authoritative score split must be a real directory")
    entries = list(split_root.iterdir())
    residue = sorted(
        entry.name
        for entry in entries
        if entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir()
    )
    if residue:
        raise RuntimeError(f"authoritative score root has residue: {residue}")
    names = {entry.name for entry in entries}
    expected_ids = set(context["ordered_ids"])
    extra = sorted(names - expected_ids)
    if extra:
        raise RuntimeError(f"score root contains episodes outside manifest: {extra}")
    records: list[dict[str, Any]] = []
    for episode_id in context["ordered_ids"]:
        if episode_id not in names:
            continue
        artifact = load_rollout_artifact(
            args.raw_root.resolve() / "calibration" / episode_id,
            expected_task=context["task"],
        )
        if not artifact.valid_reset:
            raise RuntimeError(f"scored episode has invalid reset: {episode_id}")
        _require_frozen_raw_identity(context, artifact)
        links = _expected_links_for_artifact(context, artifact)
        records.append(
            _sidecar_record(split_root / episode_id, episode_id, expected_links=links)
        )
    payload = {
        "episode_count": len(records),
        "episode_ids": [record["episode_id"] for record in records],
        "records": records,
    }
    payload["inventory_sha256"] = _sha256_bytes(_canonical(payload))
    return payload


def _records_by_id(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = inventory.get("records")
    if not isinstance(records, list):
        raise RuntimeError("inventory records are malformed")
    result = {str(record["episode_id"]): record for record in records}
    if len(result) != len(records):
        raise RuntimeError("inventory contains duplicate episode IDs")
    return result


def _assert_frozen_unchanged(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    previous = _records_by_id(frozen)
    observed = _records_by_id(current)
    for episode_id, record in previous.items():
        if episode_id not in observed:
            raise RuntimeError(f"frozen sidecar disappeared: {episode_id}")
        candidate = observed[episode_id]
        for field in ("combined_sha256", "metadata", "primitives", "links"):
            if candidate[field] != record[field]:
                raise RuntimeError(f"frozen sidecar changed ({field}): {episode_id}")


def _process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    try:
        arguments = [
            value.decode("utf-8", errors="strict")
            for value in (proc / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        stat_payload = (proc / "stat").read_text(encoding="utf-8")
        closing = stat_payload.rfind(")")
        fields = stat_payload[closing + 2 :].split()
        start_ticks = int(fields[19])
        executable = os.readlink(proc / "exe")
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return None
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "start_ticks": start_ticks,
        "executable": executable,
        "arguments": arguments,
    }


def _identity_matches(expected: Mapping[str, Any]) -> bool:
    return _process_identity(int(expected["pid"])) == dict(expected)


def _capture_serial_identities(serial_runner: Path, global_lock: Path) -> dict[str, Any]:
    target = str(serial_runner.resolve())
    matches: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        if not proc.name.isdigit():
            continue
        identity = _process_identity(int(proc.name))
        if (
            identity is not None
            and len(identity["arguments"]) >= 2
            and identity["arguments"][1] == target
        ):
            matches.append(identity)
    if len(matches) != 1:
        raise RuntimeError(f"expected one serial scorer, found {len(matches)}")
    python_identity = matches[0]
    wrapper = _process_identity(int(python_identity["parent_pid"]))
    if wrapper is None:
        raise RuntimeError("serial scorer flock wrapper is missing")
    arguments = wrapper["arguments"]
    if not arguments or Path(arguments[0]).name != "flock" or str(global_lock) not in arguments:
        raise RuntimeError("serial scorer parent is not the expected global flock wrapper")
    return {"python": python_identity, "flock_wrapper": wrapper}


def _runner_processes(serial_runner: Path) -> list[dict[str, Any]]:
    target = str(serial_runner.resolve())
    result: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        if not proc.name.isdigit():
            continue
        identity = _process_identity(int(proc.name))
        if (
            identity is not None
            and len(identity["arguments"]) >= 2
            and identity["arguments"][1] == target
        ):
            result.append(identity)
    return result


def _signal_status(pid: int) -> dict[str, Any]:
    path = Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read signal status for PID {pid}") from exc
    values: dict[str, int] = {}
    raw: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name not in {"SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt"}:
            continue
        encoded = value.strip()
        try:
            values[name] = int(encoded, 16)
        except ValueError as exc:
            raise RuntimeError(f"malformed {name} mask for PID {pid}") from exc
        raw[name] = encoded
    required = {"SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt"}
    if set(values) != required:
        raise RuntimeError(f"incomplete signal status for PID {pid}")
    return {"pid": pid, "hex_masks": raw, "integer_masks": values}


def _signal_mask_contains(status: Mapping[str, Any], field: str, signum: int) -> bool:
    masks = status.get("integer_masks")
    if not isinstance(masks, Mapping) or field not in masks:
        raise RuntimeError(f"signal status lacks {field}")
    return bool(int(masks[field]) & (1 << (signum - 1)))


def _validate_recovery_signal_status(status: Mapping[str, Any]) -> dict[str, Any]:
    sigint = signal.SIGINT
    sigterm = signal.SIGTERM
    if not _signal_mask_contains(status, "SigIgn", sigint):
        raise RuntimeError("SIGINT is not recorded as inherited ignored")
    for field in ("SigPnd", "ShdPnd", "SigBlk", "SigCgt"):
        if _signal_mask_contains(status, field, sigint):
            raise RuntimeError(f"SIGINT unexpectedly present in {field}")
    for field in ("SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt"):
        if _signal_mask_contains(status, field, sigterm):
            raise RuntimeError(f"SIGTERM unexpectedly present in {field}")
    return {
        "sigint": {
            "signal_number": int(sigint),
            "ignored": True,
            "blocked": False,
            "pending": False,
        },
        "sigterm": {
            "signal_number": int(sigterm),
            "ignored": False,
            "blocked": False,
            "pending": False,
            "custom_caught": False,
        },
        "proc_status": dict(status),
    }


def _capture_recovery_signal_status(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not _identity_matches(identity):
        raise RuntimeError("serial identity changed before signal-status capture")
    status = _signal_status(int(identity["pid"]))
    validated = _validate_recovery_signal_status(status)
    if not _identity_matches(identity):
        raise RuntimeError("serial identity changed during signal-status capture")
    return validated


def _pidfd_open_linux(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "pidfd_open", None)
    if function is None:
        raise RuntimeError("libc pidfd_open is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = int(function(pid, 0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), pid)
    return descriptor


def _pidfd_send_signal_linux(descriptor: int, signum: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "pidfd_send_signal", None)
    if function is None:
        raise RuntimeError("libc pidfd_send_signal is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = int(function(descriptor, signum, None, 0))
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), descriptor)


def _open_validated_pidfd(identity: Mapping[str, Any]) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("pidfd recovery signalling requires Linux")
    if not _identity_matches(identity):
        raise RuntimeError("serial identity changed before pidfd acquisition")
    try:
        descriptor = _pidfd_open_linux(int(identity["pid"]))
    except OSError as exc:
        raise RuntimeError("cannot acquire serial scorer pidfd") from exc
    try:
        observed = _process_identity(int(identity["pid"]))
        if observed != dict(identity):
            raise RuntimeError("serial identity changed during pidfd acquisition")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _log_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeError("serial log is not a regular non-symlink")
    info = resolved.stat()
    return {
        "path": str(resolved),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IFMT(info.st_mode),
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def _assert_same_log_object(expected: Mapping[str, Any], path: Path) -> dict[str, Any]:
    observed = _log_identity(path)
    for field in ("path", "device", "inode", "mode"):
        if observed[field] != expected.get(field):
            raise RuntimeError(f"serial log identity changed ({field})")
    return observed


def _wait_for_score_completed_record(
    log_path: Path,
    offset: int,
    serial_identity: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    buffer = b""
    with log_path.open("rb", buffering=0) as stream:
        stream.seek(offset)
        while time.monotonic() < deadline:
            if not _identity_matches(serial_identity):
                raise RuntimeError("serial scorer identity changed before boundary")
            chunk = stream.read(65536)
            if not chunk:
                time.sleep(0.02)
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                end_offset = stream.tell() - len(buffer)
                start_offset = end_offset - len(raw) - 1
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and value.get("kind") == "score_completed":
                    return {
                        "boundary_line": value,
                        "raw_line_sha256": _sha256_bytes(raw),
                        "log_start_offset": start_offset,
                        "log_end_offset": end_offset,
                    }
    raise RuntimeError("timed out waiting for a serial score_completed boundary")


def _wait_for_score_completed(
    log_path: Path,
    offset: int,
    serial_identity: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
    record = _wait_for_score_completed_record(
        log_path,
        offset,
        serial_identity,
        timeout_seconds=timeout_seconds,
    )
    return dict(record["boundary_line"]), int(record["log_end_offset"])


def _stable_inventory_log_baseline(
    context: Mapping[str, Any],
    args: argparse.Namespace,
    log_path: Path,
    serial_identity: Mapping[str, Any],
    *,
    max_attempts: int = 8,
) -> tuple[dict[str, Any], int]:
    """Bind an inventory to an EOF that saw no serial log publication during hashing."""

    for _attempt in range(max_attempts):
        if not _identity_matches(serial_identity):
            raise RuntimeError("serial identity changed while freezing log baseline")
        before = log_path.stat().st_size
        inventory = _inventory(context, args)
        after = log_path.stat().st_size
        if before == after:
            return inventory, after
    raise RuntimeError("could not freeze inventory between serial publications")


def _prevalidate_boundary_line(
    boundary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    ordered_ids: Sequence[str],
) -> None:
    if set(boundary) != {"kind", "episode_id", "sha256", "completed", "total"}:
        raise RuntimeError("score_completed boundary has unexpected fields")
    count = int(baseline["episode_count"])
    baseline_ids = list(baseline["episode_ids"])
    if baseline_ids != list(ordered_ids[:count]):
        raise RuntimeError("baseline serial sidecars are not an exact manifest prefix")
    if boundary.get("kind") != "score_completed":
        raise RuntimeError("boundary line kind drifted")
    if boundary.get("total") != len(ordered_ids):
        raise RuntimeError("boundary total differs from manifest")
    if boundary.get("completed") != count + 1:
        raise RuntimeError("boundary completed count is not the next serial episode")
    if count >= len(ordered_ids) or boundary.get("episode_id") != ordered_ids[count]:
        raise RuntimeError("boundary episode is not the next manifest episode")
    digest = boundary.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("boundary SHA-256 is malformed")


def _validate_boundary_transition(
    boundary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    frozen: Mapping[str, Any],
    ordered_ids: Sequence[str],
) -> None:
    _prevalidate_boundary_line(boundary, baseline, ordered_ids)
    baseline_ids = list(baseline["episode_ids"])
    frozen_ids = list(frozen["episode_ids"])
    if frozen["episode_count"] != baseline["episode_count"] + 1:
        raise RuntimeError("cutover published other than exactly one new sidecar")
    if frozen_ids != list(ordered_ids[: int(frozen["episode_count"])]):
        raise RuntimeError("frozen serial inventory is not an exact manifest prefix")
    new_ids = set(frozen_ids) - set(baseline_ids)
    if new_ids != {boundary["episode_id"]}:
        raise RuntimeError("cutover transition has an unexpected new sidecar set")
    record = _records_by_id(frozen)[str(boundary["episode_id"])]
    if record["combined_sha256"] != boundary["sha256"]:
        raise RuntimeError("boundary line SHA differs from published sidecar")


def _wait_identities_exit(
    identities: Sequence[Mapping[str, Any]], *, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_identity_matches(identity) for identity in identities):
            return
        time.sleep(0.05)
    live = [identity["pid"] for identity in identities if _identity_matches(identity)]
    raise RuntimeError(f"serial identities did not exit after SIGINT: {live}")


def _wait_process_instance_exit(
    identity: Mapping[str, Any], *, timeout_seconds: float, role: str
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    pid = int(identity["pid"])
    expected_start = int(identity["start_ticks"])
    while time.monotonic() < deadline:
        observed = _process_identity(pid)
        if observed is None:
            return {
                "role": role,
                "pid": pid,
                "start_ticks": expected_start,
                "exit_observed_unix_time_ns": time.time_ns(),
                "pid_reused_at_observation": False,
            }
        if int(observed["start_ticks"]) != expected_start:
            return {
                "role": role,
                "pid": pid,
                "start_ticks": expected_start,
                "exit_observed_unix_time_ns": time.time_ns(),
                "pid_reused_at_observation": True,
                "replacement_start_ticks": int(observed["start_ticks"]),
            }
        if observed != dict(identity):
            raise RuntimeError(
                f"{role} identity mutated while the same process instance remained"
            )
        time.sleep(0.05)
    raise RuntimeError(f"{role} process instance did not exit after SIGTERM: {pid}")


def _assert_process_instance_absent(identity: Mapping[str, Any], *, role: str) -> None:
    observed = _process_identity(int(identity["pid"]))
    if observed is None:
        return
    if int(observed["start_ticks"]) != int(identity["start_ticks"]):
        return
    raise RuntimeError(f"{role} process instance remains after recorded exit")


def _validate_process_exit_observation(
    observation: Mapping[str, Any], identity: Mapping[str, Any], *, role: str
) -> None:
    if (
        observation.get("role") != role
        or observation.get("pid") != identity["pid"]
        or observation.get("start_ticks") != identity["start_ticks"]
        or not isinstance(observation.get("exit_observed_unix_time_ns"), int)
        or int(observation["exit_observed_unix_time_ns"]) <= 0
        or not isinstance(observation.get("pid_reused_at_observation"), bool)
    ):
        raise RuntimeError(f"stored {role} exit observation is invalid")
    if observation["pid_reused_at_observation"]:
        replacement = observation.get("replacement_start_ticks")
        if not isinstance(replacement, int) or replacement == identity["start_ticks"]:
            raise RuntimeError(f"stored {role} PID-reuse observation is invalid")
    elif "replacement_start_ticks" in observation:
        raise RuntimeError(f"stored {role} exit observation has unexpected replacement")


def _signal_attempt_paths(execution_root: Path) -> list[Path]:
    base = execution_root / "signal-attempts"
    if not base.exists():
        return []
    if base.is_symlink() or not base.is_dir():
        raise RuntimeError("signal-attempts path is not a real directory")
    return sorted(base.glob("attempt-*-dispatched.json"))


def _complete_dispatched_interrupt(
    args: argparse.Namespace,
    identities: Mapping[str, Any],
    dispatched_path: Path,
) -> dict[str, Any]:
    dispatched = _strict_json(dispatched_path)
    if (
        dispatched.get("kind") != f"{KIND}_signal_dispatched"
        or dispatched.get("signal") != "SIGINT"
        or dispatched.get("signal_count") != 1
        or dispatched.get("os_kill_returned") is not True
        or dispatched.get("serial_python_identity") != identities["python"]
    ):
        raise RuntimeError("durable serial signal-dispatch receipt is invalid")
    _wait_identities_exit(
        [identities["python"], identities["flock_wrapper"]],
        timeout_seconds=args.serial_exit_timeout,
    )
    if _runner_processes(args.serial_runner):
        raise RuntimeError("serial scorer remains after durably dispatched SIGINT")
    suffix = dispatched_path.name.removesuffix("-dispatched.json")
    exited_path = dispatched_path.parent / f"{suffix}-exited.json"
    exited = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_signal_exit_verified",
        "dispatched_sha256": _sha256_file(dispatched_path),
        "serial_identities_exited": True,
        "runner_process_count": 0,
        "verified_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    if exited_path.exists():
        existing = _strict_json(exited_path)
        if (
            existing.get("dispatched_sha256") != exited["dispatched_sha256"]
            or existing.get("serial_identities_exited") is not True
        ):
            raise RuntimeError("existing signal-exit receipt is invalid")
        exited = existing
    else:
        _write_json_exclusive(exited_path, exited)
    interrupt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_serial_interrupt",
        "signal": "SIGINT",
        "signal_count": 1,
        "signal_dispatched_sha256": _sha256_file(dispatched_path),
        "signal_exit_verified_sha256": _sha256_file(exited_path),
        "serial_identities_exited": True,
        "runner_process_count": 0,
        "verified_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    interrupt_path = args.execution_root.resolve() / "serial-interrupt.json"
    if interrupt_path.exists():
        existing = _strict_json(interrupt_path)
        for field in (
            "signal",
            "signal_count",
            "signal_dispatched_sha256",
            "signal_exit_verified_sha256",
            "serial_identities_exited",
        ):
            if existing.get(field) != interrupt[field]:
                raise RuntimeError("existing serial-interrupt receipt is inconsistent")
        interrupt = existing
    else:
        _write_json_exclusive(interrupt_path, interrupt)
    return interrupt


def _dispatch_serial_interrupt(
    args: argparse.Namespace,
    identities: Mapping[str, Any],
    boundary_path: Path,
) -> dict[str, Any]:
    execution_root = args.execution_root.resolve()
    dispatched = _signal_attempt_paths(execution_root)
    if dispatched:
        if len(dispatched) != 1:
            raise RuntimeError("more than one durable signal dispatch exists")
        return _complete_dispatched_interrupt(args, identities, dispatched[0])
    base = execution_root / "signal-attempts"
    base.mkdir(parents=True, exist_ok=True)
    existing_intents = sorted(base.glob("attempt-*-intent.json"))
    if existing_intents:
        raise RuntimeError(
            "ambiguous signal state: durable intent exists without dispatch receipt"
        )
    attempt_index = 1
    prefix = f"attempt-{attempt_index:04d}"
    intent_path = base / f"{prefix}-intent.json"
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_signal_intent",
        "signal": "SIGINT",
        "signal_count": 1,
        "serial_python_identity": identities["python"],
        "serial_flock_identity": identities["flock_wrapper"],
        "boundary_receipt_path": str(boundary_path.resolve()),
        "boundary_receipt_sha256": _sha256_file(boundary_path),
        "created_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    _write_json_exclusive(intent_path, intent)
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial identity changed after durable signal intent")
    os.kill(int(identities["python"]["pid"]), signal.SIGINT)
    dispatched_path = base / f"{prefix}-dispatched.json"
    dispatch = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_signal_dispatched",
        "signal": "SIGINT",
        "signal_count": 1,
        "serial_python_identity": identities["python"],
        "intent_sha256": _sha256_file(intent_path),
        "os_kill_returned": True,
        "dispatched_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    _write_json_exclusive(dispatched_path, dispatch)
    return _complete_dispatched_interrupt(args, identities, dispatched_path)


def _validate_original_sigint_attempt(
    args: argparse.Namespace,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    execution_root = args.execution_root.resolve()
    intent_path = execution_root / "cutover-intent.json"
    boundary_path = execution_root / "boundary-observed.json"
    signal_intent_path = execution_root / "signal-attempts" / "attempt-0001-intent.json"
    dispatched_path = (
        execution_root / "signal-attempts" / "attempt-0001-dispatched.json"
    )
    intent = _strict_json(intent_path)
    boundary = _strict_json(boundary_path)
    signal_intent = _strict_json(signal_intent_path)
    dispatched = _strict_json(dispatched_path)
    expected_control = {
        "control_checkout": str(args.control_repo.resolve()),
        "control_head": ORIGINAL_CONTROL_HEAD,
        "implementation_commit": ORIGINAL_IMPLEMENTATION_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "script_sha256": ORIGINAL_SCRIPT_SHA256,
    }
    if intent.get("kind") != f"{KIND}_cutover_intent":
        raise RuntimeError("original cutover intent kind drifted")
    if intent.get("control") != expected_control:
        raise RuntimeError("original cutover control provenance drifted")
    if intent.get("locked_test_accessed") is not False:
        raise RuntimeError("original cutover intent lacks Locked-Test exclusion")
    identities = intent.get("serial_identities")
    if not isinstance(identities, Mapping):
        raise RuntimeError("original cutover identities are malformed")
    if boundary.get("serial_python_identity") != identities.get("python"):
        raise RuntimeError("original boundary identity differs from cutover intent")
    baseline = boundary.get("baseline_inventory")
    if not isinstance(baseline, Mapping):
        raise RuntimeError("original boundary baseline is malformed")
    if baseline != intent.get("initial_inventory"):
        raise RuntimeError("original boundary baseline differs from cutover intent")
    boundary_line = boundary.get("boundary_line")
    if not isinstance(boundary_line, Mapping):
        raise RuntimeError("original boundary line is malformed")
    _prevalidate_boundary_line(boundary_line, baseline, context["ordered_ids"])
    if (
        signal_intent.get("kind") != f"{KIND}_signal_intent"
        or signal_intent.get("signal") != "SIGINT"
        or signal_intent.get("signal_count") != 1
        or signal_intent.get("serial_python_identity") != identities.get("python")
        or signal_intent.get("serial_flock_identity") != identities.get("flock_wrapper")
        or Path(str(signal_intent.get("boundary_receipt_path"))).resolve()
        != boundary_path
        or signal_intent.get("boundary_receipt_sha256") != _sha256_file(boundary_path)
    ):
        raise RuntimeError("original SIGINT intent is invalid")
    if (
        dispatched.get("kind") != f"{KIND}_signal_dispatched"
        or dispatched.get("signal") != "SIGINT"
        or dispatched.get("signal_count") != 1
        or dispatched.get("serial_python_identity") != identities.get("python")
        or dispatched.get("intent_sha256") != _sha256_file(signal_intent_path)
        or dispatched.get("os_kill_returned") is not True
        or dispatched.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("original SIGINT dispatch is invalid")
    if (execution_root / "serial-interrupt.json").exists():
        raise RuntimeError("unexpected original serial-interrupt receipt exists")
    if (execution_root / "plan.json").exists():
        raise RuntimeError("unexpected continuation plan exists before recovery")
    return {
        "cutover_intent": intent,
        "cutover_intent_path": intent_path,
        "boundary": boundary,
        "boundary_path": boundary_path,
        "signal_intent": signal_intent,
        "signal_intent_path": signal_intent_path,
        "signal_dispatched": dispatched,
        "signal_dispatched_path": dispatched_path,
        "serial_identities": identities,
    }


def _ensure_original_sigint_ineffective_receipt(
    args: argparse.Namespace,
    original: Mapping[str, Any],
) -> dict[str, Any]:
    path = args.execution_root.resolve() / "signal-attempts" / "attempt-0001-ineffective.json"
    dispatched_path = Path(original["signal_dispatched_path"])
    expected_dispatch_sha = _sha256_file(dispatched_path)
    if path.exists():
        receipt = _strict_json(path)
        status = receipt.get("signal_status")
        observed_ns = receipt.get("observed_unix_time_ns")
        dispatched_ns = int(original["signal_dispatched"]["dispatched_unix_time_ns"])
        if (
            receipt.get("kind") != f"{KIND}_signal_ineffective"
            or receipt.get("signal") != "SIGINT"
            or receipt.get("signal_count") != 1
            or receipt.get("signal_dispatched_sha256") != expected_dispatch_sha
            or receipt.get("serial_python_identity")
            != original["serial_identities"]["python"]
            or receipt.get("serial_flock_identity")
            != original["serial_identities"]["flock_wrapper"]
            or receipt.get("exit_observed") is not False
            or receipt.get("disposition") != "inherited_ignored"
            or receipt.get("processes_alive_after_timeout") is not True
            or receipt.get("timeout_seconds") != ORIGINAL_EXIT_TIMEOUT_SECONDS
            or not isinstance(observed_ns, int)
            or observed_ns - dispatched_ns
            < int(ORIGINAL_EXIT_TIMEOUT_SECONDS * 1e9)
            or receipt.get("causal_exit_claim") is not False
            or receipt.get("locked_test_accessed") is not False
            or not isinstance(status, Mapping)
            or not isinstance(status.get("proc_status"), Mapping)
        ):
            raise RuntimeError("existing ineffective-SIGINT receipt is invalid")
        _validate_recovery_signal_status(status["proc_status"])
        return receipt
    identities = original["serial_identities"]
    status = _capture_recovery_signal_status(identities["python"])
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("flock identity changed before SIGINT closure")
    dispatched_ns = int(original["signal_dispatched"]["dispatched_unix_time_ns"])
    observed_ns = time.time_ns()
    if observed_ns - dispatched_ns < int(ORIGINAL_EXIT_TIMEOUT_SECONDS * 1e9):
        raise RuntimeError("original SIGINT timeout has not elapsed")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_signal_ineffective",
        "signal": "SIGINT",
        "signal_count": 1,
        "signal_dispatched_sha256": expected_dispatch_sha,
        "serial_python_identity": identities["python"],
        "serial_flock_identity": identities["flock_wrapper"],
        "disposition": "inherited_ignored",
        "signal_status": status,
        "exit_observed": False,
        "processes_alive_after_timeout": True,
        "timeout_seconds": ORIGINAL_EXIT_TIMEOUT_SECONDS,
        "observed_unix_time_ns": observed_ns,
        "causal_exit_claim": False,
        "locked_test_accessed": False,
    }
    _write_json_exclusive(path, receipt)
    return receipt


def _recovery_signal_paths(execution_root: Path) -> tuple[Path, Path, Path]:
    base = execution_root / "recovery-signal-attempts"
    return (
        base / "attempt-0001-intent.json",
        base / "attempt-0001-dispatched.json",
        base / "attempt-0001-exit-observed.json",
    )


def _validate_recovery_sigterm_dispatch(
    dispatched_path: Path,
    intent_path: Path,
    identities: Mapping[str, Any],
) -> dict[str, Any]:
    intent = _strict_json(intent_path)
    dispatched = _strict_json(dispatched_path)
    if (
        intent.get("kind") != f"{KIND}_recovery_signal_intent"
        or intent.get("signal") != "SIGTERM"
        or intent.get("signal_count") != 1
        or intent.get("target") != "serial_python_only"
        or intent.get("delivery_method") != "libc_pidfd_send_signal"
        or intent.get("target_start_ticks") != identities["python"]["start_ticks"]
        or intent.get("wrapper_signalled") is not False
        or intent.get("serial_python_identity") != identities["python"]
        or intent.get("serial_flock_identity") != identities["flock_wrapper"]
        or intent.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("recovery SIGTERM intent is invalid")
    signal_status = intent.get("signal_status")
    if not isinstance(signal_status, Mapping) or not isinstance(
        signal_status.get("proc_status"), Mapping
    ):
        raise RuntimeError("recovery SIGTERM intent signal status is malformed")
    _validate_recovery_signal_status(signal_status["proc_status"])
    if (
        dispatched.get("kind") != f"{KIND}_recovery_signal_dispatched"
        or dispatched.get("signal") != "SIGTERM"
        or dispatched.get("signal_count") != 1
        or dispatched.get("target_pid") != identities["python"]["pid"]
        or dispatched.get("target_start_ticks")
        != identities["python"]["start_ticks"]
        or dispatched.get("wrapper_signalled") is not False
        or dispatched.get("delivery_method") != "libc_pidfd_send_signal"
        or dispatched.get("intent_sha256") != _sha256_file(intent_path)
        or dispatched.get("os_kill_called") is not False
        or dispatched.get("pidfd_send_signal_returned") is not True
        or dispatched.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("recovery SIGTERM dispatch is invalid")
    return dispatched


def _ensure_recovery_sigterm_dispatched(
    args: argparse.Namespace,
    identities: Mapping[str, Any],
    boundary_path: Path,
    ineffective_path: Path,
    runner_sha256: str,
) -> dict[str, Any]:
    intent_path, dispatched_path, _exit_path = _recovery_signal_paths(
        args.execution_root.resolve()
    )
    attempts_root = intent_path.parent
    if attempts_root.exists() or attempts_root.is_symlink():
        if attempts_root.is_symlink() or not attempts_root.is_dir():
            raise RuntimeError("recovery signal-attempts path is not a real directory")
    else:
        _assert_execution_control_state(
            args.execution_root.resolve(), "recovery_boundary"
        )
        attempts_root.mkdir(exist_ok=False)
    _fsync_directory(args.execution_root.resolve())
    if dispatched_path.exists():
        if (args.execution_root.resolve() / "cutover-receipt.json").exists():
            phase = "cutover"
        elif _exit_path.exists():
            phase = "recovery_exit"
        else:
            phase = "recovery_dispatched"
        _assert_execution_control_state(args.execution_root.resolve(), phase)
        dispatched = _validate_recovery_sigterm_dispatch(
            dispatched_path, intent_path, identities
        )
        intent = _strict_json(intent_path)
        if (
            intent.get("runner_sha256") != runner_sha256
            or Path(str(intent.get("boundary_receipt_path"))).resolve()
            != boundary_path.resolve()
            or intent.get("boundary_receipt_sha256") != _sha256_file(boundary_path)
            or intent.get("ineffective_sigint_receipt_sha256")
            != _sha256_file(ineffective_path)
        ):
            raise RuntimeError("existing recovery signal intent provenance differs")
        return dispatched
    if intent_path.exists():
        raise RuntimeError(
            "ambiguous recovery signal state: durable intent exists without dispatch"
        )
    _assert_execution_control_state(
        args.execution_root.resolve(), "recovery_attempts_empty"
    )
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial Python identity changed before recovery intent")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("serial flock identity changed before recovery intent")
    pidfd = _open_validated_pidfd(identities["python"])
    try:
        status = _capture_recovery_signal_status(identities["python"])
        if _sha256_file(args.serial_runner.resolve()) != runner_sha256:
            raise RuntimeError("serial runner changed before recovery intent")
        intent = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_recovery_signal_intent",
            "signal": "SIGTERM",
            "signal_count": 1,
            "target": "serial_python_only",
            "target_start_ticks": identities["python"]["start_ticks"],
            "delivery_method": "libc_pidfd_send_signal",
            "wrapper_signalled": False,
            "serial_python_identity": identities["python"],
            "serial_flock_identity": identities["flock_wrapper"],
            "runner_sha256": runner_sha256,
            "boundary_receipt_path": str(boundary_path.resolve()),
            "boundary_receipt_sha256": _sha256_file(boundary_path),
            "ineffective_sigint_receipt_sha256": _sha256_file(ineffective_path),
            "signal_status": status,
            "created_unix_time_ns": time.time_ns(),
            "locked_test_accessed": False,
        }
        _write_json_exclusive(intent_path, intent)
        if not _identity_matches(identities["python"]):
            raise RuntimeError("serial Python identity changed after recovery intent")
        if not _identity_matches(identities["flock_wrapper"]):
            raise RuntimeError("serial flock identity changed after recovery intent")
        if _capture_recovery_signal_status(identities["python"]) != status:
            raise RuntimeError("signal disposition changed after recovery intent")
        if _sha256_file(args.serial_runner.resolve()) != runner_sha256:
            raise RuntimeError("serial runner changed after recovery intent")
        _pidfd_send_signal_linux(pidfd, int(signal.SIGTERM))
        dispatched = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_recovery_signal_dispatched",
            "signal": "SIGTERM",
            "signal_count": 1,
            "target_pid": identities["python"]["pid"],
            "target_start_ticks": identities["python"]["start_ticks"],
            "target": "serial_python_only",
            "delivery_method": "libc_pidfd_send_signal",
            "wrapper_signalled": False,
            "intent_sha256": _sha256_file(intent_path),
            "os_kill_called": False,
            "pidfd_send_signal_returned": True,
            "dispatched_unix_time_ns": time.time_ns(),
            "locked_test_accessed": False,
        }
        _write_json_exclusive(dispatched_path, dispatched)
        _assert_execution_control_state(
            args.execution_root.resolve(), "recovery_dispatched"
        )
        return dispatched
    finally:
        os.close(pidfd)


def _complete_recovery_sigterm(
    args: argparse.Namespace,
    identities: Mapping[str, Any],
) -> dict[str, Any]:
    intent_path, dispatched_path, exit_path = _recovery_signal_paths(
        args.execution_root.resolve()
    )
    dispatched = _validate_recovery_sigterm_dispatch(
        dispatched_path, intent_path, identities
    )
    if exit_path.exists():
        phase = (
            "cutover"
            if (args.execution_root.resolve() / "cutover-receipt.json").exists()
            else "recovery_exit"
        )
        _assert_execution_control_state(args.execution_root.resolve(), phase)
        existing = _strict_json(exit_path)
        python_observation = existing.get("python_exit_observation")
        wrapper_observation = existing.get("wrapper_exit_observation")
        if (
            existing.get("kind") != f"{KIND}_recovery_signal_exit_observed"
            or existing.get("signal") != "SIGTERM"
            or existing.get("signal_count") != 1
            or existing.get("signal_dispatched_sha256")
            != _sha256_file(dispatched_path)
            or existing.get("serial_python_identity") != identities["python"]
            or existing.get("serial_flock_identity") != identities["flock_wrapper"]
            or existing.get("python_exit_subsequently_observed") is not True
            or existing.get("wrapper_exit_subsequently_observed") is not True
            or existing.get("wrapper_signalled_by_recovery") is not False
            or existing.get("runner_process_count") != 0
            or existing.get("causal_exit_claim") is not False
            or existing.get("wrapper_natural_exit_causal_claim") is not False
            or existing.get("locked_test_accessed") is not False
            or not isinstance(python_observation, Mapping)
            or not isinstance(wrapper_observation, Mapping)
        ):
            raise RuntimeError("existing recovery exit receipt is invalid")
        _validate_process_exit_observation(
            python_observation, identities["python"], role="serial_python"
        )
        _validate_process_exit_observation(
            wrapper_observation,
            identities["flock_wrapper"],
            role="serial_flock_wrapper",
        )
        dispatched_ns = int(dispatched["dispatched_unix_time_ns"])
        python_ns = int(python_observation["exit_observed_unix_time_ns"])
        wrapper_ns = int(wrapper_observation["exit_observed_unix_time_ns"])
        if not (dispatched_ns <= python_ns <= wrapper_ns):
            raise RuntimeError("stored recovery exit observations are not monotonic")
        _assert_process_instance_absent(identities["python"], role="serial_python")
        _assert_process_instance_absent(
            identities["flock_wrapper"], role="serial_flock_wrapper"
        )
        if _runner_processes(args.serial_runner):
            raise RuntimeError("serial scorer reappeared after recovery exit receipt")
        return existing
    _assert_execution_control_state(
        args.execution_root.resolve(), "recovery_dispatched"
    )
    deadline = time.monotonic() + args.serial_exit_timeout
    python_exit = _wait_process_instance_exit(
        identities["python"],
        timeout_seconds=max(0.001, deadline - time.monotonic()),
        role="serial_python",
    )
    wrapper_exit = _wait_process_instance_exit(
        identities["flock_wrapper"],
        timeout_seconds=max(0.001, deadline - time.monotonic()),
        role="serial_flock_wrapper",
    )
    if _runner_processes(args.serial_runner):
        raise RuntimeError("serial scorer remains after recovery SIGTERM")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_recovery_signal_exit_observed",
        "signal": "SIGTERM",
        "signal_count": 1,
        "signal_dispatched_sha256": _sha256_file(dispatched_path),
        "serial_python_identity": identities["python"],
        "serial_flock_identity": identities["flock_wrapper"],
        "python_exit_subsequently_observed": True,
        "python_exit_observation": python_exit,
        "wrapper_exit_subsequently_observed": True,
        "wrapper_exit_observation": wrapper_exit,
        "wrapper_signalled_by_recovery": False,
        "runner_process_count": 0,
        "causal_exit_claim": False,
        "wrapper_natural_exit_causal_claim": False,
        "observed_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    _write_json_exclusive(exit_path, receipt)
    _assert_execution_control_state(args.execution_root.resolve(), "recovery_exit")
    return receipt


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        raise RuntimeError(f"global lock is still held: {path}")
    return descriptor


def _rename_noreplace(source: Path, destination: Path) -> None:
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError("staging and authoritative score root are not on one filesystem")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux renameat2 is required for no-overwrite publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to replace authoritative sidecar {destination}")
        raise OSError(error, os.strerror(error), str(destination))
    _fsync_directory(destination.parent)


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("worker was not started by its coordinator")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("coordinator exited while worker armed cleanup")


def _validate_evidence(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.equivalence_audit.resolve()
    benchmark_path = args.benchmark_summary.resolve()
    sample_path = args.benchmark_resource_samples.resolve()
    if _sha256_file(audit_path) != EQUIVALENCE_AUDIT_SHA256:
        raise RuntimeError("equivalence audit digest mismatch")
    if _sha256_file(benchmark_path) != BENCHMARK_SUMMARY_SHA256:
        raise RuntimeError("benchmark summary digest mismatch")
    if _sha256_file(sample_path) != BENCHMARK_RESOURCE_SAMPLES_SHA256:
        raise RuntimeError("benchmark resource-sample digest mismatch")
    audit = _strict_json(audit_path)
    benchmark = _strict_json(benchmark_path)
    required_audit = {
        "status": "pass",
        "differing_arrays_exactly_cost_arrays": True,
        "all_scientific_arrays_byte_identical": True,
        "all_non_cost_metadata_identical": True,
        "cost_differences_confined_to_runtime_resource_fields": True,
        "deterministic_cost_fields_identical": True,
        "cost_arrays_used_as_m0_m1_m2_predictors": False,
        "locked_test_accessed": False,
    }
    for field, expected in required_audit.items():
        if audit.get(field) != expected:
            raise RuntimeError(f"equivalence audit guard failed: {field}")
    if set(audit.get("cost_arrays", [])) != set(COST_ARRAY_NAMES):
        raise RuntimeError("equivalence audit cost-array set drifted")
    if (
        benchmark.get("status") != "complete"
        or benchmark.get("all_scientific_arrays_byte_identical") is not True
        or benchmark.get("serial_process_untouched") is not True
        or benchmark.get("authoritative_score_root_written") is not False
        or benchmark.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("benchmark summary guards failed")
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("benchmark resource samples are empty")
    times = [int(sample["unix_time_ns"]) for sample in samples]
    if times != sorted(times):
        raise RuntimeError("benchmark resource sample times are not monotonic")
    return {
        "equivalence_audit_path": str(audit_path),
        "equivalence_audit_sha256": EQUIVALENCE_AUDIT_SHA256,
        "benchmark_summary_path": str(benchmark_path),
        "benchmark_summary_sha256": BENCHMARK_SUMMARY_SHA256,
        "benchmark_resource_samples_path": str(sample_path),
        "benchmark_resource_samples_sha256": BENCHMARK_RESOURCE_SAMPLES_SHA256,
        "benchmark_start_unix_time_ns": min(times),
        "benchmark_end_unix_time_ns": max(times),
        "locked_test_accessed": False,
    }


def _classify_serial_modes(
    inventory: Mapping[str, Any], evidence: Mapping[str, Any], serial_start_ns: int
) -> dict[str, str]:
    start = int(evidence["benchmark_start_unix_time_ns"])
    end = int(evidence["benchmark_end_unix_time_ns"])
    modes: dict[str, str] = {}
    previous = serial_start_ns
    for record in inventory["records"]:
        publication = int(record["metadata"]["mtime_ns"])
        if publication <= previous:
            raise RuntimeError("serial sidecar publication times are not strictly increasing")
        overlap = previous <= end and publication >= start
        modes[str(record["episode_id"])] = (
            "serial_with_equivalence_benchmark_contention" if overlap else "serial"
        )
        previous = publication
    return modes


def _process_start_unix_ns(identity: Mapping[str, Any]) -> int:
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    boot_seconds = None
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            boot_seconds = int(line.split()[1])
            break
    if boot_seconds is None:
        raise RuntimeError("cannot determine Linux boot time")
    return int((boot_seconds + int(identity["start_ticks"]) / clock_ticks) * 1e9)


def _create_plan(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    control: Mapping[str, Any],
    evidence: Mapping[str, Any],
    cutover: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    completed = set(frozen["episode_ids"])
    missing = [episode_id for episode_id in context["ordered_ids"] if episode_id not in completed]
    assignments = (missing[::2], missing[1::2])
    if not missing or not all(assignments):
        raise RuntimeError("two nonempty continuation shards are required")
    selected = [episode_id for assignment in assignments for episode_id in assignment]
    if set(selected) != set(missing) or len(selected) != len(set(selected)):
        raise RuntimeError("worker partition is not an exact disjoint complement")
    modes = _classify_serial_modes(
        frozen,
        evidence,
        _process_start_unix_ns(cutover["serial_identities"]["python"]),
    )
    command_prefix = [
        str(args.python_executable.resolve()),
        str(args.serial_runner.resolve()),
        "--repo-root", str(args.repo_root.resolve()),
        "--environment-lock", str(args.environment_lock.resolve()),
        "--cache-dir", str(args.cache_dir.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--authority", str(args.authority.resolve()),
        "--raw-root", str(args.raw_root.resolve()),
        "--bound-probe", str(args.bound_probe.resolve()),
        "--score-root", str(args.score_root.resolve()),
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_plan",
        "created_unix_time_ns": time.time_ns(),
        "control": dict(control),
        "evidence": dict(evidence),
        "cutover_receipt_sha256": _sha256_bytes(_canonical(cutover)),
        "locked": {
            "repo_root": str(args.repo_root.resolve()),
            "serial_runner": str(args.serial_runner.resolve()),
            "serial_runner_sha256": _sha256_file(args.serial_runner.resolve()),
            "lock_commit": context["serial"].LOCK_COMMIT,
            "lock_tag": context["serial"].LOCK_TAG,
            "manifest_sha256": context["serial"].MANIFEST_SHA256,
            "bound_probe_sha256": context["serial"].BOUND_PROBE_SHA256,
        },
        "paths": {
            "control_repo": str(args.control_repo.resolve()),
            "execution_root": str(args.execution_root.resolve()),
            "global_lock": str(args.global_lock.resolve()),
            "python_executable": str(args.python_executable.resolve()),
            "environment_lock": str(args.environment_lock.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "manifest": str(args.manifest.resolve()),
            "authority": str(args.authority.resolve()),
            "raw_root": str(args.raw_root.resolve()),
            "bound_probe": str(args.bound_probe.resolve()),
            "score_root": str(args.score_root.resolve()),
            "feature_root": str(args.feature_root.resolve()),
        },
        "ordered_valid_episode_ids": list(context["ordered_ids"]),
        "frozen_raw_identities": [
            context["expected_raw_identities"][episode_id].to_dict()
            for episode_id in context["ordered_ids"]
        ],
        "frozen_serial_inventory": frozen,
        "frozen_serial_execution_modes": modes,
        "missing_episode_ids": missing,
        "assignments": [assignments[0], assignments[1]],
        "assignment_rule": "manifest_order_complement_alternating_indices",
        "assignment_inputs_excluded": [
            "labels", "features", "durations", "state_counts", "costs"
        ],
        "finalizer_command_prefix": command_prefix,
        "finalizer_feature_staging_base": str(
            args.execution_root.resolve() / "finalizer-staging"
        ),
        "locked_test_accessed": False,
    }
    digest = _write_json_exclusive(args.execution_root / "plan.json", plan)
    return plan, digest


def _load_plan(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = args.execution_root.resolve() / "plan.json"
    plan = _strict_json(path)
    digest = _sha256_file(path)
    if plan.get("kind") != f"{KIND}_plan" or plan.get("locked_test_accessed") is not False:
        raise RuntimeError("continuation plan guard failed")
    return plan, digest


def _assert_arguments_match_plan(
    args: argparse.Namespace, plan: Mapping[str, Any], plan_sha256: str
) -> None:
    expected_paths = {
        "control_repo": str(args.control_repo.resolve()),
        "execution_root": str(args.execution_root.resolve()),
        "global_lock": str(args.global_lock.resolve()),
        "python_executable": str(args.python_executable.resolve()),
        "environment_lock": str(args.environment_lock.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "authority": str(args.authority.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "bound_probe": str(args.bound_probe.resolve()),
        "score_root": str(args.score_root.resolve()),
        "feature_root": str(args.feature_root.resolve()),
    }
    if plan.get("paths") != expected_paths:
        raise RuntimeError("runtime paths differ from the frozen continuation plan")
    if plan["control"]["implementation_commit"] != args.expected_implementation_commit:
        raise RuntimeError("runtime implementation commit differs from plan")
    if plan["control"]["script_sha256"] != _sha256_file(Path(__file__).resolve()):
        raise RuntimeError("runtime script differs from plan")
    if plan["locked"]["serial_runner_sha256"] != _sha256_file(args.serial_runner.resolve()):
        raise RuntimeError("serial runner differs from frozen plan")
    if _sha256_file(args.execution_root.resolve() / "plan.json") != plan_sha256:
        raise RuntimeError("plan changed during runtime binding")
    evidence = plan["evidence"]
    for path_field, sha_field in (
        ("equivalence_audit_path", "equivalence_audit_sha256"),
        ("benchmark_summary_path", "benchmark_summary_sha256"),
        ("benchmark_resource_samples_path", "benchmark_resource_samples_sha256"),
    ):
        if _sha256_file(Path(evidence[path_field])) != evidence[sha_field]:
            raise RuntimeError(f"frozen evidence changed: {path_field}")


def _tree_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"staging attempt is not a real directory: {root}")
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"staging attempt contains a symlink: {entry}")
        if entry.is_dir():
            directories.append(relative)
        elif entry.is_file():
            files.append({"path": relative, **_file_record(entry)})
        else:
            raise RuntimeError(f"staging attempt contains a special file: {entry}")
    payload = {"directories": directories, "files": files}
    payload["tree_sha256"] = _sha256_bytes(_canonical(payload))
    return payload


def _next_attempt_root(
    worker_root: Path, episode_id: str, *, worker_index: int
) -> Path:
    base = worker_root / "staging" / episode_id
    base.mkdir(parents=True, exist_ok=True)
    for existing in sorted(base.glob("attempt-*")):
        if existing.is_symlink() or not existing.is_dir():
            raise RuntimeError(f"invalid existing staging attempt: {existing}")
        inventory = _tree_inventory(existing)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_abandoned_private_staging",
            "episode_id": episode_id,
            "worker_index": worker_index,
            "attempt_path": str(existing),
            "tree_inventory": inventory,
            "authoritative_score_root_written": False,
            "locked_test_accessed": False,
        }
        receipt_path = (
            worker_root
            / "abandoned-attempt-receipts"
            / episode_id
            / f"{existing.name}.json"
        )
        if receipt_path.exists():
            prior = _strict_json(receipt_path)
            if prior.get("tree_inventory") != inventory:
                raise RuntimeError(f"abandoned staging attempt changed: {existing}")
        else:
            _write_json_exclusive(receipt_path, receipt)
    for index in range(1, 1000):
        candidate = base / f"attempt-{index:04d}"
        if not candidate.exists() and not candidate.is_symlink():
            candidate.mkdir(exist_ok=False)
            return candidate
    raise RuntimeError(f"too many staging attempts for {episode_id}")


def _validate_abandoned_staging(worker_root: Path, worker_index: int) -> None:
    staging = worker_root / "staging"
    if not staging.exists():
        return
    for attempt in sorted(staging.glob("*/attempt-*")):
        episode_id = attempt.parent.name
        receipt_path = (
            worker_root
            / "abandoned-attempt-receipts"
            / episode_id
            / f"{attempt.name}.json"
        )
        receipt = _strict_json(receipt_path)
        if (
            receipt.get("worker_index") != worker_index
            or receipt.get("episode_id") != episode_id
            or receipt.get("tree_inventory") != _tree_inventory(attempt)
            or receipt.get("authoritative_score_root_written") is not False
        ):
            raise RuntimeError(f"abandoned staging receipt is invalid: {attempt}")


def _find_staged_sidecars(worker_root: Path, episode_id: str) -> list[Path]:
    base = worker_root / "staging" / episode_id
    if not base.exists():
        return []
    if base.is_symlink() or not base.is_dir():
        raise RuntimeError(f"invalid staging root for {episode_id}")
    return sorted(base.glob(f"attempt-*/scores/calibration/{episode_id}"))


def _prune_empty_parents(path: Path, stop: Path) -> None:
    current = path
    while current != stop and stop in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _promotion_paths(execution_root: Path, episode_id: str) -> tuple[Path, Path]:
    return (
        execution_root / "prepared" / f"{episode_id}.json",
        execution_root / "promotions" / f"{episode_id}.json",
    )


def _validate_record_matches_receipt(record: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    for field in ("episode_id", "combined_sha256", "metadata", "primitives", "links"):
        if record[field] != receipt["sidecar"][field]:
            raise RuntimeError(f"sidecar differs from prepared/promotion receipt: {field}")


def _worker_main(args: argparse.Namespace) -> int:
    _arm_parent_death_signal(args.coordinator_pid)
    script = Path(__file__).resolve()
    if _sha256_file(script) != args.expected_script_sha256:
        raise RuntimeError("worker script differs from coordinator-bound digest")
    plan_path = args.execution_root.resolve() / "plan.json"
    if _sha256_file(plan_path) != args.expected_plan_sha256:
        raise RuntimeError("worker plan digest mismatch")
    plan = _strict_json(plan_path)
    if plan["control"]["implementation_commit"] != args.expected_implementation_commit:
        raise RuntimeError("worker implementation commit differs from plan")
    _assert_arguments_match_plan(args, plan, args.expected_plan_sha256)
    context = _locked_context(args, validate_all_raw=False)
    if plan.get("frozen_raw_identities") != [
        context["expected_raw_identities"][episode_id].to_dict()
        for episode_id in context["ordered_ids"]
    ]:
        raise RuntimeError("worker bound-probe raw identities differ from plan")
    worker_index = args.worker_index
    if worker_index not in (0, 1):
        raise RuntimeError("worker index must be zero or one")
    assignment = tuple(plan["assignments"][worker_index])
    other = set(plan["assignments"][1 - worker_index])
    if not assignment or set(assignment) & other:
        raise RuntimeError("worker plan shards are empty or overlapping")
    expected_missing = set(plan["missing_episode_ids"])
    if set(plan["assignments"][0]) | set(plan["assignments"][1]) != expected_missing:
        raise RuntimeError("worker plan does not cover the frozen complement")
    lock_path = args.execution_root.resolve() / "locks" / f"worker{worker_index}.lock"
    lock_descriptor = _acquire_lock(lock_path)
    worker_root = args.execution_root.resolve() / f"worker{worker_index}"
    worker_root.mkdir(parents=True, exist_ok=True)
    score_split = args.score_root.resolve() / "calibration"
    try:
        from mech_int_vla.artifacts import load_rollout_artifact
        from mech_int_vla.config import ConditionSpec
        from mech_int_vla.failure_events import artifact_identity_from_rollout
        from mech_int_vla.instrumentation import SmolVLAInstrumentation
        from mech_int_vla.libero_runtime import RawLiberoEpisode
        from mech_int_vla.provenance import content_links_for
        from mech_int_vla.scoring import (
            FROZEN_TRANSFORMS,
            score_replay_to_sidecar,
        )
        from mech_int_vla.scoring_runtime import (
            SmolVLAScoringAdapter,
            factual_replay_from_artifact,
        )
        from mech_int_vla.snapshots import load_locked_smolvla, resolve_snapshot_paths

        _require_locked_python_imports(args.repo_root.resolve())

        def validate_episode(path: Path, episode_id: str, artifact: Any) -> dict[str, Any]:
            links = content_links_for(
                artifact_identity_from_rollout(artifact),
                context["bound"],
                context["root"],
                protocol=context["protocol"],
            )
            return _sidecar_record(path, episode_id, expected_links=links)

        remaining = [
            episode_id
            for episode_id in assignment
            if not (args.execution_root / "promotions" / f"{episode_id}.json").exists()
        ]
        policy_runtime = None
        if remaining:
            snapshots = resolve_snapshot_paths(
                args.environment_lock.resolve(),
                cache_dir=args.cache_dir.resolve(),
                local_files_only=True,
            )
            model_started = time.perf_counter()
            policy_runtime = load_locked_smolvla(snapshots, device="cuda")
            model_load_seconds = time.perf_counter() - model_started
        else:
            model_load_seconds = 0.0

        completed: list[dict[str, Any]] = []
        for episode_id in assignment:
            artifact = load_rollout_artifact(
                args.raw_root.resolve() / "calibration" / episode_id,
                expected_task=context["task"],
            )
            if not artifact.valid_reset:
                raise RuntimeError(f"worker saw invalid reset: {episode_id}")
            _require_frozen_raw_identity(context, artifact)
            destination = score_split / episode_id
            prepared_path, promotion_path = _promotion_paths(
                args.execution_root.resolve(), episode_id
            )
            if promotion_path.exists():
                promotion = _strict_json(promotion_path)
                if promotion.get("worker_index") != worker_index:
                    raise RuntimeError("promotion receipt worker mismatch")
                record = validate_episode(destination, episode_id, artifact)
                _validate_record_matches_receipt(record, promotion)
                completed.append(promotion)
                continue

            prepared = _strict_json(prepared_path) if prepared_path.exists() else None
            staged = _find_staged_sidecars(worker_root, episode_id)
            if len(staged) > 1:
                raise RuntimeError(f"multiple completed staging sidecars: {episode_id}")
            if destination.exists():
                if prepared is None or staged:
                    raise RuntimeError(
                        f"unreceipted authoritative destination cannot be adopted: {episode_id}"
                    )
                record = validate_episode(destination, episode_id, artifact)
                _validate_record_matches_receipt(record, prepared)
            else:
                if staged:
                    stage_path = staged[0]
                    record = validate_episode(stage_path, episode_id, artifact)
                else:
                    if prepared is not None:
                        raise RuntimeError(f"prepared sidecar disappeared: {episode_id}")
                    if policy_runtime is None:
                        raise RuntimeError("model runtime unavailable for missing episode")
                    spec = context["expected"][episode_id]
                    condition = ConditionSpec(
                        spec.condition_name,
                        spec.condition_family,
                        spec.condition_index,
                        spec.condition_parameters,
                    )
                    attempt = _next_attempt_root(
                        worker_root, episode_id, worker_index=worker_index
                    )
                    episode = RawLiberoEpisode.create(
                        context["task"],
                        base_init_state_id=spec.base_init_state_id,
                        execution=context["protocol"].split.policy_execution,
                        validity=context["protocol"].perturbations.validity,
                    )
                    instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
                    episode_started = time.perf_counter()
                    try:
                        adapter = SmolVLAScoringAdapter(
                            episode,
                            policy_runtime,
                            artifact,
                            context["bound"],
                            instrumentation,
                            reset_seed=spec.reset_seed,
                            original_condition=condition,
                            protocol=context["protocol"],
                            repo_root=context["root"],
                        )
                        replay = factual_replay_from_artifact(artifact)
                        links = content_links_for(
                            artifact_identity_from_rollout(artifact),
                            context["bound"],
                            context["root"],
                            protocol=context["protocol"],
                        )
                        result = score_replay_to_sidecar(
                            adapter,
                            replay,
                            links,
                            transforms=FROZEN_TRANSFORMS,
                            output_root=attempt / "scores",
                        )
                    finally:
                        instrumentation.remove()
                        episode.close()
                    stage_path = result.path
                    record = validate_episode(stage_path, episode_id, artifact)
                    record["worker_scoring_elapsed_seconds"] = (
                        time.perf_counter() - episode_started
                    )
                if prepared is None:
                    prepared = {
                        "schema_version": SCHEMA_VERSION,
                        "kind": f"{KIND}_prepared_sidecar",
                        "episode_id": episode_id,
                        "worker_index": worker_index,
                        "execution_mode": "two_worker",
                        "plan_sha256": args.expected_plan_sha256,
                        "script_sha256": args.expected_script_sha256,
                        "sidecar": record,
                        "locked_test_accessed": False,
                    }
                    _write_json_exclusive(prepared_path, prepared)
                else:
                    _validate_record_matches_receipt(record, prepared)
                _rename_noreplace(stage_path, destination)
                _prune_empty_parents(stage_path.parent, worker_root / "staging")
                record = validate_episode(destination, episode_id, artifact)
                _validate_record_matches_receipt(record, prepared)

            promotion = {
                **prepared,
                "kind": f"{KIND}_promotion",
                "promoted_unix_time_ns": time.time_ns(),
                "authoritative_path": str(destination),
            }
            _write_json_exclusive(promotion_path, promotion)
            completed.append(promotion)
            print(
                json.dumps(
                    {
                        "kind": "two_worker_score_promoted",
                        "episode_id": episode_id,
                        "worker_index": worker_index,
                        "completed": len(completed),
                        "total": len(assignment),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_worker_summary",
            "worker_index": worker_index,
            "worker_pid": os.getpid(),
            "assignment": list(assignment),
            "promotion_receipts": [
                str(_promotion_paths(args.execution_root.resolve(), episode_id)[1])
                for episode_id in assignment
            ],
            "model_load_seconds": model_load_seconds,
            "status": "complete",
            "locked_test_accessed": False,
        }
        summary_path = worker_root / "worker-summary.json"
        if summary_path.exists():
            existing = _strict_json(summary_path)
            if existing.get("assignment") != list(assignment) or existing.get("status") != "complete":
                raise RuntimeError("existing worker summary differs from plan")
        else:
            _write_json_exclusive(summary_path, summary)
        return 0
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _finalizer_main(args: argparse.Namespace) -> int:
    """Arm parent-death cleanup, validate the plan, then exec the locked runner."""

    _arm_parent_death_signal(args.coordinator_pid)
    script = Path(__file__).resolve()
    if _sha256_file(script) != args.expected_script_sha256:
        raise RuntimeError("finalizer wrapper script differs from frozen plan")
    plan, plan_sha256 = _load_plan(args)
    if plan_sha256 != args.expected_plan_sha256:
        raise RuntimeError("finalizer wrapper plan digest mismatch")
    _assert_arguments_match_plan(args, plan, plan_sha256)
    feature_root = args.finalizer_feature_root.resolve()
    staging_base = Path(plan["finalizer_feature_staging_base"]).resolve()
    if staging_base not in feature_root.parents:
        raise RuntimeError("finalizer feature root is outside frozen staging base")
    if feature_root.exists() or feature_root.is_symlink():
        raise RuntimeError("finalizer feature staging root must be absent")
    command = [*plan["finalizer_command_prefix"], "--feature-root", str(feature_root)]
    environment = _locked_environment(args)
    os.execve(command[0], command, environment)
    raise AssertionError("os.execve unexpectedly returned")


def _resource_sample(worker_pids: Sequence[int]) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "unix_time_ns": time.time_ns(),
        "monotonic_seconds": time.monotonic(),
        "worker_pids": list(worker_pids),
    }
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values = raw.strip().split()
        if values:
            meminfo[key] = int(values[0]) * 1024
    sample["host_memory"] = {
        "total_bytes": meminfo.get("MemTotal"),
        "available_bytes": meminfo.get("MemAvailable"),
        "used_bytes": meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0),
    }
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    if gpu is not None and gpu.returncode == 0 and gpu.stdout.strip():
        values = [float(value.strip()) for value in gpu.stdout.splitlines()[0].split(",")]
        sample["gpu"] = {
            "utilization_percent": values[0],
            "memory_used_mib": values[1],
            "memory_total_mib": values[2],
        }
    try:
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        compute = None
    rows: list[dict[str, Any]] = []
    if compute is not None and compute.returncode == 0:
        for line in compute.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit():
                rows.append({"pid": int(fields[0]), "used_memory_mib": float(fields[1])})
    sample["gpu_processes"] = rows
    if worker_pids:
        try:
            process = subprocess.run(
                [
                    "ps", "-o", "pid=,ppid=,%cpu=,rss=,vsz=,stat=,comm=", "-p",
                    ",".join(str(pid) for pid in worker_pids),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            process = None
        process_rows = []
        for line in process.stdout.splitlines() if process is not None else ():
            parts = line.split(None, 6)
            if len(parts) == 7:
                process_rows.append(
                    {
                        "pid": int(parts[0]),
                        "ppid": int(parts[1]),
                        "cpu_percent": float(parts[2]),
                        "rss_bytes": int(parts[3]) * 1024,
                        "vsz_bytes": int(parts[4]) * 1024,
                        "stat": parts[5],
                        "command": parts[6],
                    }
                )
        sample["processes"] = process_rows
    return sample


def _terminate_processes_without_sigkill(
    processes: Sequence[subprocess.Popen[Any]], *, timeout_seconds: float = 30.0
) -> None:
    live = [process for process in processes if process.poll() is None]
    for signum in (signal.SIGINT, signal.SIGTERM):
        for process in live:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            live = [process for process in live if process.poll() is None]
            if not live:
                return
            time.sleep(0.1)
    if live:
        raise RuntimeError(
            f"workers remained live after SIGINT/SIGTERM; SIGKILL forbidden: "
            f"{[process.pid for process in live]}"
        )


def _next_log_path(execution_root: Path, prefix: str) -> Path:
    logs = execution_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = logs / f"{prefix}-attempt-{index:04d}.log"
        if not path.exists() and not path.is_symlink():
            return path
    raise RuntimeError(f"too many log attempts for {prefix}")


def _quick_verify_frozen_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    for name in ("metadata", "primitives"):
        expected = record[name]
        target = path / ("metadata.json" if name == "metadata" else "primitives.npz")
        current = _file_record(target)
        if current != expected:
            raise RuntimeError(f"continuous frozen-hash guard failed: {record['episode_id']}")


def _aggregate_costs(
    inventory: Mapping[str, Any], modes: Mapping[str, str]
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {
        mode: {
            "episode_count": 0,
            "state_count": 0,
            "cuda_event_ms_sum": 0.0,
            "wall_time_ns_sum": 0,
            "forward_count": 0,
            "intervention_count": 0,
            "peak_allocated_bytes_max": 0,
            "incremental_peak_allocated_bytes_max": 0,
            "logical_activation_bytes": 0,
            "compressed_activation_bytes": 0,
        }
        for mode in EXECUTION_MODES
    }
    for record in inventory["records"]:
        episode_id = str(record["episode_id"])
        mode = modes[episode_id]
        target = aggregates[mode]
        target["episode_count"] += 1
        target["state_count"] += int(record["state_count"])
        cost = record["cost"]
        for field in (
            "cuda_event_ms_sum",
            "wall_time_ns_sum",
            "forward_count",
            "intervention_count",
            "logical_activation_bytes",
            "compressed_activation_bytes",
        ):
            target[field] += cost[field]
        for field in ("peak_allocated_bytes_max", "incremental_peak_allocated_bytes_max"):
            target[field] = max(target[field], cost[field])
    return aggregates


def _resource_summary(path: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    samples.append(value)
    gpu_util = [sample["gpu"]["utilization_percent"] for sample in samples if "gpu" in sample]
    gpu_mem = [sample["gpu"]["memory_used_mib"] for sample in samples if "gpu" in sample]
    host_used = [sample["host_memory"]["used_bytes"] for sample in samples]
    worker_rss: list[int] = []
    worker_cpu: list[float] = []
    for sample in samples:
        rows = sample.get("processes", [])
        worker_rss.append(sum(int(row["rss_bytes"]) for row in rows))
        worker_cpu.append(sum(float(row["cpu_percent"]) for row in rows))
    return {
        "sample_count": len(samples),
        "device_gpu_utilization_percent_max": max(gpu_util) if gpu_util else None,
        "device_gpu_memory_used_mib_max": max(gpu_mem) if gpu_mem else None,
        "host_used_ram_bytes_max": max(host_used) if host_used else None,
        "worker_rss_bytes_sum_max": max(worker_rss) if worker_rss else None,
        "worker_cpu_percent_sum_max": max(worker_cpu) if worker_cpu else None,
        "scope": "device totals and process sums sampled during two-worker continuation",
    }


def _validate_promotions(
    plan: Mapping[str, Any], execution_root: Path, plan_sha256: str
) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for index, assignment in enumerate(plan["assignments"]):
        for episode_id in assignment:
            receipt = _strict_json(execution_root / "promotions" / f"{episode_id}.json")
            if (
                receipt.get("kind") != f"{KIND}_promotion"
                or receipt.get("worker_index") != index
                or receipt.get("execution_mode") != "two_worker"
                or receipt.get("plan_sha256") != plan_sha256
                or receipt.get("script_sha256") != plan["control"]["script_sha256"]
                or receipt.get("locked_test_accessed") is not False
            ):
                raise RuntimeError(f"invalid promotion receipt: {episode_id}")
            counts[str(index)] += 1
    return counts


def _run_process_with_samples(
    process: subprocess.Popen[Any], sample_path: Path, *, sample_interval: float
) -> int:
    try:
        while process.poll() is None:
            _append_jsonl(sample_path, _resource_sample([process.pid]))
            time.sleep(sample_interval)
        _append_jsonl(sample_path, _resource_sample([]))
        return process.wait()
    except BaseException:
        _terminate_processes_without_sigkill([process])
        raise


def _next_finalizer_feature_root(execution_root: Path) -> Path:
    base = execution_root / "finalizer-staging"
    base.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        attempt = base / f"attempt-{index:04d}"
        if not attempt.exists() and not attempt.is_symlink():
            attempt.mkdir(exist_ok=False)
            return attempt / "features"
    raise RuntimeError("too many finalizer staging attempts")


def _next_resource_sample_path(execution_root: Path, prefix: str) -> Path:
    base = execution_root / "resource-sample-attempts"
    base.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = base / f"{prefix}-attempt-{index:04d}.jsonl"
        if not path.exists() and not path.is_symlink():
            return path
    raise RuntimeError(f"too many resource-sample attempts for {prefix}")


def _run_plan(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_sha256: str,
    global_lock_descriptor: int,
) -> int:
    fcntl.fcntl(global_lock_descriptor, fcntl.F_GETFD)
    os.set_inheritable(global_lock_descriptor, True)
    execution_root = args.execution_root.resolve()
    _assert_arguments_match_plan(args, plan, plan_sha256)
    frozen = plan["frozen_serial_inventory"]
    current = _inventory(context, args)
    _assert_frozen_unchanged(frozen, current)
    processes: list[subprocess.Popen[Any]] = []
    logs: list[Any] = []
    samples_path = _next_resource_sample_path(execution_root, "two-worker")
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    def interrupt(signum: int, _frame: Any) -> None:
        raise CoordinatorInterrupted(f"coordinator received signal {signum}")

    for signum in previous_handlers:
        signal.signal(signum, interrupt)
    try:
        for index in (0, 1):
            log_path = _next_log_path(execution_root, f"worker{index}")
            log = log_path.open("xb")
            logs.append(log)
            command = [
                str(args.python_executable.resolve()),
                str(Path(__file__).resolve()),
                "worker",
                *_common_worker_arguments(args),
                "--worker-index", str(index),
                "--coordinator-pid", str(os.getpid()),
                "--expected-script-sha256", plan["control"]["script_sha256"],
                "--expected-plan-sha256", plan_sha256,
                "--expected-implementation-commit", plan["control"]["implementation_commit"],
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(global_lock_descriptor,),
                    env=_locked_environment(args),
                )
            )
        runtime = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_runtime",
            "coordinator_pid": os.getpid(),
            "worker_pids": [process.pid for process in processes],
            "plan_sha256": plan_sha256,
            "started_unix_time_ns": time.time_ns(),
            "locked_test_accessed": False,
        }
        runtime_path = execution_root / "runtime.json"
        if runtime_path.exists():
            _append_jsonl(execution_root / "runtime-resumes.jsonl", runtime)
        else:
            _write_json_exclusive(runtime_path, runtime)
        frozen_records = list(frozen["records"])
        guard_index = 0
        last_guard = 0.0
        while any(process.poll() is None for process in processes):
            failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"continuation worker failed: {failures}")
            _append_jsonl(samples_path, _resource_sample([p.pid for p in processes]))
            now = time.monotonic()
            if frozen_records and now - last_guard >= 60.0:
                _quick_verify_frozen_record(frozen_records[guard_index % len(frozen_records)])
                guard_index += 1
                last_guard = now
            time.sleep(args.sample_interval)
        return_codes = [process.wait() for process in processes]
        _append_jsonl(samples_path, _resource_sample([]))
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"continuation workers failed: {return_codes}")
    except BaseException:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        _terminate_processes_without_sigkill(processes)
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        for log in logs:
            log.flush()
            os.fsync(log.fileno())
            log.close()

    _validate_promotions(plan, execution_root, plan_sha256)
    for index in (0, 1):
        _validate_abandoned_staging(execution_root / f"worker{index}", index)
    final_inventory = _inventory(context, args)
    if final_inventory["episode_ids"] != list(context["ordered_ids"]):
        raise RuntimeError("final authoritative inventory is not exactly 160 manifest IDs")
    _assert_frozen_unchanged(frozen, final_inventory)
    modes = dict(plan["frozen_serial_execution_modes"])
    worker_for_episode = {
        episode_id: index
        for index, assignment in enumerate(plan["assignments"])
        for episode_id in assignment
    }
    modes.update({episode_id: "two_worker" for episode_id in worker_for_episode})
    if set(modes) != set(context["ordered_ids"]):
        raise RuntimeError("execution-mode map does not cover exactly all episodes")
    execution_entries = []
    for record in final_inventory["records"]:
        episode_id = record["episode_id"]
        execution_entries.append(
            {
                "episode_id": episode_id,
                "execution_mode": modes[episode_id],
                "execution_family": (
                    "two_worker" if modes[episode_id] == "two_worker" else "serial"
                ),
                "worker_index": worker_for_episode.get(episode_id),
                "combined_sha256": record["combined_sha256"],
                "metadata_sha256": record["metadata"]["sha256"],
                "primitives_sha256": record["primitives"]["sha256"],
                "state_count": record["state_count"],
                "cost": record["cost"],
            }
        )
    execution_receipt_path = execution_root / "execution-receipt.json"
    if execution_receipt_path.exists():
        execution_receipt = _strict_json(execution_receipt_path)
        if (
            execution_receipt.get("kind") != f"{KIND}_execution_receipt"
            or execution_receipt.get("status") != "scoring_complete"
            or execution_receipt.get("closed") is not True
            or execution_receipt.get("plan_sha256") != plan_sha256
            or execution_receipt.get("episode_count") != len(execution_entries)
            or execution_receipt.get("episodes") != execution_entries
            or execution_receipt.get("cost_aggregates_by_execution_mode")
            != _aggregate_costs(final_inventory, modes)
            or execution_receipt.get("predictor_input") is not False
            or execution_receipt.get("locked_test_accessed") is not False
        ):
            raise RuntimeError("closed execution receipt fails resume validation")
    else:
        execution_receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_execution_receipt",
            "status": "scoring_complete",
            "closed": True,
            "plan_sha256": plan_sha256,
            "episode_count": len(execution_entries),
            "execution_modes": list(EXECUTION_MODES),
            "episodes": execution_entries,
            "cost_aggregates_by_execution_mode": _aggregate_costs(final_inventory, modes),
            "physical_cost_interpretation": {
                "latency_and_memory_stratified": True,
                "summed_worker_time_is_parallel_makespan": False,
                "per_process_peak_is_aggregate_device_peak": False,
                "logical_counts_and_activation_bytes_aggregable": True,
            },
            "continuation_resource_sample_path": str(samples_path),
            "continuation_resource_sample_sha256": _sha256_file(samples_path),
            "continuation_resource_summary": _resource_summary(samples_path),
            "benchmark_overhead_receipt": plan["evidence"],
            "predictor_input": False,
            "locked_test_accessed": False,
        }
        _write_json_exclusive(execution_receipt_path, execution_receipt)

    feature_summary = args.feature_root.resolve() / "score-feature-summary.json"
    if not feature_summary.exists():
        if args.feature_root.is_symlink():
            raise RuntimeError("authoritative feature root may not be a symlink")
        if args.feature_root.exists():
            if not args.feature_root.is_dir() or any(args.feature_root.iterdir()):
                raise RuntimeError("incomplete authoritative feature root is not resumable")
            args.feature_root.rmdir()
            _fsync_directory(args.feature_root.resolve().parent)
        staged_feature_root = _next_finalizer_feature_root(execution_root)
        finalizer_log_path = _next_log_path(execution_root, "unchanged-finalizer")
        finalizer_samples = _next_resource_sample_path(execution_root, "finalizer")
        with finalizer_log_path.open("xb") as log:
            command = [
                str(args.python_executable.resolve()),
                str(Path(__file__).resolve()),
                "finalizer",
                *_common_worker_arguments(args),
                "--coordinator-pid", str(os.getpid()),
                "--expected-script-sha256", plan["control"]["script_sha256"],
                "--expected-plan-sha256", plan_sha256,
                "--finalizer-feature-root", str(staged_feature_root),
            ]
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(global_lock_descriptor,),
                env=_locked_environment(args),
            )
            code = _run_process_with_samples(
                process, finalizer_samples, sample_interval=args.sample_interval
            )
            log.flush()
            os.fsync(log.fileno())
        if code != 0:
            raise RuntimeError(f"unchanged serial finalizer failed: {code}")
        staged_summary = staged_feature_root / "score-feature-summary.json"
        staged_feature = _strict_json(staged_summary)
        if (
            staged_feature.get("sidecar_count") != len(context["ordered_ids"])
            or staged_feature.get("locked_test_accessed") is not False
        ):
            raise RuntimeError("staged unchanged finalizer summary guard failed")
        _rename_noreplace(staged_feature_root, args.feature_root.resolve())
        _prune_empty_parents(
            staged_feature_root.parent,
            execution_root / "finalizer-staging",
        )
    feature = _strict_json(feature_summary)
    if (
        feature.get("sidecar_count") != len(context["ordered_ids"])
        or feature.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("unchanged serial finalizer summary guard failed")
    completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_completion",
        "status": "complete",
        "plan_sha256": plan_sha256,
        "execution_receipt_sha256": _sha256_file(execution_receipt_path),
        "feature_summary_path": str(feature_summary),
        "feature_summary_sha256": _sha256_file(feature_summary),
        "completed_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    completion_path = execution_root / "completion.json"
    if completion_path.exists():
        existing = _strict_json(completion_path)
        if existing.get("execution_receipt_sha256") != completion["execution_receipt_sha256"]:
            raise RuntimeError("completion receipt changed on resume")
    else:
        _write_json_exclusive(completion_path, completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


def _validate_paths(args: argparse.Namespace) -> None:
    execution_root = args.execution_root.resolve()
    protected = (
        ("locked repository", args.repo_root.resolve()),
        ("control repository", args.control_repo.resolve()),
        ("raw root", args.raw_root.resolve()),
        ("authoritative score root", args.score_root.resolve()),
        ("feature root", args.feature_root.resolve()),
        ("cache", args.cache_dir.resolve()),
        ("bound probe", args.bound_probe.resolve()),
        ("manifest", args.manifest.resolve()),
        ("authority", args.authority.resolve()),
        ("serial runner", args.serial_runner.resolve()),
    )
    _require_disjoint_path(execution_root, protected, name="execution root")
    if args.score_root.resolve().parent.stat().st_dev != execution_root.parent.stat().st_dev:
        raise RuntimeError("execution and score roots must reside on one filesystem")


def _cutover_main(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    control = _require_control_checkout(
        args.control_repo.resolve(), script, args.expected_implementation_commit
    )
    context = _locked_context(args, validate_all_raw=True)
    evidence = _validate_evidence(args)
    _validate_paths(args)
    if args.execution_root.exists() or args.execution_root.is_symlink():
        raise RuntimeError("execution root already exists; use resume if a plan exists")
    if args.feature_root.is_symlink():
        raise RuntimeError("feature root may not be a symlink")
    if args.feature_root.exists() and (
        not args.feature_root.is_dir() or any(args.feature_root.iterdir())
    ):
        raise RuntimeError("feature root is not an empty real directory")
    identities = _capture_serial_identities(args.serial_runner, args.global_lock.resolve())
    log_path = args.serial_log.resolve()
    if log_path.is_symlink() or not log_path.is_file():
        raise RuntimeError("serial log is not a regular file")
    initial, log_offset = _stable_inventory_log_baseline(
        context, args, log_path, identities["python"]
    )
    args.execution_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(args.execution_root.parent)
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_cutover_intent",
        "created_unix_time_ns": time.time_ns(),
        "control": control,
        "evidence": evidence,
        "serial_identities": identities,
        "serial_log": str(log_path),
        "serial_log_offset": log_offset,
        "initial_inventory": initial,
        "locked_test_accessed": False,
    }
    _write_json_exclusive(args.execution_root / "cutover-intent.json", intent)
    boundary, boundary_offset = _wait_for_score_completed(
        log_path,
        log_offset,
        identities["python"],
        timeout_seconds=args.boundary_timeout,
    )
    if boundary.get("episode_id") not in context["ordered_ids"]:
        raise RuntimeError("boundary line named an episode outside manifest")
    _prevalidate_boundary_line(boundary, initial, context["ordered_ids"])
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial scorer identity changed at boundary")
    boundary_observed = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_boundary_observed",
        "boundary_line": boundary,
        "boundary_log_offset": boundary_offset,
        "baseline_inventory": initial,
        "serial_python_identity": identities["python"],
        "observed_unix_time_ns": time.time_ns(),
        "locked_test_accessed": False,
    }
    boundary_path = args.execution_root / "boundary-observed.json"
    _write_json_exclusive(boundary_path, boundary_observed)
    _dispatch_serial_interrupt(args, identities, boundary_path)
    lock_descriptor = _acquire_lock(args.global_lock.resolve())
    try:
        frozen = _inventory(context, args)
        _assert_frozen_unchanged(initial, frozen)
        _validate_boundary_transition(
            boundary, initial, frozen, context["ordered_ids"]
        )
        cutover = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_cutover_receipt",
            "serial_identities": identities,
            "signal": "SIGINT",
            "signal_count": 1,
            "boundary_observed_sha256": _sha256_file(
                args.execution_root / "boundary-observed.json"
            ),
            "serial_interrupt_sha256": _sha256_file(
                args.execution_root / "serial-interrupt.json"
            ),
            "boundary_line": boundary_observed["boundary_line"],
            "boundary_log_offset": boundary_observed["boundary_log_offset"],
            "original_intent_inventory_sha256": initial["inventory_sha256"],
            "original_intent_episode_count": initial["episode_count"],
            "boundary_baseline_inventory_sha256": initial["inventory_sha256"],
            "boundary_baseline_episode_count": initial["episode_count"],
            "initial_inventory_sha256": initial["inventory_sha256"],
            "frozen_inventory_sha256": frozen["inventory_sha256"],
            "initial_episode_count": initial["episode_count"],
            "frozen_episode_count": frozen["episode_count"],
            "global_lock_reacquired_exclusively": True,
            "serial_processes_remaining": 0,
            "locked_test_accessed": False,
        }
        _write_json_exclusive(args.execution_root / "cutover-receipt.json", cutover)
        plan, plan_sha = _create_plan(
            args, context, control, evidence, cutover, frozen
        )
        return _run_plan(args, context, plan, plan_sha, lock_descriptor)
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_failure",
            "stage": "after_serial_boundary",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "unix_time_ns": time.time_ns(),
            "locked_test_accessed": False,
        }
        path = args.execution_root / "failure.json"
        if not path.exists():
            _write_json_exclusive(path, failure)
        raise
    finally:
        # See cutover: inherited child descriptors must retain the flock if the
        # coordinator exits before a writer.
        os.close(lock_descriptor)


def _assert_inventory_is_manifest_prefix(
    inventory: Mapping[str, Any], ordered_ids: Sequence[str]
) -> None:
    count = int(inventory["episode_count"])
    if list(inventory["episode_ids"]) != list(ordered_ids[:count]):
        raise RuntimeError("serial inventory is not an exact manifest prefix")
    if count != len(_records_by_id(inventory)):
        raise RuntimeError("serial inventory count differs from its records")


def _assert_two_worker_capacity_after_boundary(
    inventory: Mapping[str, Any], ordered_ids: Sequence[str]
) -> None:
    count = int(inventory["episode_count"])
    remaining = len(ordered_ids) - (count + 1)
    if remaining < 2:
        raise RuntimeError(
            "fewer than two continuation episodes would remain after fresh boundary"
        )


def _assert_original_boundary_retained(
    original: Mapping[str, Any], inventory: Mapping[str, Any]
) -> None:
    baseline = original["boundary"]["baseline_inventory"]
    _assert_frozen_unchanged(baseline, inventory)
    line = original["boundary"]["boundary_line"]
    records = _records_by_id(inventory)
    episode_id = str(line["episode_id"])
    if episode_id not in records:
        raise RuntimeError("original boundary sidecar disappeared")
    if records[episode_id]["combined_sha256"] != line["sha256"]:
        raise RuntimeError("original boundary sidecar hash changed")


def _validate_recovery_intent(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    control: Mapping[str, Any],
    evidence: Mapping[str, Any],
    original: Mapping[str, Any],
    ineffective_path: Path,
) -> dict[str, Any]:
    path = args.execution_root.resolve() / "recovery-cutover-intent.json"
    receipt = _strict_json(path)
    if (
        receipt.get("kind") != f"{KIND}_ignored_sigint_recovery_intent"
        or receipt.get("control") != dict(control)
        or receipt.get("evidence") != dict(evidence)
        or receipt.get("serial_identities") != original["serial_identities"]
        or receipt.get("original_cutover_intent_sha256")
        != _sha256_file(Path(original["cutover_intent_path"]))
        or receipt.get("original_boundary_sha256")
        != _sha256_file(Path(original["boundary_path"]))
        or receipt.get("original_signal_intent_sha256")
        != _sha256_file(Path(original["signal_intent_path"]))
        or receipt.get("original_signal_dispatched_sha256")
        != _sha256_file(Path(original["signal_dispatched_path"]))
        or receipt.get("original_signal_ineffective_sha256")
        != _sha256_file(ineffective_path)
        or receipt.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("recovery cutover intent provenance is invalid")
    baseline = receipt.get("baseline_inventory")
    if not isinstance(baseline, Mapping):
        raise RuntimeError("recovery baseline inventory is malformed")
    _assert_inventory_is_manifest_prefix(baseline, context["ordered_ids"])
    _assert_two_worker_capacity_after_boundary(baseline, context["ordered_ids"])
    _assert_original_boundary_retained(original, baseline)
    log_identity = receipt.get("serial_log_identity")
    if not isinstance(log_identity, Mapping):
        raise RuntimeError("recovery serial-log identity is malformed")
    observed_log = _assert_same_log_object(log_identity, Path(log_identity["path"]))
    offset = int(receipt["serial_log_offset"])
    if offset != int(log_identity["size_bytes"]) or observed_log["size_bytes"] < offset:
        raise RuntimeError("recovery serial-log offset is inconsistent")
    status = receipt.get("signal_status")
    if not isinstance(status, Mapping) or not isinstance(status.get("proc_status"), Mapping):
        raise RuntimeError("recovery baseline signal status is malformed")
    _validate_recovery_signal_status(status["proc_status"])
    if receipt.get("serial_runner_sha256") != _sha256_file(args.serial_runner.resolve()):
        raise RuntimeError("serial runner differs from recovery intent")
    return receipt


def _create_recovery_intent(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    control: Mapping[str, Any],
    evidence: Mapping[str, Any],
    original: Mapping[str, Any],
    ineffective_path: Path,
) -> dict[str, Any]:
    identities = original["serial_identities"]
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial Python identity changed before recovery baseline")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("serial flock identity changed before recovery baseline")
    log_path = args.serial_log.resolve()
    if log_path != Path(original["cutover_intent"]["serial_log"]).resolve():
        raise RuntimeError("recovery serial log differs from original intent")
    baseline, offset = _stable_inventory_log_baseline(
        context, args, log_path, identities["python"]
    )
    _assert_inventory_is_manifest_prefix(baseline, context["ordered_ids"])
    _assert_two_worker_capacity_after_boundary(baseline, context["ordered_ids"])
    _assert_original_boundary_retained(original, baseline)
    log_identity = _log_identity(log_path)
    if log_identity["size_bytes"] != offset:
        raise RuntimeError("serial log advanced while creating recovery intent")
    runner_sha = _sha256_file(args.serial_runner.resolve())
    status = _capture_recovery_signal_status(identities["python"])
    count = int(baseline["episode_count"])
    potentially_started = (
        context["ordered_ids"][count] if count < len(context["ordered_ids"]) else None
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_ignored_sigint_recovery_intent",
        "created_unix_time_ns": time.time_ns(),
        "control": dict(control),
        "evidence": dict(evidence),
        "serial_identities": identities,
        "serial_runner_sha256": runner_sha,
        "serial_log_identity": log_identity,
        "serial_log_offset": offset,
        "baseline_inventory": baseline,
        "original_cutover_intent_sha256": _sha256_file(
            Path(original["cutover_intent_path"])
        ),
        "original_boundary_sha256": _sha256_file(Path(original["boundary_path"])),
        "original_signal_intent_sha256": _sha256_file(
            Path(original["signal_intent_path"])
        ),
        "original_signal_dispatched_sha256": _sha256_file(
            Path(original["signal_dispatched_path"])
        ),
        "original_signal_ineffective_sha256": _sha256_file(ineffective_path),
        "signal_status": status,
        "next_manifest_episode_potentially_in_memory": potentially_started,
        "locked_test_accessed": False,
    }
    _write_json_exclusive(
        args.execution_root.resolve() / "recovery-cutover-intent.json", receipt
    )
    return receipt


def _validate_logged_recovery_boundary(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    recovery_intent: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = args.execution_root.resolve() / "recovery-boundary-observed.json"
    receipt = _strict_json(path)
    log_identity = recovery_intent["serial_log_identity"]
    log_path = Path(log_identity["path"])
    _assert_same_log_object(log_identity, log_path)
    record = receipt.get("log_record")
    if not isinstance(record, Mapping):
        raise RuntimeError("recovery boundary log record is malformed")
    start = int(record["log_start_offset"])
    end = int(record["log_end_offset"])
    if start < int(recovery_intent["serial_log_offset"]) or end <= start:
        raise RuntimeError("recovery boundary offsets are invalid")
    with log_path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(end - start)
    if not payload.endswith(b"\n"):
        raise RuntimeError("recovery boundary log record lacks newline")
    raw = payload[:-1]
    if _sha256_bytes(raw) != record.get("raw_line_sha256"):
        raise RuntimeError("recovery boundary raw log hash differs")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("recovery boundary log record is not JSON") from exc
    if parsed != record.get("boundary_line"):
        raise RuntimeError("recovery boundary parsed line differs from log")
    if (
        receipt.get("kind") != f"{KIND}_ignored_sigint_recovery_boundary"
        or receipt.get("recovery_intent_sha256")
        != _sha256_file(args.execution_root.resolve() / "recovery-cutover-intent.json")
        or receipt.get("baseline_inventory") != recovery_intent["baseline_inventory"]
        or receipt.get("serial_python_identity")
        != recovery_intent["serial_identities"]["python"]
        or receipt.get("serial_log_identity") != log_identity
        or receipt.get("locked_test_accessed") is not False
    ):
        raise RuntimeError("recovery boundary receipt provenance is invalid")
    _prevalidate_boundary_line(
        record["boundary_line"],
        recovery_intent["baseline_inventory"],
        context["ordered_ids"],
    )
    return receipt, path


def _observe_recovery_boundary(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    recovery_intent: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = args.execution_root.resolve() / "recovery-boundary-observed.json"
    if path.exists():
        return _validate_logged_recovery_boundary(args, context, recovery_intent)
    identities = recovery_intent["serial_identities"]
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial Python identity changed before recovery boundary")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("serial flock identity changed before recovery boundary")
    log_identity = recovery_intent["serial_log_identity"]
    log_path = Path(log_identity["path"])
    _assert_same_log_object(log_identity, log_path)
    record = _wait_for_score_completed_record(
        log_path,
        int(recovery_intent["serial_log_offset"]),
        identities["python"],
        timeout_seconds=args.boundary_timeout,
    )
    _prevalidate_boundary_line(
        record["boundary_line"],
        recovery_intent["baseline_inventory"],
        context["ordered_ids"],
    )
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial Python identity changed at recovery boundary")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("serial flock identity changed at recovery boundary")
    _capture_recovery_signal_status(identities["python"])
    _assert_same_log_object(log_identity, log_path)
    count = int(record["boundary_line"]["completed"])
    abandoned = (
        context["ordered_ids"][count] if count < len(context["ordered_ids"]) else None
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_ignored_sigint_recovery_boundary",
        "observed_unix_time_ns": time.time_ns(),
        "recovery_intent_sha256": _sha256_file(
            args.execution_root.resolve() / "recovery-cutover-intent.json"
        ),
        "serial_python_identity": identities["python"],
        "serial_flock_identity": identities["flock_wrapper"],
        "serial_log_identity": log_identity,
        "baseline_inventory": recovery_intent["baseline_inventory"],
        "log_record": record,
        "potentially_started_abandoned_episode_id": abandoned,
        "abandoned_computation_authoritative": False,
        "abandoned_computation_cost_status": "unavailable",
        "locked_test_accessed": False,
    }
    _write_json_exclusive(path, receipt)
    return _validate_logged_recovery_boundary(args, context, recovery_intent)


def _assert_no_cutover_residue(args: argparse.Namespace) -> None:
    feature_root = args.feature_root.resolve()
    if feature_root.is_symlink():
        raise RuntimeError("feature root may not be a symlink")
    if feature_root.exists() and (
        not feature_root.is_dir() or any(feature_root.iterdir())
    ):
        raise RuntimeError("feature root is not empty after serial termination")
    forbidden: list[str] = []
    for root in (args.score_root.resolve(), args.execution_root.resolve()):
        if not root.exists() or root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"cutover root is not a real directory: {root}")
        for entry in root.rglob("*"):
            name = entry.name
            if (
                name.startswith(".tmp-")
                or name == ".publish.lock"
                or name.endswith(".staging")
                or name.startswith("staging-")
                or name == "finalizer-staging"
            ):
                forbidden.append(str(entry))
    if forbidden:
        raise RuntimeError(f"cutover residue remains: {sorted(forbidden)}")


def _assert_execution_control_state(execution_root: Path, phase: str) -> None:
    base_files = {"cutover-intent.json", "boundary-observed.json"}
    base_directories = {"signal-attempts"}
    phase_files = {
        "original_dispatched": set(),
        "sigint_closed": set(),
        "recovery_intent": {"recovery-cutover-intent.json"},
        "recovery_boundary": {
            "recovery-cutover-intent.json",
            "recovery-boundary-observed.json",
        },
        "recovery_attempts_empty": {
            "recovery-cutover-intent.json",
            "recovery-boundary-observed.json",
        },
        "recovery_dispatched": {
            "recovery-cutover-intent.json",
            "recovery-boundary-observed.json",
        },
        "recovery_exit": {
            "recovery-cutover-intent.json",
            "recovery-boundary-observed.json",
        },
        "cutover": {
            "recovery-cutover-intent.json",
            "recovery-boundary-observed.json",
            "cutover-receipt.json",
        },
    }
    if phase not in phase_files:
        raise RuntimeError(f"unknown recovery control phase: {phase}")
    recovery_directory_phases = {
        "recovery_attempts_empty",
        "recovery_dispatched",
        "recovery_exit",
        "cutover",
    }
    expected_files = base_files | phase_files[phase]
    expected_directories = set(base_directories)
    if phase in recovery_directory_phases:
        expected_directories.add("recovery-signal-attempts")
    expected_names = expected_files | expected_directories
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise RuntimeError("execution control root is not a real directory")
    entries = {entry.name: entry for entry in execution_root.iterdir()}
    if set(entries) != expected_names:
        raise RuntimeError(
            f"unexpected execution control entries for {phase}: "
            f"{sorted(set(entries) ^ expected_names)}"
        )
    for name in expected_files:
        entry = entries[name]
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"execution control file is not regular: {entry}")
    for name in expected_directories:
        entry = entries[name]
        if entry.is_symlink() or not entry.is_dir():
            raise RuntimeError(f"execution control directory is not real: {entry}")
    original_names = {
        "attempt-0001-intent.json",
        "attempt-0001-dispatched.json",
    }
    if phase != "original_dispatched":
        original_names.add("attempt-0001-ineffective.json")
    original_entries = {
        entry.name: entry for entry in (execution_root / "signal-attempts").iterdir()
    }
    if set(original_entries) != original_names:
        raise RuntimeError(
            "unexpected original signal-attempt entries: "
            f"{sorted(set(original_entries) ^ original_names)}"
        )
    for entry in original_entries.values():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"original signal receipt is not regular: {entry}")
    if phase in recovery_directory_phases:
        recovery_names_by_phase = {
            "recovery_attempts_empty": set(),
            "recovery_dispatched": {
                "attempt-0001-intent.json",
                "attempt-0001-dispatched.json",
            },
            "recovery_exit": {
                "attempt-0001-intent.json",
                "attempt-0001-dispatched.json",
                "attempt-0001-exit-observed.json",
            },
            "cutover": {
                "attempt-0001-intent.json",
                "attempt-0001-dispatched.json",
                "attempt-0001-exit-observed.json",
            },
        }
        recovery_entries = {
            entry.name: entry
            for entry in (execution_root / "recovery-signal-attempts").iterdir()
        }
        expected_recovery_names = recovery_names_by_phase[phase]
        if set(recovery_entries) != expected_recovery_names:
            raise RuntimeError(
                "unexpected recovery signal-attempt entries: "
                f"{sorted(set(recovery_entries) ^ expected_recovery_names)}"
            )
        for entry in recovery_entries.values():
            if entry.is_symlink() or not entry.is_file():
                raise RuntimeError(f"recovery signal receipt is not regular: {entry}")


def _recover_ignored_sigint_main(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    control = _require_control_checkout(
        args.control_repo.resolve(), script, args.expected_implementation_commit
    )
    context = _locked_context(args, validate_all_raw=True)
    evidence = _validate_evidence(args)
    _validate_paths(args)
    execution_root = args.execution_root.resolve()
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise RuntimeError("existing execution root is not a real directory")
    recovery_intent_path = execution_root / "recovery-cutover-intent.json"
    plan_path = execution_root / "plan.json"
    if plan_path.exists():
        if _runner_processes(args.serial_runner):
            raise RuntimeError("serial runner exists after recovery plan creation")
        lock_descriptor = _acquire_lock(args.global_lock.resolve())
        try:
            plan, plan_sha = _load_plan(args)
            _assert_arguments_match_plan(args, plan, plan_sha)
            return _run_plan(args, context, plan, plan_sha, lock_descriptor)
        finally:
            os.close(lock_descriptor)
    if not recovery_intent_path.exists():
        initial_phase = (
            "sigint_closed"
            if (execution_root / "signal-attempts" / "attempt-0001-ineffective.json").exists()
            else "original_dispatched"
        )
        _assert_execution_control_state(execution_root, initial_phase)
        for relative in (
            "cutover-receipt.json",
            "runtime.json",
            "execution-receipt.json",
            "completion.json",
            "failure.json",
            "worker0",
            "worker1",
        ):
            candidate = execution_root / relative
            if candidate.exists() or candidate.is_symlink():
                raise RuntimeError(
                    f"unexpected pre-recovery control path exists: {candidate}"
                )
    original = _validate_original_sigint_attempt(args, context)
    ineffective_path = execution_root / "signal-attempts" / "attempt-0001-ineffective.json"
    _ensure_original_sigint_ineffective_receipt(args, original)
    if recovery_intent_path.exists():
        recovery_intent = _validate_recovery_intent(
            args, context, control, evidence, original, ineffective_path
        )
    else:
        _assert_execution_control_state(execution_root, "sigint_closed")
        recovery_intent = _create_recovery_intent(
            args, context, control, evidence, original, ineffective_path
        )
        _assert_execution_control_state(execution_root, "recovery_intent")
    boundary, boundary_path = _observe_recovery_boundary(
        args, context, recovery_intent
    )
    if not (execution_root / "recovery-signal-attempts").exists():
        _assert_execution_control_state(execution_root, "recovery_boundary")
    identities = recovery_intent["serial_identities"]
    _ensure_recovery_sigterm_dispatched(
        args,
        identities,
        boundary_path,
        ineffective_path,
        str(recovery_intent["serial_runner_sha256"]),
    )
    exit_receipt = _complete_recovery_sigterm(args, identities)
    _assert_execution_control_state(
        execution_root,
        "cutover" if (execution_root / "cutover-receipt.json").exists() else "recovery_exit",
    )
    lock_descriptor = _acquire_lock(args.global_lock.resolve())
    try:
        frozen = _inventory(context, args)
        baseline = recovery_intent["baseline_inventory"]
        _assert_frozen_unchanged(baseline, frozen)
        _validate_boundary_transition(
            boundary["log_record"]["boundary_line"],
            baseline,
            frozen,
            context["ordered_ids"],
        )
        _assert_original_boundary_retained(original, frozen)
        if _runner_processes(args.serial_runner):
            raise RuntimeError("serial runner reappeared after recovery")
        _assert_no_cutover_residue(args)
        cutover = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_cutover_receipt",
            "serial_identities": identities,
            "signal_sequence": [
                {
                    "signal": "SIGINT",
                    "dispatched": True,
                    "disposition": "inherited_ignored",
                    "exit_observed": False,
                    "receipt_sha256": _sha256_file(ineffective_path),
                },
                {
                    "signal": "SIGTERM",
                    "dispatched": True,
                    "process_exit_subsequently_observed": True,
                    "causal_exit_claim": False,
                    "dispatch_sha256": _sha256_file(
                        _recovery_signal_paths(execution_root)[1]
                    ),
                    "exit_observation_sha256": _sha256_file(
                        _recovery_signal_paths(execution_root)[2]
                    ),
                },
            ],
            "signal_count": 2,
            "original_boundary_observed_sha256": _sha256_file(
                Path(original["boundary_path"])
            ),
            "recovery_boundary_observed_sha256": _sha256_file(boundary_path),
            "recovery_intent_sha256": _sha256_file(recovery_intent_path),
            "recovery_exit_observation_sha256": _sha256_bytes(
                _canonical(exit_receipt)
            ),
            "boundary_line": boundary["log_record"]["boundary_line"],
            "boundary_log_offset": boundary["log_record"]["log_end_offset"],
            "original_intent_inventory_sha256": original["cutover_intent"][
                "initial_inventory"
            ]["inventory_sha256"],
            "original_intent_episode_count": original["cutover_intent"][
                "initial_inventory"
            ]["episode_count"],
            "recovery_baseline_inventory_sha256": baseline["inventory_sha256"],
            "recovery_baseline_episode_count": baseline["episode_count"],
            "frozen_inventory_sha256": frozen["inventory_sha256"],
            "frozen_episode_count": frozen["episode_count"],
            "global_lock_reacquired_exclusively": True,
            "serial_processes_remaining": 0,
            "potentially_started_abandoned_episode_id": boundary.get(
                "potentially_started_abandoned_episode_id"
            ),
            "abandoned_computation_authoritative": False,
            "abandoned_computation_cost_status": "unavailable",
            "locked_test_accessed": False,
        }
        cutover_path = execution_root / "cutover-receipt.json"
        if cutover_path.exists():
            if _strict_json(cutover_path) != cutover:
                raise RuntimeError("existing recovery cutover receipt differs")
        else:
            _write_json_exclusive(cutover_path, cutover)
        _assert_execution_control_state(execution_root, "cutover")
        plan, plan_sha = _create_plan(
            args, context, control, evidence, cutover, frozen
        )
        return _run_plan(args, context, plan, plan_sha, lock_descriptor)
    finally:
        os.close(lock_descriptor)


def _recovery_preflight_main(args: argparse.Namespace) -> int:
    """Run the ignored-SIGINT recovery guards without writing or signalling."""

    script = Path(__file__).resolve()
    control = _require_control_checkout(
        args.control_repo.resolve(), script, args.expected_implementation_commit
    )
    context = _locked_context(args, validate_all_raw=True)
    evidence = _validate_evidence(args)
    _validate_paths(args)
    execution_root = args.execution_root.resolve()
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise RuntimeError("existing execution root is not a real directory")
    _assert_execution_control_state(execution_root, "original_dispatched")
    original = _validate_original_sigint_attempt(args, context)
    identities = original["serial_identities"]
    if not _identity_matches(identities["python"]):
        raise RuntimeError("serial Python identity differs during recovery preflight")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("serial flock identity differs during recovery preflight")
    signal_status = _capture_recovery_signal_status(identities["python"])
    dispatched_ns = int(original["signal_dispatched"]["dispatched_unix_time_ns"])
    elapsed = (time.time_ns() - dispatched_ns) / 1e9
    if elapsed < ORIGINAL_EXIT_TIMEOUT_SECONDS:
        raise RuntimeError("original SIGINT timeout has not elapsed")
    inventory = _inventory(context, args)
    _assert_inventory_is_manifest_prefix(inventory, context["ordered_ids"])
    _assert_original_boundary_retained(original, inventory)
    _assert_two_worker_capacity_after_boundary(inventory, context["ordered_ids"])
    if args.serial_log.resolve() != Path(original["cutover_intent"]["serial_log"]).resolve():
        raise RuntimeError("recovery serial log differs from original intent")
    log_identity = _log_identity(args.serial_log.resolve())
    _assert_no_cutover_residue(args)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_ignored_sigint_recovery_read_only_preflight",
        "status": "pass",
        "control": control,
        "evidence": evidence,
        "serial_identities": identities,
        "signal_status": signal_status,
        "original_sigint_elapsed_seconds": elapsed,
        "current_inventory_sha256": inventory["inventory_sha256"],
        "current_episode_count": inventory["episode_count"],
        "expected_fresh_boundary_count": int(inventory["episode_count"]) + 1,
        "remaining_after_fresh_boundary": len(context["ordered_ids"])
        - (int(inventory["episode_count"]) + 1),
        "serial_log_identity": log_identity,
        "feature_root_empty": True,
        "recovery_paths_absent": True,
        "locked_test_accessed": False,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def _preflight_main(args: argparse.Namespace) -> int:
    """Run every read-only cutover guard without creating paths or signaling."""

    script = Path(__file__).resolve()
    control = _require_control_checkout(
        args.control_repo.resolve(), script, args.expected_implementation_commit
    )
    context = _locked_context(args, validate_all_raw=True)
    evidence = _validate_evidence(args)
    _validate_paths(args)
    if args.execution_root.exists() or args.execution_root.is_symlink():
        raise RuntimeError("execution root already exists before preflight")
    if args.feature_root.is_symlink():
        raise RuntimeError("feature root may not be a symlink")
    if args.feature_root.exists() and (
        not args.feature_root.is_dir() or any(args.feature_root.iterdir())
    ):
        raise RuntimeError("feature root is not an empty real directory")
    identities = _capture_serial_identities(args.serial_runner, args.global_lock.resolve())
    inventory = _inventory(context, args)
    log_path = args.serial_log.resolve()
    if log_path.is_symlink() or not log_path.is_file():
        raise RuntimeError("serial log is not a regular file")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND}_read_only_preflight",
        "status": "pass",
        "control": control,
        "evidence": evidence,
        "serial_identities": identities,
        "serial_log": str(log_path),
        "serial_log_size": log_path.stat().st_size,
        "current_inventory_sha256": inventory["inventory_sha256"],
        "current_episode_count": inventory["episode_count"],
        "manifest_episode_count": len(context["ordered_ids"]),
        "execution_root_absent": True,
        "feature_root_empty": True,
        "locked_test_accessed": False,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def _recover_plan_after_stopped_boundary(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    control: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Durably finish the cutover if the coordinator died after SIGINT."""

    execution_root = args.execution_root.resolve()
    intent = _strict_json(execution_root / "cutover-intent.json")
    if intent.get("control") != dict(control):
        raise RuntimeError("recovery control provenance differs from cutover intent")
    if _runner_processes(args.serial_runner):
        raise RuntimeError("cannot recover a stopped boundary while serial scorer exists")
    dispatched = _signal_attempt_paths(execution_root)
    if len(dispatched) != 1:
        raise RuntimeError(
            "stopped-boundary recovery requires exactly one durable signal dispatch"
        )
    dispatched_receipt = _strict_json(dispatched[0])
    prefix = dispatched[0].name.removesuffix("-dispatched.json")
    signal_intent_path = dispatched[0].parent / f"{prefix}-intent.json"
    signal_intent = _strict_json(signal_intent_path)
    if _sha256_file(signal_intent_path) != dispatched_receipt.get("intent_sha256"):
        raise RuntimeError("signal dispatch does not bind its durable intent")
    boundary_path = Path(signal_intent["boundary_receipt_path"])
    if _sha256_file(boundary_path) != signal_intent.get("boundary_receipt_sha256"):
        raise RuntimeError("signal intent does not bind its boundary receipt")
    boundary = _strict_json(boundary_path)
    if boundary.get("serial_python_identity") != intent["serial_identities"]["python"]:
        raise RuntimeError("recovery boundary identity differs from cutover intent")
    interrupt_receipt = _complete_dispatched_interrupt(
        args, intent["serial_identities"], dispatched[0]
    )
    interrupt_path = execution_root / "serial-interrupt.json"
    if (
        interrupt_receipt.get("signal") != "SIGINT"
        or interrupt_receipt.get("signal_count") != 1
        or interrupt_receipt.get("serial_identities_exited") is not True
    ):
        raise RuntimeError("serial interrupt recovery receipt is invalid")
    initial = boundary.get("baseline_inventory")
    if not isinstance(initial, dict):
        raise RuntimeError("boundary receipt lacks its exact baseline inventory")
    intent_initial = intent["initial_inventory"]
    _assert_frozen_unchanged(intent_initial, initial)
    frozen = _inventory(context, args)
    _assert_frozen_unchanged(initial, frozen)
    boundary_line = boundary["boundary_line"]
    _validate_boundary_transition(
        boundary_line, initial, frozen, context["ordered_ids"]
    )
    cutover_path = execution_root / "cutover-receipt.json"
    if cutover_path.exists():
        cutover = _strict_json(cutover_path)
    else:
        cutover = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_cutover_receipt",
            "serial_identities": intent["serial_identities"],
            "signal": "SIGINT",
            "signal_count": 1,
            "boundary_observed_sha256": _sha256_file(boundary_path),
            "serial_interrupt_sha256": _sha256_file(interrupt_path),
            "boundary_line": boundary_line,
            "boundary_log_offset": boundary["boundary_log_offset"],
            "original_intent_inventory_sha256": intent_initial["inventory_sha256"],
            "original_intent_episode_count": intent_initial["episode_count"],
            "boundary_baseline_inventory_sha256": initial["inventory_sha256"],
            "boundary_baseline_episode_count": initial["episode_count"],
            "initial_inventory_sha256": initial["inventory_sha256"],
            "frozen_inventory_sha256": frozen["inventory_sha256"],
            "initial_episode_count": initial["episode_count"],
            "frozen_episode_count": frozen["episode_count"],
            "global_lock_reacquired_exclusively": True,
            "serial_processes_remaining": 0,
            "recovered_after_coordinator_exit": True,
            "locked_test_accessed": False,
        }
        _write_json_exclusive(cutover_path, cutover)
    plan_path = execution_root / "plan.json"
    if plan_path.exists():
        return _load_plan(args)
    return _create_plan(
        args,
        context,
        control,
        intent["evidence"],
        cutover,
        frozen,
    )


def _resume_live_cutover_boundary(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    control: Mapping[str, Any],
) -> None:
    """Resume a pre-signal cutover by waiting for a fresh clean boundary."""

    execution_root = args.execution_root.resolve()
    intent = _strict_json(execution_root / "cutover-intent.json")
    if intent.get("control") != dict(control):
        raise RuntimeError("live-resume control provenance differs from cutover intent")
    identities = intent["serial_identities"]
    if not _identity_matches(identities["python"]):
        raise RuntimeError("live serial identity differs from cutover intent")
    if not _identity_matches(identities["flock_wrapper"]):
        raise RuntimeError("live flock identity differs from cutover intent")
    dispatched = _signal_attempt_paths(execution_root)
    if dispatched:
        if len(dispatched) != 1:
            raise RuntimeError("more than one durable signal dispatch exists")
        _complete_dispatched_interrupt(args, identities, dispatched[0])
        return
    signal_intents = sorted(
        (execution_root / "signal-attempts").glob("attempt-*-intent.json")
    ) if (execution_root / "signal-attempts").exists() else []
    if signal_intents:
        raise RuntimeError(
            "ambiguous signal state: intent exists without durable dispatch"
        )
    log_path = Path(intent["serial_log"])
    if log_path.is_symlink() or not log_path.is_file():
        raise RuntimeError("serial log changed before live cutover resume")
    base = execution_root / "boundary-observations"
    base.mkdir(parents=True, exist_ok=True)
    observed_paths = []
    initial_boundary = execution_root / "boundary-observed.json"
    if initial_boundary.exists():
        observed_paths.append(initial_boundary)
    observed_paths.extend(sorted(base.glob("resume-*.json")))
    if observed_paths:
        boundary_path = observed_paths[-1]
        boundary_receipt = _strict_json(boundary_path)
        if boundary_receipt.get("serial_python_identity") != identities["python"]:
            raise RuntimeError("durable boundary identity differs on live resume")
        _prevalidate_boundary_line(
            boundary_receipt["boundary_line"],
            boundary_receipt["baseline_inventory"],
            context["ordered_ids"],
        )
    else:
        baseline, log_offset = _stable_inventory_log_baseline(
            context, args, log_path, identities["python"]
        )
        boundary, boundary_offset = _wait_for_score_completed(
            log_path,
            log_offset,
            identities["python"],
            timeout_seconds=args.boundary_timeout,
        )
        _prevalidate_boundary_line(boundary, baseline, context["ordered_ids"])
        boundary_path = base / "resume-0001.json"
        boundary_receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND}_boundary_observed",
            "boundary_line": boundary,
            "boundary_log_offset": boundary_offset,
            "baseline_inventory": baseline,
            "serial_python_identity": identities["python"],
            "observed_unix_time_ns": time.time_ns(),
            "resumed_cutover": True,
            "locked_test_accessed": False,
        }
        _write_json_exclusive(boundary_path, boundary_receipt)
    _dispatch_serial_interrupt(args, identities, boundary_path)


def _resume_main(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    control = _require_control_checkout(
        args.control_repo.resolve(), script, args.expected_implementation_commit
    )
    context = _locked_context(args, validate_all_raw=True)
    _validate_paths(args)
    plan_path = args.execution_root.resolve() / "plan.json"
    live_serial = _runner_processes(args.serial_runner)
    if live_serial:
        if plan_path.exists():
            raise RuntimeError("a plan exists while a serial/finalizer runner is active")
        _resume_live_cutover_boundary(args, context, control)
    if _runner_processes(args.serial_runner):
        raise RuntimeError("serial scorer remains after live cutover resume")
    lock_descriptor = _acquire_lock(args.global_lock.resolve())
    try:
        if plan_path.exists():
            plan, plan_sha = _load_plan(args)
        else:
            plan, plan_sha = _recover_plan_after_stopped_boundary(
                args, context, control
            )
        _assert_arguments_match_plan(args, plan, plan_sha)
        return _run_plan(args, context, plan, plan_sha, lock_descriptor)
    finally:
        # Do not call LOCK_UN: a surviving inherited writer descriptor must
        # keep the global flock held after coordinator exit.
        os.close(lock_descriptor)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control-repo", type=Path, required=True)
    parser.add_argument("--expected-implementation-commit", required=True)
    parser.add_argument("--serial-runner", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bound-probe", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--global-lock", type=Path, required=True)
    parser.add_argument("--sample-interval", type=float, default=5.0)


def _common_worker_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--control-repo", str(args.control_repo.resolve()),
        "--expected-implementation-commit", args.expected_implementation_commit,
        "--serial-runner", str(args.serial_runner.resolve()),
        "--repo-root", str(args.repo_root.resolve()),
        "--python-executable", str(args.python_executable.resolve()),
        "--environment-lock", str(args.environment_lock.resolve()),
        "--cache-dir", str(args.cache_dir.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--authority", str(args.authority.resolve()),
        "--raw-root", str(args.raw_root.resolve()),
        "--bound-probe", str(args.bound_probe.resolve()),
        "--score-root", str(args.score_root.resolve()),
        "--feature-root", str(args.feature_root.resolve()),
        "--execution-root", str(args.execution_root.resolve()),
        "--global-lock", str(args.global_lock.resolve()),
        "--sample-interval", str(args.sample_interval),
    ]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    cutover = subparsers.add_parser("cutover")
    _common_arguments(cutover)
    cutover.add_argument("--equivalence-audit", type=Path, required=True)
    cutover.add_argument("--benchmark-summary", type=Path, required=True)
    cutover.add_argument("--benchmark-resource-samples", type=Path, required=True)
    cutover.add_argument("--serial-log", type=Path, required=True)
    cutover.add_argument("--boundary-timeout", type=float, default=1800.0)
    cutover.add_argument("--serial-exit-timeout", type=float, default=120.0)

    preflight = subparsers.add_parser("preflight")
    _common_arguments(preflight)
    preflight.add_argument("--equivalence-audit", type=Path, required=True)
    preflight.add_argument("--benchmark-summary", type=Path, required=True)
    preflight.add_argument("--benchmark-resource-samples", type=Path, required=True)
    preflight.add_argument("--serial-log", type=Path, required=True)

    recovery = subparsers.add_parser("recover-ignored-sigint")
    _common_arguments(recovery)
    recovery.add_argument("--equivalence-audit", type=Path, required=True)
    recovery.add_argument("--benchmark-summary", type=Path, required=True)
    recovery.add_argument("--benchmark-resource-samples", type=Path, required=True)
    recovery.add_argument("--serial-log", type=Path, required=True)
    recovery.add_argument("--boundary-timeout", type=float, default=1800.0)
    recovery.add_argument("--serial-exit-timeout", type=float, default=120.0)

    recovery_preflight = subparsers.add_parser("recover-ignored-sigint-preflight")
    _common_arguments(recovery_preflight)
    recovery_preflight.add_argument("--equivalence-audit", type=Path, required=True)
    recovery_preflight.add_argument("--benchmark-summary", type=Path, required=True)
    recovery_preflight.add_argument(
        "--benchmark-resource-samples", type=Path, required=True
    )
    recovery_preflight.add_argument("--serial-log", type=Path, required=True)

    resume = subparsers.add_parser("resume")
    _common_arguments(resume)
    resume.add_argument("--boundary-timeout", type=float, default=1800.0)
    resume.add_argument("--serial-exit-timeout", type=float, default=120.0)

    worker = subparsers.add_parser("worker")
    _common_arguments(worker)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--coordinator-pid", type=int, required=True)
    worker.add_argument("--expected-script-sha256", required=True)
    worker.add_argument("--expected-plan-sha256", required=True)

    finalizer = subparsers.add_parser("finalizer")
    _common_arguments(finalizer)
    finalizer.add_argument("--coordinator-pid", type=int, required=True)
    finalizer.add_argument("--expected-script-sha256", required=True)
    finalizer.add_argument("--expected-plan-sha256", required=True)
    finalizer.add_argument("--finalizer-feature-root", type=Path, required=True)

    args = parser.parse_args()
    if args.sample_interval <= 0.0 or not math.isfinite(args.sample_interval):
        raise RuntimeError("sample interval must be finite and positive")
    if args.mode == "worker":
        return _worker_main(args)
    if args.mode == "finalizer":
        return _finalizer_main(args)
    if args.mode == "preflight":
        return _preflight_main(args)
    if args.mode == "cutover":
        if args.boundary_timeout <= 0.0 or not math.isfinite(args.boundary_timeout):
            raise RuntimeError("boundary timeout must be finite and positive")
        if args.serial_exit_timeout <= 0.0 or not math.isfinite(args.serial_exit_timeout):
            raise RuntimeError("serial exit timeout must be finite and positive")
        return _cutover_main(args)
    if args.mode == "recover-ignored-sigint":
        for name in (
            "boundary_timeout",
            "serial_exit_timeout",
        ):
            value = float(getattr(args, name))
            if value <= 0.0 or not math.isfinite(value):
                raise RuntimeError(f"{name.replace('_', ' ')} must be finite and positive")
        return _recover_ignored_sigint_main(args)
    if args.mode == "recover-ignored-sigint-preflight":
        return _recovery_preflight_main(args)
    if args.boundary_timeout <= 0.0 or not math.isfinite(args.boundary_timeout):
        raise RuntimeError("boundary timeout must be finite and positive")
    if args.serial_exit_timeout <= 0.0 or not math.isfinite(args.serial_exit_timeout):
        raise RuntimeError("serial exit timeout must be finite and positive")
    return _resume_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
