#!/usr/bin/env python3
"""Isolated two-worker benchmark for immutable Calibration score replays.

The authoritative serial scorer is never controlled by this program.  Each
benchmark worker consumes only already-scored Calibration episodes, acquires a
worker-specific lock, and writes to a fresh worker-specific output root.  The
orchestrator compares every published byte with the authoritative reference
sidecars and records wall-clock throughput plus GPU/CPU/RAM telemetry.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import ConditionSpec
from mech_int_vla.failure_events import artifact_identity_from_rollout
from mech_int_vla.instrumentation import SmolVLAInstrumentation
from mech_int_vla.libero_runtime import RawLiberoEpisode
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


SCHEMA_VERSION = 1
BENCHMARK_KIND = "calibration_two_worker_scoring_benchmark"
COST_ARRAY_NAMES = (
    "original_cost",
    "transformed_cost",
    "intervention_minus_cost",
    "intervention_plus_cost",
)


class BenchmarkInterrupted(BaseException):
    """Raised by signal handlers so benchmark children are always cleaned up."""


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite benchmark artifact {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(payload)


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
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


def _terminate_benchmark_processes(
    processes: Sequence[subprocess.Popen[Any]], *, timeout_seconds: float = 30.0
) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while live and time.monotonic() < deadline:
        live = [process for process in live if process.poll() is None]
        if live:
            time.sleep(0.1)
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue


def _arm_parent_death_signal(expected_parent_pid: int) -> None:
    """Make a worker receive SIGTERM if its orchestrator disappears."""

    if os.getppid() != expected_parent_pid:
        raise RuntimeError("benchmark worker was not started by its orchestrator")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("benchmark orchestrator exited while worker armed cleanup")


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
        if closing < 0:
            return None
        fields_after_command = stat_payload[closing + 2 :].split()
        start_ticks = int(fields_after_command[19])
        executable = os.readlink(proc / "exe")
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return None
    return {
        "pid": pid,
        "parent_pid": int(fields_after_command[1]),
        "start_ticks": start_ticks,
        "executable": executable,
        "arguments": arguments,
    }


def _capture_serial_identity(serial_runner: Path) -> dict[str, Any]:
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
        raise RuntimeError(
            f"expected exactly one authoritative serial scorer, found {len(matches)}"
        )
    return matches[0]


def _serial_identity_matches(expected: Mapping[str, Any]) -> bool:
    current = _process_identity(int(expected["pid"]))
    return current == dict(expected)


def _load_serial_runner(path: Path) -> ModuleType:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        "calibration_serial_runner_18d6494", resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import serial runner {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_require_authority", "BOUND_PROBE_SHA256", "MANIFEST_SHA256"):
        if not hasattr(module, name):
            raise RuntimeError(f"serial runner lacks required symbol {name}")
    return module


def _acquire_worker_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        raise RuntimeError(f"benchmark worker lock is already held: {path}")
    return descriptor


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--serial-runner", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bound-probe", type=Path, required=True)


def _worker_main(args: argparse.Namespace) -> int:
    _arm_parent_death_signal(args.orchestrator_pid)
    benchmark_script = Path(__file__).resolve()
    if _sha256_file(benchmark_script) != args.expected_benchmark_script_sha256:
        raise RuntimeError("benchmark script differs from orchestrator-bound digest")
    lock_descriptor = _acquire_worker_lock(args.worker_lock.resolve())
    try:
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
        expected = {episode.episode_id: episode for episode in manifest.episodes}
        episode_ids = tuple(args.episode_id)
        if not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise RuntimeError("worker episode IDs must be a nonempty unique sequence")
        if any(episode_id not in expected for episode_id in episode_ids):
            raise RuntimeError("worker received an episode outside Calibration authority")

        worker_root = args.worker_root.resolve()
        score_root = worker_root / "scores"
        _require_disjoint_path(
            worker_root,
            (
                ("repository", root),
                ("raw root", args.raw_root.resolve()),
                ("reference score root", args.reference_score_root.resolve()),
                ("cache root", args.cache_dir.resolve()),
                ("manifest", args.manifest.resolve()),
                ("authority", args.authority.resolve()),
                ("bound probe", args.bound_probe.resolve()),
                ("environment lock", args.environment_lock.resolve()),
                ("serial runner", args.serial_runner.resolve()),
            ),
            name="worker root",
        )
        if score_root.exists() or (worker_root / "summary.json").exists():
            raise RuntimeError("benchmark worker output already exists")
        for episode_id in episode_ids:
            reference = (
                args.reference_score_root.resolve() / "calibration" / episode_id
            )
            load_scoring_sidecar(reference, expected_episode_id=episode_id)

        model_started = time.perf_counter()
        snapshots = resolve_snapshot_paths(
            args.environment_lock.resolve(),
            cache_dir=args.cache_dir.resolve(),
            local_files_only=True,
        )
        policy_runtime = load_locked_smolvla(snapshots, device="cuda")
        model_load_seconds = time.perf_counter() - model_started

        results: list[dict[str, Any]] = []
        worker_started = time.perf_counter()
        for episode_id in episode_ids:
            episode_started = time.perf_counter()
            spec = expected[episode_id]
            condition = ConditionSpec(
                spec.condition_name,
                spec.condition_family,
                spec.condition_index,
                spec.condition_parameters,
            )
            raw_path = args.raw_root.resolve() / "calibration" / episode_id
            artifact = load_rollout_artifact(raw_path, expected_task=task)
            if not artifact.valid_reset:
                raise RuntimeError(f"benchmark episode has invalid reset: {episode_id}")
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
            finally:
                instrumentation.remove()
                episode.close()
            results.append(
                {
                    "episode_id": episode_id,
                    "elapsed_seconds": time.perf_counter() - episode_started,
                    "result_sha256": result.sha256,
                    "scored_state_count": len(result.scored_control_steps),
                    "sidecar_path": str(result.path),
                }
            )
            print(json.dumps(results[-1], sort_keys=True), flush=True)

        if _sha256_file(benchmark_script) != args.expected_benchmark_script_sha256:
            raise RuntimeError("benchmark script changed while worker was running")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": "calibration_scoring_benchmark_worker",
            "worker_index": args.worker_index,
            "worker_pid": os.getpid(),
            "worker_lock": str(args.worker_lock.resolve()),
            "benchmark_script_sha256": args.expected_benchmark_script_sha256,
            "serial_runner_sha256": _sha256_file(args.serial_runner.resolve()),
            "model_load_seconds": model_load_seconds,
            "scoring_elapsed_seconds": time.perf_counter() - worker_started,
            "episodes": results,
            "locked_test_accessed": False,
        }
        _write_exclusive(worker_root / "summary.json", _canonical(summary))
        return 0
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts:
            values[key] = int(parts[0]) * 1024
    return values


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _resource_sample(worker_pids: Sequence[int], serial_pid: int | None) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "monotonic_seconds": time.monotonic(),
        "unix_time_ns": time.time_ns(),
        "worker_pids": list(worker_pids),
        "serial_pid": serial_pid,
    }
    meminfo = _read_meminfo()
    sample["host_memory"] = {
        "total_bytes": meminfo.get("MemTotal"),
        "available_bytes": meminfo.get("MemAvailable"),
        "used_bytes": (
            meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0)
        ),
    }
    sample["load_average"] = [
        float(value) for value in Path("/proc/loadavg").read_text().split()[:3]
    ]

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
        fields = [field.strip() for field in gpu.stdout.strip().splitlines()[0].split(",")]
        if len(fields) == 3:
            sample["gpu"] = {
                "utilization_percent": _parse_float(fields[0]),
                "memory_used_mib": _parse_float(fields[1]),
                "memory_total_mib": _parse_float(fields[2]),
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
    compute_rows: list[dict[str, Any]] = []
    if compute is not None and compute.returncode == 0:
        for line in compute.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0].isdigit():
                compute_rows.append(
                    {"pid": int(fields[0]), "used_memory_mib": _parse_float(fields[1])}
                )
    sample["gpu_processes"] = compute_rows

    requested_pids = list(worker_pids)
    if serial_pid is not None:
        requested_pids.append(serial_pid)
    process_rows: list[dict[str, Any]] = []
    if requested_pids:
        try:
            result = subprocess.run(
                [
                    "ps",
                    "-o",
                    "pid=,ppid=,%cpu=,rss=,vsz=,stat=,comm=",
                    "-p",
                    ",".join(str(pid) for pid in requested_pids),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        for line in result.stdout.splitlines() if result is not None else ():
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


def _cost_summary(sidecar: Any) -> dict[str, float | int]:
    cuda_ms = 0.0
    wall_ns = 0.0
    forwards = 0.0
    interventions = 0.0
    for name in COST_ARRAY_NAMES:
        array = np.asarray(sidecar.arrays[name], dtype=np.float64)
        cuda_ms += float(np.nansum(array[..., 0]))
        wall_ns += float(np.nansum(array[..., 1]))
        forwards += float(np.nansum(array[..., 2]))
        interventions += float(np.nansum(array[..., 3]))
    return {
        "model_cuda_seconds": cuda_ms / 1000.0,
        "model_wall_seconds": wall_ns / 1e9,
        "forward_count": int(round(forwards)),
        "intervention_count": int(round(interventions)),
    }


def _array_hashes(sidecar: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in sidecar.arrays.items():
        array = np.ascontiguousarray(value)
        descriptor = f"{array.dtype.str}|{array.shape}|".encode("ascii")
        result[name] = _sha256_bytes(descriptor + array.tobytes(order="C"))
    return result


def _finite_values(samples: Sequence[Mapping[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for sample in samples:
        current: Any = sample
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)) and math.isfinite(float(current)):
            values.append(float(current))
    return values


def _resource_summary(
    samples: Sequence[Mapping[str, Any]], worker_pids: Sequence[int], serial_pid: int | None
) -> dict[str, Any]:
    gpu_util = _finite_values(samples, ("gpu", "utilization_percent"))
    gpu_mem = _finite_values(samples, ("gpu", "memory_used_mib"))
    host_used = _finite_values(samples, ("host_memory", "used_bytes"))
    combined_cpu: list[float] = []
    combined_rss: list[float] = []
    per_worker_rss: dict[int, list[float]] = {pid: [] for pid in worker_pids}
    per_worker_gpu: dict[int, list[float]] = {pid: [] for pid in worker_pids}
    serial_cpu: list[float] = []
    serial_rss: list[float] = []
    serial_gpu: list[float] = []
    for sample in samples:
        rows = [
            row
            for row in sample.get("processes", [])
            if row.get("pid") in worker_pids
        ]
        combined_cpu.append(sum(float(row["cpu_percent"]) for row in rows))
        combined_rss.append(sum(float(row["rss_bytes"]) for row in rows))
        for row in rows:
            per_worker_rss[int(row["pid"])].append(float(row["rss_bytes"]))
        if serial_pid is not None:
            matching_serial = [
                row
                for row in sample.get("processes", [])
                if row.get("pid") == serial_pid
            ]
            if matching_serial:
                serial_cpu.append(float(matching_serial[0]["cpu_percent"]))
                serial_rss.append(float(matching_serial[0]["rss_bytes"]))
        for row in sample.get("gpu_processes", []):
            pid = row.get("pid")
            value = row.get("used_memory_mib")
            if pid in per_worker_gpu and isinstance(value, (int, float)):
                per_worker_gpu[int(pid)].append(float(value))
            if pid == serial_pid and isinstance(value, (int, float)):
                serial_gpu.append(float(value))

    def stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "max": None}
        return {"mean": statistics.fmean(values), "max": max(values)}

    return {
        "sample_count": len(samples),
        "gpu_utilization_scope": "device_total_including_authoritative_serial_scorer",
        "gpu_utilization_percent": stats(gpu_util),
        "total_gpu_memory_used_mib": stats(gpu_mem),
        "benchmark_worker_cpu_percent_sum": stats(combined_cpu),
        "benchmark_worker_rss_bytes_sum": stats(combined_rss),
        "host_used_ram_bytes": stats(host_used),
        "per_worker_peak_rss_bytes": {
            str(pid): max(values) if values else None
            for pid, values in per_worker_rss.items()
        },
        "per_worker_peak_gpu_memory_mib": {
            str(pid): max(values) if values else None
            for pid, values in per_worker_gpu.items()
        },
        "authoritative_serial_pid": serial_pid,
        "authoritative_serial_cpu_percent": stats(serial_cpu),
        "authoritative_serial_rss_bytes": stats(serial_rss),
        "authoritative_serial_gpu_memory_mib": stats(serial_gpu),
    }


def _orchestrator_main(args: argparse.Namespace) -> int:
    benchmark_script = Path(__file__).resolve()
    benchmark_script_sha256 = _sha256_file(benchmark_script)
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
        raise RuntimeError("bound probe differs from Calibration authority")

    assignments = (tuple(args.worker0_episode), tuple(args.worker1_episode))
    selected = tuple(episode for assignment in assignments for episode in assignment)
    if not all(assignments) or len(selected) != len(set(selected)):
        raise RuntimeError("two nonempty disjoint worker assignments are required")
    ordered_ids = [episode.episode_id for episode in manifest.episodes]
    if any(episode_id not in ordered_ids for episode_id in selected):
        raise RuntimeError("benchmark selection is outside Calibration manifest")

    reference_root = args.reference_score_root.resolve()
    reference: dict[str, dict[str, Any]] = {}
    for episode_id in selected:
        index = ordered_ids.index(episode_id)
        if index == 0:
            raise RuntimeError("cannot derive a publication interval for first episode")
        previous_id = ordered_ids[index - 1]
        destination = reference_root / "calibration" / episode_id
        previous = reference_root / "calibration" / previous_id
        sidecar = load_scoring_sidecar(destination, expected_episode_id=episode_id)
        load_scoring_sidecar(previous, expected_episode_id=previous_id)
        current_mtime = (destination / "metadata.json").stat().st_mtime_ns
        previous_mtime = (previous / "metadata.json").stat().st_mtime_ns
        interval = (current_mtime - previous_mtime) / 1e9
        if interval <= 0.0:
            raise RuntimeError("reference publication intervals are not increasing")
        costs = _cost_summary(sidecar)
        interval_overhead = interval - float(costs["model_wall_seconds"])
        if not 0.0 <= interval_overhead <= 120.0:
            raise RuntimeError(
                "reference publication interval contains an implausible pause/restart gap"
            )
        reference[episode_id] = {
            "sidecar_path": str(destination),
            "previous_episode_id": previous_id,
            "serial_publication_interval_seconds": interval,
            "serial_interval_overhead_seconds": interval_overhead,
            "metadata_mtime_ns": current_mtime,
            "previous_metadata_path": str(previous / "metadata.json"),
            "previous_metadata_mtime_ns": previous_mtime,
            "metadata_sha256": sidecar.metadata_sha256,
            "primitives_sha256": sidecar.primitives_sha256,
            "array_sha256": _array_hashes(sidecar),
            "cost": costs,
            "state_count": int(sidecar.metadata["capture"]["state_count"]),
        }

    benchmark_root = args.benchmark_root.resolve()
    worker_lock_root = args.worker_lock_root.resolve()
    protected_paths = (
        ("repository", root),
        ("raw root", args.raw_root.resolve()),
        ("reference score root", reference_root),
        ("cache root", args.cache_dir.resolve()),
        ("manifest", args.manifest.resolve()),
        ("authority", args.authority.resolve()),
        ("bound probe", args.bound_probe.resolve()),
        ("environment lock", args.environment_lock.resolve()),
        ("serial runner", args.serial_runner.resolve()),
        ("benchmark script", benchmark_script),
    )
    _require_disjoint_path(benchmark_root, protected_paths, name="benchmark root")
    _require_disjoint_path(worker_lock_root, protected_paths, name="worker lock root")
    _require_disjoint_path(
        worker_lock_root, (("benchmark root", benchmark_root),), name="worker lock root"
    )
    serial_identity = _capture_serial_identity(args.serial_runner)
    if benchmark_root.exists() or benchmark_root.is_symlink():
        raise RuntimeError(f"benchmark root already exists: {benchmark_root}")
    if worker_lock_root.exists() or worker_lock_root.is_symlink():
        raise RuntimeError(f"worker lock root already exists: {worker_lock_root}")
    benchmark_root.mkdir(parents=True, exist_ok=False)
    worker_lock_root.mkdir(parents=True, exist_ok=False)
    worker_roots = (benchmark_root / "worker0", benchmark_root / "worker1")
    for worker_root in worker_roots:
        worker_root.mkdir(exist_ok=False)

    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{BENCHMARK_KIND}_plan",
        "benchmark_script": str(benchmark_script),
        "benchmark_script_sha256": benchmark_script_sha256,
        "serial_runner": str(args.serial_runner.resolve()),
        "serial_runner_sha256": _sha256_file(args.serial_runner.resolve()),
        "manifest_sha256": serial.MANIFEST_SHA256,
        "bound_probe_sha256": serial.BOUND_PROBE_SHA256,
        "reference_score_root": str(reference_root),
        "assignments": [list(value) for value in assignments],
        "reference": reference,
        "authoritative_serial_identity": serial_identity,
        "watchdog_seconds": args.max_wall_seconds,
        "output_isolation": {
            "benchmark_root": str(benchmark_root),
            "worker_roots": [str(value) for value in worker_roots],
            "authoritative_score_root_written": False,
        },
        "locked_test_accessed": False,
    }
    _write_exclusive(benchmark_root / "plan.json", _canonical(plan))

    common = [
        "--expected-benchmark-script-sha256", benchmark_script_sha256,
        "--orchestrator-pid", str(os.getpid()),
        "--serial-runner", str(args.serial_runner.resolve()),
        "--repo-root", str(root),
        "--environment-lock", str(args.environment_lock.resolve()),
        "--cache-dir", str(args.cache_dir.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--authority", str(args.authority.resolve()),
        "--raw-root", str(args.raw_root.resolve()),
        "--bound-probe", str(args.bound_probe.resolve()),
        "--reference-score-root", str(reference_root),
    ]
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    benchmark_started = time.perf_counter()
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    }

    def interrupt(signum: int, _frame: Any) -> None:
        raise BenchmarkInterrupted(f"benchmark orchestrator received signal {signum}")

    for signum in previous_handlers:
        signal.signal(signum, interrupt)
    try:
        for index, (worker_root, episode_ids) in enumerate(zip(worker_roots, assignments, strict=True)):
            log = (worker_root / "worker.log").open("xb")
            logs.append(log)
            command = [
                sys.executable,
                str(benchmark_script),
                "worker",
                *common,
                "--worker-index", str(index),
                "--worker-root", str(worker_root),
                "--worker-lock", str(worker_lock_root / f"worker{index}.lock"),
            ]
            for episode_id in episode_ids:
                command.extend(["--episode-id", episode_id])
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )

        worker_pids = [process.pid for process in processes]
        serial_pid = int(serial_identity["pid"])
        samples: list[dict[str, Any]] = []
        while any(process.poll() is None for process in processes):
            if not _serial_identity_matches(serial_identity):
                raise RuntimeError("authoritative serial scorer identity changed")
            if time.perf_counter() - benchmark_started > args.max_wall_seconds:
                raise RuntimeError("two-worker benchmark exceeded wall-clock watchdog")
            failed = [
                process.returncode
                for process in processes
                if process.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"benchmark worker failed early: {failed}")
            sample_started = time.monotonic()
            samples.append(_resource_sample(worker_pids, serial_pid))
            remaining = args.sample_interval - (time.monotonic() - sample_started)
            if remaining > 0.0:
                time.sleep(remaining)
        samples.append(_resource_sample(worker_pids, serial_pid))
        return_codes = [process.wait() for process in processes]
        if not _serial_identity_matches(serial_identity):
            raise RuntimeError("authoritative serial scorer changed before benchmark receipt")
    except BaseException:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        _terminate_benchmark_processes(processes)
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        for log in logs:
            log.flush()
            os.fsync(log.fileno())
            log.close()
    benchmark_elapsed = time.perf_counter() - benchmark_started
    _write_exclusive(benchmark_root / "resource-samples.json", _canonical(samples))

    if any(code != 0 for code in return_codes):
        failure = {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{BENCHMARK_KIND}_failure",
            "return_codes": return_codes,
            "worker_pids": worker_pids,
            "elapsed_seconds": benchmark_elapsed,
            "locked_test_accessed": False,
        }
        _write_exclusive(benchmark_root / "benchmark-failure.json", _canonical(failure))
        raise RuntimeError(f"benchmark worker failed: {return_codes}")

    comparisons: dict[str, dict[str, Any]] = {}
    worker_summaries: list[dict[str, Any]] = []
    for worker_root in worker_roots:
        worker_summaries.append(json.loads((worker_root / "summary.json").read_text()))
    worker_for_episode = {
        episode_id: index
        for index, assignment in enumerate(assignments)
        for episode_id in assignment
    }
    for episode_id in selected:
        benchmark_path = (
            worker_roots[worker_for_episode[episode_id]]
            / "scores"
            / "calibration"
            / episode_id
        )
        benchmark_sidecar = load_scoring_sidecar(
            benchmark_path, expected_episode_id=episode_id
        )
        reference_sidecar = load_scoring_sidecar(
            Path(reference[episode_id]["sidecar_path"]),
            expected_episode_id=episode_id,
        )
        reference_metadata_path = Path(reference[episode_id]["sidecar_path"]) / "metadata.json"
        if (
            reference_sidecar.metadata_sha256 != reference[episode_id]["metadata_sha256"]
            or reference_sidecar.primitives_sha256
            != reference[episode_id]["primitives_sha256"]
            or reference_metadata_path.stat().st_mtime_ns
            != reference[episode_id]["metadata_mtime_ns"]
            or Path(reference[episode_id]["previous_metadata_path"]).stat().st_mtime_ns
            != reference[episode_id]["previous_metadata_mtime_ns"]
        ):
            raise RuntimeError("authoritative reference sidecar changed during benchmark")
        benchmark_arrays = _array_hashes(benchmark_sidecar)
        array_equal = {
            name: benchmark_arrays[name] == reference[episode_id]["array_sha256"][name]
            for name in benchmark_arrays
        }
        files = {}
        for filename in ("metadata.json", "primitives.npz"):
            reference_file = Path(reference[episode_id]["sidecar_path"]) / filename
            benchmark_file = benchmark_path / filename
            reference_digest = _sha256_file(reference_file)
            benchmark_digest = _sha256_file(benchmark_file)
            files[filename] = {
                "reference_sha256": reference_digest,
                "benchmark_sha256": benchmark_digest,
                "byte_identical": (
                    reference_digest == benchmark_digest
                    and reference_file.read_bytes() == benchmark_file.read_bytes()
                ),
            }
        comparisons[episode_id] = {
            "files": files,
            "all_files_byte_identical": all(
                value["byte_identical"] for value in files.values()
            ),
            "array_byte_identity": array_equal,
            "all_arrays_byte_identical": all(array_equal.values()),
            "all_non_cost_arrays_byte_identical": all(
                equal for name, equal in array_equal.items() if name not in COST_ARRAY_NAMES
            ),
            "differing_arrays": sorted(name for name, equal in array_equal.items() if not equal),
            "benchmark_cost": _cost_summary(benchmark_sidecar),
        }

    serial_seconds = sum(
        float(reference[episode_id]["serial_publication_interval_seconds"])
        for episode_id in selected
    )
    identity_preserved = _serial_identity_matches(serial_identity)
    if not identity_preserved:
        raise RuntimeError("authoritative serial scorer changed before summary publication")
    if _sha256_file(benchmark_script) != benchmark_script_sha256:
        raise RuntimeError("benchmark script changed before summary publication")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": BENCHMARK_KIND,
        "status": "complete",
        "benchmark_script_sha256": benchmark_script_sha256,
        "assignments": [list(value) for value in assignments],
        "episode_count": len(selected),
        "state_count": sum(int(reference[value]["state_count"]) for value in selected),
        "serial_baseline": {
            "publication_interval_seconds": serial_seconds,
            "episodes_per_hour": len(selected) * 3600.0 / serial_seconds,
            "source": "authoritative_sidecar_publication_mtime_deltas",
        },
        "two_worker_benchmark": {
            "elapsed_seconds_including_model_load": benchmark_elapsed,
            "episodes_per_hour_including_model_load": len(selected) * 3600.0 / benchmark_elapsed,
            "speedup_including_model_load": serial_seconds / benchmark_elapsed,
            "worker_summaries": worker_summaries,
        },
        "comparison": comparisons,
        "all_sidecars_byte_identical": all(
            value["all_files_byte_identical"] for value in comparisons.values()
        ),
        "all_scientific_arrays_byte_identical": all(
            value["all_non_cost_arrays_byte_identical"] for value in comparisons.values()
        ),
        "resources": _resource_summary(samples, worker_pids, serial_pid),
        "serial_process_untouched": identity_preserved,
        "serial_identity_preserved": identity_preserved,
        "authoritative_score_root_written": False,
        "locked_test_accessed": False,
    }
    digest = _write_exclusive(
        benchmark_root / "benchmark-summary.json", _canonical(summary)
    )
    print(json.dumps({**summary, "benchmark_summary_sha256": digest}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    orchestrator = subparsers.add_parser("orchestrate")
    _common_parser(orchestrator)
    orchestrator.add_argument("--reference-score-root", type=Path, required=True)
    orchestrator.add_argument("--benchmark-root", type=Path, required=True)
    orchestrator.add_argument("--worker-lock-root", type=Path, required=True)
    orchestrator.add_argument("--worker0-episode", action="append", required=True)
    orchestrator.add_argument("--worker1-episode", action="append", required=True)
    orchestrator.add_argument("--sample-interval", type=float, default=1.0)
    orchestrator.add_argument("--max-wall-seconds", type=float, default=2400.0)

    worker = subparsers.add_parser("worker")
    _common_parser(worker)
    worker.add_argument("--reference-score-root", type=Path, required=True)
    worker.add_argument("--worker-root", type=Path, required=True)
    worker.add_argument("--worker-lock", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--episode-id", action="append", required=True)
    worker.add_argument("--orchestrator-pid", type=int, required=True)
    worker.add_argument("--expected-benchmark-script-sha256", required=True)

    args = parser.parse_args()
    if args.mode == "worker":
        return _worker_main(args)
    if args.sample_interval <= 0.0 or not math.isfinite(args.sample_interval):
        raise RuntimeError("sample interval must be finite and positive")
    if args.max_wall_seconds <= 0.0 or not math.isfinite(args.max_wall_seconds):
        raise RuntimeError("watchdog must be finite and positive")
    return _orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
