from __future__ import annotations

import importlib.util
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "ops"
    / "calibration_two_worker_continuation_18d6494.py"
)
SPEC = importlib.util.spec_from_file_location("calibration_continuation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


def _record(episode_id: str, mtime_ns: int, value: int = 1) -> dict[str, object]:
    file_record = {
        "sha256": f"{value:064x}",
        "size_bytes": value,
        "device": 1,
        "inode": value,
        "mtime_ns": mtime_ns,
        "ctime_ns": mtime_ns,
    }
    return {
        "episode_id": episode_id,
        "path": f"/score/{episode_id}",
        "metadata": dict(file_record),
        "primitives": dict(file_record),
        "combined_sha256": f"{value + 100:064x}",
        "state_count": value,
        "links": {"probe_sha256": "a" * 64},
        "cost": {
            "cuda_event_ms_sum": float(value),
            "wall_time_ns_sum": value * 10,
            "forward_count": value * 2,
            "intervention_count": value * 3,
            "peak_allocated_bytes_max": value * 4,
            "incremental_peak_allocated_bytes_max": value * 5,
            "logical_activation_bytes": value * 6,
            "compressed_activation_bytes": value * 7,
        },
    }


class ContinuationControlTests(unittest.TestCase):
    @staticmethod
    def _recovery_signal_status(*, catch_sigterm: bool = False) -> dict[str, object]:
        def bit(signum: int) -> int:
            return 1 << (int(signum) - 1)

        values = {
            "SigPnd": 0,
            "ShdPnd": 0,
            "SigBlk": 0,
            "SigIgn": bit(continuation.signal.SIGINT),
            "SigCgt": bit(continuation.signal.SIGTERM) if catch_sigterm else 0,
        }
        return {
            "pid": 101,
            "hex_masks": {name: f"{value:016x}" for name, value in values.items()},
            "integer_masks": values,
        }

    def test_recovery_signal_status_requires_ignored_int_and_default_term(self) -> None:
        status = self._recovery_signal_status()
        result = continuation._validate_recovery_signal_status(status)
        self.assertTrue(result["sigint"]["ignored"])
        self.assertFalse(result["sigterm"]["custom_caught"])

        caught = self._recovery_signal_status(catch_sigterm=True)
        with self.assertRaisesRegex(RuntimeError, "SIGTERM unexpectedly present in SigCgt"):
            continuation._validate_recovery_signal_status(caught)

    def test_serial_cost_modes_use_interval_overlap(self) -> None:
        inventory = {
            "records": [
                _record("a", 100, 1),
                _record("b", 200, 2),
                _record("c", 300, 3),
                _record("d", 400, 4),
            ]
        }
        evidence = {
            "benchmark_start_unix_time_ns": 150,
            "benchmark_end_unix_time_ns": 250,
        }
        self.assertEqual(
            continuation._classify_serial_modes(inventory, evidence, 0),
            {
                "a": "serial",
                "b": "serial_with_equivalence_benchmark_contention",
                "c": "serial_with_equivalence_benchmark_contention",
                "d": "serial",
            },
        )

    def test_cost_aggregation_is_stratified_and_uses_max_for_peaks(self) -> None:
        records = [_record("a", 100, 2), _record("b", 200, 3), _record("c", 300, 4)]
        result = continuation._aggregate_costs(
            {"records": records},
            {"a": "serial", "b": "serial", "c": "two_worker"},
        )
        self.assertEqual(result["serial"]["episode_count"], 2)
        self.assertEqual(result["serial"]["cuda_event_ms_sum"], 5.0)
        self.assertEqual(result["serial"]["forward_count"], 10)
        self.assertEqual(result["serial"]["peak_allocated_bytes_max"], 12)
        self.assertEqual(result["two_worker"]["episode_count"], 1)
        self.assertEqual(
            result["serial_with_equivalence_benchmark_contention"]["episode_count"], 0
        )

    def test_frozen_inventory_rejects_hash_or_identity_change(self) -> None:
        frozen = {"records": [_record("a", 100, 1)]}
        same = {"records": [_record("a", 100, 1), _record("b", 200, 2)]}
        continuation._assert_frozen_unchanged(frozen, same)
        changed = {"records": [_record("a", 101, 1)]}
        with self.assertRaisesRegex(RuntimeError, "frozen sidecar changed"):
            continuation._assert_frozen_unchanged(frozen, changed)

    def test_recovery_requires_two_nonempty_shards_before_signal(self) -> None:
        continuation._assert_two_worker_capacity_after_boundary(
            {"episode_count": 1}, ["a", "b", "c", "d"]
        )
        with self.assertRaisesRegex(RuntimeError, "fewer than two continuation"):
            continuation._assert_two_worker_capacity_after_boundary(
                {"episode_count": 2}, ["a", "b", "c", "d"]
            )

    def test_execution_control_allowlist_rejects_extra_signal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cutover-intent.json").write_text("{}", encoding="utf-8")
            (root / "boundary-observed.json").write_text("{}", encoding="utf-8")
            attempts = root / "signal-attempts"
            attempts.mkdir()
            (attempts / "attempt-0001-intent.json").write_text("{}", encoding="utf-8")
            (attempts / "attempt-0001-dispatched.json").write_text(
                "{}", encoding="utf-8"
            )
            continuation._assert_execution_control_state(
                root, "original_dispatched"
            )
            (attempts / "attempt-0002-dispatched.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected original signal"):
                continuation._assert_execution_control_state(
                    root, "original_dispatched"
                )

    def test_process_exit_wait_rejects_same_instance_identity_mutation(self) -> None:
        identity = {
            "pid": 101,
            "parent_pid": 100,
            "start_ticks": 1234,
            "executable": "/python",
            "arguments": ["/python", "runner.py"],
        }
        mutated = dict(identity, parent_pid=1)
        original = continuation._process_identity
        continuation._process_identity = lambda _pid: mutated
        try:
            with self.assertRaisesRegex(RuntimeError, "identity mutated"):
                continuation._wait_process_instance_exit(
                    identity, timeout_seconds=0.1, role="serial_python"
                )
        finally:
            continuation._process_identity = original

    def test_pidfd_acquisition_rejects_reused_target_before_signal(self) -> None:
        identity = {
            "pid": 101,
            "parent_pid": 100,
            "start_ticks": 1001,
            "executable": "/python",
            "arguments": ["/python", "runner.py"],
        }
        replacement = dict(identity, start_ticks=2002)
        original_platform = continuation.sys.platform
        original_match = continuation._identity_matches
        original_process = continuation._process_identity
        original_pidfd_open = continuation._pidfd_open_linux
        original_close = continuation.os.close
        closed: list[int] = []
        continuation.sys.platform = "linux"
        continuation._identity_matches = lambda _identity: True
        continuation._process_identity = lambda _pid: replacement
        continuation._pidfd_open_linux = lambda _pid: 42
        continuation.os.close = closed.append
        try:
            with self.assertRaisesRegex(RuntimeError, "changed during pidfd"):
                continuation._open_validated_pidfd(identity)
        finally:
            continuation.sys.platform = original_platform
            continuation._identity_matches = original_match
            continuation._process_identity = original_process
            continuation.os.close = original_close
            continuation._pidfd_open_linux = original_pidfd_open
        self.assertEqual(closed, [42])

    @unittest.skipUnless(sys.platform.startswith("linux"), "pidfd is Linux-only")
    def test_libc_pidfd_delivery_path_with_signal_zero(self) -> None:
        descriptor = continuation._pidfd_open_linux(os.getpid())
        try:
            continuation._pidfd_send_signal_linux(descriptor, 0)
        finally:
            os.close(descriptor)

    @unittest.skipUnless(sys.platform.startswith("linux"), "renameat2 is Linux-only")
    def test_rename_noreplace_publishes_once_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "authoritative"
            parent.mkdir()
            first = root / "first"
            first.mkdir()
            (first / "payload").write_text("first", encoding="utf-8")
            destination = parent / "episode"
            continuation._rename_noreplace(first, destination)
            self.assertEqual((destination / "payload").read_text(), "first")

            second = root / "second"
            second.mkdir()
            (second / "payload").write_text("second", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                continuation._rename_noreplace(second, destination)
            self.assertEqual((destination / "payload").read_text(), "first")
            self.assertTrue(second.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "flock inheritance is Linux-tested")
    def test_inherited_writer_fd_keeps_global_flock_after_parent_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "global.lock"
            held = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.set_inheritable(held, True)
            read_fd, write_fd = os.pipe()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; os.read(int(sys.argv[1]),1)",
                    str(read_fd),
                ],
                pass_fds=(held, read_fd),
            )
            os.close(read_fd)
            os.close(held)
            competitor = os.open(lock_path, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.write(write_fd, b"x")
                os.close(write_fd)
                child.wait(timeout=5)
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
                try:
                    os.close(write_fd)
                except OSError:
                    pass
                os.close(competitor)

    def test_boundary_wait_returns_only_flushed_score_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "serial.log"
            path.write_bytes(b"")
            original = continuation._identity_matches
            continuation._identity_matches = lambda _identity: True

            def writer() -> None:
                time.sleep(0.05)
                with path.open("ab", buffering=0) as stream:
                    stream.write(b"not json\n")
                    stream.write(json.dumps({"kind": "score_resume_validated"}).encode() + b"\n")
                    stream.write(
                        json.dumps(
                            {
                                "kind": "score_completed",
                                "episode_id": "episode-22",
                                "sha256": "a" * 64,
                            }
                        ).encode()
                        + b"\n"
                    )

            thread = threading.Thread(target=writer)
            thread.start()
            try:
                value, offset = continuation._wait_for_score_completed(
                    path, 0, {"pid": 1}, timeout_seconds=2.0
                )
            finally:
                continuation._identity_matches = original
                thread.join()
            self.assertEqual(value["episode_id"], "episode-22")
            self.assertGreater(offset, 0)

    def test_boundary_transition_requires_exact_next_id_count_and_hash(self) -> None:
        baseline_records = [_record("a", 100, 1), _record("b", 200, 2)]
        boundary_record = _record("c", 300, 3)
        boundary_record["combined_sha256"] = "d" * 64
        baseline = {
            "episode_count": 2,
            "episode_ids": ["a", "b"],
            "records": baseline_records,
        }
        frozen = {
            "episode_count": 3,
            "episode_ids": ["a", "b", "c"],
            "records": [*baseline_records, boundary_record],
        }
        line = {
            "kind": "score_completed",
            "episode_id": "c",
            "sha256": "d" * 64,
            "completed": 3,
            "total": 4,
        }
        continuation._validate_boundary_transition(line, baseline, frozen, ["a", "b", "c", "d"])
        malformed = dict(line, completed=4)
        with self.assertRaisesRegex(RuntimeError, "completed count"):
            continuation._validate_boundary_transition(
                malformed, baseline, frozen, ["a", "b", "c", "d"]
            )
        extra = dict(frozen)
        extra["episode_count"] = 4
        extra["episode_ids"] = ["a", "b", "c", "d"]
        extra["records"] = [*frozen["records"], _record("d", 400, 4)]
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            continuation._validate_boundary_transition(
                line, baseline, extra, ["a", "b", "c", "d"]
            )

    def test_signal_intent_without_dispatch_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "signal-attempts"
            attempts.mkdir()
            (attempts / "attempt-0001-intent.json").write_text("{}", encoding="utf-8")
            args = Namespace(execution_root=root)
            with self.assertRaisesRegex(RuntimeError, "ambiguous signal state"):
                continuation._dispatch_serial_interrupt(
                    args,
                    {"python": {"pid": 1}, "flock_wrapper": {"pid": 2}},
                    root / "unused-boundary.json",
                )

    def test_legacy_sigint_dispatch_shape_remains_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "signal-attempts"
            attempts.mkdir()
            identities = {
                "python": {"pid": 101, "parent_pid": 100, "start_ticks": 1001},
                "flock_wrapper": {
                    "pid": 100,
                    "parent_pid": 1,
                    "start_ticks": 1000,
                },
            }
            dispatched_path = attempts / "attempt-0001-dispatched.json"
            continuation._write_json_exclusive(
                dispatched_path,
                {
                    "schema_version": continuation.SCHEMA_VERSION,
                    "kind": f"{continuation.KIND}_signal_dispatched",
                    "signal": "SIGINT",
                    "signal_count": 1,
                    "serial_python_identity": identities["python"],
                    "intent_sha256": "a" * 64,
                    "os_kill_returned": True,
                    "dispatched_unix_time_ns": 1,
                    "locked_test_accessed": False,
                },
            )
            original_wait = continuation._wait_identities_exit
            original_runners = continuation._runner_processes
            continuation._wait_identities_exit = lambda *_args, **_kwargs: None
            continuation._runner_processes = lambda _runner: []
            try:
                receipt = continuation._complete_dispatched_interrupt(
                    Namespace(
                        execution_root=root,
                        serial_exit_timeout=1.0,
                        serial_runner=Path("/runner.py"),
                    ),
                    identities,
                    dispatched_path,
                )
            finally:
                continuation._wait_identities_exit = original_wait
                continuation._runner_processes = original_runners
            self.assertEqual(receipt["signal"], "SIGINT")
            self.assertEqual(receipt["signal_count"], 1)

    def test_existing_recovery_dispatch_is_validated_without_resignal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "recovery-signal-attempts"
            attempts.mkdir()
            boundary = root / "recovery-boundary-observed.json"
            ineffective = root / "attempt-0001-ineffective.json"
            boundary.write_text('{"boundary":true}', encoding="utf-8")
            ineffective.write_text('{"ineffective":true}', encoding="utf-8")
            identities = {
                "python": {"pid": 101, "parent_pid": 100, "start_ticks": 1001},
                "flock_wrapper": {
                    "pid": 100,
                    "parent_pid": 1,
                    "start_ticks": 1000,
                },
            }
            runner_sha = "a" * 64
            signal_status = continuation._validate_recovery_signal_status(
                self._recovery_signal_status()
            )
            intent_path = attempts / "attempt-0001-intent.json"
            intent = {
                "schema_version": continuation.SCHEMA_VERSION,
                "kind": f"{continuation.KIND}_recovery_signal_intent",
                "signal": "SIGTERM",
                "signal_count": 1,
                "target": "serial_python_only",
                "target_start_ticks": 1001,
                "delivery_method": "libc_pidfd_send_signal",
                "wrapper_signalled": False,
                "serial_python_identity": identities["python"],
                "serial_flock_identity": identities["flock_wrapper"],
                "runner_sha256": runner_sha,
                "boundary_receipt_path": str(boundary.resolve()),
                "boundary_receipt_sha256": continuation._sha256_file(boundary),
                "ineffective_sigint_receipt_sha256": continuation._sha256_file(
                    ineffective
                ),
                "signal_status": signal_status,
                "created_unix_time_ns": 1,
                "locked_test_accessed": False,
            }
            continuation._write_json_exclusive(intent_path, intent)
            dispatched_path = attempts / "attempt-0001-dispatched.json"
            continuation._write_json_exclusive(
                dispatched_path,
                {
                    "schema_version": continuation.SCHEMA_VERSION,
                    "kind": f"{continuation.KIND}_recovery_signal_dispatched",
                    "signal": "SIGTERM",
                    "signal_count": 1,
                    "target_pid": 101,
                    "target_start_ticks": 1001,
                    "target": "serial_python_only",
                    "delivery_method": "libc_pidfd_send_signal",
                    "wrapper_signalled": False,
                    "intent_sha256": continuation._sha256_file(intent_path),
                    "os_kill_called": False,
                    "pidfd_send_signal_returned": True,
                    "dispatched_unix_time_ns": 2,
                    "locked_test_accessed": False,
                },
            )
            original_pidfd_send = continuation._pidfd_send_signal_linux
            continuation._pidfd_send_signal_linux = lambda *_args: self.fail(
                "a durable recovery dispatch must never be re-signalled"
            )
            original_state_guard = continuation._assert_execution_control_state
            continuation._assert_execution_control_state = lambda *_args: None
            try:
                receipt = continuation._ensure_recovery_sigterm_dispatched(
                    Namespace(execution_root=root),
                    identities,
                    boundary,
                    ineffective,
                    runner_sha,
                )
            finally:
                continuation._pidfd_send_signal_linux = original_pidfd_send
                continuation._assert_execution_control_state = original_state_guard
            self.assertTrue(receipt["pidfd_send_signal_returned"])

    def test_existing_recovery_exit_receipt_survives_restart_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = root / "recovery-signal-attempts"
            attempts.mkdir()
            identities = {
                "python": {
                    "pid": 101,
                    "parent_pid": 100,
                    "start_ticks": 1001,
                    "executable": "/python",
                    "arguments": ["/python", "runner.py"],
                },
                "flock_wrapper": {
                    "pid": 100,
                    "parent_pid": 1,
                    "start_ticks": 1000,
                    "executable": "/flock",
                    "arguments": ["/flock", "lock", "/python", "runner.py"],
                },
            }
            status = continuation._validate_recovery_signal_status(
                self._recovery_signal_status()
            )
            intent_path = attempts / "attempt-0001-intent.json"
            continuation._write_json_exclusive(
                intent_path,
                {
                    "schema_version": continuation.SCHEMA_VERSION,
                    "kind": f"{continuation.KIND}_recovery_signal_intent",
                    "signal": "SIGTERM",
                    "signal_count": 1,
                    "target": "serial_python_only",
                    "target_start_ticks": 1001,
                    "delivery_method": "libc_pidfd_send_signal",
                    "wrapper_signalled": False,
                    "serial_python_identity": identities["python"],
                    "serial_flock_identity": identities["flock_wrapper"],
                    "signal_status": status,
                    "locked_test_accessed": False,
                },
            )
            dispatched_path = attempts / "attempt-0001-dispatched.json"
            continuation._write_json_exclusive(
                dispatched_path,
                {
                    "schema_version": continuation.SCHEMA_VERSION,
                    "kind": f"{continuation.KIND}_recovery_signal_dispatched",
                    "signal": "SIGTERM",
                    "signal_count": 1,
                    "target_pid": 101,
                    "target_start_ticks": 1001,
                    "target": "serial_python_only",
                    "delivery_method": "libc_pidfd_send_signal",
                    "wrapper_signalled": False,
                    "intent_sha256": continuation._sha256_file(intent_path),
                    "os_kill_called": False,
                    "pidfd_send_signal_returned": True,
                    "dispatched_unix_time_ns": 2,
                    "locked_test_accessed": False,
                },
            )
            exit_path = attempts / "attempt-0001-exit-observed.json"
            stored = {
                "schema_version": continuation.SCHEMA_VERSION,
                "kind": f"{continuation.KIND}_recovery_signal_exit_observed",
                "signal": "SIGTERM",
                "signal_count": 1,
                "signal_dispatched_sha256": continuation._sha256_file(dispatched_path),
                "serial_python_identity": identities["python"],
                "serial_flock_identity": identities["flock_wrapper"],
                "python_exit_subsequently_observed": True,
                "python_exit_observation": {
                    "role": "serial_python",
                    "pid": 101,
                    "start_ticks": 1001,
                    "exit_observed_unix_time_ns": 3,
                    "pid_reused_at_observation": False,
                },
                "wrapper_exit_subsequently_observed": True,
                "wrapper_exit_observation": {
                    "role": "serial_flock_wrapper",
                    "pid": 100,
                    "start_ticks": 1000,
                    "exit_observed_unix_time_ns": 4,
                    "pid_reused_at_observation": False,
                },
                "wrapper_signalled_by_recovery": False,
                "runner_process_count": 0,
                "causal_exit_claim": False,
                "wrapper_natural_exit_causal_claim": False,
                "observed_unix_time_ns": 4,
                "locked_test_accessed": False,
            }
            continuation._write_json_exclusive(exit_path, stored)
            original_identity = continuation._process_identity
            original_runners = continuation._runner_processes
            original_state_guard = continuation._assert_execution_control_state
            continuation._process_identity = lambda _pid: None
            continuation._runner_processes = lambda _runner: []
            continuation._assert_execution_control_state = lambda *_args: None
            try:
                observed = continuation._complete_recovery_sigterm(
                    Namespace(
                        execution_root=root,
                        serial_runner=Path("/runner.py"),
                        serial_exit_timeout=1.0,
                    ),
                    identities,
                )
            finally:
                continuation._process_identity = original_identity
                continuation._runner_processes = original_runners
                continuation._assert_execution_control_state = original_state_guard
            self.assertEqual(observed, stored)


if __name__ == "__main__":
    unittest.main()
