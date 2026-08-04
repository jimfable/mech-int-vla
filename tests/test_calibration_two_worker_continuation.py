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


if __name__ == "__main__":
    unittest.main()
