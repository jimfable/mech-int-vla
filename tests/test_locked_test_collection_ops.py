from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "ops" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERVISOR = _load_script("locked_test_supervisor")


def test_entrypoints_import_without_external_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for name in {
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }:
        environment.pop(name, None)
    for script in ("locked_test_cell.py", "locked_test_supervisor.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ops" / script), "--help"],
            cwd=Path("/"),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--repo-root" in result.stdout


def test_cell_import_sets_required_runtime_environment() -> None:
    code = (
        "import importlib.util,json;"
        f"p={str(ROOT / 'ops/locked_test_cell.py')!r};"
        "s=importlib.util.spec_from_file_location('cell',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps({k:__import__('os').environ.get(k) "
        "for k in m.REQUIRED_RUNTIME_ENVIRONMENT},sort_keys=True))"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for name in (
        "MUJOCO_GL",
        "MUJOCO_EGL_DEVICE_ID",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "MUJOCO_GL": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_global_lock_rejects_concurrent_supervisor(tmp_path: Path) -> None:
    lock_path = tmp_path / "supervisor.lock"
    with SUPERVISOR._exclusive_supervisor_lock(lock_path):
        code = (
            "import importlib.util,pathlib;"
            f"p={str(ROOT / 'ops/locked_test_supervisor.py')!r};"
            "s=importlib.util.spec_from_file_location('sup',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"q=pathlib.Path({str(lock_path)!r});"
            "\nwith m._exclusive_supervisor_lock(q): pass"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0
    assert "another Locked Test supervisor" in result.stderr


def test_repo_file_binding_requires_exact_tracked_head_bytes(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    environment_lock = tmp_path / "environment.lock"
    cell = tmp_path / "ops" / "locked_test_cell.py"
    environment_lock.write_text("locked = true\n", encoding="utf-8")
    cell.write_text("print('cell')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "environment.lock", "ops/locked_test_cell.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assert (
        SUPERVISOR._require_exact_tracked_repo_file(
            tmp_path, environment_lock, Path("environment.lock")
        )
        == environment_lock
    )
    environment_lock.write_text("locked = false\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes differ from HEAD"):
        SUPERVISOR._require_exact_tracked_repo_file(
            tmp_path, environment_lock, Path("environment.lock")
        )


def test_remote_runtime_preflight_is_machine_readable_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "environment-gpu.freeze"
    freeze.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        SUPERVISOR,
        "_require_exact_tracked_repo_file",
        lambda root, supplied, relative: freeze,
    )
    monkeypatch.setattr(SUPERVISOR, "_sha256_file", lambda path: SUPERVISOR.GPU_FREEZE_SHA256)
    monkeypatch.setattr(
        SUPERVISOR.platform, "python_version", lambda: SUPERVISOR.EXPECTED_PYTHON_VERSION
    )
    monkeypatch.setattr(SUPERVISOR.sys, "prefix", "/venv/main")
    monkeypatch.setattr(
        SUPERVISOR.importlib.metadata,
        "version",
        lambda distribution: SUPERVISOR.REQUIRED_DISTRIBUTIONS[distribution],
    )
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu130",
        version=SimpleNamespace(cuda=SUPERVISOR.EXPECTED_CUDA_VERSION),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5090",
            get_device_capability=lambda index: (12, 0),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        SUPERVISOR.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0],
            0,
            stdout=(
                "0, NVIDIA GeForce RTX 5090, 580.159.03, 12.0\n"
            ),
            stderr="",
        ),
    )
    for name, value in SUPERVISOR.REQUIRED_COLLECTION_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    result = SUPERVISOR._validate_remote_runtime(tmp_path)
    assert result["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result["cuda_version"] == "13.0"
    assert result["environment_gpu_freeze_sha256"] == SUPERVISOR.GPU_FREEZE_SHA256

    # The host driver is recorded, but was never a frozen protocol input.  A
    # rebuilt instance may use another CUDA-13-compatible driver while exposing
    # the same checked runtime, device and compute capability.
    monkeypatch.setattr(
        SUPERVISOR.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0],
            0,
            stdout="0, NVIDIA GeForce RTX 5090, 590.44.01, 12.0\n",
            stderr="",
        ),
    )
    assert SUPERVISOR._validate_remote_runtime(tmp_path)["gpu_driver_version"] == "590.44.01"

    monkeypatch.setattr(
        SUPERVISOR.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout="0, NVIDIA GeForce RTX 4090, 580.159.03, 8.9\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="GPU freeze mismatch"):
        SUPERVISOR._validate_remote_runtime(tmp_path)


class _Episode:
    def __init__(self, episode_id: str):
        self.episode_id = episode_id

    def to_dict(self) -> dict[str, str]:
        return {"episode_id": self.episode_id}


def _authority() -> dict[str, object]:
    return {
        "head_commit": SUPERVISOR.COLLECTION_COMMIT,
        "tag_commit": SUPERVISOR.COLLECTION_COMMIT,
        "lock_payload_sha256": SUPERVISOR.LOCK_PAYLOAD_SHA256,
        "reality_gate_receipt_sha256": SUPERVISOR.REALITY_GATE_RECEIPT_SHA256,
        "orientation_eligibility_sha256": SUPERVISOR.ORIENTATION_SHA256,
        "reality_gate_lock_receipt_sha256": SUPERVISOR.LOCK_RECEIPT_SHA256,
        "failure_event_freeze_sha256": SUPERVISOR.FAILURE_FREEZE_SHA256,
        "calibration_manifest_sha256": SUPERVISOR.MANIFEST_SHA256,
        "locked_test_accessed": True,
    }


def _runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, plan: bool):
    episodes = (_Episode("episode-0"), _Episode("episode-1"))
    manifest_payload = {"episodes": [item.to_dict() for item in episodes]}
    manifest = SimpleNamespace(
        episodes=episodes,
        sha256=SUPERVISOR.MANIFEST_SHA256,
        to_dict=lambda: manifest_payload,
    )
    artifact_root = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    receipt = tmp_path / "locked-test-complete.json"
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps(_authority()), encoding="utf-8")
    reality = tmp_path / "locks" / "reality_gate_frozen.json"
    reality.parent.mkdir()
    reality.write_text("{}", encoding="utf-8")
    environment_lock = tmp_path / "environment.lock"
    environment_lock.write_text("fixture", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    cell = tmp_path / "cell.py"
    cell.write_text("pass\n", encoding="utf-8")
    args = Namespace(
        environment_lock=environment_lock,
        manifest=manifest_path,
        authority=authority,
        cache_dir=cache,
        artifact_root=artifact_root,
        cell_script=cell,
        log_dir=log_dir,
        completion_receipt=receipt,
        plan_only=plan,
    )
    protocol = SimpleNamespace(task_order=SimpleNamespace(tasks=(object(),)))
    monkeypatch.setattr(SUPERVISOR, "_require_locked_checkout", lambda root: None)
    monkeypatch.setattr(
        SUPERVISOR,
        "_validate_remote_runtime",
        lambda root: {
            "gpu_name": SUPERVISOR.EXPECTED_GPU_NAME,
            "cuda_version": SUPERVISOR.EXPECTED_CUDA_VERSION,
        },
    )
    monkeypatch.setattr(SUPERVISOR, "_canonical_json_sha_file", lambda path: SUPERVISOR.LOCK_PAYLOAD_SHA256)
    monkeypatch.setattr(SUPERVISOR, "load_protocol_config", lambda path: protocol)
    monkeypatch.setattr(
        SUPERVISOR,
        "load_model_input_lock",
        lambda path: SimpleNamespace(policy_revision=SUPERVISOR.POLICY_REVISION),
    )
    monkeypatch.setattr(SUPERVISOR, "_load_manifest", lambda path: manifest_payload)
    monkeypatch.setattr(SUPERVISOR, "reconstruct_episode_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(SUPERVISOR, "_validate_disk_space", lambda *a: 99_000_000_000)
    monkeypatch.setattr(
        SUPERVISOR,
        "resolve_snapshot_paths",
        lambda *a, **k: SimpleNamespace(policy=cache / "policy", base_vlm=cache / "vlm"),
    )

    def validate(destination: Path, task: object, expected: _Episode):
        if (destination / "corrupt").exists():
            raise RuntimeError("corrupt artifact")
        if not destination.is_dir():
            raise RuntimeError("missing artifact")
        return {
            "episode_id": expected.episode_id,
            "metadata_sha256": "m",
            "trajectory_sha256": "t",
            "valid_reset": True,
            "success": False,
            "control_steps": 1,
        }

    monkeypatch.setattr(SUPERVISOR, "_validate_artifact", validate)
    return args, episodes


def test_plan_only_validates_resume_without_gpu_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args, episodes = _runner_fixture(tmp_path, monkeypatch, plan=True)
    existing = args.artifact_root / "locked_test" / episodes[0].episode_id
    existing.mkdir(parents=True)
    monkeypatch.setattr(
        SUPERVISOR.subprocess,
        "run",
        lambda *a, **k: pytest.fail("plan-only must not launch a child"),
    )
    assert SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock") == 0
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "locked_test_plan_validated"
    assert output["resume_episodes"] == 1
    assert output["missing_episodes"] == 1


def test_fake_child_collection_resumes_and_writes_locked_test_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, episodes = _runner_fixture(tmp_path, monkeypatch, plan=False)
    (args.artifact_root / "locked_test" / episodes[0].episode_id).mkdir(parents=True)

    def fake_child(command: list[str], **kwargs: object):
        index = int(command[command.index("--index") + 1])
        (args.artifact_root / "locked_test" / episodes[index].episode_id).mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(SUPERVISOR.subprocess, "run", fake_child)
    assert SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock") == 0
    payload = json.loads(args.completion_receipt.read_text(encoding="utf-8"))
    assert payload["kind"] == "locked_test_collection_receipt"
    assert [item["episode_id"] for item in payload["episodes"]] == [
        "episode-0",
        "episode-1",
    ]


def test_corrupt_resume_child_failure_staging_and_extra_dirs_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, episodes = _runner_fixture(tmp_path, monkeypatch, plan=False)
    collection = args.artifact_root / "locked_test"
    corrupt = collection / episodes[0].episode_id
    corrupt.mkdir(parents=True)
    (corrupt / "corrupt").touch()
    with pytest.raises(RuntimeError, match="corrupt artifact"):
        SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock")
    (corrupt / "corrupt").unlink()
    extra = collection / "not-in-manifest"
    extra.mkdir()
    with pytest.raises(RuntimeError, match="non-manifest entries"):
        SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock")
    extra.rmdir()
    staging = collection / f".{episodes[1].episode_id}.tmp-interrupted"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="unexplained staging"):
        SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock")
    staging.rmdir()

    def child_failure(*args: object, **kwargs: object):
        raise subprocess.CalledProcessError(9, ["fake-cell"])

    monkeypatch.setattr(SUPERVISOR.subprocess, "run", child_failure)
    with pytest.raises(subprocess.CalledProcessError):
        SUPERVISOR._run_locked(args, tmp_path, tmp_path / "lock")
    assert not args.completion_receipt.exists()


def test_receipt_refuses_mismatch_and_preserves_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "locked-test-complete.json"
    SUPERVISOR._write_receipt(path, {"kind": "locked_test_collection_receipt", "x": 1})
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="differs"):
        SUPERVISOR._write_receipt(
            path, {"kind": "locked_test_collection_receipt", "x": 2}
        )
    assert path.read_bytes() == original
