from __future__ import annotations

import hashlib
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from mech_int_vla.config import load_protocol_config
from mech_int_vla.guard import (
    CalibrationGuardError,
    LockedTestGuardError,
    assert_calibration_ready,
    assert_locked_test_ready,
)

ROOT = Path(__file__).parents[1]
POLICY = "31d453f7edd78c839a8bbc39744a292686daf0de"
ARTIFACT_CONTENTS = {
    "bound_probe": b"source-bound-probe-artifact",
    "feature_reference_arrays": b"calibration-reference-arrays",
    "feature_reference_metadata": b"calibration-reference-metadata",
    "predictor_bundle": b"canonical-m0-m1-m2-predictor-bundle",
    "predictor_metadata": b"canonical-m0-m1-m2-predictor-metadata",
    "probe": b"probe-artifact",
    "reality_gate_manifest": b"reality-gate-manifest",
    "calibration_manifest": b"calibration-manifest",
}


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def selected_task(task) -> dict[str, object]:
    return {
        "rank": task.rank,
        "suite": task.suite,
        "task_id": task.task_id,
        "language": task.language,
        "primary_object": task.primary_object,
        "planar_symmetry_order": task.planar_symmetry_order,
    }


def reality_payload(task) -> dict[str, object]:
    return {
        "selected_task": selected_task(task),
        "selected_variable": {"name": "theta_rel", "symmetry_order": 2},
        "policy_revision": POLICY,
    }


def calibration_payload(task) -> dict[str, object]:
    return {
        **reality_payload(task),
        "representation_probe": {
            "candidate": "vlm_context",
            "ridge_alpha": 0.1,
            "coefficient_hash": digest("probe"),
        },
        "predictor": {
            "family": "logistic_regression",
            "hyperparameters": {"C": 1.0},
            "coefficient_hash": digest("predictor-coefficients"),
        },
        "artifact_hashes": {
            name: {
                "path": f"artifacts/frozen/{name}.bin",
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
            for name, contents in ARTIFACT_CONTENTS.items()
        },
        "alarm_thresholds": {"m0": 0.5, "m1": 0.55, "m2": 0.6},
        "patch_strength": 0.5,
        "calibration_metrics": {
            "m0": {"log_loss": 0.6, "brier": 0.21, "auroc": 0.7},
            "m1": {"log_loss": 0.5, "brier": 0.18, "auroc": 0.75},
            "m2": {"log_loss": 0.45, "brier": 0.16, "auroc": 0.8},
        },
    }


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def locked_repo(
    tmp_path: Path, *, path: Path, tag: str, payload: dict[str, object]
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Protocol Test")
    git(repo, "config", "user.email", "protocol@example.invalid")
    freeze = repo / path
    freeze.parent.mkdir(parents=True)
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    for name, contents in ARTIFACT_CONTENTS.items():
        artifact = repo / "artifacts" / "frozen" / f"{name}.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(contents)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "freeze stage")
    git(repo, "tag", tag)
    return repo


@pytest.fixture(scope="module")
def protocol():
    return load_protocol_config(ROOT / "configs")


def test_calibration_guard_returns_parsed_reality_gate_payload(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.calibration_guard
    payload = reality_payload(task)
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(CalibrationGuardError, match="lock payload keys differ"):
        assert_calibration_ready(
            repo,
            config,
            protocol=protocol,
            task=task,
            policy_revision=POLICY,
        )


def test_calibration_guard_rejects_task_semantic_mismatch(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    payload = reality_payload(task)
    payload["selected_task"]["language"] = "wrong task language"
    config = protocol.split.calibration_guard
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(CalibrationGuardError, match="language"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_calibration_guard_rejects_policy_revision_mismatch(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    payload = reality_payload(task)
    payload["policy_revision"] = "f" * 40
    config = protocol.split.calibration_guard
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(CalibrationGuardError, match="does not match"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_calibration_guard_rejects_dirty_worktree(protocol, tmp_path: Path) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.calibration_guard
    repo = locked_repo(
        tmp_path,
        path=config.required_file,
        tag=config.required_tag,
        payload=reality_payload(task),
    )
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")

    with pytest.raises(CalibrationGuardError, match="clean worktree"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_calibration_guard_rejects_tag_not_exactly_at_head(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.calibration_guard
    repo = locked_repo(
        tmp_path,
        path=config.required_file,
        tag=config.required_tag,
        payload=reality_payload(task),
    )
    (repo / "later.txt").write_text("later", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "commit after lock")

    with pytest.raises(CalibrationGuardError, match="exactly at HEAD"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_calibration_guard_rejects_ignored_untracked_freeze(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.calibration_guard
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Protocol Test")
    git(repo, "config", "user.email", "protocol@example.invalid")
    (repo / ".gitignore").write_text("locks/\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "ignore locks")
    git(repo, "tag", config.required_tag)
    freeze = repo / config.required_file
    freeze.parent.mkdir()
    freeze.write_text(json.dumps(reality_payload(task)), encoding="utf-8")

    with pytest.raises(CalibrationGuardError, match="tracked in the lock commit"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_calibration_guard_rejects_missing_or_empty_freeze(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.calibration_guard
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")

    with pytest.raises(CalibrationGuardError, match="freeze file is missing"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )

    freeze = repo / config.required_file
    freeze.parent.mkdir()
    freeze.write_text("{}", encoding="utf-8")
    with pytest.raises(CalibrationGuardError, match="nonempty JSON object"):
        assert_calibration_ready(
            repo, config, protocol=protocol, task=task, policy_revision=POLICY
        )


def test_locked_guard_returns_parsed_complete_payload(protocol, tmp_path: Path) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    receipt = assert_locked_test_ready(
        repo,
        config,
        task=task,
        policy_revision=POLICY,
        selection=protocol.split.calibration_selection,
    )

    assert receipt.payload == payload


def test_locked_guard_rejects_task_and_policy_mismatches(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    payload["selected_task"]["primary_object"] = "wrong_object"
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )
    with pytest.raises(LockedTestGuardError, match="primary_object"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )

    second = tmp_path / "second"
    second.mkdir()
    policy_payload = calibration_payload(task)
    policy_payload["policy_revision"] = "f" * 40
    repo = locked_repo(
        second,
        path=config.required_file,
        tag=config.required_tag,
        payload=policy_payload,
    )
    with pytest.raises(LockedTestGuardError, match="policy_revision does not match"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["representation_probe"].update(
                {"candidate": "unregistered_layer"}
            ),
            "five configured candidates",
        ),
        (
            lambda payload: payload.update({"patch_strength": 0.75}),
            "patch_strength",
        ),
        (
            lambda payload: payload["predictor"].update({"family": "random_forest"}),
            "predictor.family",
        ),
        (
            lambda payload: payload["predictor"].update(
                {"hyperparameters": {"C": 0.333}}
            ),
            "hyperparameters.C",
        ),
        (
            lambda payload: payload["predictor"].update({"coefficient_hash": "abc123"}),
            "coefficient_hash",
        ),
        (
            lambda payload: payload["artifact_hashes"].pop("calibration_manifest"),
            "artifact_hashes is missing",
        ),
        (
            lambda payload: payload["artifact_hashes"].update(
                {"probe": digest("probe-artifact")}
            ),
            "must have exactly: path, sha256",
        ),
        (
            lambda payload: payload.update(
                {"alarm_thresholds": {"m0": 0.5, "m1": 1.2, "m2": 0.6}}
            ),
            "alarm_thresholds.m1",
        ),
        (
            lambda payload: payload.update(
                {"calibration_metrics": {"m0": {}, "m1": {}, "m2": {}}}
            ),
            "calibration_metrics.m0",
        ),
        (
            lambda payload: payload.update({"selected_variable": "PENDING"}),
            "placeholder",
        ),
        (
            lambda payload: payload.update(
                {"selected_variable": {"name": "favorite_color"}}
            ),
            "not preregistered",
        ),
        (
            lambda payload: payload.update(
                {"selected_variable": {"name": "theta_rel", "symmetry_order": 1}}
            ),
            "symmetry_order",
        ),
        (
            lambda payload: payload["calibration_metrics"]["m1"].update(
                {"log_loss": -0.1}
            ),
            "log_loss must be finite and nonnegative",
        ),
        (
            lambda payload: payload["calibration_metrics"]["m0"].update(
                {"brier": -0.1}
            ),
            "brier must be finite and nonnegative",
        ),
        (
            lambda payload: payload["calibration_metrics"]["m0"].update({"brier": 1.1}),
            "brier must be in",
        ),
        (
            lambda payload: payload["calibration_metrics"]["m2"].update({"auroc": 1.1}),
            "auroc must be in",
        ),
        (
            lambda payload: payload["calibration_metrics"].update(
                {"PENDING": {"value": 1.0}}
            ),
            "placeholder key",
        ),
    ],
)
def test_locked_guard_rejects_invalid_frozen_selection(
    protocol, tmp_path: Path, mutation, message: str
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = deepcopy(calibration_payload(task))
    mutation(payload)
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(LockedTestGuardError, match=message):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )


def test_locked_guard_accepts_configured_histogram_boosting(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    payload["predictor"] = {
        "family": "histogram_gradient_boosting",
        "hyperparameters": {
            "learning_rate": 0.03,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 0,
            "max_iter": 200,
        },
        "coefficient_hash": digest("boosting-model"),
    }
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    assert_locked_test_ready(
        repo,
        config,
        task=task,
        policy_revision=POLICY,
        selection=protocol.split.calibration_selection,
    )


def test_locked_guard_requires_histogram_boosting_max_iter_exactly_200(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    payload["predictor"] = {
        "family": "histogram_gradient_boosting",
        "hyperparameters": {
            "learning_rate": 0.03,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 0,
            "max_iter": 199,
        },
        "coefficient_hash": digest("boosting-model"),
    }
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(LockedTestGuardError, match="exactly 200"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )


def test_locked_guard_accepts_only_the_exact_never_alarm_sentinel(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    payload["alarm_thresholds"]["m1"] = math.nextafter(1.0, math.inf)
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )
    assert_locked_test_ready(
        repo,
        config,
        task=task,
        policy_revision=POLICY,
        selection=protocol.split.calibration_selection,
    )

    second = tmp_path / "second"
    second.mkdir()
    invalid = calibration_payload(task)
    invalid["alarm_thresholds"]["m1"] = math.nextafter(
        math.nextafter(1.0, math.inf), math.inf
    )
    repo = locked_repo(
        second, path=config.required_file, tag=config.required_tag, payload=invalid
    )
    with pytest.raises(LockedTestGuardError, match="never-alarm sentinel"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda declaration: declaration.update({"path": "../outside.bin"}),
            "safe repository-relative path",
        ),
        (
            lambda declaration: declaration.update({"path": "/tmp/outside.bin"}),
            "safe repository-relative path",
        ),
        (
            lambda declaration: declaration.update(
                {"path": "artifacts/frozen/missing.bin"}
            ),
            "missing or unreadable",
        ),
        (
            lambda declaration: declaration.update({"sha256": digest("wrong")}),
            "does not match the artifact bytes",
        ),
    ],
)
def test_locked_guard_rejects_unsafe_missing_or_mismatched_artifacts(
    protocol, tmp_path: Path, mutate, message: str
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard
    payload = calibration_payload(task)
    mutate(payload["artifact_hashes"]["probe"])
    repo = locked_repo(
        tmp_path, path=config.required_file, tag=config.required_tag, payload=payload
    )

    with pytest.raises(LockedTestGuardError, match=message):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )


def test_locked_guard_rejects_symlink_and_untracked_artifacts(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    config = protocol.split.locked_test_guard

    symlink_case = tmp_path / "symlink"
    symlink_case.mkdir()
    payload = calibration_payload(task)
    payload["artifact_hashes"]["probe"]["path"] = "artifacts/frozen/probe-link.bin"
    repo = locked_repo(
        symlink_case,
        path=config.required_file,
        tag=config.required_tag,
        payload=payload,
    )
    (repo / "artifacts/frozen/probe-link.bin").symlink_to("probe.bin")
    git(repo, "add", "artifacts/frozen/probe-link.bin")
    git(repo, "commit", "--amend", "-qm", "freeze symlink")
    git(repo, "tag", "-f", config.required_tag)
    with pytest.raises(LockedTestGuardError, match="must not contain symlinks"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )

    untracked_case = tmp_path / "untracked"
    untracked_case.mkdir()
    payload = calibration_payload(task)
    payload["artifact_hashes"]["probe"]["path"] = "artifacts/frozen/untracked.bin"
    repo = locked_repo(
        untracked_case,
        path=config.required_file,
        tag=config.required_tag,
        payload=payload,
    )
    untracked = repo / "artifacts/frozen/untracked.bin"
    untracked.write_bytes(ARTIFACT_CONTENTS["probe"])
    (repo / ".gitignore").write_text(
        "artifacts/frozen/untracked.bin\n", encoding="utf-8"
    )
    git(repo, "add", ".gitignore")
    git(repo, "commit", "--amend", "-qm", "freeze with ignored artifact")
    git(repo, "tag", "-f", config.required_tag)
    with pytest.raises(LockedTestGuardError, match="tracked in the lock commit"):
        assert_locked_test_ready(
            repo,
            config,
            task=task,
            policy_revision=POLICY,
            selection=protocol.split.calibration_selection,
        )
