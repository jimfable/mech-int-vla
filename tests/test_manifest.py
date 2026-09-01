from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from mech_int_vla import SplitName, generate_episode_manifest, load_protocol_config
from mech_int_vla.config import ProtocolConfigError
from mech_int_vla.guard import CalibrationGuardError
from mech_int_vla.manifest import reconstruct_episode_manifest

ROOT = Path(__file__).parents[1]
POLICY = "31d453f7edd78c839a8bbc39744a292686daf0de"
COMMIT = "a" * 40
ARTIFACT_CONTENTS = {
    "bound_probe": b"source-bound-probe-artifact",
    "calibration_activation_reference_arrays": b"calibration-activation-arrays",
    "calibration_activation_reference_metadata": b"calibration-activation-metadata",
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


def freeze_payload(task) -> dict[str, object]:
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
            "coefficient_hash": digest("predictor"),
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


def ready_repo(tmp_path: Path, guard, payload: dict[str, object]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Protocol Test")
    git("config", "user.email", "protocol@example.invalid")
    freeze = repo / guard.required_file
    freeze.parent.mkdir()
    freeze.write_text(json.dumps(payload), encoding="utf-8")
    for name, contents in ARTIFACT_CONTENTS.items():
        artifact = repo / "artifacts" / "frozen" / f"{name}.bin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(contents)
    git("add", ".")
    git("commit", "-qm", "freeze stage")
    git("tag", guard.required_tag)
    return repo, git("rev-parse", "HEAD")


@pytest.fixture(scope="module")
def protocol():
    return load_protocol_config(ROOT / "configs")


def test_discovery_manifest_is_deterministic_unique_and_balanced(protocol) -> None:
    task = protocol.task_order.tasks[0]
    first = generate_episode_manifest(
        SplitName.DISCOVERY,
        task,
        protocol,
        policy_revision=POLICY,
        code_commit=COMMIT,
    )
    second = generate_episode_manifest(
        "discovery", task, protocol, policy_revision=POLICY, code_commit=COMMIT
    )

    assert first.sha256 == second.sha256
    assert len(first.episodes) == 40
    assert len({episode.reset_seed for episode in first.episodes}) == 40
    yaw = [
        episode
        for episode in first.episodes
        if episode.condition_family == "object_yaw"
    ]
    assert Counter(episode.condition_name for episode in yaw) == {
        "yaw_neg_15": 5,
        "yaw_pos_15": 5,
        "yaw_neg_30": 5,
        "yaw_pos_30": 5,
        "yaw_neg_45": 5,
        "yaw_pos_45": 5,
    }
    per_init = defaultdict(list)
    for episode in yaw:
        per_init[episode.base_init_state_id].append(
            abs(episode.condition_parameters["value"])
        )
    assert all(sorted(magnitudes) == [15, 30, 45] for magnitudes in per_init.values())


def test_calibration_has_eight_cells_and_balanced_cardinal_directions(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    _repo, head = ready_repo(
        tmp_path, protocol.split.calibration_guard, reality_payload(task)
    )
    manifest = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=POLICY,
        code_commit=head,
    )

    assert len(manifest.episodes) == 160
    assert {episode.base_init_state_id for episode in manifest.episodes} == set(
        range(10, 30)
    )
    assert all(
        len([e for e in manifest.episodes if e.base_init_state_id == init_id]) == 8
        for init_id in range(10, 30)
    )
    translations = [
        episode
        for episode in manifest.episodes
        if episode.condition_family == "object_planar_translation"
    ]
    assert Counter(
        e.condition_parameters["assigned_direction"] for e in translations
    ) == {
        "x_positive": 5,
        "x_negative": 5,
        "y_positive": 5,
        "y_negative": 5,
    }
    with pytest.raises(TypeError):
        translations[0].condition_parameters["assigned_direction"] = "x_positive"
    assert manifest.episodes[0].noise_seed("output", 0) == manifest.episodes[
        0
    ].noise_seed("output", 0)


def test_historical_protected_manifest_can_be_revalidated_after_head_advances(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    repo, lock_commit = ready_repo(
        tmp_path, protocol.split.calibration_guard, reality_payload(task)
    )
    original = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=POLICY,
        code_commit=lock_commit,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-qm", "later lock"],
        check=True,
    )

    with pytest.raises(CalibrationGuardError, match="tag.*HEAD"):
        generate_episode_manifest(
            SplitName.CALIBRATION,
            task,
            protocol,
            policy_revision=POLICY,
            code_commit=lock_commit,
            repo_root=repo,
        )
    reconstructed = reconstruct_episode_manifest(
        SplitName.CALIBRATION,
        task,
        protocol,
        policy_revision=POLICY,
        code_commit=lock_commit,
    )
    assert reconstructed.to_dict() == original.to_dict()


def test_locked_manifest_fails_closed_without_repo(protocol) -> None:
    with pytest.raises(ProtocolConfigError, match="repo_root"):
        generate_episode_manifest(
            SplitName.LOCKED_TEST,
            protocol.task_order.tasks[0],
            protocol,
            policy_revision=POLICY,
            code_commit=COMMIT,
        )


def test_calibration_manifest_fails_closed_without_repo(protocol) -> None:
    with pytest.raises(ProtocolConfigError, match="repo_root"):
        generate_episode_manifest(
            SplitName.CALIBRATION,
            protocol.task_order.tasks[0],
            protocol,
            policy_revision=POLICY,
            code_commit=COMMIT,
        )


def test_locked_manifest_after_guard_has_held_out_cells_and_balanced_diagonals(
    protocol, tmp_path: Path
) -> None:
    task = protocol.task_order.tasks[0]
    repo, head = ready_repo(
        tmp_path, protocol.split.locked_test_guard, freeze_payload(task)
    )

    manifest = generate_episode_manifest(
        SplitName.LOCKED_TEST,
        task,
        protocol,
        policy_revision=POLICY,
        code_commit=head,
        repo_root=repo,
    )

    assert len(manifest.episodes) == 160
    assert {episode.base_init_state_id for episode in manifest.episodes} == set(
        range(30, 50)
    )
    assert Counter(episode.condition_family for episode in manifest.episodes) == {
        "iid": 20,
        "object_yaw": 80,
        "object_planar_translation": 20,
        "agentview_camera_extrinsic": 40,
    }
    translations = [
        episode
        for episode in manifest.episodes
        if episode.condition_family == "object_planar_translation"
    ]
    assert set(
        Counter(
            e.condition_parameters["assigned_direction"] for e in translations
        ).values()
    ) == {5}
