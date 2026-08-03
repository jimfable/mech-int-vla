from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.artifacts import (
    ACTIVATION_CANDIDATES,
    FRAME_SCALAR_FEATURE_NAMES,
    ArtifactValidationError,
    CohortManifest,
    assemble_probe_cohort,
    load_rollout_artifact,
)
from mech_int_vla.config import TaskSpec

POLICY_REVISION = "1" * 40
BASE_REVISION = "2" * 40
CODE_COMMIT = "3" * 40
TASK = TaskSpec(
    rank=1,
    suite="libero_10",
    task_id=5,
    language="pick up the book and place it in the back compartment",
    primary_object="black_book",
    planar_symmetry_order=2,
)


def _quaternion_xyzw(yaw: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def _quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _write_artifact(
    root: Path,
    *,
    base_init_state_id: int,
    condition_index: int,
    success: bool = False,
    valid: bool = True,
    action_count: int = 6,
    eef_yaw: float = 0.0,
    object_yaw: float = 0.0,
    policy_revision: str = POLICY_REVISION,
    code_commit: str = CODE_COMMIT,
) -> Path:
    if not valid:
        action_count = 0
    split = "calibration"
    episode_id = (
        f"libero_10-task5-{split}-init{base_init_state_id:02d}-cell{condition_index}"
    )
    directory = root / split / episode_id
    directory.mkdir(parents=True)
    frame_count = action_count + 1
    activation_steps = np.arange(0, action_count, 5, dtype=np.int32)
    terminated = np.zeros(action_count, dtype=np.bool_)
    truncated = np.zeros(action_count, dtype=np.bool_)
    if valid:
        (terminated if success else truncated)[-1] = True
    rewards = np.zeros(action_count, dtype=np.float32)
    if success:
        rewards[-1] = 1.0
    frame_success = np.zeros(frame_count, dtype=np.bool_)
    if success:
        frame_success[-1] = True

    steps = np.arange(frame_count, dtype=np.int32)
    simulator_time = steps.astype(np.float64) * 0.05
    eef_position = np.tile(np.asarray([0.0, 0.0, 1.0]), (frame_count, 1))
    object_position = np.tile(np.asarray([0.1, 0.0, 0.9]), (frame_count, 1))
    goal_position = np.tile(np.asarray([0.3, 0.0, 0.9]), (frame_count, 1))
    phase = np.full(frame_count, "pregrasp", dtype="U16")
    scalar = np.zeros((frame_count, len(FRAME_SCALAR_FEATURE_NAMES)), dtype=np.float32)
    scalar[:, 0] = steps / max(action_count, 1)
    scalar[:, 1] = simulator_time
    scalar[:, 2] = np.linalg.norm(eef_position - object_position, axis=1)
    scalar[:, 3] = np.linalg.norm(object_position - goal_position, axis=1)
    scalar[:, 4] = 0.04
    scalar[:, 7] = frame_success
    scalar[:, 8] = 1.0
    scalar[:, 12] = math.sin(2 * (eef_yaw - object_yaw))
    scalar[:, 13] = math.cos(2 * (eef_yaw - object_yaw))
    scalar[:, 14] = math.sin(2 * object_yaw)
    scalar[:, 15] = math.cos(2 * object_yaw)
    arrays: dict[str, np.ndarray] = {
        "actions": np.arange(action_count * 7, dtype=np.float32).reshape(
            action_count, 7
        ),
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "activation_control_step": activation_steps,
        "frame_control_step": steps,
        "frame_simulator_time": simulator_time,
        "frame_policy_state": np.arange(frame_count * 8, dtype=np.float32).reshape(
            frame_count, 8
        ),
        "frame_agentview_image": np.zeros((frame_count, 360, 360, 3), dtype=np.uint8),
        "frame_robot0_eye_in_hand_image": np.zeros(
            (frame_count, 360, 360, 3), dtype=np.uint8
        ),
        "frame_eef_position": eef_position,
        "frame_eef_quaternion_xyzw": np.tile(
            _quaternion_xyzw(eef_yaw), (frame_count, 1)
        ),
        "frame_primary_object_position": object_position,
        "frame_primary_object_quaternion_wxyz": np.tile(
            _quaternion_wxyz(object_yaw), (frame_count, 1)
        ),
        "frame_gripper_qpos": np.full((frame_count, 2), 0.02, dtype=np.float64),
        "frame_gripper_qvel": np.zeros((frame_count, 2), dtype=np.float64),
        "frame_primary_gripper_contact": np.zeros(frame_count, dtype=np.bool_),
        "frame_primary_grasped": np.zeros(frame_count, dtype=np.bool_),
        "frame_task_success": frame_success,
        "frame_phase": phase,
        "frame_scalar_features": scalar,
        "frame_goal_present": np.ones(frame_count, dtype=np.bool_),
        "frame_goal_position": goal_position,
        "frame_goal_quaternion_wxyz": np.tile(_quaternion_wxyz(0.0), (frame_count, 1)),
        "frame_task_predicates": frame_success.reshape(frame_count, 1),
    }
    for candidate_index, candidate in enumerate(ACTIVATION_CANDIDATES):
        arrays[f"activation_{candidate}"] = (
            np.arange(action_count * 3, dtype=np.float32).reshape(action_count, 3)[
                activation_steps
            ]
            + 100 * candidate_index
            + 10 * condition_index
            if valid
            else np.empty((0, 0), dtype=np.float32)
        )

    status = "invalid_reset" if not valid else "success" if success else "truncated"
    metadata = {
        "schema_version": 1,
        "episode": {
            "episode_id": episode_id,
            "suite": "libero_10",
            "task_id": 5,
            "task_rank": 1,
            "split": split,
            "base_init_state_id": base_init_state_id,
            "condition_index": condition_index,
            "condition_name": f"condition_{condition_index}",
            "condition_family": "yaw",
            "condition_parameters": {"value": float(condition_index)},
            "reset_seed": 101000 + base_init_state_id * 10 + condition_index,
            "inference_seed": 12345 + condition_index,
            "policy_revision": policy_revision,
            "code_commit": code_commit,
        },
        "task_language": "pick up the book and place it in the back compartment",
        "task": {
            "rank": 1,
            "suite": "libero_10",
            "task_id": 5,
            "language": "pick up the book and place it in the back compartment",
            "primary_object": "black_book",
            "planar_symmetry_order": 2,
        },
        "condition": {
            "name": f"condition_{condition_index}",
            "family": "yaw",
            "index": condition_index,
            "parameters": {"value": float(condition_index)},
        },
        "model": {
            "policy_revision": policy_revision,
            "base_vlm_revision": BASE_REVISION,
        },
        "execution": {
            "n_action_steps": 1,
            "max_steps": max(action_count, 1),
            "reset_noop_steps": 10,
            "settle_actions": [[0.0] * 6 + [-1.0] for _ in range(10)],
            "closed_loop_replanning": True,
        },
        "validity": {
            "valid": valid,
            "reasons": [] if valid else ["primary_object_penetration"],
            "finite": True,
            "deepest_primary_penetration_m": 0.0 if valid else 0.01,
            "linear_speed_m_s": 0.0,
            "angular_speed_rad_s": 0.0,
            "in_workspace": True,
            "initial_success": False,
        },
        "validity_retry": (
            {"performed": False}
            if valid
            else {
                "performed": True,
                "same_reset_seed_and_condition": True,
                "validity": {
                    "valid": False,
                    "reasons": ["primary_object_penetration"],
                    "finite": True,
                    "deepest_primary_penetration_m": 0.01,
                    "linear_speed_m_s": 0.0,
                    "angular_speed_rad_s": 0.0,
                    "in_workspace": True,
                    "initial_success": False,
                },
                "settle_actions": [[0.0] * 6 + [-1.0] for _ in range(10)],
                "agrees_on_invalidity": True,
                "agrees_on_reasons": True,
            }
        ),
        "outcome": {
            "status": status,
            "success": success,
            "terminated": bool(success),
            "truncated": bool(valid and not success),
            "control_steps": action_count,
            "reward_sum": float(rewards.sum()),
            "terminal_state_preserved": valid,
        },
        "capture": {
            "policy_select_calls": action_count,
            "scored_policy_select_calls": len(activation_steps),
            "score_stride_steps": 5,
            "instrumented_internal_calls": len(activation_steps) * 11,
            "activation_candidates": list(ACTIVATION_CANDIDATES),
            "activation_dtype": "float32",
            "frame_count": frame_count,
            "frame_scalar_feature_names": list(FRAME_SCALAR_FEATURE_NAMES),
            "task_predicate_names": ["goal_satisfied"],
            "raw_images_stored": True,
            "raw_image_encoding": "lossless_uint8_npz_deflate",
            "raw_image_observation_keys": [
                "agentview_image",
                "robot0_eye_in_hand_image",
            ],
            "raw_image_shape": [360, 360, 3],
        },
        "files": {"trajectory": "trajectory.npz"},
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    return directory


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path / "trajectory.npz", **arrays)


def _mutate_metadata(path: Path, mutation) -> None:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_load_validates_and_hashes_both_inputs(tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path, base_init_state_id=20, condition_index=0, success=True
    )
    artifact = load_rollout_artifact(path)

    assert artifact.episode_id == path.name
    assert artifact.valid_reset
    assert artifact.success
    assert artifact.action_count == 6
    assert (
        artifact.hashes.metadata_sha256
        == hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest()
    )
    assert (
        artifact.hashes.trajectory_sha256
        == hashlib.sha256((path / "trajectory.npz").read_bytes()).hexdigest()
    )
    assert not artifact.arrays["actions"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        artifact.arrays["actions"][0, 0] = 0.0
    with pytest.raises(TypeError):
        artifact.metadata["episode"]["suite"] = "mutated"
    with pytest.raises(TypeError):
        artifact.metadata["capture"]["activation_candidates"][0] = "mutated"


def test_manifest_recursively_freezes_a_copy_and_serializes_canonically() -> None:
    source = {"nested": {"items": [3, 1], "enabled": True}}
    manifest = CohortManifest(source)
    before = manifest.sha256
    source["nested"]["items"][0] = 99

    assert manifest.sha256 == before
    assert json.loads(manifest.canonical_json()) == {
        "nested": {"enabled": True, "items": [3, 1]}
    }
    with pytest.raises(TypeError):
        manifest.payload["nested"]["enabled"] = False
    with pytest.raises(TypeError):
        manifest.payload["nested"]["items"][0] = 99


def test_authoritative_task_checks_object_and_symmetry(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    wrong_task = TaskSpec(
        rank=TASK.rank,
        suite=TASK.suite,
        task_id=TASK.task_id,
        language=TASK.language,
        primary_object=TASK.primary_object,
        planar_symmetry_order=1,
    )

    with pytest.raises(ArtifactValidationError, match="symmetry does not match"):
        load_rollout_artifact(path, expected_task=wrong_task)
    with pytest.raises(ArtifactValidationError, match="multiple of the stored stride"):
        assemble_probe_cohort(
            [path],
            requested_episode_ids=[path.name],
            expected_task=TASK,
            stride=3,
        )


def test_cohort_uses_pre_action_stride_and_wraps_relative_yaw(tmp_path: Path) -> None:
    failure = _write_artifact(
        tmp_path,
        base_init_state_id=21,
        condition_index=1,
        eef_yaw=-math.pi + 0.1,
        object_yaw=math.pi - 0.1,
    )
    success = _write_artifact(
        tmp_path,
        base_init_state_id=20,
        condition_index=0,
        success=True,
        eef_yaw=0.7,
        object_yaw=0.2,
    )
    requested = [failure.name, success.name]

    cohort = assemble_probe_cohort(
        [failure, success],
        requested_episode_ids=requested,
        expected_task=TASK,
    )
    reversed_cohort = assemble_probe_cohort(
        [success, failure],
        requested_episode_ids=list(reversed(requested)),
        expected_task=TASK,
    )

    assert cohort.samples.n_rows == 4
    assert cohort.control_step.tolist() == [0, 5, 0, 5]
    assert cohort.base_init_state_id.tolist() == [20, 20, 21, 21]
    assert cohort.episode_id.tolist() == [
        success.name,
        success.name,
        failure.name,
        failure.name,
    ]
    assert cohort.failure_label.tolist() == [False, False, True, True]
    assert np.allclose(cohort.theta_rel, [0.5, 0.5, 0.2, 0.2])
    assert set(cohort.activation_features) == set(ACTIVATION_CANDIDATES)
    assert all(matrix.shape == (4, 3) for matrix in cohort.activation_features.values())
    assert cohort.activation_features["vlm_context"][:, 0].tolist() == [
        0.0,
        15.0,
        10.0,
        25.0,
    ]
    assert cohort.manifest.payload["selection"]["outcome_conditioned"] is False
    assert len(cohort.manifest_sha256) == 64
    assert cohort.manifest_sha256 == reversed_cohort.manifest_sha256
    assert cohort.manifest.canonical_json() == reversed_cohort.manifest.canonical_json()
    original_manifest_hash = cohort.manifest_sha256
    with pytest.raises(TypeError):
        cohort.manifest.payload["episodes"][0]["episode_id"] = "mutated"
    assert cohort.manifest_sha256 == original_manifest_hash


def test_invalid_resets_are_excluded_and_explicitly_listed(tmp_path: Path) -> None:
    valid = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    invalid = _write_artifact(
        tmp_path, base_init_state_id=21, condition_index=1, valid=False
    )
    cohort = assemble_probe_cohort(
        [invalid, valid],
        requested_episode_ids=[valid.name, invalid.name],
        expected_task=TASK,
        stride=5,
    )

    assert cohort.valid_episode_ids == (valid.name,)
    assert cohort.invalid_reset_episode_ids == (invalid.name,)
    assert cohort.episode_id.tolist() == [valid.name, valid.name]
    assert cohort.control_step.tolist() == [0, 5]
    assert cohort.manifest.payload["invalid_reset_episode_ids"] == (invalid.name,)


def test_invalid_retry_audit_must_be_complete_and_consistent(tmp_path: Path) -> None:
    invalid = _write_artifact(
        tmp_path, base_init_state_id=21, condition_index=1, valid=False
    )
    _mutate_metadata(
        invalid,
        lambda metadata: metadata["validity_retry"].update(agrees_on_invalidity=False),
    )

    with pytest.raises(
        ArtifactValidationError, match="inconsistent with retry validity"
    ):
        load_rollout_artifact(invalid, expected_task=TASK)


@pytest.mark.parametrize(
    ("requested_selector", "paths_selector", "match"),
    [
        (
            lambda one, two: [one.name, two.name],
            lambda one, two: [one],
            "missing requested",
        ),
        (
            lambda one, two: [one.name],
            lambda one, two: [one, two],
            "unexpected artifacts",
        ),
        (
            lambda one, two: [one.name],
            lambda one, two: [one, one],
            "duplicate episode IDs",
        ),
    ],
)
def test_cohort_fails_on_missing_unexpected_or_duplicate_artifacts(
    tmp_path: Path, requested_selector, paths_selector, match: str
) -> None:
    one = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    two = _write_artifact(tmp_path, base_init_state_id=21, condition_index=1)
    with pytest.raises(ArtifactValidationError, match=match):
        assemble_probe_cohort(
            paths_selector(one, two),
            requested_episode_ids=requested_selector(one, two),
            expected_task=TASK,
        )


@pytest.mark.parametrize("mixed_field", ["policy", "code"])
def test_cohort_fails_on_mixed_policy_or_code(tmp_path: Path, mixed_field: str) -> None:
    one = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    kwargs = (
        {"policy_revision": "9" * 40}
        if mixed_field == "policy"
        else {"code_commit": "8" * 40}
    )
    two = _write_artifact(tmp_path, base_init_state_id=21, condition_index=1, **kwargs)
    with pytest.raises(ArtifactValidationError, match="artifacts mix"):
        assemble_probe_cohort(
            [one, two],
            requested_episode_ids=[one.name, two.name],
            expected_task=TASK,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda metadata: metadata.update(schema_version=2), "schema_version"),
        (
            lambda metadata: metadata["condition"].update(name="wrong"),
            "condition.name",
        ),
        (
            lambda metadata: metadata["outcome"].update(success="false"),
            "outcome.success",
        ),
        (
            lambda metadata: metadata["capture"].update(policy_select_calls=5),
            "policy_select_calls",
        ),
        (
            lambda metadata: metadata["capture"].update(raw_images_stored=False),
            "raw_images_stored",
        ),
    ],
)
def test_loader_rejects_metadata_schema_and_cross_reference_drift(
    tmp_path: Path, mutation, match: str
) -> None:
    path = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    _mutate_metadata(path, mutation)
    with pytest.raises(ArtifactValidationError, match=match):
        load_rollout_artifact(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda arrays: arrays.update(
                frame_control_step=np.asarray([0, 1, 2, 4, 4, 5, 6], dtype=np.int32)
            ),
            "sequential from zero",
        ),
        (
            lambda arrays: arrays.update(
                activation_control_step=np.asarray([0, 4], dtype=np.int32)
            ),
            "strictly increasing pre-action stride-5",
        ),
        (
            lambda arrays: arrays.update(actions=np.zeros((6, 6), dtype=np.float32)),
            "must have shape",
        ),
        (
            lambda arrays: arrays.pop("frame_phase"),
            "missing frame_phase",
        ),
        (
            lambda arrays: arrays.update(
                frame_agentview_image=np.zeros((7, 360, 360, 3), dtype=np.float32)
            ),
            "frame_agentview_image.*dtype uint8",
        ),
        (
            lambda arrays: arrays.update(
                frame_robot0_eye_in_hand_image=np.zeros(
                    (7, 180, 180, 3), dtype=np.uint8
                )
            ),
            "frame_robot0_eye_in_hand_image.*shape",
        ),
        (
            lambda arrays: arrays.update(
                activation_vlm_context=np.zeros((1, 3), dtype=np.float32)
            ),
            "activation_vlm_context.*shape",
        ),
    ],
)
def test_loader_rejects_incomplete_or_misaligned_trajectory_arrays(
    tmp_path: Path, mutation, match: str
) -> None:
    path = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    arrays = _read_arrays(path)
    mutation(arrays)
    _write_arrays(path, arrays)
    with pytest.raises(ArtifactValidationError, match=match):
        load_rollout_artifact(path)


def test_loader_uses_allow_pickle_false(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    arrays = _read_arrays(path)
    arrays["actions"] = np.asarray([[object()] * 7] * 6, dtype=object)
    _write_arrays(path, arrays)

    with pytest.raises(ArtifactValidationError, match="safe NumPy archive"):
        load_rollout_artifact(path)


def test_loader_rejects_non_atomic_directory_contents_and_path_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    (path / "partial.tmp").write_text("incomplete", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="unexpected partial.tmp"):
        load_rollout_artifact(path)

    (path / "partial.tmp").unlink()
    renamed = path.with_name("wrong-episode")
    path.rename(renamed)
    with pytest.raises(ArtifactValidationError, match="directory name"):
        load_rollout_artifact(renamed)


def test_feature_width_must_be_stable_across_valid_episodes(tmp_path: Path) -> None:
    one = _write_artifact(tmp_path, base_init_state_id=20, condition_index=0)
    two = _write_artifact(tmp_path, base_init_state_id=21, condition_index=1)
    arrays = _read_arrays(two)
    arrays["activation_vlm_context"] = np.zeros((2, 4), dtype=np.float32)
    _write_arrays(two, arrays)

    with pytest.raises(ArtifactValidationError, match="feature width changed"):
        assemble_probe_cohort(
            [one, two],
            requested_episode_ids=[one.name, two.name],
            expected_task=TASK,
        )
