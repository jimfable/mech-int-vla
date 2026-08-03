from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from mech_int_vla.artifacts import ArtifactHashes, RolloutArtifact
from mech_int_vla.determinism import hash_seed
from mech_int_vla.feature_pipeline import (
    FeaturePipelineError,
    build_calibration_features,
    build_locked_test_features,
)
from mech_int_vla.features import M0_FEATURE_NAMES, M1_FEATURE_NAMES
from mech_int_vla.probes import (
    DEFAULT_CANDIDATE_PREFERENCE,
    FROZEN_RIDGE_ALPHA_GRID,
    AlphaCVResult,
    CandidateCVResult,
    CenteredCircularRidge,
    ProbeArtifact,
)
from mech_int_vla.scoring import COST_FIELDS, FROZEN_TRANSFORMS, LoadedScoringSidecar


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _probe(candidate: str = "vlm_context") -> ProbeArtifact:
    model = CenteredCircularRidge(
        alpha=0.1,
        symmetry_order=1,
        feature_center=np.zeros(2),
        target_center=np.asarray([0.5, -0.25]),
        coefficient=np.eye(2),
    )
    selected_index = DEFAULT_CANDIDATE_PREFERENCE.index(candidate)
    candidate_results = []
    for candidate_index, name in enumerate(DEFAULT_CANDIDATE_PREFERENCE):
        base = 0.1 if candidate_index == selected_index else 1.0 + candidate_index
        alpha_results = []
        for alpha_index, alpha in enumerate(FROZEN_RIDGE_ALPHA_GRID):
            score = base + abs(alpha_index - 3) * 0.01
            folds = (score,) * 5
            mean = float(np.mean(folds))
            standard_error = float(np.std(folds, ddof=1) / math.sqrt(5))
            alpha_results.append(AlphaCVResult(alpha, folds, mean, standard_error))
        selected = alpha_results[3]
        candidate_results.append(
            CandidateCVResult(
                name,
                tuple(alpha_results),
                0.1,
                selected.mean_mae_rad,
                selected.standard_error_rad,
            )
        )
    selected_result = candidate_results[selected_index]
    return ProbeArtifact(
        model=model,
        candidate=candidate,
        alpha_grid=FROZEN_RIDGE_ALPHA_GRID,
        candidate_preference=DEFAULT_CANDIDATE_PREFERENCE,
        candidate_results=tuple(candidate_results),
        one_standard_error_threshold_rad=(
            selected_result.mean_mae_rad + selected_result.standard_error_rad
        ),
        fold_test_groups=((10,), (11,), (12,), (13,), (14,)),
        training_rows=10,
        training_episodes=5,
        training_base_init_state_ids=(10, 11, 12, 13, 14),
    )


def _readonly_arrays(values: dict[str, np.ndarray]) -> MappingProxyType:
    frozen = {}
    for name, value in values.items():
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        frozen[name] = copied
    return MappingProxyType(frozen)


def _raw_artifact(
    *,
    episode_id: str,
    split: str,
    base_init: int,
    success: bool,
    episode_offset: float,
    valid: bool = True,
    task_id: int = 5,
    policy_revision: str = "pinned-policy",
    code_commit: str = "c" * 40,
) -> RolloutArtifact:
    action_count = 6 if valid else 0
    frame_count = action_count + 1
    steps = np.arange(frame_count, dtype=np.int32)
    activation_steps = np.arange(0, action_count, 5, dtype=np.int32)
    eef_position = np.column_stack(
        (
            episode_offset + steps * 0.01,
            np.zeros(frame_count),
            np.ones(frame_count),
        )
    )
    object_position = eef_position + np.asarray([0.1, 0.2, -0.1])
    goal_position = object_position + np.asarray([0.3, -0.1, 0.05])
    identity_wxyz = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frame_count, 1))
    identity_xyzw = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (frame_count, 1))
    terminated = np.zeros(action_count, dtype=np.bool_)
    truncated = np.zeros(action_count, dtype=np.bool_)
    if valid:
        (terminated if success else truncated)[-1] = True
    status = "invalid_reset" if not valid else "success" if success else "truncated"
    frame_success = np.zeros(frame_count, dtype=np.bool_)
    if valid and success:
        frame_success[-1] = True
    arrays = {
        "actions": np.full((action_count, 7), episode_offset, dtype=np.float32),
        "activation_control_step": activation_steps,
        "frame_phase": np.full(frame_count, "pregrasp", dtype="U16"),
        "frame_eef_position": eef_position.astype(np.float64),
        "frame_eef_quaternion_xyzw": identity_xyzw.astype(np.float64),
        "frame_primary_object_position": object_position.astype(np.float64),
        "frame_primary_object_quaternion_wxyz": identity_wxyz.astype(np.float64),
        "frame_goal_present": np.ones(frame_count, dtype=np.bool_),
        "frame_goal_position": goal_position.astype(np.float64),
        "frame_goal_quaternion_wxyz": identity_wxyz.astype(np.float64),
        "frame_gripper_qpos": np.full((frame_count, 2), 0.02 + episode_offset / 100),
        "frame_primary_gripper_contact": np.zeros(frame_count, dtype=np.bool_),
        "frame_primary_grasped": np.zeros(frame_count, dtype=np.bool_),
    }
    metadata_hash = _sha(f"raw-metadata:{episode_id}:{success}:{valid}:{task_id}")
    trajectory_hash = _sha(f"raw-trajectory:{episode_id}:{success}:{valid}")
    metadata = {
        "schema_version": 1,
        "episode": {
            "episode_id": episode_id,
            "suite": "libero_10",
            "task_id": task_id,
            "task_rank": 1,
            "split": split,
            "base_init_state_id": base_init,
            "policy_revision": policy_revision,
            "code_commit": code_commit,
        },
        "task_language": "place the book",
        "task": {
            "rank": 1,
            "suite": "libero_10",
            "task_id": task_id,
            "language": "place the book",
            "primary_object": "black_book",
            "planar_symmetry_order": 1,
        },
        "model": {
            "policy_revision": policy_revision,
            "base_vlm_revision": "pinned-vlm",
        },
        "validity": {"valid": valid},
        "execution": {"max_steps": 520},
        "outcome": {
            "status": status,
            "success": success if valid else False,
            "control_steps": action_count,
        },
    }
    return RolloutArtifact(
        path=Path("/synthetic/raw") / split / episode_id,
        metadata=metadata,
        arrays=_readonly_arrays(arrays),
        hashes=ArtifactHashes(metadata_hash, trajectory_hash),
    )


def _cost_array(shape: tuple[int, ...], *, intervention: bool) -> np.ndarray:
    row = np.asarray([1.0, 10.0, 1.0, float(intervention), 100.0, 20.0, 8.0, 8.0])
    return np.broadcast_to(row, (*shape, len(COST_FIELDS))).copy()


def _score_sidecar(
    raw: RolloutArtifact,
    probe: ProbeArtifact,
    *,
    episode_offset: float,
    unavailable_transform: int | None = None,
    unavailable_intervention_draw: int | None = None,
    config_sha256: str = _sha("config"),
    code_sha256: str = _sha("code"),
    activation_width: int = 2,
) -> LoadedScoringSidecar:
    episode_id = raw.episode_id
    split = str(raw.metadata["episode"]["split"])
    action_count = raw.action_count
    steps = np.arange(0, action_count, 5, dtype=np.int32)
    state_count = len(steps)
    original_actions = np.empty((state_count, 8, 10, 7), dtype=np.float32)
    original_activation = np.empty((state_count, 8, activation_width), dtype=np.float32)
    for state_index, step in enumerate(steps):
        for draw in range(8):
            base = episode_offset + state_index * 0.2 + draw * 0.01
            original_actions[state_index, draw] = base
            original_activation[state_index, draw] = np.linspace(
                1.0 + base, 2.0 + base, activation_width
            )
    transformed_actions = np.empty((state_count, 6, 4, 10, 7), dtype=np.float32)
    transformed_activation = np.empty(
        (state_count, 6, 4, activation_width), dtype=np.float32
    )
    for transform in range(6):
        transformed_actions[:, transform] = (
            original_actions[:, :4] + (transform + 1) * 0.1
        )
        transformed_activation[:, transform] = (
            original_activation[:, :4] + (transform + 1) * 0.05
        )
    transform_mask = np.ones((state_count, 6), dtype=np.bool_)
    transformed_cost = _cost_array((state_count, 6, 4), intervention=False)
    if unavailable_transform is not None:
        transform_mask[:, unavailable_transform] = False
        transformed_actions[:, unavailable_transform] = np.nan
        transformed_activation[:, unavailable_transform] = np.nan
        transformed_cost[:, unavailable_transform] = np.nan
    intervention_mask = np.ones((state_count, 4), dtype=np.bool_)
    minus_actions = original_actions[:, :4] - 0.2
    plus_actions = original_actions[:, :4] + 0.2
    minus_activation = original_activation[:, :4] - 0.1
    plus_activation = original_activation[:, :4] + 0.1
    minus_cost = _cost_array((state_count, 4), intervention=True)
    plus_cost = _cost_array((state_count, 4), intervention=True)
    if unavailable_intervention_draw is not None:
        intervention_mask[:, unavailable_intervention_draw] = False
        for value in (
            minus_actions,
            plus_actions,
            minus_activation,
            plus_activation,
            minus_cost,
            plus_cost,
        ):
            value[:, unavailable_intervention_draw] = np.nan
    noise_seed = np.asarray(
        [
            [
                hash_seed("score-noise-v1", episode_id, int(step), draw)
                for draw in range(8)
            ]
            for step in steps
        ],
        dtype=np.int64,
    )
    arrays = {
        "control_step": steps,
        "noise_seed": noise_seed,
        "transform_available": transform_mask,
        "intervention_available": intervention_mask,
        "original_actions": original_actions,
        "transformed_actions": transformed_actions,
        "intervention_minus_actions": minus_actions,
        "intervention_plus_actions": plus_actions,
        "original_activation": original_activation,
        "transformed_activation": transformed_activation,
        "intervention_minus_activation": minus_activation,
        "intervention_plus_activation": plus_activation,
        "original_cost": _cost_array((state_count, 8), intervention=False),
        "transformed_cost": transformed_cost,
        "intervention_minus_cost": minus_cost,
        "intervention_plus_cost": plus_cost,
    }
    links = {
        "raw_metadata_sha256": raw.hashes.metadata_sha256,
        "raw_trajectory_sha256": raw.hashes.trajectory_sha256,
        "probe_sha256": probe.sha256(),
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
    }
    metadata = {
        "schema_version": 1,
        "episode_id": episode_id,
        "split": split,
        "links": links,
        "protocol": {
            "score_stride_steps": 5,
            "original_draws": 8,
            "common_draws": 4,
            "chunk_actions": 10,
            "action_dimension": 7,
            "transform_order": [
                {
                    "name": transform.name,
                    "family": transform.family,
                    "value": transform.value,
                }
                for transform in FROZEN_TRANSFORMS
            ],
            "intervention_degrees": [-10.0, 10.0],
        },
        "capture": {
            "state_count": state_count,
            "selected_activation_width": activation_width,
        },
        "factual_outcome": {
            "status": raw.metadata["outcome"]["status"],
            "success": raw.success,
            "control_steps": action_count,
        },
    }
    return LoadedScoringSidecar(
        path=Path("/synthetic/scores") / split / episode_id,
        metadata=metadata,
        arrays=_readonly_arrays(arrays),
        metadata_sha256=_sha(f"score-metadata:{episode_id}:{raw.success}"),
        primitives_sha256=_sha(f"score-primitives:{episode_id}:{raw.success}"),
    )


def _paired_cohort(
    *,
    split: str,
    probe: ProbeArtifact,
    successes: tuple[bool, ...] = (True, True, False),
    base_inits: tuple[int, ...] = (10, 10, 11),
    unavailable_transform: int | None = None,
    unavailable_intervention_draw: int | None = None,
    code_commit: str = "c" * 40,
) -> tuple[list[RolloutArtifact], list[LoadedScoringSidecar]]:
    raws = []
    scores = []
    for index, (success, base_init) in enumerate(
        zip(successes, base_inits, strict=True)
    ):
        episode_id = f"{split}-task5-init{base_init}-cell{index}"
        raw = _raw_artifact(
            episode_id=episode_id,
            split=split,
            base_init=base_init,
            success=success,
            episode_offset=float(index + 1),
            code_commit=code_commit,
        )
        raws.append(raw)
        scores.append(
            _score_sidecar(
                raw,
                probe,
                episode_offset=float(index + 1),
                unavailable_transform=unavailable_transform,
                unavailable_intervention_draw=unavailable_intervention_draw,
            )
        )
    return raws, scores


def test_calibration_pipeline_exact_features_mapping_and_oof_exclusion() -> None:
    probe = _probe("early_expert_t1_0")
    raws, scores = _paired_cohort(split="calibration", probe=probe)
    bundle, cohort = build_calibration_features(
        list(reversed(raws)), list(reversed(scores)), probe
    )

    assert cohort.split == "calibration"
    assert cohort.m0_names == M0_FEATURE_NAMES
    assert cohort.m1_names == M1_FEATURE_NAMES
    assert len(cohort.records) == 6
    assert [
        (record.episode_id, record.control_step) for record in cohort.records
    ] == sorted((raw.episode_id, step) for raw in raws for step in (0, 5))
    first = cohort.records[0]
    # The first two transforms add 0.1 and 0.2 in every action dimension.
    unit = np.linalg.norm(0.1 / bundle.action_scale.values) / math.sqrt(7)
    assert first.m0.values[0] == pytest.approx(1.5 * unit)
    assert first.m0.values[1] == pytest.approx(2.0 * unit)
    assert first.m1.as_dict()["m1_object_minus_eef_x"] == pytest.approx(0.1)
    source_score = next(
        score for score in scores if score.metadata["episode_id"] == first.episode_id
    )
    expected_probe = probe.model.predict_raw(
        source_score.arrays["original_activation"][0]
    )
    expected_norm = np.mean(np.linalg.norm(expected_probe, axis=1))
    assert first.m2.as_dict()["m2_probe_resultant_norm_mean"] == pytest.approx(
        expected_norm
    )
    assert "m2_probe_flow_noise_circular_dispersion" in first.m2.names

    # Both episodes sharing init 10 are absent from one another's OOF references.
    init10_record = next(
        record for record in cohort.records if record.base_init_state_id == 10
    )
    init10_coverage = bundle.coverage_states
    eligible_episode_ids = {
        state.episode_id
        for state in init10_coverage
        if state.base_init_state_id != init10_record.base_init_state_id
    }
    assert eligible_episode_ids == {
        raw.episode_id
        for raw in raws
        if raw.metadata["episode"]["base_init_state_id"] == 11
    }


def test_masks_propagate_to_m0_m2_and_interventions_fail_closed() -> None:
    probe = _probe()
    raws, scores = _paired_cohort(
        split="calibration",
        probe=probe,
        unavailable_transform=5,
        unavailable_intervention_draw=2,
    )
    _, cohort = build_calibration_features(raws, scores, probe)
    for record in cohort.records:
        assert "m2_probe_flow_noise_circular_dispersion" not in record.m2.names
        assert np.isnan(record.m0.as_dict()["m0_object_action_drift_mean"])
        assert np.isnan(
            record.m2.as_dict()["m2_object_probe_equivariance_error_mean_rad"]
        )
        assert np.isnan(record.m2.as_dict()["m2_probe_controllability_gain_per_rad"])
        assert np.isnan(record.m2.as_dict()["m2_probe_intervention_specificity_ratio"])


def test_deterministic_hashes_matrices_and_bundle_are_immutable() -> None:
    probe = _probe()
    raws, scores = _paired_cohort(split="calibration", probe=probe)
    first_bundle, first = build_calibration_features(raws, scores, probe)
    second_bundle, second = build_calibration_features(
        list(reversed(raws)), list(reversed(scores)), probe
    )
    assert first_bundle.metadata_sha256 == second_bundle.metadata_sha256
    assert first.provenance_sha256 == second.provenance_sha256
    assert np.array_equal(first.m2_matrix, second.m2_matrix, equal_nan=True)
    with pytest.raises(ValueError):
        first.m0_matrix[0, 0] = 0
    with pytest.raises(ValueError):
        first_bundle.action_scale.values[0] = 0
    with pytest.raises(ValueError):
        first_bundle.coverage_states[0].vector[0] = 0


def test_locked_test_uses_full_bundle_and_test_label_cannot_change_features() -> None:
    probe = _probe()
    calibration_raws, calibration_scores = _paired_cohort(
        split="calibration", probe=probe
    )
    bundle, _ = build_calibration_features(calibration_raws, calibration_scores, probe)
    failed_raws, failed_scores = _paired_cohort(
        split="locked_test",
        probe=probe,
        successes=(False,),
        base_inits=(30,),
        code_commit="d" * 40,
    )
    successful_raws, successful_scores = _paired_cohort(
        split="locked_test",
        probe=probe,
        successes=(True,),
        base_inits=(30,),
        code_commit="d" * 40,
    )
    failed = build_locked_test_features(failed_raws, failed_scores, probe, bundle)
    successful = build_locked_test_features(
        successful_raws, successful_scores, probe, bundle
    )
    assert np.array_equal(failed.m0_matrix, successful.m0_matrix, equal_nan=True)
    assert np.array_equal(failed.m1_matrix, successful.m1_matrix, equal_nan=True)
    assert np.array_equal(failed.m2_matrix, successful.m2_matrix, equal_nan=True)
    assert failed.records[0].terminal_failure_label is True
    assert successful.records[0].terminal_failure_label is False
    assert failed.reference_bundle_sha256 == bundle.metadata_sha256
    assert successful.reference_bundle_sha256 == bundle.metadata_sha256
    assert failed.cohort_identity.code_commit == "d" * 40
    assert bundle.cohort_identity.code_commit == "c" * 40


def test_locked_test_rejects_reference_probe_task_and_cohort_mismatch() -> None:
    probe = _probe()
    calibration_raws, calibration_scores = _paired_cohort(
        split="calibration", probe=probe
    )
    bundle, _ = build_calibration_features(calibration_raws, calibration_scores, probe)
    test_raws, _ = _paired_cohort(
        split="locked_test", probe=probe, successes=(False,), base_inits=(30,)
    )

    other_probe = _probe("early_expert_t1_0")
    relinked_scores = [_score_sidecar(test_raws[0], other_probe, episode_offset=1.0)]
    with pytest.raises(FeaturePipelineError, match="probe differs"):
        build_locked_test_features(test_raws, relinked_scores, other_probe, bundle)

    wrong_task = _raw_artifact(
        episode_id=test_raws[0].episode_id,
        split="locked_test",
        base_init=30,
        success=False,
        episode_offset=1.0,
        task_id=2,
    )
    wrong_task_score = _score_sidecar(wrong_task, probe, episode_offset=1.0)
    with pytest.raises(FeaturePipelineError, match="task differs"):
        build_locked_test_features([wrong_task], [wrong_task_score], probe, bundle)

    wrong_policy = _raw_artifact(
        episode_id=test_raws[0].episode_id,
        split="locked_test",
        base_init=30,
        success=False,
        episode_offset=1.0,
        policy_revision="different-policy",
    )
    wrong_policy_score = _score_sidecar(wrong_policy, probe, episode_offset=1.0)
    with pytest.raises(FeaturePipelineError, match="cohort differs"):
        build_locked_test_features([wrong_policy], [wrong_policy_score], probe, bundle)

    wrong_source_score = _score_sidecar(
        test_raws[0],
        probe,
        episode_offset=1.0,
        code_sha256=_sha("different-scoring-source"),
    )
    with pytest.raises(FeaturePipelineError, match="cohort differs"):
        build_locked_test_features(test_raws, [wrong_source_score], probe, bundle)


def test_one_to_one_duplicates_and_split_fail_before_reduction() -> None:
    probe = _probe()
    raws, scores = _paired_cohort(split="calibration", probe=probe)
    with pytest.raises(FeaturePipelineError, match="one-to-one"):
        build_calibration_features(raws, scores[:-1], probe)
    with pytest.raises(FeaturePipelineError, match="duplicate episode"):
        build_calibration_features([*raws, raws[0]], scores, probe)
    locked_raws, locked_scores = _paired_cohort(
        split="locked_test", probe=probe, successes=(False,), base_inits=(30,)
    )
    with pytest.raises(FeaturePipelineError, match="must all be calibration"):
        build_calibration_features(locked_raws, locked_scores, probe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw_metadata_link", "metadata hash link"),
        ("raw_trajectory_link", "trajectory hash link"),
        ("probe_link", "different probe"),
        ("stride", "stride control steps"),
        ("width", "probe width"),
        ("outcome", "factual outcomes"),
        ("transform_nan", "finiteness"),
        ("noise_seed", "noise seeds"),
    ],
)
def test_linkage_state_and_tamper_mismatches_fail_closed(
    mutation: str, message: str
) -> None:
    probe = _probe()
    raws, scores = _paired_cohort(split="calibration", probe=probe)
    score = scores[0]
    metadata = {
        **score.metadata,
        "links": dict(score.metadata["links"]),
        "capture": dict(score.metadata["capture"]),
        "factual_outcome": dict(score.metadata["factual_outcome"]),
    }
    arrays = {name: np.array(value, copy=True) for name, value in score.arrays.items()}
    if mutation == "raw_metadata_link":
        metadata["links"]["raw_metadata_sha256"] = _sha("stale")
    elif mutation == "raw_trajectory_link":
        metadata["links"]["raw_trajectory_sha256"] = _sha("stale")
    elif mutation == "probe_link":
        metadata["links"]["probe_sha256"] = _sha("wrong-probe")
    elif mutation == "stride":
        arrays["control_step"][1] = 4
    elif mutation == "width":
        metadata["capture"]["selected_activation_width"] = 3
    elif mutation == "outcome":
        metadata["factual_outcome"]["success"] = not raws[0].success
    elif mutation == "transform_nan":
        arrays["transformed_activation"][0, 0, 0, 0] = np.nan
    elif mutation == "noise_seed":
        arrays["noise_seed"][0, 0] += 1
    tampered = LoadedScoringSidecar(
        score.path,
        metadata,
        _readonly_arrays(arrays),
        score.metadata_sha256,
        score.primitives_sha256,
    )
    with pytest.raises(FeaturePipelineError, match=message):
        build_calibration_features(raws, [tampered, *scores[1:]], probe)


def test_invalid_rollout_and_bad_intervention_mask_fail_closed() -> None:
    probe = _probe()
    invalid = _raw_artifact(
        episode_id="calibration-invalid",
        split="calibration",
        base_init=10,
        success=False,
        episode_offset=1.0,
        valid=False,
    )
    invalid_score = _score_sidecar(invalid, probe, episode_offset=1.0)
    with pytest.raises(FeaturePipelineError, match="invalid or empty"):
        build_calibration_features([invalid], [invalid_score], probe)

    raws, scores = _paired_cohort(split="calibration", probe=probe)
    arrays = {
        name: np.array(value, copy=True) for name, value in scores[0].arrays.items()
    }
    arrays["intervention_available"][0, 0] = False
    tampered = replace(scores[0], arrays=_readonly_arrays(arrays))
    with pytest.raises(FeaturePipelineError, match="finiteness"):
        build_calibration_features(raws, [tampered, *scores[1:]], probe)
