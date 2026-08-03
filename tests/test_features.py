from __future__ import annotations

import math

import numpy as np
import pytest

from mech_int_vla.features import (
    COVERAGE_FEATURE_NAMES,
    COVERAGE_VECTOR_NAMES,
    INTERVENTION_ORDER,
    M0_FEATURE_NAMES,
    M1_FEATURE_NAMES,
    M1_RAW_FEATURE_NAMES,
    M2_EXPERT_FEATURE_NAMES,
    M2_EXPERT_NOISE_FEATURE_NAME,
    M2_INTERVENTION_FEATURE_NAMES,
    PROBE_NORM_FLOOR,
    TRANSFORM_ORDER,
    ActionScale,
    CoverageState,
    FeatureValidationError,
    M0Primitives,
    M1PoseState,
    M2Primitives,
    NamedFeatureRow,
    ProbeNormState,
    assemble_feature_hierarchy,
    canonical_feature_metadata_sha256,
    compute_coverage_features,
    compute_m0_features,
    compute_m2_features,
    construct_m1_raw_pose,
    fit_coverage_reference,
    fit_episode_equal_action_scale,
    query_coverage_features,
    robust_probe_norm_distance,
)


def _scale(values: np.ndarray | None = None) -> ActionScale:
    return ActionScale(
        np.ones(7) if values is None else values,
        np.zeros(7, dtype=np.bool_),
        ("calibration-episode",),
    )


def _yaw_wxyz(angle: float) -> np.ndarray:
    return np.asarray([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _yaw_xyzw(angle: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def _rpy_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _pose(*, goal: bool = True, phase: str = "pregrasp") -> NamedFeatureRow:
    state = M1PoseState.from_arrays(
        eef_position=[1.0, 2.0, 3.0],
        eef_quaternion_xyzw=_yaw_xyzw(math.pi / 2),
        object_position=[2.0, 4.0, 6.0],
        object_quaternion_wxyz=-_yaw_wxyz(math.pi),
        goal_position=[4.0, 7.0, 10.0] if goal else None,
        goal_quaternion_wxyz=_yaw_wxyz(3 * math.pi / 2) if goal else None,
        gripper_qpos=[0.02, 0.04],
        primary_contact=True,
        backend_grasp=False,
        normalized_step=0.25,
        phase=phase,
        symmetry_order=2,
    )
    return construct_m1_raw_pose(state)


def _raw_with_coverage_value(value: float, phase: str = "pregrasp") -> NamedFeatureRow:
    raw = _pose(phase=phase)
    values = raw.values.copy()
    for name in COVERAGE_VECTOR_NAMES:
        values[M1_RAW_FEATURE_NAMES.index(name)] = 0.0
    values[M1_RAW_FEATURE_NAMES.index("m1_object_minus_eef_x")] = value
    for candidate_phase in ("pregrasp", "grasped", "transport", "placed"):
        values[M1_RAW_FEATURE_NAMES.index(f"m1_phase_{candidate_phase}")] = float(
            candidate_phase == phase
        )
    return NamedFeatureRow(M1_RAW_FEATURE_NAMES, values, raw.metadata)


def _coverage_state(
    index: int,
    *,
    value: float = 0.0,
    success: bool = True,
    phase: str = "pregrasp",
    split: str = "calibration",
    base_init: int | None = None,
    episode_id: str | None = None,
) -> CoverageState:
    return CoverageState.create(
        episode_id=episode_id or f"ep{index:02d}",
        base_init_state_id=index if base_init is None else base_init,
        control_step=0,
        split=split,
        phase=phase,
        success=success,
        raw_pose=_raw_with_coverage_value(value, phase),
    )


def test_episode_equal_action_scale_uses_population_moments_and_floor() -> None:
    actions = np.zeros((3, 8, 10, 7))
    actions[1, ..., 0] = 2.0
    actions[2, ..., 0] = 4.0
    actions[..., 1] = 9.0
    result = fit_episode_equal_action_scale(actions, ["a", "b", "b"])

    # Episode A has moments (0,0), episode B has moments (3,10).
    assert result.values[0] == pytest.approx(math.sqrt(5.0 - 1.5**2))
    assert result.values[1] == 1.0
    assert result.replaced_by_one.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert result.episode_ids == ("a", "b")
    assert len(result.metadata_sha256) == 64


def test_action_scale_rejects_nonfinite_shape_and_empty_episode_id() -> None:
    actions = np.zeros((1, 8, 10, 7))
    with pytest.raises(FeatureValidationError, match="shape"):
        fit_episode_equal_action_scale(actions[:, :, :9], ["a"])
    actions[0, 0, 0, 0] = np.nan
    with pytest.raises(FeatureValidationError, match="finite"):
        fit_episode_equal_action_scale(actions, ["a"])
    actions[0, 0, 0, 0] = 0.0
    with pytest.raises(FeatureValidationError, match="nonempty"):
        fit_episode_equal_action_scale(actions, [""])


def test_m0_fixed_reductions_and_feature_order() -> None:
    original = np.zeros((8, 10, 7))
    for draw in range(8):
        original[draw, :, 0] = draw
    transformed = np.empty((6, 4, 10, 7))
    for transform, offset in enumerate((1, 2, 3, 4, 5, 6)):
        transformed[transform] = original[:4] + offset
    row = compute_m0_features(
        M0Primitives.from_arrays(
            original_actions=original,
            transformed_actions=transformed,
            transformed_available=np.ones(6, dtype=np.bool_),
        ),
        _scale(),
    )

    assert row.names == M0_FEATURE_NAMES
    assert row.values[:6] == pytest.approx([1.5, 2.0, 3.5, 4.0, 5.5, 6.0])
    assert row.values[6:8] == pytest.approx([1.5, 3.5])
    assert row.values[8] == pytest.approx(52.5)
    pairwise = np.asarray(
        [
            abs(first - second) / math.sqrt(7)
            for first in range(8)
            for second in range(first + 1, 8)
        ]
    )
    assert row.values[9] == pytest.approx(np.mean(pairwise))
    assert row.values[10] == pytest.approx(np.percentile(pairwise, 90, method="linear"))
    assert row.values[11] == pytest.approx(3.5 / math.sqrt(7))
    assert row.values[12] == 0.0


def test_m0_masks_are_fail_closed_and_primitives_are_immutable() -> None:
    original = np.zeros((8, 10, 7))
    transformed = np.zeros((6, 4, 10, 7))
    transformed[1] = np.nan
    primitives = M0Primitives.from_arrays(
        original_actions=original,
        transformed_actions=transformed,
        transformed_available=[True, False, True, True, True, True],
    )
    original[0, 0, 0] = 123.0
    assert primitives.original_actions[0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        primitives.original_actions[0, 0, 0] = 1.0
    row = compute_m0_features(primitives, _scale())
    assert np.isnan(row.values[[0, 1, 6]]).all()
    assert np.isfinite(row.values[[2, 3, 4, 5, 7]]).all()
    with pytest.raises(FeatureValidationError, match="only NaN"):
        M0Primitives.from_arrays(
            original_actions=np.zeros((8, 10, 7)),
            transformed_actions=np.zeros((6, 4, 10, 7)),
            transformed_available=[False, True, True, True, True, True],
        )
    with pytest.raises(FeatureValidationError, match="frozen order"):
        M0Primitives.from_arrays(
            original_actions=np.zeros((8, 10, 7)),
            transformed_actions=np.zeros((6, 4, 10, 7)),
            transformed_available=np.ones(6, dtype=np.bool_),
            transform_order=tuple(reversed(TRANSFORM_ORDER)),
        )


def test_m1_pose_coordinates_quaternion_conventions_and_schema() -> None:
    row = _pose()
    values = row.as_dict()
    assert row.names == M1_RAW_FEATURE_NAMES
    assert row.values[:3] == pytest.approx([1.0, 2.0, 3.0])
    assert row.values[3:6] == pytest.approx([2.0, 3.0, 4.0])
    expected_relative = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    assert row.values[6:10] == pytest.approx(expected_relative)
    assert row.values[10:14] == pytest.approx(expected_relative)
    assert values["m1_symmetry_eef_object_yaw_sin"] == pytest.approx(0.0, abs=1e-12)
    assert values["m1_symmetry_eef_object_yaw_cos"] == pytest.approx(-1.0)
    assert values["m1_symmetry_object_goal_yaw_sin"] == pytest.approx(0.0, abs=1e-12)
    assert values["m1_symmetry_object_goal_yaw_cos"] == pytest.approx(-1.0)
    assert values["m1_mean_two_finger_opening"] == pytest.approx(0.03)
    assert row.values[-4:].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_m1_missing_goal_is_explicit_and_bad_quaternions_fail() -> None:
    row = _pose(goal=False)
    assert np.isnan(row.values[3:6]).all()
    assert np.isnan(row.values[10:14]).all()
    assert np.isnan(row.values[16:18]).all()
    assert row.as_dict()["m1_goal_present"] == 0.0
    with pytest.raises(FeatureValidationError, match="both be present"):
        M1PoseState.from_arrays(
            eef_position=[0, 0, 0],
            eef_quaternion_xyzw=[0, 0, 0, 1],
            object_position=[0, 0, 0],
            object_quaternion_wxyz=[1, 0, 0, 0],
            goal_position=[0, 0, 0],
            goal_quaternion_wxyz=None,
            gripper_qpos=[0, 0],
            primary_contact=False,
            backend_grasp=False,
            normalized_step=0,
            phase="pregrasp",
            symmetry_order=1,
        )
    with pytest.raises(FeatureValidationError, match="unit length"):
        M1PoseState.from_arrays(
            eef_position=[0, 0, 0],
            eef_quaternion_xyzw=[0, 0, 0, 2],
            object_position=[0, 0, 0],
            object_quaternion_wxyz=[1, 0, 0, 0],
            goal_position=None,
            goal_quaternion_wxyz=None,
            gripper_qpos=[0, 0],
            primary_contact=False,
            backend_grasp=False,
            normalized_step=0,
            phase="pregrasp",
            symmetry_order=1,
        )


def test_m1_symmetry_yaw_comes_from_relative_quaternion_with_roll_pitch() -> None:
    eef_wxyz = _rpy_wxyz(0.4, 0.5, 0.7)
    object_wxyz = _rpy_wxyz(-0.2, 0.3, -0.6)
    goal_wxyz = _rpy_wxyz(0.1, -0.4, 1.1)
    row = construct_m1_raw_pose(
        M1PoseState.from_arrays(
            eef_position=[0, 0, 0],
            eef_quaternion_xyzw=[
                eef_wxyz[1],
                eef_wxyz[2],
                eef_wxyz[3],
                eef_wxyz[0],
            ],
            object_position=[1, 0, 0],
            object_quaternion_wxyz=object_wxyz,
            goal_position=[2, 0, 0],
            goal_quaternion_wxyz=goal_wxyz,
            gripper_qpos=[0, 0],
            primary_contact=False,
            backend_grasp=False,
            normalized_step=0.5,
            phase="transport",
            symmetry_order=1,
        )
    )
    q_eef_object = row.values[6:10]
    q_object_goal = row.values[10:14]

    def quaternion_yaw(quaternion: np.ndarray) -> float:
        w, x, y, z = quaternion
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    relative_yaw = quaternion_yaw(q_eef_object)
    relative_goal_yaw = quaternion_yaw(q_object_goal)
    assert row.values[14:16] == pytest.approx(
        [math.sin(relative_yaw), math.cos(relative_yaw)]
    )
    assert row.values[16:18] == pytest.approx(
        [math.sin(relative_goal_yaw), math.cos(relative_goal_yaw)]
    )
    # With nonzero roll/pitch, relative-frame yaw is not world Euler-yaw subtraction.
    assert abs(relative_yaw - (-0.6 - 0.7)) > 0.05
    assert abs(relative_goal_yaw - (1.1 - -0.6)) > 0.05


def test_coverage_group_exclusions_neighbor_counts_and_deterministic_ties() -> None:
    references = [
        _coverage_state(
            index,
            value=0.0,
            success=index >= 10,
            base_init=0 if index in {0, 1} else index,
        )
        for index in range(30)
    ]
    query = _coverage_state(
        99,
        value=0.0,
        split="calibration",
        base_init=0,
        episode_id="ep00",
    )
    fit = fit_coverage_reference(
        references,
        exclude_episode_id=query.episode_id,
        exclude_base_init_state_id=query.base_init_state_id,
    )
    assert all(reference.base_init_state_id != 0 for reference in fit.references)
    row = query_coverage_features(query, fit)
    assert row.names == COVERAGE_FEATURE_NAMES
    assert row.values[0] == 0.0
    assert row.values[1] == 0.0
    # ep02..ep26 are the deterministic first 25 ties: ep02..ep09 fail (8/25).
    assert row.values[2] == pytest.approx(8 / 25)
    assert fit.scale_replaced_by_one.all()


def test_coverage_locked_test_uses_full_fit_and_missing_rows_fail_closed() -> None:
    references = [_coverage_state(index, value=float(index)) for index in range(25)]
    locked_query = _coverage_state(99, value=0.0, split="locked_test")
    full_fit = fit_coverage_reference(references)
    assert np.isfinite(query_coverage_features(locked_query, full_fit).values).all()
    excluded_fit = fit_coverage_reference(
        references, exclude_episode_id="ep00", exclude_base_init_state_id=0
    )
    with pytest.raises(FeatureValidationError, match="full Calibration"):
        query_coverage_features(locked_query, excluded_fit)

    missing = CoverageState(
        "locked-missing",
        99,
        0,
        "locked_test",
        "pregrasp",
        False,
        np.full(len(COVERAGE_VECTOR_NAMES), np.nan),
    )
    assert np.isnan(query_coverage_features(missing, full_fit).values).all()
    references_with_missing = [
        *references,
        CoverageState(
            "missing-reference",
            100,
            0,
            "calibration",
            "pregrasp",
            True,
            np.full(len(COVERAGE_VECTOR_NAMES), np.nan),
        ),
    ]
    assert len(fit_coverage_reference(references_with_missing).references) == 25


def test_coverage_insufficient_neighbor_families_are_nan() -> None:
    references = [
        _coverage_state(index, value=float(index), success=index < 4)
        for index in range(24)
    ]
    query = _coverage_state(99, value=0.0, split="locked_test")
    row = compute_coverage_features(query, references)
    assert np.isnan(row.values[0])
    assert np.isfinite(row.values[1])
    assert np.isnan(row.values[2])


def test_coverage_fit_hash_commits_to_reference_outcome_and_vector() -> None:
    references = [_coverage_state(index, value=float(index)) for index in range(6)]
    baseline = fit_coverage_reference(references)
    reversed_fit = fit_coverage_reference(tuple(reversed(references)))
    assert np.array_equal(baseline.mean, reversed_fit.mean)
    assert np.array_equal(baseline.scale, reversed_fit.scale)
    assert baseline.reference_records_sha256 == reversed_fit.reference_records_sha256
    assert baseline.metadata_sha256 == reversed_fit.metadata_sha256

    first = references[0]
    changed_outcome = [
        CoverageState(
            first.episode_id,
            first.base_init_state_id,
            first.control_step,
            first.split,
            first.phase,
            not first.success,
            first.vector,
        ),
        *references[1:],
    ]
    outcome_fit = fit_coverage_reference(changed_outcome)
    assert baseline.reference_records_sha256 != outcome_fit.reference_records_sha256
    assert baseline.metadata_sha256 != outcome_fit.metadata_sha256

    changed_vector = first.vector.copy()
    changed_vector[0] = np.nextafter(changed_vector[0], math.inf)
    vector_references = [
        CoverageState(
            first.episode_id,
            first.base_init_state_id,
            first.control_step,
            first.split,
            first.phase,
            first.success,
            changed_vector,
        ),
        *references[1:],
    ]
    vector_fit = fit_coverage_reference(vector_references)
    assert baseline.reference_records_sha256 != vector_fit.reference_records_sha256
    assert baseline.metadata_sha256 != vector_fit.metadata_sha256


def _m2_primitives(*, noise_dependent: bool, object_error: float = 0.0) -> M2Primitives:
    original = np.tile(np.asarray([2.0, 0.0]), (8, 1))
    transformed = np.tile(np.asarray([2.0, 0.0]), (6, 4, 1))
    for index, delta in enumerate((math.radians(-15), math.radians(15))):
        angle = -2 * delta + object_error
        transformed[4 + index, :, 0] = 2 * math.cos(angle)
        transformed[4 + index, :, 1] = 2 * math.sin(angle)
    intervention = np.zeros((2, 4, 10, 7))
    intervention[0, ..., 0] = -2.0
    intervention[1, ..., 0] = 2.0
    intervention[0, ..., 5] = -1.0
    intervention[1, ..., 5] = 1.0
    return M2Primitives.from_arrays(
        original_probe_vectors=original,
        transformed_probe_vectors=transformed,
        transformed_available=np.ones(6, dtype=np.bool_),
        intervention_actions=intervention,
        intervention_available=np.ones(4, dtype=np.bool_),
        symmetry_order=2,
        noise_dependent=noise_dependent,
    )


def test_m2_exact_circle_norm_and_central_difference_summaries() -> None:
    row = compute_m2_features(
        _m2_primitives(noise_dependent=True),
        _scale(),
        robust_norm_distance=3.0,
    )
    assert M2_EXPERT_NOISE_FEATURE_NAME in row.names
    assert row.values[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)
    assert row.values[3:6] == pytest.approx([2.0, 3.0, 0.0])
    assert row.values[6] == pytest.approx(2.0 / math.radians(20))
    assert row.values[7] == pytest.approx(2.0)

    physical_error = compute_m2_features(
        _m2_primitives(noise_dependent=False, object_error=0.1),
        _scale(),
        robust_norm_distance=0.0,
    )
    assert M2_EXPERT_NOISE_FEATURE_NAME not in physical_error.names
    assert physical_error.values[0] == pytest.approx(0.05)


def test_m2_zero_vectors_unavailable_masks_and_zero_yaw_are_nan() -> None:
    primitives = _m2_primitives(noise_dependent=True)
    original = primitives.original_probe_vectors.copy()
    original[0] = 0.0
    zero_probe = M2Primitives.from_arrays(
        original_probe_vectors=original,
        transformed_probe_vectors=primitives.transformed_probe_vectors,
        transformed_available=primitives.transformed_available,
        intervention_actions=primitives.intervention_actions,
        intervention_available=np.ones(4, dtype=np.bool_),
        symmetry_order=2,
        noise_dependent=True,
    )
    row = compute_m2_features(zero_probe, _scale(), robust_norm_distance=np.nan)
    assert np.isnan(row.values[[0, 1, 2, 4, 5, 6, 7]]).all()

    unavailable = primitives.transformed_probe_vectors.copy()
    unavailable[2] = np.nan
    masked = M2Primitives.from_arrays(
        original_probe_vectors=primitives.original_probe_vectors,
        transformed_probe_vectors=unavailable,
        transformed_available=[True, True, False, True, True, True],
        intervention_actions=primitives.intervention_actions,
        intervention_available=np.ones(4, dtype=np.bool_),
        symmetry_order=2,
        noise_dependent=False,
    )
    assert np.isnan(
        compute_m2_features(masked, _scale(), robust_norm_distance=0).as_dict()[
            "m2_camera_probe_circular_dispersion"
        ]
    )

    no_yaw = primitives.intervention_actions.copy()
    no_yaw[..., 5] = 0.0
    zero_denominator = M2Primitives.from_arrays(
        original_probe_vectors=primitives.original_probe_vectors,
        transformed_probe_vectors=primitives.transformed_probe_vectors,
        transformed_available=primitives.transformed_available,
        intervention_actions=no_yaw,
        intervention_available=np.ones(4, dtype=np.bool_),
        symmetry_order=2,
        noise_dependent=False,
        intervention_order=INTERVENTION_ORDER,
    )
    result = compute_m2_features(zero_denominator, _scale(), robust_norm_distance=0)
    assert result.as_dict()["m2_probe_controllability_gain_per_rad"] == 0.0
    assert np.isnan(result.as_dict()["m2_probe_intervention_specificity_ratio"])


def test_m2_probe_norm_floor_boundary_is_frozen() -> None:
    baseline = _m2_primitives(noise_dependent=True)
    at_floor_vectors = baseline.original_probe_vectors.copy()
    at_floor_vectors[0] = [PROBE_NORM_FLOOR, 0.0]
    at_floor = M2Primitives.from_arrays(
        original_probe_vectors=at_floor_vectors,
        transformed_probe_vectors=baseline.transformed_probe_vectors,
        transformed_available=baseline.transformed_available,
        intervention_actions=baseline.intervention_actions,
        intervention_available=baseline.intervention_available,
        symmetry_order=2,
        noise_dependent=True,
    )
    at_floor_result = compute_m2_features(at_floor, _scale(), robust_norm_distance=0)
    assert np.isnan(at_floor_result.values[[0, 1, 2, 5, 6, 7]]).all()

    above_floor_vectors = baseline.original_probe_vectors.copy()
    above_floor_vectors[0] = [
        np.nextafter(PROBE_NORM_FLOOR, math.inf),
        0.0,
    ]
    above_floor = M2Primitives.from_arrays(
        original_probe_vectors=above_floor_vectors,
        transformed_probe_vectors=baseline.transformed_probe_vectors,
        transformed_available=baseline.transformed_available,
        intervention_actions=baseline.intervention_actions,
        intervention_available=baseline.intervention_available,
        symmetry_order=2,
        noise_dependent=True,
    )
    above_floor_result = compute_m2_features(
        above_floor, _scale(), robust_norm_distance=0
    )
    assert np.isfinite(above_floor_result.values).all()


def test_m2_intervention_draw_mask_is_exact_immutable_and_conservative() -> None:
    complete = _m2_primitives(noise_dependent=False)
    assert complete.intervention_available.tolist() == [True, True, True, True]
    with pytest.raises(ValueError):
        complete.intervention_available[0] = False

    mixed_actions = complete.intervention_actions.copy()
    mixed_actions[:, 2] = np.nan
    mixed_mask = np.asarray([True, True, False, True])
    mixed = M2Primitives.from_arrays(
        original_probe_vectors=complete.original_probe_vectors,
        transformed_probe_vectors=complete.transformed_probe_vectors,
        transformed_available=complete.transformed_available,
        intervention_actions=mixed_actions,
        intervention_available=mixed_mask,
        symmetry_order=2,
        noise_dependent=False,
    )
    mixed_mask[:] = True
    assert mixed.intervention_available.tolist() == [True, True, False, True]
    mixed_result = compute_m2_features(mixed, _scale(), robust_norm_distance=0)
    assert np.isnan(mixed_result.values[-len(M2_INTERVENTION_FEATURE_NAMES) :]).all()

    unavailable = M2Primitives.from_arrays(
        original_probe_vectors=complete.original_probe_vectors,
        transformed_probe_vectors=complete.transformed_probe_vectors,
        transformed_available=complete.transformed_available,
        intervention_actions=np.full((2, 4, 10, 7), np.nan),
        intervention_available=np.zeros(4, dtype=np.bool_),
        symmetry_order=2,
        noise_dependent=False,
    )
    unavailable_result = compute_m2_features(
        unavailable, _scale(), robust_norm_distance=0
    )
    assert np.isnan(
        unavailable_result.values[-len(M2_INTERVENTION_FEATURE_NAMES) :]
    ).all()

    with pytest.raises(FeatureValidationError, match="shape"):
        M2Primitives.from_arrays(
            original_probe_vectors=complete.original_probe_vectors,
            transformed_probe_vectors=complete.transformed_probe_vectors,
            transformed_available=complete.transformed_available,
            intervention_actions=complete.intervention_actions,
            intervention_available=True,
            symmetry_order=2,
            noise_dependent=False,
        )
    with pytest.raises(FeatureValidationError, match="boolean dtype"):
        M2Primitives.from_arrays(
            original_probe_vectors=complete.original_probe_vectors,
            transformed_probe_vectors=complete.transformed_probe_vectors,
            transformed_available=complete.transformed_available,
            intervention_actions=complete.intervention_actions,
            intervention_available=np.ones(4, dtype=np.int64),
            symmetry_order=2,
            noise_dependent=False,
        )
    with pytest.raises(FeatureValidationError, match="unavailable.*NaN"):
        M2Primitives.from_arrays(
            original_probe_vectors=complete.original_probe_vectors,
            transformed_probe_vectors=complete.transformed_probe_vectors,
            transformed_available=complete.transformed_available,
            intervention_actions=complete.intervention_actions,
            intervention_available=np.asarray([True, True, False, True]),
            symmetry_order=2,
            noise_dependent=False,
        )


def test_robust_probe_norm_distance_oof_floor_and_minimum() -> None:
    references = [
        ProbeNormState(f"ep{index}", index, 0, "calibration", "pregrasp", True, value)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0))
    ]
    query = ProbeNormState("query", 99, 0, "calibration", "pregrasp", False, 6.0)
    assert robust_probe_norm_distance(query, references) == pytest.approx(3 / 1.4826)
    assert math.isnan(robust_probe_norm_distance(query, references[:4]))

    constant = [
        ProbeNormState(f"const{index}", index, 0, "calibration", "pregrasp", True, 2.0)
        for index in range(5)
    ]
    constant_query = ProbeNormState(
        "locked", 99, 0, "locked_test", "pregrasp", False, 2.0 + 1e-8
    )
    assert robust_probe_norm_distance(constant_query, constant) == pytest.approx(1.0)

    leaked = [
        *references,
        ProbeNormState("query", 100, 5, "calibration", "pregrasp", True, 1e6),
        ProbeNormState("same-base", 99, 0, "calibration", "pregrasp", True, 1e6),
    ]
    assert robust_probe_norm_distance(query, leaked) == pytest.approx(3 / 1.4826)


def test_canonical_schema_hashes_are_stable_and_component_specific() -> None:
    first = canonical_feature_metadata_sha256("M0")
    second = canonical_feature_metadata_sha256("M0")
    assert first == second
    assert len(first) == 64
    assert first != canonical_feature_metadata_sha256("M2_vlm")
    with pytest.raises(FeatureValidationError, match="unknown"):
        canonical_feature_metadata_sha256("future-unfrozen-component")


def test_feature_hierarchy_freezes_append_order_and_shared_values() -> None:
    original = np.zeros((8, 10, 7))
    m0 = compute_m0_features(
        M0Primitives.from_arrays(
            original_actions=original,
            transformed_actions=np.zeros((6, 4, 10, 7)),
            transformed_available=np.ones(6, dtype=np.bool_),
        ),
        _scale(),
    )
    raw = _pose()
    coverage = NamedFeatureRow(
        COVERAGE_FEATURE_NAMES,
        np.asarray([1.0, 2.0, 0.25]),
        {"schema_version": 1, "component": "test-coverage"},
    )
    m2 = compute_m2_features(
        _m2_primitives(noise_dependent=True),
        _scale(),
        robust_norm_distance=0.0,
    )
    hierarchy = assemble_feature_hierarchy(m0, raw, coverage, m2)
    assert hierarchy.m1.names == M1_FEATURE_NAMES
    assert hierarchy.m2.names == M2_EXPERT_FEATURE_NAMES
    assert np.array_equal(
        hierarchy.m0.values,
        hierarchy.m1.values[: len(M0_FEATURE_NAMES)],
        equal_nan=True,
    )
    assert np.array_equal(
        hierarchy.m1.values,
        hierarchy.m2.values[: len(M1_FEATURE_NAMES)],
        equal_nan=True,
    )
    assert len(hierarchy.metadata_sha256) == 64
