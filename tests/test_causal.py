from __future__ import annotations

import math

import numpy as np
import pytest

from mech_int_vla.causal import (
    CandidateState,
    CausalAnalysisError,
    cluster_bootstrap_rate_interval,
    compare_random_controls,
    evaluate_confirmatory_causal_claim,
    five_nearest_neighbor_distance,
    iter_norm_matched_random_shifts,
    norm_matched_random_subspaces,
    off_manifold_flag,
    orthogonal_probe_projector,
    pair_eligibility,
    patch_activation,
    probe_patch_shift,
    select_pairs,
    select_pairs_for_three_seeds,
    summarize_action_effect,
    symmetry_aware_orientation_difference,
)


def state(
    candidate_id: str,
    *,
    angle_deg: float,
    phase: str = "pregrasp",
    contact: bool = False,
    gripper: float = 0.5,
    eef: tuple[float, float, float] = (0.0, 0.0, 0.0),
    obj: tuple[float, float, float] = (0.0, 0.0, 0.0),
    time: float = 0.2,
    predicates: dict[str, bool] | None = None,
    symmetry: int = 1,
) -> CandidateState:
    return CandidateState.create(
        candidate_id=candidate_id,
        base_init_id=int(candidate_id.lstrip("s") or 0),
        phase=phase,
        contact=contact,
        gripper_opening=gripper,
        eef_position=eef,
        object_position=obj,
        normalized_time=time,
        non_primary_predicates=predicates or {"drawer_open": False},
        orientation_rad=math.radians(angle_deg),
        symmetry_order=symmetry,
    )


def test_projector_and_patch_use_probe_row_space_without_mutation() -> None:
    coefficient = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    recipient = np.array([1.0, 1.0, 5.0])
    donor = np.array([3.0, -1.0, 100.0])
    projector = orthogonal_probe_projector(coefficient)
    np.testing.assert_allclose(projector, np.diag([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(projector @ projector, projector)
    np.testing.assert_allclose(
        probe_patch_shift(recipient, donor, coefficient, alpha=0.5), [1, -1, 0]
    )
    np.testing.assert_allclose(
        patch_activation(recipient, donor, coefficient, alpha=0.5), [2, 0, 5]
    )
    np.testing.assert_array_equal(recipient, [1, 1, 5])
    assert not projector.flags.writeable


def test_projector_fails_closed_unless_probe_row_space_has_rank_two() -> None:
    with pytest.raises(CausalAnalysisError, match="rank exactly 2, got 1"):
        orthogonal_probe_projector([[1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(CausalAnalysisError, match="rank exactly 2, got 0"):
        orthogonal_probe_projector(np.zeros((2, 3)))


def test_random_subspaces_are_seeded_rank_two_and_shift_norm_matched() -> None:
    coefficient = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    difference = np.array([3.0, 4.0, 5.0, 6.0])
    for alpha in (0.25, 0.5):
        first = norm_matched_random_subspaces(
            coefficient, difference, seed=19, alpha=alpha, count=4
        )
        second = norm_matched_random_subspaces(
            coefficient, difference, seed=19, alpha=alpha, count=4
        )
        assert len(first) == 4
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left.projector, right.projector)
            np.testing.assert_array_equal(left.matched_shift, right.matched_shift)
            np.testing.assert_allclose(
                left.projector @ left.projector, left.projector, atol=1e-12
            )
            assert np.linalg.matrix_rank(left.projector, tol=1e-10) == 2
            assert left.alpha == alpha
            assert left.target_projected_norm == pytest.approx(5.0)
            assert left.matched_patch_norm == pytest.approx(alpha * 5.0)
            assert np.linalg.norm(left.matched_shift) == pytest.approx(alpha * 5.0)
            assert not left.matched_shift.flags.writeable
    with pytest.raises(CausalAnalysisError, match="frozen values"):
        norm_matched_random_subspaces(
            coefficient, difference, seed=19, alpha=0.75, count=1
        )


def test_streaming_random_shifts_equal_dense_projector_controls() -> None:
    coefficient = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    difference = np.array([3.0, 4.0, 5.0, 6.0])
    dense = norm_matched_random_subspaces(
        coefficient, difference, seed=23, alpha=0.25, count=5
    )
    streaming = tuple(
        iter_norm_matched_random_shifts(
            coefficient, difference, seed=23, alpha=0.25, count=5
        )
    )
    for expected, observed in zip(dense, streaming, strict=True):
        np.testing.assert_allclose(observed.matched_shift, expected.matched_shift)
        np.testing.assert_allclose(
            observed.orthonormal_basis.T @ observed.orthonormal_basis,
            np.eye(2),
            atol=1e-12,
        )
        assert observed.raw_projected_norm == pytest.approx(
            expected.raw_projected_norm
        )
        assert not observed.orthonormal_basis.flags.writeable


def test_candidate_defensively_copies_and_canonicalizes_predicates() -> None:
    predicates = {"z": True, "a": False}
    candidate = CandidateState.create(
        candidate_id="s1",
        base_init_id=1,
        phase="grasped",
        contact=True,
        gripper_opening=0.4,
        eef_position=[1, 2, 3],
        object_position=[4, 5, 6],
        normalized_time=0.5,
        non_primary_predicates=predicates,
        orientation_rad=0.2,
        symmetry_order=2,
    )
    predicates["a"] = True
    assert candidate.non_primary_predicates == (("a", False), ("z", True))
    with pytest.raises(CausalAnalysisError, match="sorted order"):
        CandidateState(
            "bad",
            1,
            "phase",
            False,
            0.0,
            (0, 0, 0),
            (0, 0, 0),
            0.0,
            (("z", False), ("a", True)),
            0.0,
            1,
        )
    with pytest.raises(CausalAnalysisError, match="positive integer"):
        CandidateState.create(
            candidate_id="invalid-symmetry",
            base_init_id=1,
            phase="phase",
            contact=False,
            gripper_opening=0.0,
            eef_position=(0, 0, 0),
            object_position=(0, 0, 0),
            normalized_time=0.0,
            non_primary_predicates={},
            orientation_rad=0.0,
            symmetry_order=True,
        )


def test_symmetry_aware_orientation_and_exact_eligibility_boundaries() -> None:
    assert math.degrees(
        symmetry_aware_orientation_difference(
            math.radians(170), math.radians(-170), symmetry_order=1
        )
    ) == pytest.approx(20.0)
    assert math.degrees(
        symmetry_aware_orientation_difference(
            math.radians(10), math.radians(100), symmetry_order=2
        )
    ) == pytest.approx(90.0)

    recipient = state("s1", angle_deg=0)
    at_30 = state(
        "s2", angle_deg=30, gripper=0.51, eef=(0.02, 0, 0), obj=(0, 0.02, 0), time=0.3
    )
    at_90 = state("s3", angle_deg=90)
    assert pair_eligibility(recipient, at_30).eligible
    assert pair_eligibility(recipient, at_90).eligible
    at_5 = state("s4", angle_deg=5)
    below_5 = state("s5", angle_deg=4.999)
    assert not pair_eligibility(recipient, at_5, mode="matched_control").eligible
    assert pair_eligibility(recipient, below_5, mode="matched_control").eligible


def test_pair_eligibility_exposes_every_failed_constraint() -> None:
    recipient = state("s1", angle_deg=0)
    donor = state(
        "s2",
        angle_deg=20,
        phase="placed",
        contact=True,
        gripper=0.511,
        eef=(0.021, 0, 0),
        obj=(0, 0.021, 0),
        time=0.301,
        predicates={"drawer_open": True},
    )
    result = pair_eligibility(recipient, donor)
    assert not result.eligible
    assert set(result.reasons) == {
        "phase",
        "contact",
        "gripper_opening",
        "eef_position",
        "object_position",
        "normalized_time",
        "non_primary_predicates",
        "confirmatory_orientation",
    }


def test_pair_selection_is_order_independent_seeded_and_never_reuses_states() -> None:
    candidates = [
        state(f"s{index}", angle_deg=0 if index % 2 else 45, time=0.2 + index * 0.0001)
        for index in range(1, 13)
    ]
    forward = select_pairs(candidates, seed=7)
    reverse = select_pairs(list(reversed(candidates)), seed=7)
    assert forward == reverse
    used = [
        value for pair in forward.pairs for value in (pair.recipient_id, pair.donor_id)
    ]
    assert len(used) == len(set(used))
    assert len(forward.pairs) == 6
    assert forward.eligible_edges_not_selected == forward.eligible_edge_count - 6
    assert select_pairs(candidates, seed=8).pairs != forward.pairs


def test_pair_selection_caps_each_of_exactly_three_seeds() -> None:
    candidates = [
        state(f"s{index}", angle_deg=0 if index % 2 else 45) for index in range(1, 51)
    ]
    result = select_pairs_for_three_seeds(candidates, seeds=[11, 12, 13])
    assert [len(selection.pairs) for selection in result.selections] == [20, 20, 20]
    assert result.attempted_pairs == result.maximum_pairs == 60
    with pytest.raises(CausalAnalysisError, match="exactly three"):
        select_pairs_for_three_seeds(candidates, seeds=[1, 2])
    with pytest.raises(CausalAnalysisError, match="distinct"):
        select_pairs_for_three_seeds(candidates, seeds=[1, 1, 2])


def test_action_summary_uses_first_ten_yaw_sign_and_standardized_specificity() -> None:
    recipient = np.zeros((12, 7))
    donor = np.zeros((12, 7))
    patched = np.zeros((12, 7))
    donor[:10, 5] = 4.0
    patched[:10, 5] = 2.0
    patched[:10, 0] = 0.3
    patched[10:, 0] = 1000.0  # excluded from the fixed first-ten summary
    result = summarize_action_effect(
        recipient, donor, patched, action_scale=[2, 1, 1, 1, 1, 2, 1]
    )
    assert result.target_effect == pytest.approx(1.0)
    assert result.natural_target_effect == pytest.approx(2.0)
    assert result.donor_aligned_target_effect == pytest.approx(1.0)
    assert result.off_target_ratio == pytest.approx(0.15)
    assert result.temporal_yaw_dot_product == pytest.approx(8.0)
    assert result.sign_correct

    negative_donor = donor.copy()
    negative_patch = patched.copy()
    negative_donor[:10, 5] *= -1
    negative_patch[:10, 5] *= -1
    aligned = summarize_action_effect(
        recipient, negative_donor, negative_patch, action_scale=np.ones(7)
    )
    assert aligned.target_effect < 0
    assert aligned.donor_aligned_target_effect > 0
    assert aligned.sign_correct

    # Product of the separately averaged yaw shifts is positive, but the frozen
    # temporal dot product is negative and must decide sign correctness.
    formulation_recipient = np.zeros((10, 7))
    formulation_donor = np.zeros((10, 7))
    formulation_patch = np.zeros((10, 7))
    formulation_donor[:, 5] = [-1.0] * 9 + [10.0]
    formulation_patch[:, 5] = [1.0] * 9 + [-1.0]
    formulation = summarize_action_effect(
        formulation_recipient,
        formulation_donor,
        formulation_patch,
        action_scale=np.ones(7),
    )
    assert formulation.target_effect * formulation.natural_target_effect > 0
    assert formulation.temporal_yaw_dot_product < 0
    assert not formulation.sign_correct

    zero_target = summarize_action_effect(
        recipient, donor, np.zeros((12, 7)), action_scale=np.ones(7)
    )
    assert math.isinf(zero_target.off_target_ratio)
    assert not zero_target.sign_correct


def test_random_percentile_and_off_manifold_boundaries_are_strict() -> None:
    comparison = compare_random_controls(10.0, np.arange(10.0))
    assert comparison.exceeds_95th_percentile
    tie = compare_random_controls(8.55, np.arange(10.0))
    assert tie.percentile_95_threshold == pytest.approx(8.55)
    assert not tie.exceeds_95th_percentile

    reference = np.arange(6.0)[:, None]
    assert five_nearest_neighbor_distance([0.0], reference) == pytest.approx(2.0)
    assert not off_manifold_flag(
        patched_five_nn_distance=2.0, natural_95th_percentile=2.0
    ).off_manifold
    assert off_manifold_flag(
        patched_five_nn_distance=2.01, natural_95th_percentile=2.0
    ).off_manifold


def test_cluster_bootstrap_rate_is_deterministic_and_clustered() -> None:
    signs = np.array([True, True, False, False])
    clusters = ["a", "a", "b", "b"]
    first = cluster_bootstrap_rate_interval(signs, clusters, seed=5, replicates=500)
    second = cluster_bootstrap_rate_interval(
        signs[::-1], clusters[::-1], seed=5, replicates=500
    )
    assert first == second
    assert first.estimate == 0.5
    assert first.lower == 0.0
    assert first.upper == 1.0


def test_confirmatory_decision_succeeds_only_when_every_exact_criterion_passes() -> (
    None
):
    effects = np.full(60, 2.0)
    signs = np.ones(60, dtype=bool)
    clusters = np.repeat(np.arange(20), 3)
    seeds = np.tile(np.array([101, 102, 103]), 20)
    result = evaluate_confirmatory_causal_claim(
        effects,
        signs,
        clusters,
        seeds,
        random_control_effects=np.linspace(-1, 1, 1000),
        pair_off_target_ratios=np.full(60, 0.25),
        supporting_layer_effects={
            "vlm_context": 1.0,
            "early_expert": 1.0,
            "late_expert": -1.0,
        },
        expected_pairing_seeds=[101, 102, 103],
        bootstrap_seed=9,
        bootstrap_replicates=200,
    )
    assert result.status == "success"
    assert result.succeeds
    assert result.sign_interval is not None and result.sign_interval.lower > 0.5
    assert result.positive_seed_count == 3
    assert result.supporting_layer_count == 2


def test_confirmatory_fails_closed_on_pair_count_and_strict_boundaries() -> None:
    empty = evaluate_confirmatory_causal_claim(
        np.array([], dtype=float),
        np.array([], dtype=bool),
        [],
        [],
        random_control_effects=np.zeros(1000),
        pair_off_target_ratios=np.array([], dtype=float),
        supporting_layer_effects={
            "vlm_context": 0.0,
            "early_expert": 0.0,
            "late_expert": 0.0,
        },
        expected_pairing_seeds=[1, 2, 3],
        bootstrap_seed=1,
    )
    assert empty.status == "inconclusive"
    assert empty.valid_pairs == 0

    inconclusive = evaluate_confirmatory_causal_claim(
        np.ones(29),
        np.ones(29, dtype=bool),
        list(range(29)),
        np.resize([1, 2, 3], 29),
        random_control_effects=np.zeros(1000),
        pair_off_target_ratios=np.zeros(29),
        supporting_layer_effects={
            "vlm_context": 1.0,
            "early_expert": 1.0,
            "late_expert": 1.0,
        },
        expected_pairing_seeds=[1, 2, 3],
        bootstrap_seed=1,
        bootstrap_replicates=10,
    )
    assert inconclusive.status == "inconclusive"
    assert inconclusive.sign_interval is None

    signs = np.array([True] * 15 + [False] * 15)
    failed = evaluate_confirmatory_causal_claim(
        np.ones(30),
        signs,
        np.arange(30),
        np.repeat([1, 2, 3], 10),
        random_control_effects=np.zeros(1000),
        pair_off_target_ratios=np.full(30, 0.250001),
        supporting_layer_effects={
            "vlm_context": 1.0,
            "early_expert": -1.0,
            "late_expert": -1.0,
        },
        expected_pairing_seeds=[1, 2, 3],
        bootstrap_seed=1,
        bootstrap_replicates=100,
    )
    assert not failed.succeeds
    assert not failed.specificity_passes
    assert not failed.sign_passes
    assert not failed.layer_support_passes
