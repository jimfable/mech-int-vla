from __future__ import annotations

import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.causal import CandidateState, select_pairs

MODULE_PATH = Path(__file__).parents[1] / "ops" / "locked_test_causal.py"
SPEC = importlib.util.spec_from_file_location("locked_test_causal_ops", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
causal_ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = causal_ops
SPEC.loader.exec_module(causal_ops)


def state(candidate_id: str, angle: float, *, time: float = 0.2) -> CandidateState:
    return CandidateState.create(
        candidate_id=candidate_id,
        base_init_id=int(candidate_id.removeprefix("s")),
        phase="pregrasp",
        contact=False,
        gripper_opening=0.5,
        eef_position=(0.0, 0.0, 0.0),
        object_position=(0.0, 0.0, 0.0),
        normalized_time=time,
        non_primary_predicates={"placed": False},
        orientation_rad=math.radians(angle),
        symmetry_order=1,
    )


def test_fast_pair_selection_matches_frozen_greedy_selection() -> None:
    candidates = [
        state(f"s{index}", 0.0 if index % 2 else 45.0, time=0.2 + index / 10000)
        for index in range(1, 41)
    ]
    expected = select_pairs(candidates, seed=19).pairs
    observed = causal_ops._fast_pairs(candidates, seed=19, mode="confirmatory")
    assert observed == tuple(
        (pair.recipient_id, pair.donor_id, pair.orientation_difference_deg)
        for pair in expected
    )


def evidence(index: int, *, selected: float = 2.0) -> dict:
    seed = causal_ops.PAIRING_SEEDS[index % 3]
    controls = np.linspace(-1.0, 1.0, causal_ops.RANDOM_CONTROL_COUNT)
    return {
        "pair": {
            "pair_index": index,
            "seed": seed,
            "base_init_state_id": index % 20,
            "valid": True,
        },
        "selected_patch": {
            "sign_correct": True,
            "donor_aligned_target_effect": selected,
            "off_target_ratio": 0.25,
            "off_target_ratio_status": "finite",
        },
        "off_manifold": {"off_manifold": False},
        "matched_control": {"sign_correct": False},
        "random_controls": [
            {
                "control_index": control_index,
                "donor_aligned_target_effect": float(value),
            }
            for control_index, value in enumerate(controls)
        ],
    }


def test_causal_summary_pools_same_random_index_over_pairs() -> None:
    rows = [evidence(index) for index in range(60)]
    summary = causal_ops.summarize_causal_evidence(rows)
    assert summary["status"] == "complete"
    assert summary["valid_pairs"] == 60
    assert summary["random_control_95th_percentile"] == pytest.approx(0.9)
    assert summary["random_control_passes"]
    assert summary["specificity_passes"]
    assert summary["sign_passes"]
    assert summary["positive_seed_count"] == 3
    assert summary["seed_stability_passes"]
    assert summary["sign_interval"]["replicates"] == 10_000


def test_random_controls_cannot_silently_become_zero_distribution() -> None:
    rows = [evidence(index) for index in range(30)]
    rows[0]["random_controls"] = rows[0]["random_controls"][:-1]
    with pytest.raises(causal_ops.LockedTestCausalError, match="incomplete"):
        causal_ops.random_control_distribution(rows)


def test_zero_yaw_effect_has_canonical_nullable_ratio() -> None:
    recipient = np.zeros((10, 7))
    donor = np.zeros((10, 7))
    donor[:, 5] = 1.0
    summary = causal_ops.summarize_action_effect(
        recipient, donor, recipient, action_scale=np.ones(7)
    )
    payload = causal_ops._effect_payload(summary)
    assert payload["off_target_ratio"] is None
    assert payload["off_target_ratio_status"] == "infinite_zero_yaw_effect"


def test_dose_summary_keeps_fixed_alpha_cell_order_and_empty_cells() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "pair": {
                    "pair_index": index,
                    "seed": causal_ops.PAIRING_SEEDS[index % 3],
                    "condition_index": index,
                    "base_init_state_id": index,
                    "valid": index != 7,
                    **({} if index != 7 else {"invalid_reason": "test"}),
                },
                "alphas": (
                    [
                        {
                            "alpha": alpha,
                            "sign_correct": True,
                            "donor_aligned_target_effect": alpha,
                            "off_target_ratio": 0.2,
                            "off_target_ratio_status": "finite",
                        }
                        for alpha in causal_ops.SENSITIVITY_ALPHAS
                    ]
                    if index != 7
                    else None
                ),
            }
        )
    summary = causal_ops.summarize_dose_evidence(rows)
    assert [(row["alpha"], row["condition_index"]) for row in summary] == [
        (alpha, cell)
        for alpha in causal_ops.SENSITIVITY_ALPHAS
        for cell in range(8)
    ]
    empty = summary[7]
    assert empty["pair_indices"] == [7]
    assert empty["valid_pairs"] == 0
    assert empty["sign_correct_rate"] is None
    assert empty["median_off_target_ratio_status"] == "unavailable_no_valid_pairs"
    assert not empty["specificity_passes"]


def test_content_addressed_tree_resume_and_tamper_detection(tmp_path: Path) -> None:
    staging = tmp_path / "run.incomplete"
    evidence_payload = causal_ops._canonical({"pair": 0})
    evidence_sha = causal_ops._sha256(evidence_payload)
    causal_ops._write_exclusive(
        staging / "evidence" / f"{evidence_sha}.json", evidence_payload
    )
    receipt = {"evidence_hashes": [evidence_sha], "value": 1}
    path, digest = causal_ops._publish_tree(
        tmp_path / "published",
        receipt_name="causal.json",
        receipt=receipt,
        staging=staging,
    )
    resumed = causal_ops._publish_tree(
        tmp_path / "published",
        receipt_name="causal.json",
        receipt=receipt,
        staging=staging,
    )
    assert resumed == (path, digest)
    (path.parent / "evidence" / f"{evidence_sha}.json").write_bytes(b"tampered")
    with pytest.raises(causal_ops.LockedTestCausalError, match="published evidence differs"):
        causal_ops._publish_tree(
            tmp_path / "published",
            receipt_name="causal.json",
            receipt=receipt,
            staging=staging,
        )


def test_pair_runner_checkpoints_all_random_controls_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions = np.zeros((10, 7), dtype=np.float32)
    donor_actions = actions.copy()
    donor_actions[:, 5] = 1.0
    matched_actions = actions.copy()
    matched_actions[:, 5] = 0.1

    def sidecar_state(_root: Path, candidate_id: str):
        if candidate_id == "recipient@0000":
            return np.zeros(4), actions, 17
        if candidate_id == "donor@0000":
            return np.array([1.0, 1.0, 0.0, 0.0]), donor_actions, 18
        assert candidate_id == "matched@0000"
        return np.array([0.1, 0.1, 0.0, 0.0]), matched_actions, 19

    runtime_calls = 0

    @contextmanager
    def recipient_runtime(*_args, **_kwargs):
        nonlocal runtime_calls
        runtime_calls += 1
        yield object(), object(), object(), "expert_layer_4", 0

    def action_chunk(_adapter, _processed, _noise, *, shift, **_kwargs):
        result = actions.copy()
        result[:, 5] = float(np.asarray(shift)[0])
        return result

    monkeypatch.setattr(causal_ops, "_sidecar_state", sidecar_state)
    monkeypatch.setattr(causal_ops, "_recipient_runtime", recipient_runtime)
    monkeypatch.setattr(causal_ops, "_action_chunk", action_chunk)
    pair = causal_ops.PlannedPair(
        pair_index=0,
        seed=causal_ops.PAIRING_SEEDS[0],
        condition_index=0,
        base_init_state_id=1,
        recipient_id="recipient@0000",
        donor_id="donor@0000",
        orientation_difference_degrees=45.0,
        matched_donor_id="matched@0000",
        matched_orientation_difference_degrees=1.0,
    )
    common = {
        "source": {"manifest_sha256": "a" * 64},
        "coefficient": np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
        "action_scale": np.ones(7),
        "natural": np.vstack([np.zeros((5, 4)), np.ones((1, 4))]),
        "natural_95": 1.0,
    }
    args = type("Args", (), {"score_root": tmp_path})()
    staging = tmp_path / "causal.incomplete"
    first = causal_ops._run_causal_pair(
        common, args, object(), staging, pair
    )
    assert runtime_calls == 1
    row, digest, relative = first
    assert len(row["random_controls"]) == 1000
    assert row["selected_patch"]["alpha"] == 0.25
    assert causal_ops._sha256((staging / relative).read_bytes()) == digest
    assert len(list((staging / "progress" / "pair-000").glob("random-*.npz"))) == 40

    monkeypatch.setattr(
        causal_ops,
        "_recipient_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reran GPU")),
    )
    assert causal_ops._run_causal_pair(
        common, args, object(), staging, pair
    ) == first
