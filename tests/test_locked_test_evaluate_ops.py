from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mech_int_vla.config import SplitName, TaskSpec
from mech_int_vla.evaluation import EvaluationError
from mech_int_vla.manifest import EpisodeSpec, Manifest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "locked_test_evaluate.py"
SPEC = importlib.util.spec_from_file_location("locked_test_evaluate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluate_ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_ops)

CAUSAL_SCRIPT = ROOT / "ops" / "locked_test_causal.py"
CAUSAL_SPEC = importlib.util.spec_from_file_location(
    "locked_test_causal_for_evaluator_contract", CAUSAL_SCRIPT
)
assert CAUSAL_SPEC is not None and CAUSAL_SPEC.loader is not None
causal_ops = importlib.util.module_from_spec(CAUSAL_SPEC)
sys.modules[CAUSAL_SPEC.name] = causal_ops
CAUSAL_SPEC.loader.exec_module(causal_ops)


SHA = "a" * 64


def manifest() -> Manifest:
    task = TaskSpec(
        rank=1,
        suite="libero_10",
        task_id=5,
        language="place the black book",
        primary_object="black_book",
        planar_symmetry_order=2,
    )
    episodes = tuple(
        EpisodeSpec(
            suite=task.suite,
            task_id=task.task_id,
            task_rank=task.rank,
            split=SplitName.LOCKED_TEST,
            base_init_state_id=init_id,
            condition_index=cell,
            condition_name=f"cell-{cell}",
            condition_family="iid" if cell == 0 else "perturbation",
            condition_parameters={"cell": cell},
            reset_seed=300_000 + init_id * 10 + cell,
            inference_seed=400_000 + init_id * 10 + cell,
            policy_revision="3" * 40,
            code_commit="1" * 40,
        )
        for init_id in range(30, 50)
        for cell in range(8)
    )
    return Manifest(1, SplitName.LOCKED_TEST, task, episodes)


def inputs(*, invalid: tuple[str, ...] = ()):
    frozen = manifest()
    invalid_set = set(invalid)
    records = []
    raw = {}
    for specification in frozen.episodes:
        episode_id = specification.episode_id
        label = int(specification.base_init_state_id % 2 == 0)
        is_valid = episode_id not in invalid_set
        raw[episode_id] = {
            "valid_reset": is_valid,
            "label": label,
            "failure_step": 100 if label and is_valid else None,
            "metadata_sha256": "b" * 64,
            "trajectory_sha256": "c" * 64,
            "categories": {name: False for name in evaluate_ops.VALIDITY_CATEGORIES},
        }
        if not is_valid:
            continue
        for step in range(0, 205, 5):
            # Three already-frozen predictors.  These depend on the fixture's
            # outcome only to make expected test metrics simple; production
            # source provenance is separately required to prove no label access.
            probabilities = {
                "M0": 0.60 if label else 0.40,
                "M1": 0.80 if label else 0.20,
                "M2": 0.90 if label else 0.10,
            }
            records.append(
                {
                    "episode_id": episode_id,
                    "base_init_state_id": specification.base_init_state_id,
                    "control_step": step,
                    "terminal_failure_label": bool(label),
                    "source_hashes": {
                        "episode_id": episode_id,
                        "raw_metadata_sha256": "b" * 64,
                        "raw_trajectory_sha256": "c" * 64,
                        "score_metadata_sha256": "d" * 64,
                        "score_primitives_sha256": "e" * 64,
                    },
                    "probabilities": probabilities,
                }
            )
    invalid_rows = [
        {
            "episode_id": episode_id,
            "base_init_state_id": next(item.base_init_state_id for item in frozen.episodes if item.episode_id == episode_id),
            "condition_index": next(item.condition_index for item in frozen.episodes if item.episode_id == episode_id),
        }
        for episode_id in sorted(invalid)
    ]
    prediction_payload = {
        "schema_version": 1,
        "kind": "locked_test_frozen_predictions",
        "source": {
            "manifest_sha256": frozen.sha256,
            "bound_probe_sha256": SHA,
            "score_allocation_sha256": SHA,
            "feature_cohort_sha256": SHA,
            "reference_bundle_sha256": SHA,
            "predictor_bundle_sha256": SHA,
            "predictor_metadata_sha256": SHA,
            "calibration_data_sha256": SHA,
            "calibration_freeze_sha256": SHA,
            "label_source": "feature_cohort_terminal_outcome_joined_only_during_prediction_serialization",
            "prediction_rule": "frozen_all_calibration_predictor_applied_without_label_argument",
        },
        "records": records,
        "invalid_resets": invalid_rows,
        "counts": {
            "attempted_episodes": 160,
            "valid_episodes": 160 - len(invalid),
            "invalid_resets": len(invalid),
            "state_rows": len(records),
        },
    }
    cells = [
        {
            "condition_index": cell,
            "episodes": 20,
            "invalid_resets": sum(row["condition_index"] == cell for row in invalid_rows),
            "invalid_reset_rate": sum(row["condition_index"] == cell for row in invalid_rows) / 20,
            "envelope_violation_counts": {name: 0 for name in evaluate_ops.VALIDITY_CATEGORIES},
            "envelope_violation_rates": {name: 0.0 for name in evaluate_ops.VALIDITY_CATEGORIES},
            "valid": True,
        }
        for cell in range(8)
    ]
    causal = {
        "selected_layer_summary": {"specificity_passes": True},
        "supporting_layers": {"multi_layer_support_available": False},
        "confirmatory": {"status": "unsupported", "succeeds": False},
    }
    sensitivity = {
        "rollout_diagnostics": {
            "status": "unavailable_preaccess_missing_position_trace",
            "reason": "frozen_position_decoder_and_all_object_trace_absent",
        },
        "dose_by_difficulty": [],
        "broken_successes": {
            "status": "unavailable",
            "reason": "patched_closed_loop_outcome_not_defined",
        },
    }
    cost = {"stages": [], "budget_gate_stops": []}
    return frozen, raw, cells, prediction_payload, causal, sensitivity, cost


def report(*, invalid: tuple[str, ...] = ()):
    frozen, raw, cells, predictions, causal, sensitivity, cost = inputs(invalid=invalid)
    return evaluate_ops.build_report(
        manifest=frozen,
        manifest_sha256=frozen.sha256,
        raw=raw,
        integrity_cells=cells,
        prediction_payload=predictions,
        prediction_sha256=SHA,
        causal_payload=causal,
        causal_sha256=SHA,
        sensitivity_payload=sensitivity,
        sensitivity_sha256=SHA,
        cost_payload=cost,
        cost_sha256=SHA,
        thresholds={"M0": 0.5, "M1": 0.5, "M2": 0.5},
        bootstrap_replicates=50,
    )


def test_report_has_exact_runbook_order_and_deterministic_bytes() -> None:
    first = report()
    second = report()
    assert [section["number"] for section in first["sections"]] == list(range(1, 11))
    assert tuple(section["title"] for section in first["sections"]) == evaluate_ops.SECTION_TITLES
    assert evaluate_ops._canonical(first) == evaluate_ops._canonical(second)
    diagnostics = first["sections"][8]["subsections"][0]["result"]
    assert diagnostics == {
        "status": "unavailable_preaccess_missing_position_trace",
        "reason": "frozen_position_decoder_and_all_object_trace_absent",
    }
    decision = first["sections"][9]["measured"]
    assert decision["multi_layer_support_available"] is False
    assert decision["positive_confirmatory_causal_claim_succeeds"] is False
    assert first["evaluation_protocol"] == {
        "bootstrap_seed": 260803,
        "bootstrap_replicates": 50,
        "bootstrap_confidence": 0.9,
        "primary_steps": [0, 50, 100, 150, 200],
        "model_training_or_selection_performed": False,
    }


def test_content_addressed_report_publish_is_verified_on_resume(tmp_path: Path) -> None:
    value = report()
    first_path, first_sha = evaluate_ops._publish(value, tmp_path)
    second_path, second_sha = evaluate_ops._publish(value, tmp_path)
    assert (first_path, first_sha) == (second_path, second_sha)
    assert first_path.parent.name == first_sha


def test_allowed_invalid_reset_is_retained_in_integrity_and_excluded_everywhere() -> None:
    invalid_id = manifest().episodes[0].episode_id
    result = report(invalid=(invalid_id,))
    integrity = result["sections"][0]
    primary = result["sections"][1]["result"]
    assert integrity["artifact_count"] == 160
    assert integrity["valid_episodes"] == 159
    assert integrity["invalid_episode_ids"] == [invalid_id]
    assert primary["episodes"] == 159
    assert result["sections"][2]["models"]["M0"]["episodes"] == 159


def test_more_than_ten_percent_invalid_in_one_cell_fails_exact_coverage() -> None:
    frozen, raw, _, predictions, *_ = inputs(
        invalid=tuple(
            item.episode_id
            for item in manifest().episodes
            if item.condition_index == 0
        )[:3]
    )
    with pytest.raises(EvaluationError, match="exceed 10%"):
        evaluate_ops._prediction_episodes(predictions, frozen, raw)


def test_prediction_label_or_exact_episode_coverage_cannot_drift() -> None:
    frozen, raw, _, predictions, *_ = inputs()
    predictions["records"][0]["terminal_failure_label"] = not predictions["records"][0]["terminal_failure_label"]
    with pytest.raises(evaluate_ops.LockedTestEvaluationError, match="prediction/raw identity mismatch"):
        evaluate_ops._prediction_episodes(predictions, frozen, raw)

    _, raw, _, predictions, *_ = inputs()
    predictions["records"] = predictions["records"][1:]
    with pytest.raises(evaluate_ops.LockedTestEvaluationError, match="cadence is incomplete"):
        evaluate_ops._prediction_episodes(predictions, frozen, raw)


def test_evaluator_contains_no_fit_or_locked_label_prediction_path() -> None:
    source = Path(evaluate_ops.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "mech_int_vla.predictors" not in imports
    assert not {name for name in calls if name.startswith("fit_")}


def test_missing_required_receipt_stops_before_raw_or_metrics(monkeypatch, tmp_path: Path) -> None:
    seen = []

    def fake_load(path, digest, **kwargs):
        del digest, kwargs
        seen.append(path.name)
        if path.name == "causal.json":
            raise evaluate_ops.LockedTestEvaluationError("required causal receipt missing")
        return {}

    monkeypatch.setattr(evaluate_ops, "_load_addressed_json", fake_load)
    monkeypatch.setattr(
        evaluate_ops,
        "_load_raw_inputs",
        lambda *args, **kwargs: pytest.fail("raw outcomes opened before receipt preflight"),
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("manifest", "predictions", "freeze", "gate", "causal", "sensitivity", "cost")}
    with pytest.raises(evaluate_ops.LockedTestEvaluationError, match="causal receipt"):
        evaluate_ops.evaluate(
            manifest_path=paths["manifest"], manifest_sha256=SHA,
            raw_root=tmp_path, predictions_path=paths["predictions"], predictions_sha256=SHA,
            calibration_freeze_path=paths["freeze"], calibration_freeze_sha256=SHA,
            reality_gate_lock_path=paths["gate"], reality_gate_lock_sha256=SHA,
            causal_receipt_path=paths["causal"], causal_receipt_sha256=SHA,
            sensitivity_receipt_path=paths["sensitivity"], sensitivity_receipt_sha256=SHA,
            cost_receipt_path=paths["cost"], cost_receipt_sha256=SHA,
        )
    assert seen == ["manifest.json", "predictions.json", "freeze.json", "gate.json", "causal.json"]


def sensitivity_contract(*, diagnostic_status: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "locked_test_sensitivity_receipt",
        "source": {
            "manifest_sha256": SHA,
            "prediction_receipt_sha256": "b" * 64,
            "causal_receipt_sha256": "c" * 64,
            "alphas": [0.5, 1.0],
            "pairing_seeds": [260803, 260804, 260805],
            "patch_rule": "same_selected_layer_pair_plan_no_refit",
        },
        "evidence_hashes": [],
        "dose_evidence": [],
        "dose_by_difficulty": [],
        "rollout_diagnostics": {
            "status": diagnostic_status,
            "reason": "frozen_position_decoder_and_all_object_trace_absent",
        },
        "broken_successes": {
            "status": "unavailable",
            "reason": "patched_closed_loop_outcome_not_defined",
        },
    }


def test_position_diagnostic_unavailable_marker_is_exact_and_nonblocking(tmp_path: Path) -> None:
    with pytest.raises(
        evaluate_ops.LockedTestEvaluationError,
        match="exact pre-access missing-position-trace marker",
    ):
        evaluate_ops._validate_sensitivity_receipt(
            sensitivity_contract(diagnostic_status="unavailable"),
            tmp_path / "sensitivity.json", SHA, "b" * 64, "c" * 64,
        )


def test_producer_receipts_and_evidence_are_evaluator_compatible(tmp_path: Path) -> None:
    frozen, _, _, predictions, *_ = inputs()
    prediction_source = predictions["source"]
    raw_by_episode = {}
    for record in predictions["records"]:
        hashes = record["source_hashes"]
        raw_by_episode[record["episode_id"]] = {
            "episode_id": record["episode_id"],
            "raw_metadata_sha256": hashes["raw_metadata_sha256"],
            "raw_trajectory_sha256": hashes["raw_trajectory_sha256"],
        }
    raw_inventory = [raw_by_episode[name] for name in sorted(raw_by_episode)]
    causal_source = {
        "manifest_sha256": frozen.sha256,
        "prediction_receipt_sha256": SHA,
        "raw_inventory_sha256": evaluate_ops._sha256(
            evaluate_ops._canonical(raw_inventory)
        ),
        "score_allocation_sha256": prediction_source["score_allocation_sha256"],
        "bound_probe_sha256": prediction_source["bound_probe_sha256"],
        "calibration_reference_sha256": prediction_source["reference_bundle_sha256"],
        "calibration_activation_reference_sha256": (
            "cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c"
        ),
        "alpha": 0.25,
        "pairing_seeds": [260803, 260804, 260805],
        "random_subspaces_per_pair": 1000,
        "matched_donor_rule": "orientation_difference_degrees < 5",
        "pair_selection_rule": "outcome_blind_frozen_state_matching",
    }
    causal_source_sha = evaluate_ops._sha256(evaluate_ops._canonical(causal_source))
    causal_rows = []
    causal_pointers = []
    causal_hashes = []
    causal_staging = tmp_path / "causal.incomplete"
    (causal_staging / "evidence").mkdir(parents=True)
    controls = np.linspace(-1.0, 1.0, 1000)
    for index in range(60):
        recipient = frozen.episodes[index]
        donor = frozen.episodes[(index + 1) % len(frozen.episodes)]
        causal_valid = index != 59
        row = {
            "schema_version": 1,
            "kind": "locked_test_causal_pair_evidence",
            "source_sha256": causal_source_sha,
            "pair": {
                "pair_index": index,
                "seed": [260803, 260804, 260805][index // 20],
                "condition_index": recipient.condition_index,
                "base_init_state_id": recipient.base_init_state_id,
                "recipient_id": f"{recipient.episode_id}@0000",
                "donor_id": f"{donor.episode_id}@0000" if causal_valid else None,
                "orientation_difference_degrees": 45.0 if causal_valid else None,
                "valid": causal_valid,
                **(
                    {}
                    if causal_valid
                    else {"invalid_reason": "no_eligible_confirmatory_donor"}
                ),
            },
            "selected_patch": (
                {
                    "alpha": 0.25,
                    "sign_correct": True,
                    "donor_aligned_target_effect": 2.0,
                    "off_target_ratio": 0.2,
                    "off_target_ratio_status": "finite",
                }
                if causal_valid else None
            ),
            "off_manifold": (
                {
                    "patched_five_nn_distance": 1.0,
                    "natural_95th_percentile": 2.0,
                    "off_manifold": False,
                }
                if causal_valid else None
            ),
            "matched_control": (
                {
                    "donor_id": f"{donor.episode_id}@0000",
                    "orientation_difference_degrees": 4.999,
                    "sign_correct": False,
                    "donor_aligned_target_effect": 0.0,
                }
                if causal_valid else None
            ),
            "random_controls": (
                [
                    {
                        "control_index": control_index,
                        "donor_aligned_target_effect": float(effect),
                    }
                    for control_index, effect in enumerate(controls)
                ]
                if causal_valid else []
            ),
        }
        payload = causal_ops._canonical(row)
        digest = causal_ops._sha256(payload)
        (causal_staging / "evidence" / f"{digest}.json").write_bytes(payload)
        causal_rows.append(row)
        causal_hashes.append(digest)
        causal_pointers.append(
            {
                "pair_index": index,
                "seed": row["pair"]["seed"],
                "condition_index": recipient.condition_index,
                "base_init_state_id": recipient.base_init_state_id,
                    "valid": causal_valid,
                "evidence_sha256": digest,
                "evidence_path": f"evidence/{digest}.json",
            }
        )
    causal_receipt = {
        "schema_version": 1,
        "kind": "locked_test_causal_patching_receipt",
        "source": causal_source,
        "evidence_hashes": causal_hashes,
        "pairs": causal_pointers,
        "selected_layer_summary": causal_ops.summarize_causal_evidence(causal_rows),
        "supporting_layers": {
            "status": "unavailable",
            "reason": "frozen_supporting_layer_coefficients_absent",
            "multi_layer_support_available": False,
            "layer_support_passes": False,
        },
        "confirmatory": {
            "status": "unsupported",
            "succeeds": False,
            "reason": "frozen_supporting_layer_coefficients_absent",
        },
    }
    causal_path, causal_sha = causal_ops._publish_tree(
        tmp_path / "causal", receipt_name="causal.json",
        receipt=causal_receipt, staging=causal_staging,
    )
    causal_receipt = causal_ops._read_canonical_json(causal_path, causal_sha)
    evaluate_ops._validate_causal_receipt(
        causal_receipt, causal_path, frozen, frozen.sha256, SHA, predictions
    )

    sensitivity_source = {
        "manifest_sha256": frozen.sha256,
        "prediction_receipt_sha256": SHA,
        "causal_receipt_sha256": causal_sha,
        "alphas": [0.5, 1.0],
        "pairing_seeds": [260803, 260804, 260805],
        "patch_rule": "same_selected_layer_pair_plan_no_refit",
    }
    sensitivity_source_sha = evaluate_ops._sha256(
        evaluate_ops._canonical(sensitivity_source)
    )
    sensitivity_rows = []
    sensitivity_pointers = []
    sensitivity_hashes = []
    sensitivity_staging = tmp_path / "sensitivity.incomplete"
    (sensitivity_staging / "evidence").mkdir(parents=True)
    for index, causal_row in enumerate(causal_rows):
        pair = causal_row["pair"]
        dose_valid = pair["valid"] and pair["condition_index"] != 7
        dose_invalid_reason = (
            pair.get("invalid_reason", "no_eligible_matched_donor")
            if not dose_valid else None
        )
        row = {
            "schema_version": 1,
            "kind": "locked_test_sensitivity_pair_evidence",
            "source_sha256": sensitivity_source_sha,
            "pair": {
                "pair_index": index,
                "seed": pair["seed"],
                "condition_index": pair["condition_index"],
                "base_init_state_id": pair["base_init_state_id"],
                "valid": dose_valid,
                **({} if dose_valid else {"invalid_reason": dose_invalid_reason}),
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
                    for alpha in (0.5, 1.0)
                ]
                if dose_valid
                else None
            ),
        }
        payload = causal_ops._canonical(row)
        digest = causal_ops._sha256(payload)
        (sensitivity_staging / "evidence" / f"{digest}.json").write_bytes(payload)
        sensitivity_rows.append(row)
        sensitivity_hashes.append(digest)
        sensitivity_pointers.append(
            {
                "pair_index": index,
                "evidence_sha256": digest,
                "evidence_path": f"evidence/{digest}.json",
            }
        )
    sensitivity_receipt = {
        "schema_version": 1,
        "kind": "locked_test_sensitivity_receipt",
        "source": sensitivity_source,
        "evidence_hashes": sensitivity_hashes,
        "dose_evidence": sensitivity_pointers,
        "dose_by_difficulty": causal_ops.summarize_dose_evidence(sensitivity_rows),
        "rollout_diagnostics": {
            "status": "unavailable_preaccess_missing_position_trace",
            "reason": "frozen_position_decoder_and_all_object_trace_absent",
        },
        "broken_successes": {
            "status": "unavailable",
            "reason": "patched_closed_loop_outcome_not_defined",
        },
    }
    sensitivity_path, _ = causal_ops._publish_tree(
        tmp_path / "sensitivity", receipt_name="sensitivity.json",
        receipt=sensitivity_receipt, staging=sensitivity_staging,
    )
    sensitivity_receipt = causal_ops._read_canonical_json(sensitivity_path)
    evaluate_ops._validate_sensitivity_receipt(
        sensitivity_receipt, sensitivity_path, frozen.sha256, SHA, causal_sha
    )
    with pytest.raises(
        evaluate_ops.LockedTestEvaluationError,
        match="60 dose-evidence rows",
    ):
        evaluate_ops._validate_sensitivity_receipt(
            sensitivity_contract(
                diagnostic_status="unavailable_preaccess_missing_position_trace"
            ),
            tmp_path / "sensitivity.json", SHA, "b" * 64, "c" * 64,
        )
