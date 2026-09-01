from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

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
        "confirmatory": {"status": "complete", "specificity_passes": True}
    }
    sensitivity = {
        "rollout_diagnostics": {}, "dose_by_difficulty": [], "broken_successes": {}
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
