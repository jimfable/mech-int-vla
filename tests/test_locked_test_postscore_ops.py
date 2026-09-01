from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "locked_test_postscore.py"
SPEC = importlib.util.spec_from_file_location("locked_test_postscore", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
postscore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(postscore)
SHA = "a" * 64


def bound_payload(*, target: str = "[cos(s*theta_rel),sin(s*theta_rel)]"):
    return {
        "numerical_probe": {
            "metadata": {
                "target": target,
                "selection": {"candidate": "early_expert_t1_0"},
                "candidate_results": [
                    {"candidate": name, "mean_mae_rad": 0.2}
                    for name in (
                        "vlm_context",
                        "early_expert_t1_0",
                        "early_expert_t0_5",
                        "late_expert_t1_0",
                        "late_expert_t0_5",
                    )
                ],
            }
        }
    }


def test_authorized_limitations_are_not_capability_blockers_or_proxies() -> None:
    assessment = postscore.capability_assessment(
        bound_payload(), activation_reference_loaded=True
    )
    assert assessment["blockers"] == []
    assert [item["code"] for item in assessment["limitations"]] == [
        "unavailable_preaccess_missing_position_trace",
        "multi_layer_support_unavailable",
    ]
    assert assessment["limitations"][0]["reason"] == (
        "frozen_position_decoder_and_all_object_trace_absent"
    )
    assert assessment["limitations"][1]["multi_layer_support_available"] is False
    message = "\n".join(item["message"] for item in assessment["limitations"])
    assert "must not contain" in message
    assert "positive confirmatory causal claim is deterministically unsupported/false" in message


def test_activation_reference_remains_the_only_hard_capability_blocker() -> None:
    assessment = postscore.capability_assessment(
        bound_payload(), activation_reference_loaded=False
    )
    assert [item["code"] for item in assessment["blockers"]] == [
        "calibration_natural_activation_reference_missing"
    ]


def test_activation_reference_loader_enforces_exact_cross_bindings(monkeypatch) -> None:
    loaded = type(
        "Loaded",
        (),
        {
            "metadata": {
                "counts": {"episodes": 160, "rows": 9455, "width": 720},
                "source": {
                    "bound_probe_sha256": SHA,
                    "feature_reference_sha256": "b" * 64,
                },
                "selection": {"labels_used": False, "refit_performed": False},
            }
        },
    )()
    module = type(
        "Module", (), {"load_activation_reference": staticmethod(lambda *a, **k: loaded)}
    )
    monkeypatch.setattr(postscore, "_load_activation_reference_module", lambda: module)
    activation_sha = postscore.CALIBRATION_ACTIVATION_REFERENCE_SHA256
    assert postscore._load_activation_reference(
        Path("natural"), activation_sha,
        bound_probe_sha256=SHA,
        calibration_reference_sha256="b" * 64,
    ) is loaded
    loaded.metadata["counts"]["rows"] = 9454
    with pytest.raises(postscore.PostscoreError, match="9455 x 720"):
        postscore._load_activation_reference(
            Path("natural"), activation_sha,
            bound_probe_sha256=SHA,
            calibration_reference_sha256="b" * 64,
        )


def test_cost_receipt_is_strict_deterministic_and_evaluator_compatible(tmp_path: Path) -> None:
    rows = [
        f"{name},{index + 1},{index / 10},{index / 100}"
        for index, name in enumerate(postscore.STAGE_ORDER)
    ]
    first = postscore.build_cost_receipt(
        manifest_sha256=SHA,
        prediction_receipt_sha256="b" * 64,
        stage_rows=rows,
        budget_gate_stops=["scoring paused at the 24-hour budget gate"],
    )
    second = postscore.build_cost_receipt(
        manifest_sha256=SHA,
        prediction_receipt_sha256="b" * 64,
        stage_rows=rows,
        budget_gate_stops=["scoring paused at the 24-hour budget gate"],
    )
    assert postscore._canonical(first) == postscore._canonical(second)
    first_path, first_digest = postscore._publish(tmp_path, "cost-receipt.json", first)
    second_path, second_digest = postscore._publish(tmp_path, "cost-receipt.json", second)
    assert (first_path, first_digest) == (second_path, second_digest)
    assert first_path.parent.name == first_digest
    loaded = json.loads(first_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in loaded["stages"]] == list(postscore.STAGE_ORDER)
    evaluator = postscore._load_evaluator_module()
    evaluator._validate_cost_receipt(loaded, SHA, "b" * 64)


@pytest.mark.parametrize(
    "rows,match",
    [
        (["collection,1,1,1"], "exactly five"),
        (
            [
                "scoring,1,1,1",
                "collection,1,1,1",
                "evaluation,1,1,1",
                "causal_patching,1,1,1",
                "sensitivity,1,1,1",
            ],
            "frozen order",
        ),
        (
            [
                "collection,-1,1,1",
                "scoring,1,1,1",
                "evaluation,1,1,1",
                "causal_patching,1,1,1",
                "sensitivity,1,1,1",
            ],
            "finite",
        ),
    ],
)
def test_cost_evidence_fails_closed(rows, match) -> None:
    with pytest.raises(postscore.PostscoreError, match=match):
        postscore.build_cost_receipt(
            manifest_sha256=SHA,
            prediction_receipt_sha256="b" * 64,
            stage_rows=rows,
            budget_gate_stops=[],
        )


def test_cli_self_bootstraps_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=Path("/"),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout
    assert "cost" in result.stdout


def test_preflight_never_starts_gpu_work_when_capability_is_blocked(monkeypatch) -> None:
    monkeypatch.setattr(
        postscore,
        "validate_scientific_inputs",
        lambda **kwargs: {
            "limitations": [],
            "blockers": [
                {"code": "undefined_metric", "message": "required evidence is not frozen"}
            ],
        },
    )
    namespace = type(
        "Args",
        (),
        {
            "manifest": Path("manifest.json"), "manifest_sha256": SHA,
            "predictions": Path("predictions.json"), "predictions_sha256": SHA,
            "raw_root": Path("raw"), "score_root": Path("score"),
            "cohort": Path("cohort"), "cohort_sha256": SHA,
            "bound_probe": Path("bound.json"), "bound_probe_sha256": SHA,
            "calibration_reference": Path("reference"),
            "calibration_reference_sha256": SHA,
            "calibration_activation_reference": Path("activation-reference"),
            "calibration_activation_reference_sha256": SHA,
            "calibration_freeze": Path("calibration-freeze.json"),
            "calibration_freeze_sha256": SHA,
        },
    )()
    with pytest.raises(postscore.PostscoreError, match="undefined_metric"):
        postscore._preflight_command(namespace)
