#!/usr/bin/env python3
"""Fail-closed capability gate and accounting producer for post-scoring.

The GPU causal producer lives in :mod:`ops.locked_test_causal`.  This companion
validates its immutable inputs before it can run and produces the one receipt
which legitimately comes from operator evidence: stage accounting.  Two
prospectively approved limitations are represented explicitly rather than by
proxy numbers: the position-trace diagnostic is unavailable and a missing
supporting-layer probe makes the positive multi-layer claim unsupported.  The
selected-layer experiment remains mandatory in both cases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.config import SplitName
from mech_int_vla.feature_artifacts import (
    load_feature_cohort,
    load_feature_reference_bundle,
)
from mech_int_vla.scoring import load_scoring_sidecar

SCHEMA_VERSION = 1
PAIRING_SEEDS = (260_803, 260_804, 260_805)
STAGE_ORDER = ("collection", "scoring", "evaluation", "causal_patching", "sensitivity")
POSITION_DIAGNOSTIC_STATUS = "unavailable_preaccess_missing_position_trace"
POSITION_DIAGNOSTIC_REASON = "frozen_position_decoder_and_all_object_trace_absent"
SUPPORTING_LAYER_STATUS = "unavailable"
SUPPORTING_LAYER_REASON = "frozen_supporting_layer_coefficients_absent"
CALIBRATION_ACTIVATION_REFERENCE_SHA256 = (
    "cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c"
)


class PostscoreError(RuntimeError):
    """Fail-closed error for malformed evidence or an undefined analysis."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PostscoreError(f"value is not finite canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PostscoreError(f"evidence file is absent or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise PostscoreError("expected digest must be a lowercase SHA-256")
    if not path.is_file() or path.is_symlink():
        raise PostscoreError(f"required JSON evidence is absent or unsafe: {path}")
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise PostscoreError(f"content digest mismatch for {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostscoreError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostscoreError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict) or payload != _canonical(value):
        raise PostscoreError(f"{path} is not one canonical JSON object")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostscoreError(f"{where} must be an object")
    return value


def _finite(value: Any, where: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool):
        raise PostscoreError(f"{where} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PostscoreError(f"{where} must be numeric") from exc
    if not math.isfinite(result) or result < lower:
        raise PostscoreError(f"{where} must be finite and >= {lower}")
    return result


def _load_evaluator_module() -> Any:
    path = REPO_ROOT / "ops" / "locked_test_evaluate.py"
    spec = importlib.util.spec_from_file_location("locked_test_evaluate_for_postscore", path)
    if spec is None or spec.loader is None:
        raise PostscoreError("could not load the locked evaluation contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_activation_reference_module() -> Any:
    path = REPO_ROOT / "ops" / "build_calibration_activation_reference.py"
    spec = importlib.util.spec_from_file_location(
        "calibration_activation_reference_for_postscore", path
    )
    if spec is None or spec.loader is None:
        raise PostscoreError("could not load the Calibration activation-reference contract")
    module = importlib.util.module_from_spec(spec)
    # Dataclass construction consults sys.modules while the file executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _load_activation_reference(
    path: Path,
    expected_sha256: str,
    *,
    bound_probe_sha256: str,
    calibration_reference_sha256: str,
) -> Any:
    """Load and cross-bind the full pre-access natural activation matrix."""

    if not _is_sha256(expected_sha256):
        raise PostscoreError("activation-reference digest must be a lowercase SHA-256")
    if expected_sha256 != CALIBRATION_ACTIVATION_REFERENCE_SHA256:
        raise PostscoreError("activation reference is not the final frozen cb210e artifact")
    module = _load_activation_reference_module()
    try:
        loaded = module.load_activation_reference(
            path.resolve(), expected_sha256=expected_sha256
        )
    except Exception as exc:
        raise PostscoreError(f"invalid Calibration activation reference: {exc}") from exc
    metadata = _mapping(loaded.metadata, "activation_reference.metadata")
    counts = _mapping(metadata.get("counts"), "activation_reference.counts")
    if counts != {"episodes": 160, "rows": 9455, "width": 720}:
        raise PostscoreError(
            "Calibration activation reference must contain exactly 160 episodes and "
            "a 9455 x 720 natural matrix"
        )
    source = _mapping(metadata.get("source"), "activation_reference.source")
    if source.get("bound_probe_sha256") != bound_probe_sha256:
        raise PostscoreError("activation reference is bound to another BoundProbe")
    if source.get("feature_reference_sha256") != calibration_reference_sha256:
        raise PostscoreError("activation reference is bound to another feature reference")
    selection = _mapping(metadata.get("selection"), "activation_reference.selection")
    if selection.get("labels_used") is not False or selection.get("refit_performed") is not False:
        raise PostscoreError("activation reference used labels or refitting")
    return loaded


def _publish(output_root: Path, filename: str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    canonical = _canonical(payload)
    digest = _sha256(canonical)
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise PostscoreError("output root is not a safe directory")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / digest
    target = destination / filename
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or {item.name for item in destination.iterdir()} != {filename}
            or target.is_symlink()
            or target.read_bytes() != canonical
        ):
            raise PostscoreError(f"content-addressed output differs: {destination}")
        return target, digest
    destination.mkdir()
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
    return target, digest


def parse_stage_evidence(values: Sequence[str]) -> list[dict[str, Any]]:
    """Parse exact ``name,wall_seconds,gpu_hours,instance_charges`` rows."""

    if len(values) != len(STAGE_ORDER):
        raise PostscoreError(
            "cost accounting requires exactly five --stage rows in frozen order"
        )
    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        parts = raw.split(",")
        if len(parts) != 4 or parts[0] != STAGE_ORDER[index]:
            raise PostscoreError(
                "stage evidence must use name,wall_seconds,gpu_hours,instance_charges "
                f"in frozen order {list(STAGE_ORDER)}"
            )
        stages.append(
            {
                "name": parts[0],
                "wall_seconds": _finite(parts[1], f"{parts[0]}.wall_seconds"),
                "gpu_hours": _finite(parts[2], f"{parts[0]}.gpu_hours"),
                "instance_charges": _finite(parts[3], f"{parts[0]}.instance_charges"),
            }
        )
    return stages


def build_cost_receipt(
    *,
    manifest_sha256: str,
    prediction_receipt_sha256: str,
    stage_rows: Sequence[str],
    budget_gate_stops: Sequence[str],
) -> dict[str, Any]:
    if not _is_sha256(manifest_sha256) or not _is_sha256(prediction_receipt_sha256):
        raise PostscoreError("cost source digests must be lowercase SHA-256 values")
    if any(not isinstance(item, str) or not item.strip() for item in budget_gate_stops):
        raise PostscoreError("budget-gate stop descriptions must be non-empty strings")
    if len(set(budget_gate_stops)) != len(budget_gate_stops):
        raise PostscoreError("budget-gate stop descriptions must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_cost_receipt",
        "source": {
            "manifest_sha256": manifest_sha256,
            "prediction_receipt_sha256": prediction_receipt_sha256,
            "evidence_source": "operator_supplied_stage_accounting",
        },
        "stages": parse_stage_evidence(stage_rows),
        "budget_gate_stops": list(budget_gate_stops),
    }


def capability_assessment(
    bound_probe_payload: Mapping[str, Any],
    *,
    activation_reference_loaded: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Classify frozen capabilities without converting limitations to proxies."""

    limitations: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    numerical = _mapping(bound_probe_payload.get("numerical_probe"), "bound.numerical_probe")
    metadata = _mapping(numerical.get("metadata"), "bound.numerical_probe.metadata")
    target = metadata.get("target")
    selection = _mapping(metadata.get("selection"), "bound probe selection")
    selected_candidate = selection.get("candidate")
    if target != "relative_primary_object_position_xyz":
        limitations.append(
            {
                "code": POSITION_DIAGNOSTIC_STATUS,
                "reason": POSITION_DIAGNOSTIC_REASON,
                "authorized": True,
                "message": (
                    "The frozen probe target is "
                    f"{target!r}; there is no pre-access position decoder and all-object "
                    "position trace. Section 9a must emit its exact unavailable marker and "
                    "must not contain angular, coverage, or other proxy values."
                ),
            }
        )
    candidate_results = metadata.get("candidate_results")
    has_supporting_parameters = bool(metadata.get("supporting_layer_parameters"))
    if not has_supporting_parameters:
        limitations.append(
            {
                "code": "multi_layer_support_unavailable",
                "reason": SUPPORTING_LAYER_REASON,
                "authorized": True,
                "multi_layer_support_available": False,
                "message": (
                    "Executable coefficients exist only for selected candidate "
                    f"{selected_candidate!r}; the "
                    f"{len(candidate_results) if isinstance(candidate_results, list) else 0} "
                    "candidate-result rows contain selection metrics, not frozen coefficients. "
                    "Selected-layer patching remains mandatory, but the positive confirmatory "
                    "causal claim is deterministically unsupported/false and no refit is allowed."
                ),
            }
        )
    if not activation_reference_loaded:
        blockers.append(
            {
                "code": "calibration_natural_activation_reference_missing",
                "authorized": False,
                "message": (
                    "A hash-bound Calibration activation-reference directory with the full "
                    "natural 9455 x 720 matrix and per-episode source hashes is mandatory for "
                    "the 5-NN/off-manifold computation."
                ),
            }
        )
    return {"limitations": limitations, "blockers": blockers}


def capability_blockers(
    bound_probe_payload: Mapping[str, Any],
    calibration_reference_metadata: Mapping[str, Any] | None = None,
    *,
    activation_reference_loaded: bool | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only hard blockers.

    The old feature-reference ``arrays.members`` value is deliberately ignored:
    it is not the newly frozen natural-activation reference.  Callers must pass
    ``activation_reference_loaded=True`` only after the content-addressed
    activation-reference loader has verified the directory.
    """

    del calibration_reference_metadata
    return capability_assessment(
        bound_probe_payload,
        activation_reference_loaded=bool(activation_reference_loaded),
    )["blockers"]


def validate_scientific_inputs(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    predictions_path: Path,
    predictions_sha256: str,
    raw_root: Path,
    score_root: Path,
    cohort_path: Path,
    cohort_sha256: str,
    bound_probe_path: Path,
    bound_probe_sha256: str,
    calibration_reference_path: Path,
    calibration_reference_sha256: str,
    calibration_activation_reference_path: Path,
    calibration_activation_reference_sha256: str,
    calibration_freeze_path: Path,
    calibration_freeze_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Validate exact content bindings, then return capability assessment."""

    evaluator = _load_evaluator_module()
    manifest_payload = evaluator._load_addressed_json(manifest_path, manifest_sha256)
    manifest = evaluator._manifest_from_payload(manifest_payload)
    predictions = evaluator._load_addressed_json(
        predictions_path, predictions_sha256, directory_addressed=True
    )
    evaluator._validate_prediction_receipt(predictions, manifest_sha256, predictions_sha256)
    source = _mapping(predictions["source"], "predictions.source")
    freeze = _strict_json(calibration_freeze_path, calibration_freeze_sha256)
    if source.get("calibration_freeze_sha256") != calibration_freeze_sha256:
        raise PostscoreError("prediction receipt is bound to another Calibration freeze")
    if source.get("feature_cohort_sha256") != cohort_sha256:
        raise PostscoreError("prediction/cohort content binding differs")
    if source.get("bound_probe_sha256") != bound_probe_sha256:
        raise PostscoreError("prediction/bound-probe content binding differs")
    if source.get("reference_bundle_sha256") != calibration_reference_sha256:
        raise PostscoreError("prediction/Calibration-reference content binding differs")

    cohort = load_feature_cohort(cohort_path, expected_sha256=cohort_sha256)
    if cohort.split != SplitName.LOCKED_TEST.value:
        raise PostscoreError("feature cohort is not Locked Test")
    reference = load_feature_reference_bundle(
        calibration_reference_path, expected_sha256=calibration_reference_sha256
    )
    if cohort.reference_bundle_sha256 != reference.metadata_sha256:
        # The cohort stores the logical reference provenance digest while the
        # prediction source stores the directory content address. Both must be
        # checked; neither can be substituted for the other.
        raise PostscoreError("cohort logical Calibration-reference binding differs")

    bound_payload = _strict_json(bound_probe_path, bound_probe_sha256)
    _load_activation_reference(
        calibration_activation_reference_path,
        calibration_activation_reference_sha256,
        bound_probe_sha256=bound_probe_sha256,
        calibration_reference_sha256=calibration_reference_sha256,
    )
    frozen_hashes = _mapping(freeze.get("artifact_hashes"), "calibration_freeze.artifact_hashes")
    frozen_activation_metadata = _mapping(
        frozen_hashes.get("calibration_activation_reference_metadata"),
        "artifact_hashes.calibration_activation_reference_metadata",
    )
    frozen_activation_arrays = _mapping(
        frozen_hashes.get("calibration_activation_reference_arrays"),
        "artifact_hashes.calibration_activation_reference_arrays",
    )
    expected_activation_bindings = (
        (
            frozen_activation_metadata,
            _sha256_file(calibration_activation_reference_path / "metadata.json"),
        ),
        (
            frozen_activation_arrays,
            _sha256_file(calibration_activation_reference_path / "arrays.npz"),
        ),
    )
    if any(item.get("sha256") != expected for item, expected in expected_activation_bindings):
        raise PostscoreError(
            "final Calibration freeze does not bind the supplied activation-reference bytes"
        )
    for frozen_item, filename in (
        (frozen_activation_metadata, "metadata.json"),
        (frozen_activation_arrays, "arrays.npz"),
    ):
        frozen_path = frozen_item.get("path")
        if (
            not isinstance(frozen_path, str)
            or Path(frozen_path).name != filename
            or Path(frozen_path).parent.name != calibration_activation_reference_sha256
        ):
            raise PostscoreError(
                "Calibration freeze activation-reference path/content address differs"
            )
    expected = {item.episode_id: item for item in manifest.episodes}
    split_raw = raw_root / SplitName.LOCKED_TEST.value
    split_scores = score_root / SplitName.LOCKED_TEST.value
    raw_dirs = {item.name for item in split_raw.iterdir() if item.is_dir()}
    if raw_dirs != set(expected):
        raise PostscoreError("raw artifact topology is not the exact 160-episode manifest")
    invalid_ids = {item["episode_id"] for item in predictions["invalid_resets"]}
    records_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for record in predictions["records"]:
        records_by_episode.setdefault(record["episode_id"], []).append(record)
    for episode_id in sorted(expected):
        raw = load_rollout_artifact(split_raw / episode_id, expected_task=manifest.task)
        if raw.valid_reset != (episode_id not in invalid_ids):
            raise PostscoreError(f"prediction validity inventory differs for {episode_id}")
        if episode_id in invalid_ids:
            continue
        rows = records_by_episode.get(episode_id)
        if not rows:
            raise PostscoreError(f"valid episode lacks prediction rows: {episode_id}")
        hashes = rows[0]["source_hashes"]
        if (
            hashes["raw_metadata_sha256"] != raw.hashes.metadata_sha256
            or hashes["raw_trajectory_sha256"] != raw.hashes.trajectory_sha256
        ):
            raise PostscoreError(f"prediction/raw hashes differ for {episode_id}")
        sidecar = load_scoring_sidecar(
            split_scores / episode_id, expected_episode_id=episode_id
        )
        if (
            _sha256_file(sidecar.path / "metadata.json")
            != hashes["score_metadata_sha256"]
            or _sha256_file(sidecar.path / "primitives.npz")
            != hashes["score_primitives_sha256"]
        ):
            raise PostscoreError(f"prediction/score hashes differ for {episode_id}")
    del reference
    return capability_assessment(bound_payload, activation_reference_loaded=True)


def _cost_command(args: argparse.Namespace) -> int:
    receipt = build_cost_receipt(
        manifest_sha256=args.manifest_sha256,
        prediction_receipt_sha256=args.predictions_sha256,
        stage_rows=args.stage,
        budget_gate_stops=args.budget_gate_stop,
    )
    path, digest = _publish(args.output_root.resolve(), "cost-receipt.json", receipt)
    print(
        _canonical(
            {
                "kind": "locked_test_cost_receipt_complete",
                "receipt_path": str(path),
                "receipt_sha256": digest,
            }
        ).decode("utf-8"),
        flush=True,
    )
    return 0


def _preflight_command(args: argparse.Namespace) -> int:
    assessment = validate_scientific_inputs(
        manifest_path=args.manifest.resolve(), manifest_sha256=args.manifest_sha256,
        predictions_path=args.predictions.resolve(), predictions_sha256=args.predictions_sha256,
        raw_root=args.raw_root.resolve(), score_root=args.score_root.resolve(),
        cohort_path=args.cohort.resolve(), cohort_sha256=args.cohort_sha256,
        bound_probe_path=args.bound_probe.resolve(), bound_probe_sha256=args.bound_probe_sha256,
        calibration_reference_path=args.calibration_reference.resolve(),
        calibration_reference_sha256=args.calibration_reference_sha256,
        calibration_activation_reference_path=args.calibration_activation_reference.resolve(),
        calibration_activation_reference_sha256=args.calibration_activation_reference_sha256,
        calibration_freeze_path=args.calibration_freeze.resolve(),
        calibration_freeze_sha256=args.calibration_freeze_sha256,
    )
    blockers = assessment["blockers"]
    if blockers:
        raise PostscoreError(
            "post-scoring scientific inputs cannot define every required receipt:\n"
            + "\n".join(f"- {item['code']}: {item['message']}" for item in blockers)
        )
    # Reaching this branch would mean a future, properly frozen artifact set has
    # all required capabilities. GPU execution is intentionally not silently
    # started by a preflight command.
    print(
        _canonical(
            {
                "kind": "locked_test_postscore_preflight_passed",
                "limitations": assessment["limitations"],
            }
        ).decode(),
        flush=True,
    )
    return 0


def _capabilities_command(args: argparse.Namespace) -> int:
    bound = _strict_json(args.bound_probe.resolve(), args.bound_probe_sha256)
    reference_path = args.calibration_reference.resolve()
    if reference_path.name != args.calibration_reference_sha256:
        raise PostscoreError("Calibration-reference directory is not its declared content address")
    load_feature_reference_bundle(
        reference_path, expected_sha256=args.calibration_reference_sha256
    )
    activation_reference = _load_activation_reference(
        args.calibration_activation_reference.resolve(),
        args.calibration_activation_reference_sha256,
        bound_probe_sha256=args.bound_probe_sha256,
        calibration_reference_sha256=args.calibration_reference_sha256,
    )
    del activation_reference
    assessment = capability_assessment(bound, activation_reference_loaded=True)
    blockers = assessment["blockers"]
    if blockers:
        raise PostscoreError(
            "frozen artifacts cannot define the mandatory post-scoring analyses:\n"
            + "\n".join(f"- {item['code']}: {item['message']}" for item in blockers)
        )
    print(
        _canonical(
            {
                "kind": "locked_test_postscore_capabilities_passed",
                "limitations": assessment["limitations"],
            }
        ).decode(),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cost = subparsers.add_parser("cost", help="publish hash-bound operator cost evidence")
    cost.add_argument("--manifest-sha256", required=True)
    cost.add_argument("--predictions-sha256", required=True)
    cost.add_argument(
        "--stage", action="append", required=True,
        help="name,wall_seconds,gpu_hours,instance_charges; repeat in frozen order",
    )
    cost.add_argument("--budget-gate-stop", action="append", default=[])
    cost.add_argument("--output-root", type=Path, required=True)
    cost.set_defaults(handler=_cost_command)

    preflight = subparsers.add_parser(
        "preflight", help="validate all scientific evidence before GPU patching"
    )
    for name in (
        "manifest", "predictions", "cohort", "bound-probe",
        "calibration-reference", "calibration-activation-reference",
        "calibration-freeze",
    ):
        preflight.add_argument(f"--{name}", type=Path, required=True)
        preflight.add_argument(f"--{name}-sha256", required=True)
    preflight.add_argument("--raw-root", type=Path, required=True)
    preflight.add_argument("--score-root", type=Path, required=True)
    preflight.set_defaults(handler=_preflight_command)

    capabilities = subparsers.add_parser(
        "capabilities",
        help="check frozen probe/reference capability before Locked Test collection",
    )
    capabilities.add_argument("--bound-probe", type=Path, required=True)
    capabilities.add_argument("--bound-probe-sha256", required=True)
    capabilities.add_argument("--calibration-reference", type=Path, required=True)
    capabilities.add_argument("--calibration-reference-sha256", required=True)
    capabilities.add_argument(
        "--calibration-activation-reference", type=Path, required=True
    )
    capabilities.add_argument(
        "--calibration-activation-reference-sha256", required=True
    )
    capabilities.set_defaults(handler=_capabilities_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
