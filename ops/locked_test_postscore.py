#!/usr/bin/env python3
"""Evidence producer and fail-closed capability gate for Locked Test post-scoring.

The cost subcommand produces the one post-scoring input that can be derived
from operator-supplied accounting evidence.  The preflight subcommand validates
all scientific inputs and refuses to start GPU intervention work when the
frozen artifacts cannot define a required analysis.  It never substitutes an
orientation probe for a position/identity probe and never treats an unfrozen
supporting layer as confirmatory evidence.
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


def capability_blockers(
    bound_probe_payload: Mapping[str, Any],
    calibration_reference_metadata: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return protocol blockers visible from immutable Calibration evidence.

    A blocker is a scientific-definition failure, not a missing implementation.
    The messages name the exact extra frozen evidence a future amendment would
    need.  Returning these instead of proxy metrics is the central fail-closed
    guarantee of this producer.
    """

    blockers: list[dict[str, str]] = []
    numerical = _mapping(bound_probe_payload.get("numerical_probe"), "bound.numerical_probe")
    metadata = _mapping(numerical.get("metadata"), "bound.numerical_probe.metadata")
    target = metadata.get("target")
    selection = _mapping(metadata.get("selection"), "bound probe selection")
    selected_candidate = selection.get("candidate")
    if target != "relative_primary_object_position_xyz":
        blockers.append(
            {
                "code": "rollout_diagnostic_probe_target_incompatible",
                "message": (
                    "Amendment 9a requires decoded object position to define nearest-object "
                    "identity and position-error/remaining-distance. The frozen probe target is "
                    f"{target!r}, not relative_primary_object_position_xyz; an orientation/error "
                    "proxy is forbidden. A pre-Locked-Test Calibration position probe and all-object "
                    "pose trace would have to be frozen explicitly."
                ),
            }
        )
    candidate_results = metadata.get("candidate_results")
    has_supporting_parameters = bool(metadata.get("supporting_layer_parameters"))
    if not has_supporting_parameters:
        blockers.append(
            {
                "code": "supporting_layer_probe_parameters_missing",
                "message": (
                    "PREREG section 10 requires effect-direction support at two of three "
                    "activation locations. The bound artifact contains executable coefficients "
                    f"only for selected candidate {selected_candidate!r}; candidate_results "
                    f"({len(candidate_results) if isinstance(candidate_results, list) else 0} rows) "
                    "contain CV metrics but no frozen non-selected-layer coefficients. Refitting "
                    "after Locked Test access is prohibited."
                ),
            }
        )
    arrays = _mapping(calibration_reference_metadata.get("arrays"), "reference.arrays")
    members = _mapping(arrays.get("members"), "reference.arrays.members")
    if "selected_natural_activation" not in members:
        blockers.append(
            {
                "code": "calibration_natural_activation_reference_missing",
                "message": (
                    "The frozen feature reference contains coverage/probe-norm summaries but no "
                    "selected natural Calibration activation matrix. The required activation-space "
                    "5-NN off-manifold threshold therefore needs the hash-bound Calibration score "
                    "sidecars named by the reference source hashes; the feature reference alone "
                    "cannot produce it."
                ),
            }
        )
    return blockers


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
) -> list[dict[str, str]]:
    """Validate exact content bindings, then return scientific blockers."""

    evaluator = _load_evaluator_module()
    manifest_payload = evaluator._load_addressed_json(manifest_path, manifest_sha256)
    manifest = evaluator._manifest_from_payload(manifest_payload)
    predictions = evaluator._load_addressed_json(
        predictions_path, predictions_sha256, directory_addressed=True
    )
    evaluator._validate_prediction_receipt(predictions, manifest_sha256, predictions_sha256)
    source = _mapping(predictions["source"], "predictions.source")
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
    if cohort.reference_bundle_sha256 != reference.provenance_sha256:
        # The cohort stores the logical reference provenance digest while the
        # prediction source stores the directory content address. Both must be
        # checked; neither can be substituted for the other.
        raise PostscoreError("cohort logical Calibration-reference binding differs")

    bound_payload = _strict_json(bound_probe_path, bound_probe_sha256)
    reference_metadata = _strict_json(
        calibration_reference_path / "metadata.json",
        _sha256_file(calibration_reference_path / "metadata.json"),
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
    return capability_blockers(bound_payload, reference_metadata)


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
    blockers = validate_scientific_inputs(
        manifest_path=args.manifest.resolve(), manifest_sha256=args.manifest_sha256,
        predictions_path=args.predictions.resolve(), predictions_sha256=args.predictions_sha256,
        raw_root=args.raw_root.resolve(), score_root=args.score_root.resolve(),
        cohort_path=args.cohort.resolve(), cohort_sha256=args.cohort_sha256,
        bound_probe_path=args.bound_probe.resolve(), bound_probe_sha256=args.bound_probe_sha256,
        calibration_reference_path=args.calibration_reference.resolve(),
        calibration_reference_sha256=args.calibration_reference_sha256,
    )
    if blockers:
        raise PostscoreError(
            "post-scoring scientific inputs cannot define every required receipt:\n"
            + "\n".join(f"- {item['code']}: {item['message']}" for item in blockers)
        )
    # Reaching this branch would mean a future, properly frozen artifact set has
    # all required capabilities. GPU execution is intentionally not silently
    # started by a preflight command.
    print(_canonical({"kind": "locked_test_postscore_preflight_passed"}).decode(), flush=True)
    return 0


def _capabilities_command(args: argparse.Namespace) -> int:
    bound = _strict_json(args.bound_probe.resolve(), args.bound_probe_sha256)
    reference_path = args.calibration_reference.resolve()
    if reference_path.name != args.calibration_reference_sha256:
        raise PostscoreError("Calibration-reference directory is not its declared content address")
    load_feature_reference_bundle(
        reference_path, expected_sha256=args.calibration_reference_sha256
    )
    metadata_path = reference_path / "metadata.json"
    reference_metadata = _strict_json(
        metadata_path, _sha256_file(metadata_path)
    )
    blockers = capability_blockers(bound, reference_metadata)
    if blockers:
        raise PostscoreError(
            "frozen artifacts cannot define the mandatory post-scoring analyses:\n"
            + "\n".join(f"- {item['code']}: {item['message']}" for item in blockers)
        )
    print(_canonical({"kind": "locked_test_postscore_capabilities_passed"}).decode(), flush=True)
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
    for name in ("manifest", "predictions", "cohort", "bound-probe", "calibration-reference"):
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
    capabilities.set_defaults(handler=_capabilities_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
