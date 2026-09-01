#!/usr/bin/env python3
"""Resumable Locked Test causal and sensitivity evidence producer.

The ``causal`` subcommand executes the selected-layer alpha=0.25 intervention,
the matched-donor control, and the registered 1,000 random-subspace controls in
that order.  The ``sensitivity`` subcommand is deliberately separate and accepts
the already-published causal receipt before executing alpha 0.5/1.0.  This keeps
the Section 11 analysis order executable rather than merely documentary.

No learned object is fitted here.  Pairing uses factual state metadata only,
every GPU result is checkpointed, and final receipts are immutable,
content-addressed trees.  The unavailable secondary evidence fixed by the
2026-09-01 amendment is emitted only as exact schema markers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.causal import (
    RANDOM_CONTROL_COUNT,
    CandidateState,
    cluster_bootstrap_rate_interval,
    compare_random_controls,
    five_nearest_neighbor_distance,
    iter_norm_matched_random_shifts,
    off_manifold_flag,
    pair_eligibility,
    patch_activation,
    probe_patch_shift,
    summarize_action_effect,
)
from mech_int_vla.config import ConditionSpec, SplitName, load_protocol_config
from mech_int_vla.feature_artifacts import (
    load_feature_cohort,
    load_feature_reference_bundle,
)
from mech_int_vla.features import _relative_quaternion, _xyzw_to_wxyz, _yaw_wxyz
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.probe_artifacts import load_bound_probe_artifact
from mech_int_vla.scoring import CHUNK_ACTIONS, load_scoring_sidecar

SCHEMA_VERSION = 1
COLLECTION_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
LOCKED_MANIFEST_SHA256 = "1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49"
PAIRING_SEEDS = (260_803, 260_804, 260_805)
FROZEN_ALPHA = 0.25
SENSITIVITY_ALPHAS = (0.5, 1.0)
PAIR_LIMIT_PER_SEED = 20
RANDOM_CHUNK_SIZE = 25
BOOTSTRAP_SEED = 260_803
BOOTSTRAP_REPLICATES = 10_000
POSITION_STATUS = "unavailable_preaccess_missing_position_trace"
POSITION_REASON = "frozen_position_decoder_and_all_object_trace_absent"
SUPPORTING_REASON = "frozen_supporting_layer_coefficients_absent"
BROKEN_SUCCESS_REASON = "patched_closed_loop_outcome_not_defined"


class LockedTestCausalError(RuntimeError):
    """Raised on any stale binding, malformed checkpoint, or undefined result."""


@dataclass(frozen=True)
class PlannedPair:
    pair_index: int
    seed: int
    condition_index: int
    base_init_state_id: int
    recipient_id: str
    donor_id: str | None
    orientation_difference_degrees: float | None
    matched_donor_id: str | None
    matched_orientation_difference_degrees: float | None
    plan_invalid_reason: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LockedTestCausalError(f"value is not finite canonical JSON: {exc}") from exc


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
        raise LockedTestCausalError(f"required regular file is absent or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LockedTestCausalError(f"required JSON is absent or unsafe: {path}")
    payload = path.read_bytes()
    if expected_sha256 is not None and (
        not _is_sha256(expected_sha256) or _sha256(payload) != expected_sha256
    ):
        raise LockedTestCausalError(f"content digest mismatch for {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LockedTestCausalError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockedTestCausalError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or payload != _canonical(value):
        raise LockedTestCausalError(f"{path} is not one canonical JSON object")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise LockedTestCausalError(f"checkpoint differs from deterministic bytes: {path}")
        return
    _write_exclusive(path, payload)


def _split_candidate(candidate_id: str) -> tuple[str, int]:
    episode_id, separator, step = candidate_id.rpartition("@")
    if not separator or not episode_id:
        raise LockedTestCausalError(f"invalid candidate identity: {candidate_id}")
    try:
        control_step = int(step)
    except ValueError as exc:
        raise LockedTestCausalError(f"invalid candidate step: {candidate_id}") from exc
    if control_step < 0 or control_step % 5:
        raise LockedTestCausalError(f"candidate is off five-step cadence: {candidate_id}")
    return episode_id, control_step


def _fast_pairs(
    candidates: Sequence[CandidateState], *, seed: int, mode: str
) -> tuple[tuple[str, str, float], ...]:
    """Return the exact greedy selection without materializing every O(n^2) edge."""

    ordered = sorted(candidates, key=lambda state: state.candidate_id)
    by_id = {state.candidate_id: state for state in ordered}
    if len(by_id) != len(ordered):
        raise LockedTestCausalError("candidate identities are not unique")
    traversal = _seeded_traversal(ordered, seed)
    unused = set(by_id)
    selected: list[tuple[str, str, float]] = []
    for recipient in traversal:
        if len(selected) == PAIR_LIMIT_PER_SEED:
            break
        if recipient.candidate_id not in unused:
            continue
        options: list[tuple[float, str, float]] = []
        for donor in ordered:
            if donor.candidate_id == recipient.candidate_id or donor.candidate_id not in unused:
                continue
            eligibility = pair_eligibility(recipient, donor, mode=mode)
            if eligibility.eligible:
                options.append(
                    (
                        eligibility.standardized_matching_distance,
                        donor.candidate_id,
                        eligibility.orientation_difference_deg,
                    )
                )
        if not options:
            continue
        _, donor_id, orientation = min(options, key=lambda item: (item[0], item[1]))
        selected.append((recipient.candidate_id, donor_id, float(orientation)))
        unused.remove(recipient.candidate_id)
        unused.remove(donor_id)
    return tuple(selected)


def _seeded_traversal(
    candidates: Sequence[CandidateState], seed: int
) -> list[CandidateState]:
    return sorted(
        candidates,
        key=lambda state: (
            hashlib.sha256(f"{seed}\0{state.candidate_id}".encode()).digest(),
            state.candidate_id,
        ),
    )


def _matched_donors(
    confirmatory: Sequence[tuple[int, str, str, float]],
    candidates: Sequence[CandidateState],
) -> dict[tuple[int, str], tuple[str, float] | None]:
    """Choose the nearest <5-degree donor for each fixed confirmatory recipient."""

    ordered = sorted(candidates, key=lambda state: state.candidate_id)
    by_id = {state.candidate_id: state for state in ordered}
    result: dict[tuple[int, str], tuple[str, float] | None] = {}
    used_donors_by_seed: dict[int, set[str]] = defaultdict(set)
    reserved_recipients = {
        seed: {
            recipient_id
            for row_seed, recipient_id, _, _ in confirmatory
            if row_seed == seed
        }
        for seed in PAIRING_SEEDS
    }
    for seed, recipient_id, _, _ in confirmatory:
        recipient = by_id[recipient_id]
        options: list[tuple[float, str, float]] = []
        for donor in ordered:
            if (
                donor.candidate_id in used_donors_by_seed[seed]
                or donor.candidate_id in reserved_recipients[seed]
                or donor.candidate_id == recipient_id
            ):
                continue
            eligibility = pair_eligibility(recipient, donor, mode="matched_control")
            if eligibility.eligible:
                options.append(
                    (
                        eligibility.standardized_matching_distance,
                        donor.candidate_id,
                        eligibility.orientation_difference_deg,
                    )
                )
        if not options:
            result[(seed, recipient_id)] = None
            continue
        _, donor_id, orientation = min(options, key=lambda item: (item[0], item[1]))
        used_donors_by_seed[seed].add(donor_id)
        result[(seed, recipient_id)] = (donor_id, float(orientation))
    return result


def build_pair_plan(
    candidates: Sequence[CandidateState], episode_specs: Mapping[str, Any]
) -> tuple[PlannedPair, ...]:
    selected: list[tuple[int, str, str, float]] = []
    slots: list[tuple[int, str, str | None, float | None, str | None]] = []
    for seed in PAIRING_SEEDS:
        seed_pairs = [
            (seed, recipient, donor, orientation)
            for recipient, donor, orientation in _fast_pairs(
                candidates, seed=seed, mode="confirmatory"
            )
        ]
        selected.extend(seed_pairs)
        slots.extend(
            (row_seed, recipient, donor, orientation, None)
            for row_seed, recipient, donor, orientation in seed_pairs
        )
        missing = PAIR_LIMIT_PER_SEED - len(seed_pairs)
        used = {
            candidate_id
            for _, recipient, donor, _ in seed_pairs
            for candidate_id in (recipient, donor)
        }
        fillers = [
            candidate.candidate_id
            for candidate in _seeded_traversal(candidates, seed)
            if candidate.candidate_id not in used
        ][:missing]
        if len(fillers) != missing:
            raise LockedTestCausalError("not enough candidate states for 20 pair attempts")
        slots.extend(
            (seed, recipient, None, None, "no_eligible_confirmatory_donor")
            for recipient in fillers
        )
    matched = _matched_donors(selected, candidates)
    plan: list[PlannedPair] = []
    for pair_index, (seed, recipient, donor, orientation, invalid_reason) in enumerate(slots):
        recipient_episode, _ = _split_candidate(recipient)
        specification = episode_specs[recipient_episode]
        control = None if donor is None else matched[(seed, recipient)]
        plan.append(
            PlannedPair(
                pair_index=pair_index,
                seed=seed,
                condition_index=int(specification.condition_index),
                base_init_state_id=int(specification.base_init_state_id),
                recipient_id=recipient,
                donor_id=donor,
                orientation_difference_degrees=orientation,
                matched_donor_id=None if control is None else control[0],
                matched_orientation_difference_degrees=(
                    None if control is None else control[1]
                ),
                plan_invalid_reason=invalid_reason,
            )
        )
    if len(plan) != 60:
        raise LockedTestCausalError("pair plan did not retain exactly 60 attempt slots")
    return tuple(plan)


def _effect_payload(summary: Any) -> dict[str, Any]:
    ratio = float(summary.off_target_ratio)
    if math.isfinite(ratio):
        ratio_value: float | None = ratio
        ratio_status = "finite"
    else:
        ratio_value = None
        ratio_status = "infinite_zero_yaw_effect"
    return {
        "sign_correct": bool(summary.sign_correct),
        "target_effect": float(summary.target_effect),
        "natural_target_effect": float(summary.natural_target_effect),
        "donor_aligned_target_effect": float(summary.donor_aligned_target_effect),
        "standardized_mean_change": list(summary.standardized_mean_change),
        "off_target_change": float(summary.off_target_change),
        "off_target_ratio": ratio_value,
        "off_target_ratio_status": ratio_status,
        "temporal_yaw_dot_product": float(summary.temporal_yaw_dot_product),
    }


def _ratio_for_decision(payload: Mapping[str, Any]) -> float:
    value = payload.get("off_target_ratio")
    return math.inf if value is None else float(value)


def random_control_distribution(
    evidence: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    valid = [row for row in evidence if row["pair"].get("valid") is True]
    if not valid:
        return np.zeros(RANDOM_CONTROL_COUNT, dtype=np.float64)
    for row in valid:
        controls = row.get("random_controls")
        if (
            not isinstance(controls, list)
            or len(controls) != RANDOM_CONTROL_COUNT
            or [item.get("control_index") for item in controls]
            != list(range(RANDOM_CONTROL_COUNT))
        ):
            raise LockedTestCausalError("random control evidence is incomplete or misordered")
    matrix = np.asarray(
        [
            [float(item["donor_aligned_target_effect"]) for item in row["random_controls"]]
            for row in valid
        ],
        dtype=np.float64,
    )
    if matrix.shape != (len(valid), RANDOM_CONTROL_COUNT) or not np.isfinite(matrix).all():
        raise LockedTestCausalError("random control evidence is incomplete or nonfinite")
    return np.median(matrix, axis=0)


def _nullable_median(values: Sequence[float]) -> tuple[float | None, str]:
    if not values:
        return None, "unavailable_no_valid_pairs"
    result = float(np.median(np.asarray(values, dtype=np.float64)))
    return (
        (result, "finite")
        if math.isfinite(result)
        else (None, "infinite_zero_yaw_effect")
    )


def summarize_causal_evidence(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in evidence if row["pair"].get("valid") is True]
    effects = np.asarray(
        [row["selected_patch"]["donor_aligned_target_effect"] for row in valid],
        dtype=np.float64,
    )
    signs = np.asarray([row["selected_patch"]["sign_correct"] for row in valid], dtype=bool)
    ratios = [_ratio_for_decision(row["selected_patch"]) for row in valid]
    random_distribution = random_control_distribution(evidence)
    median_effect = float(np.median(effects)) if effects.size else None
    median_ratio, median_ratio_status = _nullable_median(ratios)
    seed_rows = {
        seed: [row for row in valid if row["pair"]["seed"] == seed]
        for seed in PAIRING_SEEDS
    }
    matched_signs = [row["matched_control"]["sign_correct"] for row in valid]
    off_flags = [row["off_manifold"]["off_manifold"] for row in valid]
    positive_seed_count = sum(
        bool(rows)
        and float(
            np.median(
                [row["selected_patch"]["donor_aligned_target_effect"] for row in rows]
            )
        )
        > 0.0
        for rows in seed_rows.values()
    )
    result: dict[str, Any] = {
        "status": (
            "complete" if len(valid) >= 30 else "inconclusive_insufficient_valid_pairs"
        ),
        "attempted_pairs": 60,
        "valid_pairs": len(valid),
        "sign_correct_count": int(np.count_nonzero(signs)),
        "sign_correct_rate": None if not valid else float(np.mean(signs)),
        "sign_interval": None,
        "median_donor_aligned_target_effect": median_effect,
        "median_off_target_ratio": median_ratio,
        "median_off_target_ratio_status": median_ratio_status,
        "off_manifold_rate": None if not valid else float(np.mean(off_flags)),
        "matched_control_sign_rate": None if not valid else float(np.mean(matched_signs)),
        "random_control_95th_percentile": (
            None if not valid else float(np.percentile(random_distribution, 95.0))
        ),
        "random_control_passes": False,
        "specificity_passes": False,
        "sign_passes": False,
        "positive_seed_count": positive_seed_count,
        "seed_stability_passes": False,
    }
    if not valid:
        return result
    clusters = [row["pair"]["base_init_state_id"] for row in valid]
    interval = (
        cluster_bootstrap_rate_interval(
            signs,
            clusters,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            confidence=0.90,
        )
        if len(set(clusters)) >= 2
        else None
    )
    comparison = compare_random_controls(float(median_effect), random_distribution)
    specificity = median_ratio is not None and median_ratio <= 0.25
    sign_passes = (
        interval is not None and interval.estimate > 0.5 and interval.lower > 0.5
    )
    seed_passes = positive_seed_count >= 2
    result.update(
        {
            "sign_interval": None if interval is None else asdict(interval),
            "random_control_passes": comparison.exceeds_95th_percentile,
            "specificity_passes": specificity,
            "sign_passes": sign_passes,
            "seed_stability_passes": seed_passes,
        }
    )
    return result


def _build_candidates(cohort: Any, raw_root: Path, task: Any) -> list[CandidateState]:
    by_episode: dict[str, list[Any]] = defaultdict(list)
    for record in cohort.records:
        by_episode[record.episode_id].append(record)
    candidates: list[CandidateState] = []
    for episode_id in sorted(by_episode):
        artifact = load_rollout_artifact(
            raw_root / SplitName.LOCKED_TEST.value / episode_id,
            expected_task=task,
        )
        if not artifact.valid_reset:
            raise LockedTestCausalError(f"invalid reset appears in feature cohort: {episode_id}")
        first_record = by_episode[episode_id][0]
        source = first_record.source_hashes
        if (
            source.raw_metadata_sha256 != artifact.hashes.metadata_sha256
            or source.raw_trajectory_sha256 != artifact.hashes.trajectory_sha256
        ):
            raise LockedTestCausalError(f"cohort/raw hash binding differs for {episode_id}")
        arrays = artifact.arrays
        max_steps = int(artifact.metadata["execution"]["max_steps"])
        phase = arrays["frame_phase"]
        eef_position = arrays["frame_eef_position"]
        eef_quaternion = arrays["frame_eef_quaternion_xyzw"]
        object_position = arrays["frame_primary_object_position"]
        object_quaternion = arrays["frame_primary_object_quaternion_wxyz"]
        gripper = arrays["frame_gripper_qpos"]
        contact = arrays["frame_primary_gripper_contact"]
        predicates = arrays["frame_task_predicates"]
        for record in by_episode[episode_id]:
            if record.source_hashes != source:
                raise LockedTestCausalError(f"state source hashes drift within {episode_id}")
            step = int(record.control_step)
            relative = _relative_quaternion(
                _xyzw_to_wxyz(np.asarray(eef_quaternion[step], dtype=np.float64)),
                np.asarray(object_quaternion[step], dtype=np.float64),
            )
            candidates.append(
                CandidateState.create(
                    candidate_id=f"{episode_id}@{step:04d}",
                    base_init_id=int(record.base_init_state_id),
                    phase=str(phase[step]),
                    contact=bool(contact[step]),
                    gripper_opening=float(np.mean(gripper[step])),
                    eef_position=tuple(float(value) for value in eef_position[step]),
                    object_position=tuple(float(value) for value in object_position[step]),
                    normalized_time=step / max_steps,
                    non_primary_predicates={
                        f"predicate_{index}": bool(value)
                        for index, value in enumerate(predicates[step])
                    },
                    orientation_rad=float(_yaw_wxyz(relative)),
                    symmetry_order=int(cohort.task_identity.symmetry_order),
                )
            )
    expected = [
        (record.episode_id, int(record.control_step)) for record in cohort.records
    ]
    observed = [_split_candidate(candidate.candidate_id) for candidate in candidates]
    if observed != expected:
        raise LockedTestCausalError("candidate membership differs from feature cohort")
    return candidates


def _load_activation_reference(
    path: Path, expected_sha256: str, *, expected_width: int
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Load the immutable Calibration natural-activation reference.

    The builder uses a two-file content-addressed directory.  This loader keeps
    its contract intentionally small: the full canonical metadata, exact arrays
    hash, selected-candidate matrix, and precomputed leave-self-out natural p95
    must all agree.  Any additional metadata remains hash-bound and auditable.
    """

    directory = path.resolve()
    module_path = REPO_ROOT / "ops" / "build_calibration_activation_reference.py"
    spec = importlib.util.spec_from_file_location(
        "locked_test_calibration_activation_reference", module_path
    )
    if spec is None or spec.loader is None:
        raise LockedTestCausalError("cannot load activation-reference contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        loaded = module.load_activation_reference(
            directory, expected_sha256=expected_sha256
        )
    except Exception as exc:
        raise LockedTestCausalError(f"invalid activation reference: {exc}") from exc
    metadata = dict(loaded.metadata)
    natural = np.asarray(loaded.arrays["activation_vectors"], dtype=np.float64)
    if (
        natural.ndim != 2
        or natural.shape[0] < 6
        or natural.shape[1] != expected_width
        or not np.isfinite(natural).all()
    ):
        raise LockedTestCausalError("natural activation matrix shape/content differs")
    geometry = metadata.get("geometry")
    threshold = (
        geometry.get("natural_95th_percentile")
        if isinstance(geometry, Mapping)
        else None
    )
    if threshold is None:
        raise LockedTestCausalError("activation reference lacks frozen natural p95")
    natural_95 = float(threshold)
    if not math.isfinite(natural_95) or natural_95 < 0:
        raise LockedTestCausalError("activation reference natural p95 is invalid")
    distances = loaded.arrays.get("natural_five_nn_distance")
    if distances is not None:
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim != 1 or not np.isfinite(distances).all():
            raise LockedTestCausalError("natural 5-NN distance vector is invalid")
        if not math.isclose(float(np.percentile(distances, 95.0)), natural_95, abs_tol=1e-12):
            raise LockedTestCausalError("activation reference p95 disagrees with distances")
    return natural, natural_95, metadata


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    manifest = reconstruct_episode_manifest(
        SplitName.LOCKED_TEST,
        task,
        protocol,
        policy_revision=POLICY_REVISION,
        code_commit=COLLECTION_COMMIT,
    )
    manifest_bytes = _canonical(manifest.to_dict())
    if (
        args.manifest_sha256 != LOCKED_MANIFEST_SHA256
        or manifest.sha256 != LOCKED_MANIFEST_SHA256
        or _sha256_file(args.manifest.resolve()) != LOCKED_MANIFEST_SHA256
        or args.manifest.resolve().read_bytes() != manifest_bytes
    ):
        raise LockedTestCausalError("Locked Test manifest does not reconstruct exactly")
    predictions = _read_canonical_json(
        args.predictions.resolve(), args.predictions_sha256
    )
    if args.predictions.resolve().parent.name != args.predictions_sha256:
        raise LockedTestCausalError("prediction receipt directory is not content addressed")
    prediction_source = predictions.get("source")
    if not isinstance(prediction_source, Mapping):
        raise LockedTestCausalError("prediction source is absent")
    if prediction_source.get("manifest_sha256") != LOCKED_MANIFEST_SHA256:
        raise LockedTestCausalError("predictions are bound to another manifest")
    cohort = load_feature_cohort(
        args.feature_cohort.resolve(), args.feature_cohort_sha256
    )
    if cohort.split != SplitName.LOCKED_TEST.value:
        raise LockedTestCausalError("feature cohort is not Locked Test")
    if prediction_source.get("feature_cohort_sha256") != args.feature_cohort_sha256:
        raise LockedTestCausalError("prediction/cohort binding differs")
    expected_episode_ids = {episode.episode_id for episode in manifest.episodes}
    cohort_episode_ids = {record.episode_id for record in cohort.records}
    invalid_rows = predictions.get("invalid_resets")
    if not isinstance(invalid_rows, list):
        raise LockedTestCausalError("prediction invalid-reset inventory is absent")
    invalid_episode_ids = {row.get("episode_id") for row in invalid_rows}
    if (
        None in invalid_episode_ids
        or cohort_episode_ids & invalid_episode_ids
        or cohort_episode_ids | invalid_episode_ids != expected_episode_ids
    ):
        raise LockedTestCausalError("prediction/cohort validity membership differs")
    raw_split_root = args.raw_root.resolve() / SplitName.LOCKED_TEST.value
    if not raw_split_root.is_dir() or raw_split_root.is_symlink():
        raise LockedTestCausalError("Locked Test raw split root is absent or unsafe")
    raw_episode_ids = {
        item.name for item in raw_split_root.iterdir() if item.is_dir() and not item.is_symlink()
    }
    if raw_episode_ids != expected_episode_ids:
        raise LockedTestCausalError("raw topology differs from the exact 160-episode manifest")
    score_split_root = args.score_root.resolve() / SplitName.LOCKED_TEST.value
    if not score_split_root.is_dir() or score_split_root.is_symlink():
        raise LockedTestCausalError("Locked Test score split root is absent or unsafe")
    score_episode_ids = {
        item.name for item in score_split_root.iterdir() if item.is_dir() and not item.is_symlink()
    }
    if score_episode_ids != cohort_episode_ids:
        raise LockedTestCausalError("score topology differs from valid feature-cohort membership")
    bound = load_bound_probe_artifact(
        args.bound_probe.resolve(),
        protocol=protocol,
        repo_root=root,
        expected_sha256=args.bound_probe_sha256,
    )
    if prediction_source.get("bound_probe_sha256") != args.bound_probe_sha256:
        raise LockedTestCausalError("prediction/bound-probe binding differs")
    reference = load_feature_reference_bundle(
        args.calibration_feature_reference.resolve(),
        args.calibration_feature_reference_sha256,
    )
    if prediction_source.get("reference_bundle_sha256") != args.calibration_feature_reference_sha256:
        raise LockedTestCausalError("prediction/feature-reference binding differs")
    if cohort.reference_bundle_sha256 != reference.metadata_sha256:
        raise LockedTestCausalError("cohort logical feature-reference binding differs")
    coefficient = np.asarray(bound.probe.model.coefficient, dtype=np.float64)
    natural, natural_95, activation_metadata = _load_activation_reference(
        args.calibration_activation_reference.resolve(),
        args.calibration_activation_reference_sha256,
        expected_width=coefficient.shape[1],
    )
    selection = activation_metadata.get("selection")
    selected_candidate = (
        selection.get("selected_candidate") if isinstance(selection, Mapping) else None
    )
    if selected_candidate != bound.probe.candidate:
        raise LockedTestCausalError("activation reference uses another selected candidate")
    candidates = _build_candidates(cohort, args.raw_root.resolve(), task)
    episode_specs = {episode.episode_id: episode for episode in manifest.episodes}
    plan = build_pair_plan(candidates, episode_specs)
    raw_inventory = []
    sources_by_episode = {record.episode_id: record.source_hashes for record in cohort.records}
    for episode_id in sorted(sources_by_episode):
        sidecar = load_scoring_sidecar(
            args.score_root.resolve() / SplitName.LOCKED_TEST.value / episode_id,
            expected_episode_id=episode_id,
        )
        source_hashes = sources_by_episode[episode_id]
        if (
            sidecar.metadata_sha256 != source_hashes.score_metadata_sha256
            or sidecar.primitives_sha256 != source_hashes.score_primitives_sha256
            or sidecar.metadata["links"]["raw_metadata_sha256"]
            != source_hashes.raw_metadata_sha256
            or sidecar.metadata["links"]["raw_trajectory_sha256"]
            != source_hashes.raw_trajectory_sha256
        ):
            raise LockedTestCausalError(f"cohort/score binding differs for {episode_id}")
        raw_inventory.append(
            {
                "episode_id": episode_id,
                "raw_metadata_sha256": source_hashes.raw_metadata_sha256,
                "raw_trajectory_sha256": source_hashes.raw_trajectory_sha256,
            }
        )
    source = {
        "manifest_sha256": LOCKED_MANIFEST_SHA256,
        "prediction_receipt_sha256": args.predictions_sha256,
        "raw_inventory_sha256": _sha256(_canonical(raw_inventory)),
        "score_allocation_sha256": prediction_source.get("score_allocation_sha256"),
        "bound_probe_sha256": args.bound_probe_sha256,
        "calibration_reference_sha256": args.calibration_feature_reference_sha256,
        "calibration_activation_reference_sha256": args.calibration_activation_reference_sha256,
        "alpha": FROZEN_ALPHA,
        "pairing_seeds": list(PAIRING_SEEDS),
        "random_subspaces_per_pair": RANDOM_CONTROL_COUNT,
        "matched_donor_rule": "orientation_difference_degrees < 5",
        "pair_selection_rule": "outcome_blind_frozen_state_matching",
    }
    if not _is_sha256(source["score_allocation_sha256"]):
        raise LockedTestCausalError("prediction score-allocation digest is invalid")
    return {
        "root": root,
        "protocol": protocol,
        "task": task,
        "manifest": manifest,
        "episode_specs": episode_specs,
        "cohort": cohort,
        "bound": bound,
        "coefficient": coefficient,
        "action_scale": np.asarray(reference.action_scale.values, dtype=np.float64),
        "natural": natural,
        "natural_95": natural_95,
        "plan": plan,
        "source": source,
    }


def _sidecar_state(
    score_root: Path, candidate_id: str
) -> tuple[np.ndarray, np.ndarray, int]:
    episode_id, control_step = _split_candidate(candidate_id)
    sidecar = load_scoring_sidecar(
        score_root / SplitName.LOCKED_TEST.value / episode_id,
        expected_episode_id=episode_id,
    )
    indices = np.flatnonzero(sidecar.arrays["control_step"] == control_step)
    if indices.size != 1:
        raise LockedTestCausalError(f"sidecar state lookup is not unique: {candidate_id}")
    index = int(indices[0])
    activation = np.asarray(
        sidecar.arrays["original_activation"][index], dtype=np.float64
    ).mean(axis=0)
    actions = np.asarray(
        sidecar.arrays["original_actions"][index, 0], dtype=np.float32
    ).copy()
    noise_seed = int(sidecar.arrays["noise_seed"][index, 0])
    if (
        activation.ndim != 1
        or not np.isfinite(activation).all()
        or actions.shape != (CHUNK_ACTIONS, 7)
        or not np.isfinite(actions).all()
        or noise_seed < 0
    ):
        raise LockedTestCausalError(f"sidecar state is malformed: {candidate_id}")
    return activation, actions, noise_seed


def _action_chunk(
    adapter: Any,
    processed: Any,
    noise: Any,
    *,
    shift: np.ndarray,
    location: str,
    denoising_step: int | None,
) -> np.ndarray:
    import torch

    from mech_int_vla.scoring import _rng_equal, _rng_state
    from mech_int_vla.scoring_runtime import _clone_batch, _runtime_value_equal

    instrumentation = adapter.instrumentation
    instrumentation.clear()
    queue_before = adapter.policy_queue_state()
    rng_before = _rng_state()
    noise_before = noise.detach().clone()
    tensor = torch.as_tensor(
        np.array(shift, dtype=np.float32, copy=True), device=adapter.device
    )
    policy = adapter.policy_runtime.policy
    with (
        torch.inference_mode(),
        instrumentation,
        adapter._capture_selected_only(),
        instrumentation.patch(location, tensor, denoising_step=denoising_step),
    ):
        normalized = policy._get_action_chunk(
            _clone_batch(processed), noise=noise
        )
    postprocessed = adapter.policy_runtime.postprocessor(normalized)
    actions = (
        postprocessed.detach().to(device="cpu", dtype=torch.float32).numpy()
        if isinstance(postprocessed, torch.Tensor)
        else np.asarray(postprocessed)
    )
    if actions.shape == (1, 50, 7):
        actions = actions[0]
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise LockedTestCausalError("patched policy output is not finite float32 (50,7)")
    if not _runtime_value_equal(queue_before, adapter.policy_queue_state()):
        raise LockedTestCausalError("patched private inference mutated policy queues")
    if not _rng_equal(rng_before, _rng_state()):
        raise LockedTestCausalError("patched private inference mutated process RNG state")
    if not torch.equal(noise_before, noise):
        raise LockedTestCausalError("patched private inference mutated explicit noise")
    return actions[:CHUNK_ACTIONS].copy()


@contextmanager
def _recipient_runtime(
    common: Mapping[str, Any],
    args: argparse.Namespace,
    policy_runtime: Any,
    pair: PlannedPair,
    noise_seed: int,
) -> Iterator[tuple[Any, Any, Any, str, int | None]]:
    from mech_int_vla.instrumentation import SmolVLAInstrumentation
    from mech_int_vla.libero_runtime import RawLiberoEpisode
    from mech_int_vla.scoring import _compare_frames
    from mech_int_vla.scoring_runtime import (
        SmolVLAScoringAdapter,
        candidate_target,
        factual_replay_from_artifact,
    )

    recipient_episode, recipient_step = _split_candidate(pair.recipient_id)
    specification = common["episode_specs"][recipient_episode]
    artifact = load_rollout_artifact(
        args.raw_root.resolve() / SplitName.LOCKED_TEST.value / recipient_episode,
        expected_task=common["task"],
    )
    episode = RawLiberoEpisode.create(
        common["task"],
        base_init_state_id=specification.base_init_state_id,
        execution=common["protocol"].split.policy_execution,
        validity=common["protocol"].perturbations.validity,
    )
    instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
    try:
        adapter = SmolVLAScoringAdapter(
            episode,
            policy_runtime,
            artifact,
            common["bound"],
            instrumentation,
            reset_seed=specification.reset_seed,
            original_condition=ConditionSpec(
                specification.condition_name,
                specification.condition_family,
                specification.condition_index,
                specification.condition_parameters,
            ),
            protocol=common["protocol"],
            repo_root=common["root"],
        )
        replay = factual_replay_from_artifact(artifact)
        frame = adapter.reset_replay()
        _compare_frames(frame, replay.frames[0], "causal replay reset")
        for step_index in range(recipient_step):
            transition = adapter.step_replay(replay.actions[step_index])
            frame = transition.frame
            _compare_frames(frame, replay.frames[step_index + 1], "causal replay step")
        if frame.control_step != recipient_step:
            raise LockedTestCausalError("causal replay stopped at the wrong state")
        adapter.begin_score_state()
        processed = adapter.process_observation(frame)
        noise = adapter.noise_for_seed(noise_seed)
        location, denoising_step = candidate_target(common["bound"].probe.candidate)
        yield adapter, processed, noise, location, denoising_step
    finally:
        instrumentation.remove()
        instrumentation.clear()
        episode.close()


def _load_policy_runtime(args: argparse.Namespace) -> Any:
    from mech_int_vla.snapshots import load_locked_smolvla, resolve_snapshot_paths

    snapshots = resolve_snapshot_paths(
        args.environment_lock.resolve(),
        cache_dir=args.cache_dir.resolve(),
        local_files_only=True,
    )
    return load_locked_smolvla(snapshots, device="cuda")


def _random_chunk_path(progress_root: Path, pair_index: int, start: int) -> Path:
    stop = min(start + RANDOM_CHUNK_SIZE, RANDOM_CONTROL_COUNT)
    return progress_root / f"pair-{pair_index:03d}" / f"random-{start:04d}-{stop - 1:04d}.npz"


def _safe_npz_bytes(**arrays: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _load_random_progress(progress_root: Path, pair_index: int) -> dict[int, float]:
    values: dict[int, float] = {}
    for start in range(0, RANDOM_CONTROL_COUNT, RANDOM_CHUNK_SIZE):
        path = _random_chunk_path(progress_root, pair_index, start)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise LockedTestCausalError(f"random checkpoint is unsafe: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"control_index", "donor_aligned_target_effect"}:
                    raise LockedTestCausalError("random checkpoint members differ")
                indices = np.asarray(archive["control_index"], dtype=np.int64)
                effects = np.asarray(
                    archive["donor_aligned_target_effect"], dtype=np.float64
                )
        except (OSError, ValueError) as exc:
            raise LockedTestCausalError(f"invalid random checkpoint {path}: {exc}") from exc
        stop = min(start + RANDOM_CHUNK_SIZE, RANDOM_CONTROL_COUNT)
        if (
            not np.array_equal(indices, np.arange(start, stop, dtype=np.int64))
            or effects.shape != indices.shape
            or not np.isfinite(effects).all()
        ):
            raise LockedTestCausalError(f"random checkpoint content differs: {path}")
        for index, effect in zip(indices.tolist(), effects.tolist(), strict=True):
            if index in values:
                raise LockedTestCausalError("duplicate random checkpoint index")
            values[index] = float(effect)
    return values


def _write_random_chunk(
    progress_root: Path, pair_index: int, start: int, effects: Sequence[float]
) -> None:
    stop = min(start + RANDOM_CHUNK_SIZE, RANDOM_CONTROL_COUNT)
    if len(effects) != stop - start or not np.isfinite(effects).all():
        raise LockedTestCausalError("cannot checkpoint incomplete random-control chunk")
    path = _random_chunk_path(progress_root, pair_index, start)
    payload = _safe_npz_bytes(
        control_index=np.arange(start, stop, dtype=np.int64),
        donor_aligned_target_effect=np.asarray(effects, dtype=np.float64),
    )
    if path.exists():
        # Loading performs the semantic check; byte identity is not required from
        # ZIP containers because the arrays themselves are the frozen evidence.
        _load_random_progress(progress_root, pair_index)
        return
    _write_exclusive(path, payload)


def _source_sha256(source: Mapping[str, Any]) -> str:
    return _sha256(_canonical(source))


def _pair_identity(pair: PlannedPair, *, valid: bool, invalid_reason: str | None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "pair_index": pair.pair_index,
        "seed": pair.seed,
        "condition_index": pair.condition_index,
        "base_init_state_id": pair.base_init_state_id,
        "recipient_id": pair.recipient_id,
        "donor_id": pair.donor_id,
        "orientation_difference_degrees": pair.orientation_difference_degrees,
        "valid": valid,
    }
    if not valid:
        value["invalid_reason"] = invalid_reason
    return value


def _invalid_evidence(pair: PlannedPair, source_sha: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_causal_pair_evidence",
        "source_sha256": source_sha,
        "pair": _pair_identity(pair, valid=False, invalid_reason=reason),
        "selected_patch": None,
        "off_manifold": None,
        "matched_control": None,
        "random_controls": [],
    }


def _publish_evidence_checkpoint(
    staging: Path, pair: PlannedPair, evidence: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    payload = _canonical(evidence)
    digest = _sha256(payload)
    relative = f"evidence/{digest}.json"
    evidence_path = staging / relative
    if evidence_path.exists():
        if evidence_path.is_symlink() or evidence_path.read_bytes() != payload:
            raise LockedTestCausalError("pair evidence content address collides")
    else:
        _write_exclusive(evidence_path, payload)
    pointer = {
        "pair_index": pair.pair_index,
        "evidence_sha256": digest,
        "evidence_path": relative,
    }
    _write_or_verify_json(staging / "pair-receipts" / f"{pair.pair_index:03d}.json", pointer)
    return dict(evidence), digest, relative


def _load_evidence_checkpoint(
    staging: Path, pair: PlannedPair, source_sha: str
) -> tuple[dict[str, Any], str, str] | None:
    pointer_path = staging / "pair-receipts" / f"{pair.pair_index:03d}.json"
    if not pointer_path.exists():
        return None
    pointer = _read_canonical_json(pointer_path)
    if pointer.get("pair_index") != pair.pair_index:
        raise LockedTestCausalError("pair checkpoint index differs")
    digest = pointer.get("evidence_sha256")
    relative = pointer.get("evidence_path")
    if not _is_sha256(digest) or relative != f"evidence/{digest}.json":
        raise LockedTestCausalError("pair evidence pointer is malformed")
    evidence = _read_canonical_json(staging / relative, digest)
    if (
        evidence.get("source_sha256") != source_sha
        or evidence.get("pair", {}).get("pair_index") != pair.pair_index
    ):
        raise LockedTestCausalError("pair evidence source/identity differs")
    return evidence, digest, relative


def _run_causal_pair(
    common: Mapping[str, Any],
    args: argparse.Namespace,
    policy_runtime: Any,
    staging: Path,
    pair: PlannedPair,
) -> tuple[dict[str, Any], str, str]:
    source_sha = _source_sha256(common["source"])
    completed = _load_evidence_checkpoint(staging, pair, source_sha)
    if completed is not None:
        return completed
    if pair.plan_invalid_reason is not None or pair.matched_donor_id is None:
        return _publish_evidence_checkpoint(
            staging,
            pair,
            _invalid_evidence(
                pair,
                source_sha,
                pair.plan_invalid_reason or "no_eligible_matched_donor",
            ),
        )

    score_root = args.score_root.resolve()
    recipient_activation, recipient_actions, noise_seed = _sidecar_state(
        score_root, pair.recipient_id
    )
    donor_activation, donor_actions, _ = _sidecar_state(score_root, pair.donor_id)
    matched_activation, matched_actions, _ = _sidecar_state(
        score_root, pair.matched_donor_id
    )
    difference = donor_activation - recipient_activation
    core_path = staging / "progress" / f"pair-{pair.pair_index:03d}" / "core.json"
    core = _read_canonical_json(core_path) if core_path.exists() else None
    random_effects = _load_random_progress(staging / "progress", pair.pair_index)
    if core is not None and (
        core.get("source_sha256") != source_sha
        or core.get("pair_index") != pair.pair_index
    ):
        raise LockedTestCausalError("causal core checkpoint binding differs")
    if core is None or len(random_effects) != RANDOM_CONTROL_COUNT:
        with _recipient_runtime(
            common, args, policy_runtime, pair, noise_seed
        ) as (adapter, processed, noise, location, denoising_step):
            if core is None:
                selected_shift = probe_patch_shift(
                    recipient_activation,
                    donor_activation,
                    common["coefficient"],
                    alpha=FROZEN_ALPHA,
                )
                selected_actions = _action_chunk(
                    adapter,
                    processed,
                    noise,
                    shift=selected_shift,
                    location=location,
                    denoising_step=denoising_step,
                )
                selected = summarize_action_effect(
                    recipient_actions,
                    donor_actions,
                    selected_actions,
                    action_scale=common["action_scale"],
                )
                patched_activation = patch_activation(
                    recipient_activation,
                    donor_activation,
                    common["coefficient"],
                    alpha=FROZEN_ALPHA,
                )
                patched_distance = five_nearest_neighbor_distance(
                    patched_activation, common["natural"]
                )
                manifold = off_manifold_flag(
                    patched_five_nn_distance=patched_distance,
                    natural_95th_percentile=common["natural_95"],
                )
                matched_shift = probe_patch_shift(
                    recipient_activation,
                    matched_activation,
                    common["coefficient"],
                    alpha=FROZEN_ALPHA,
                )
                matched_patched_actions = _action_chunk(
                    adapter,
                    processed,
                    noise,
                    shift=matched_shift,
                    location=location,
                    denoising_step=denoising_step,
                )
                matched = summarize_action_effect(
                    recipient_actions,
                    matched_actions,
                    matched_patched_actions,
                    action_scale=common["action_scale"],
                )
                core = {
                    "source_sha256": source_sha,
                    "pair_index": pair.pair_index,
                    "selected_patch": {
                        key: value
                        for key, value in _effect_payload(selected).items()
                        if key
                        in {
                            "sign_correct",
                            "donor_aligned_target_effect",
                            "off_target_ratio",
                            "off_target_ratio_status",
                        }
                    }
                    | {"alpha": FROZEN_ALPHA},
                    "off_manifold": {
                        "patched_five_nn_distance": patched_distance,
                        "natural_95th_percentile": common["natural_95"],
                        "off_manifold": bool(manifold.off_manifold),
                    },
                    "matched_control": {
                        "donor_id": pair.matched_donor_id,
                        "orientation_difference_degrees": pair.matched_orientation_difference_degrees,
                        "sign_correct": bool(matched.sign_correct),
                        "donor_aligned_target_effect": float(
                            matched.donor_aligned_target_effect
                        ),
                    },
                }
                _write_or_verify_json(core_path, core)

            pending_start: int | None = None
            pending_effects: list[float] = []
            controls = iter_norm_matched_random_shifts(
                common["coefficient"],
                difference,
                seed=pair.seed,
                alpha=FROZEN_ALPHA,
                count=RANDOM_CONTROL_COUNT,
            )
            for control_index, control in enumerate(controls):
                if control_index in random_effects:
                    continue
                start = (control_index // RANDOM_CHUNK_SIZE) * RANDOM_CHUNK_SIZE
                if pending_start is None:
                    pending_start = start
                if pending_start != start:
                    raise LockedTestCausalError("random checkpoint chunk ordering drifted")
                patched_actions = _action_chunk(
                    adapter,
                    processed,
                    noise,
                    shift=control.matched_shift,
                    location=location,
                    denoising_step=denoising_step,
                )
                summary = summarize_action_effect(
                    recipient_actions,
                    donor_actions,
                    patched_actions,
                    action_scale=common["action_scale"],
                )
                pending_effects.append(float(summary.donor_aligned_target_effect))
                stop = min(start + RANDOM_CHUNK_SIZE, RANDOM_CONTROL_COUNT)
                if control_index + 1 == stop:
                    _write_random_chunk(
                        staging / "progress",
                        pair.pair_index,
                        start,
                        pending_effects,
                    )
                    for offset, effect in enumerate(pending_effects):
                        random_effects[start + offset] = effect
                    pending_start = None
                    pending_effects = []
            if pending_effects:
                raise LockedTestCausalError("unfinished random-control chunk")
    if core is None or set(random_effects) != set(range(RANDOM_CONTROL_COUNT)):
        raise LockedTestCausalError("causal pair did not complete all evidence")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_causal_pair_evidence",
        "source_sha256": source_sha,
        "pair": _pair_identity(pair, valid=True, invalid_reason=None),
        "selected_patch": core["selected_patch"],
        "off_manifold": core["off_manifold"],
        "matched_control": core["matched_control"],
        "random_controls": [
            {
                "control_index": index,
                "donor_aligned_target_effect": random_effects[index],
            }
            for index in range(RANDOM_CONTROL_COUNT)
        ],
    }
    return _publish_evidence_checkpoint(staging, pair, evidence)


def _verify_tree(
    destination: Path,
    *,
    receipt_name: str,
    receipt_bytes: bytes,
    evidence_bytes: Mapping[str, bytes],
) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise LockedTestCausalError("published receipt destination is unsafe")
    expected_root = {receipt_name, "evidence"}
    if {item.name for item in destination.iterdir()} != expected_root:
        raise LockedTestCausalError("published receipt topology differs")
    if (destination / receipt_name).read_bytes() != receipt_bytes:
        raise LockedTestCausalError("published receipt bytes differ")
    evidence_root = destination / "evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise LockedTestCausalError("published evidence root is unsafe")
    if {item.name for item in evidence_root.iterdir()} != set(evidence_bytes):
        raise LockedTestCausalError("published evidence membership differs")
    for name, payload in evidence_bytes.items():
        path = evidence_root / name
        if path.is_symlink() or path.read_bytes() != payload:
            raise LockedTestCausalError(f"published evidence differs: {name}")


def _publish_tree(
    output_root: Path,
    *,
    receipt_name: str,
    receipt: Mapping[str, Any],
    staging: Path,
) -> tuple[Path, str]:
    receipt_bytes = _canonical(receipt)
    digest = _sha256(receipt_bytes)
    destination = output_root / digest
    evidence_bytes: dict[str, bytes] = {}
    for evidence_hash in receipt["evidence_hashes"]:
        name = f"{evidence_hash}.json"
        source = staging / "evidence" / name
        payload = source.read_bytes()
        if _sha256(payload) != evidence_hash:
            raise LockedTestCausalError("staged evidence digest differs")
        evidence_bytes[name] = payload
    output_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_tree(
            destination,
            receipt_name=receipt_name,
            receipt_bytes=receipt_bytes,
            evidence_bytes=evidence_bytes,
        )
        return destination / receipt_name, digest
    temporary = output_root / f".{digest}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise LockedTestCausalError(f"publication staging already exists: {temporary}")
    temporary.mkdir()
    try:
        _write_exclusive(temporary / receipt_name, receipt_bytes)
        (temporary / "evidence").mkdir()
        for name, payload in evidence_bytes.items():
            _write_exclusive(temporary / "evidence" / name, payload)
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_tree(
        destination,
        receipt_name=receipt_name,
        receipt_bytes=receipt_bytes,
        evidence_bytes=evidence_bytes,
    )
    return destination / receipt_name, digest


def run_causal(args: argparse.Namespace) -> tuple[Path, str]:
    common = _load_inputs(args)
    staging = args.output_root.resolve().with_name(args.output_root.resolve().name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_causal_pair_plan",
        "source": common["source"],
        "pairs": [asdict(pair) for pair in common["plan"]],
    }
    _write_or_verify_json(staging / "plan.json", plan_payload)
    policy_runtime: Any | None = None
    rows: list[dict[str, Any]] = []
    evidence_hashes: list[str] = []
    pair_receipts: list[dict[str, Any]] = []
    for pair in common["plan"]:
        completed = _load_evidence_checkpoint(
            staging, pair, _source_sha256(common["source"])
        )
        if completed is None:
            if pair.matched_donor_id is not None and policy_runtime is None:
                policy_runtime = _load_policy_runtime(args)
            completed = _run_causal_pair(
                common, args, policy_runtime, staging, pair
            )
        evidence, evidence_hash, relative = completed
        rows.append(evidence)
        evidence_hashes.append(evidence_hash)
        pair_receipts.append(
            {
                "pair_index": pair.pair_index,
                "seed": pair.seed,
                "condition_index": pair.condition_index,
                "base_init_state_id": pair.base_init_state_id,
                "valid": bool(evidence["pair"]["valid"]),
                "evidence_sha256": evidence_hash,
                "evidence_path": relative,
            }
        )
        print(
            json.dumps(
                {
                    "kind": "locked_test_causal_pair_complete",
                    "pair_index": pair.pair_index,
                    "completed": pair.pair_index + 1,
                    "total": 60,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected_summary = summarize_causal_evidence(rows)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_causal_patching_receipt",
        "source": common["source"],
        "evidence_hashes": evidence_hashes,
        "pairs": pair_receipts,
        "selected_layer_summary": selected_summary,
        "supporting_layers": {
            "status": "unavailable",
            "reason": SUPPORTING_REASON,
            "multi_layer_support_available": False,
            "layer_support_passes": False,
        },
        "confirmatory": {
            "status": "unsupported",
            "succeeds": False,
            "reason": SUPPORTING_REASON,
        },
    }
    return _publish_tree(
        args.output_root.resolve(),
        receipt_name="causal.json",
        receipt=receipt,
        staging=staging,
    )


def _load_causal_receipt(
    path: Path, expected_sha256: str, common_source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    receipt = _read_canonical_json(path, expected_sha256)
    if path.parent.name != expected_sha256:
        raise LockedTestCausalError("causal receipt directory address differs")
    if (
        receipt.get("kind") != "locked_test_causal_patching_receipt"
        or receipt.get("source") != common_source
    ):
        raise LockedTestCausalError("causal receipt source differs from sensitivity inputs")
    summaries = receipt.get("pairs")
    hashes = receipt.get("evidence_hashes")
    if not isinstance(summaries, list) or not isinstance(hashes, list) or len(summaries) != 60:
        raise LockedTestCausalError("causal receipt does not contain 60 pairs")
    if [row.get("evidence_sha256") for row in summaries] != hashes:
        raise LockedTestCausalError("causal evidence hash order differs")
    evidence: dict[int, dict[str, Any]] = {}
    for index, summary in enumerate(summaries):
        digest = summary.get("evidence_sha256")
        relative = summary.get("evidence_path")
        if (
            summary.get("pair_index") != index
            or not _is_sha256(digest)
            or relative != f"evidence/{digest}.json"
        ):
            raise LockedTestCausalError("causal pair summary is malformed")
        row = _read_canonical_json(path.parent / relative, digest)
        if row.get("pair", {}).get("pair_index") != index:
            raise LockedTestCausalError("causal evidence pair index differs")
        evidence[index] = row
    return receipt, evidence


def _sensitivity_pair_identity(
    pair: PlannedPair, *, valid: bool, invalid_reason: str | None
) -> dict[str, Any]:
    value = {
        "pair_index": pair.pair_index,
        "seed": pair.seed,
        "condition_index": pair.condition_index,
        "base_init_state_id": pair.base_init_state_id,
        "valid": valid,
    }
    if not valid:
        value["invalid_reason"] = invalid_reason
    return value


def _run_sensitivity_pair(
    common: Mapping[str, Any],
    args: argparse.Namespace,
    policy_runtime: Any,
    staging: Path,
    pair: PlannedPair,
    causal_evidence: Mapping[str, Any],
    sensitivity_source: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    source_sha = _source_sha256(sensitivity_source)
    completed = _load_evidence_checkpoint(staging, pair, source_sha)
    if completed is not None:
        return completed
    if causal_evidence["pair"]["valid"] is not True:
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "kind": "locked_test_sensitivity_pair_evidence",
            "source_sha256": source_sha,
            "pair": _sensitivity_pair_identity(
                pair,
                valid=False,
                invalid_reason=str(causal_evidence["pair"].get("invalid_reason")),
            ),
            "alphas": None,
        }
        return _publish_evidence_checkpoint(staging, pair, evidence)

    score_root = args.score_root.resolve()
    recipient_activation, recipient_actions, noise_seed = _sidecar_state(
        score_root, pair.recipient_id
    )
    donor_activation, donor_actions, _ = _sidecar_state(score_root, pair.donor_id)
    alpha_rows: list[dict[str, Any]] = []
    with _recipient_runtime(common, args, policy_runtime, pair, noise_seed) as (
        adapter,
        processed,
        noise,
        location,
        denoising_step,
    ):
        for alpha in SENSITIVITY_ALPHAS:
            shift = probe_patch_shift(
                recipient_activation,
                donor_activation,
                common["coefficient"],
                alpha=alpha,
            )
            actions = _action_chunk(
                adapter,
                processed,
                noise,
                shift=shift,
                location=location,
                denoising_step=denoising_step,
            )
            summary = _effect_payload(
                summarize_action_effect(
                    recipient_actions,
                    donor_actions,
                    actions,
                    action_scale=common["action_scale"],
                )
            )
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "sign_correct": summary["sign_correct"],
                    "donor_aligned_target_effect": summary[
                        "donor_aligned_target_effect"
                    ],
                    "off_target_ratio": summary["off_target_ratio"],
                    "off_target_ratio_status": summary["off_target_ratio_status"],
                }
            )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_sensitivity_pair_evidence",
        "source_sha256": source_sha,
        "pair": _sensitivity_pair_identity(pair, valid=True, invalid_reason=None),
        "alphas": alpha_rows,
    }
    return _publish_evidence_checkpoint(staging, pair, evidence)


def summarize_dose_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for alpha in SENSITIVITY_ALPHAS:
        for condition_index in range(8):
            all_rows = [
                row
                for row in evidence
                if row["pair"]["condition_index"] == condition_index
            ]
            valid_rows = [row for row in all_rows if row["pair"]["valid"] is True]
            alpha_rows = [
                next(item for item in row["alphas"] if item["alpha"] == alpha)
                for row in valid_rows
            ]
            ratios = [_ratio_for_decision(row) for row in alpha_rows]
            median_ratio, ratio_status = _nullable_median(ratios)
            effects = [row["donor_aligned_target_effect"] for row in alpha_rows]
            signs = [row["sign_correct"] for row in alpha_rows]
            result.append(
                {
                    "alpha": alpha,
                    "condition_index": condition_index,
                    "pair_indices": [row["pair"]["pair_index"] for row in all_rows],
                    "valid_pairs": len(valid_rows),
                    "sign_correct_count": int(np.count_nonzero(signs)),
                    "sign_correct_rate": None if not signs else float(np.mean(signs)),
                    "median_donor_aligned_target_effect": (
                        None if not effects else float(np.median(effects))
                    ),
                    "median_off_target_ratio": median_ratio,
                    "median_off_target_ratio_status": ratio_status,
                    "specificity_passes": (
                        median_ratio is not None and median_ratio <= 0.25
                    ),
                }
            )
    return result


def run_sensitivity(args: argparse.Namespace) -> tuple[Path, str]:
    common = _load_inputs(args)
    causal_receipt, causal_evidence = _load_causal_receipt(
        args.causal_receipt.resolve(),
        args.causal_receipt_sha256,
        common["source"],
    )
    sensitivity_source = {
        "manifest_sha256": LOCKED_MANIFEST_SHA256,
        "prediction_receipt_sha256": args.predictions_sha256,
        "causal_receipt_sha256": args.causal_receipt_sha256,
        "alphas": list(SENSITIVITY_ALPHAS),
        "pairing_seeds": list(PAIRING_SEEDS),
        "patch_rule": "same_selected_layer_pair_plan_no_refit",
    }
    staging = args.output_root.resolve().with_name(args.output_root.resolve().name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(
        staging / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "locked_test_sensitivity_pair_plan",
            "source": sensitivity_source,
            "causal_evidence_hashes": causal_receipt["evidence_hashes"],
            "pairs": [asdict(pair) for pair in common["plan"]],
        },
    )
    policy_runtime: Any | None = None
    evidence_rows: list[dict[str, Any]] = []
    evidence_hashes: list[str] = []
    evidence_receipts: list[dict[str, Any]] = []
    for pair in common["plan"]:
        completed = _load_evidence_checkpoint(
            staging, pair, _source_sha256(sensitivity_source)
        )
        if completed is None:
            if (
                causal_evidence[pair.pair_index]["pair"]["valid"] is True
                and policy_runtime is None
            ):
                policy_runtime = _load_policy_runtime(args)
            completed = _run_sensitivity_pair(
                common,
                args,
                policy_runtime,
                staging,
                pair,
                causal_evidence[pair.pair_index],
                sensitivity_source,
            )
        evidence, digest, relative = completed
        evidence_rows.append(evidence)
        evidence_hashes.append(digest)
        evidence_receipts.append(
            {
                "pair_index": pair.pair_index,
                "evidence_sha256": digest,
                "evidence_path": relative,
            }
        )
        print(
            json.dumps(
                {
                    "kind": "locked_test_sensitivity_pair_complete",
                    "pair_index": pair.pair_index,
                    "completed": pair.pair_index + 1,
                    "total": 60,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "locked_test_sensitivity_receipt",
        "source": sensitivity_source,
        "evidence_hashes": evidence_hashes,
        "dose_evidence": evidence_receipts,
        "dose_by_difficulty": summarize_dose_evidence(evidence_rows),
        "rollout_diagnostics": {
            "status": POSITION_STATUS,
            "reason": POSITION_REASON,
        },
        "broken_successes": {
            "status": "unavailable",
            "reason": BROKEN_SUCCESS_REASON,
        },
    }
    return _publish_tree(
        args.output_root.resolve(),
        receipt_name="sensitivity.json",
        receipt=receipt,
        staging=staging,
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--feature-cohort", type=Path, required=True)
    parser.add_argument("--feature-cohort-sha256", required=True)
    parser.add_argument("--bound-probe", type=Path, required=True)
    parser.add_argument("--bound-probe-sha256", required=True)
    parser.add_argument("--calibration-feature-reference", type=Path, required=True)
    parser.add_argument("--calibration-feature-reference-sha256", required=True)
    parser.add_argument("--calibration-activation-reference", type=Path, required=True)
    parser.add_argument("--calibration-activation-reference-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce resumable, content-addressed Locked Test causal evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    causal = subparsers.add_parser("causal", help="run selected-layer confirmatory evidence")
    _common_arguments(causal)
    sensitivity = subparsers.add_parser(
        "sensitivity", help="run post-causal alpha-by-condition sensitivity"
    )
    _common_arguments(sensitivity)
    sensitivity.add_argument("--causal-receipt", type=Path, required=True)
    sensitivity.add_argument("--causal-receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.manifest_sha256 != LOCKED_MANIFEST_SHA256:
        raise LockedTestCausalError(
            f"--manifest-sha256 must equal frozen {LOCKED_MANIFEST_SHA256}"
        )
    if args.command == "causal":
        path, digest = run_causal(args)
    else:
        path, digest = run_sensitivity(args)
    print(
        json.dumps(
            {
                "kind": f"locked_test_{args.command}_complete",
                "receipt_path": str(path),
                "receipt_sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
