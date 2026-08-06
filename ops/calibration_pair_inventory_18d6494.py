#!/usr/bin/env python3
"""Phase 1 of the Section 8 causal protocol: Calibration donor/recipient pair inventory.

This is a read-only counting gate.  It builds one ``CandidateState`` per scored
Calibration state from the immutable raw trajectories, then reports how many
valid confirmatory donor/recipient pairs exist under the frozen matching
tolerances.  It chooses no alpha, patches nothing, and runs no model.

``PREREG.md:356`` declares patching *inconclusive rather than negative* below 30
valid pairs, so this inventory can end the causal phase before any GPU time is
spent.

Exhaustive pairing is O(n^2) over 9,455 states (~44M comparisons), which is not
tractable in pure Python.  Eligibility is therefore evaluated in two stages:
states are first bucketed by the criteria that must match *exactly* (phase,
contact, non-primary predicates, symmetry order), then a vectorised numpy
prefilter applies the numeric tolerances with a deliberately slack margin, and
every surviving candidate pair is finally confirmed by the frozen
``pair_eligibility`` itself.  The prefilter can only ever admit a superset, so
the reported counts are exactly those of the frozen implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mech_int_vla.causal import (
    CONFIRMATORY_PAIR_MINIMUM,
    CandidateState,
    pair_eligibility,
)
from mech_int_vla.feature_artifacts import load_feature_cohort
from mech_int_vla.features import (
    _relative_quaternion,
    _xyzw_to_wxyz,
    _yaw_wxyz,
)

SCHEMA_VERSION = 1
EXPECTED_COHORT_SHA256 = (
    "989f67f8b18dbc7349dc85bc4552cfc0d4c0bbf379b0d2fd5e33ce0ef82446e0"
)
# Frozen tolerances (causal.pair_eligibility); the prefilter widens them.
GRIPPER_TOLERANCE = 0.01
EEF_TOLERANCE = 0.02
OBJECT_TOLERANCE = 0.02
TIME_TOLERANCE = 0.10
PREFILTER_SLACK = 1.0 + 1e-6
ORIENTATION_MIN_DEG = 30.0
ORIENTATION_MAX_DEG = 90.0


class PairInventoryError(RuntimeError):
    """Raised when an input does not match the frozen artifacts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PairInventoryError(message)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _build_candidates(cohort, raw_root: Path) -> list[CandidateState]:
    """One CandidateState per scored Calibration state, from raw trajectories.

    ``cohort`` is a fully validated ``FeatureCohort``: the loader has already
    verified the artifact bytes against the expected digest, so the rows and
    identifier types here are exactly the fit-time ones.
    """

    symmetry_order = int(cohort.task_identity.symmetry_order)
    by_episode: dict[str, list] = defaultdict(list)
    for record in cohort.records:
        by_episode[record.episode_id].append(record)

    candidates: list[CandidateState] = []
    for episode_id in sorted(by_episode):
        episode_dir = raw_root / "calibration" / episode_id
        metadata = json.loads(
            (episode_dir / "metadata.json").read_text(encoding="utf-8")
        )
        max_steps = int(metadata["execution"]["max_steps"])
        with np.load(episode_dir / "trajectory.npz") as handle:
            phase_array = handle["frame_phase"]
            eef_position = handle["frame_eef_position"]
            eef_quaternion_xyzw = handle["frame_eef_quaternion_xyzw"]
            object_position = handle["frame_primary_object_position"]
            object_quaternion_wxyz = handle["frame_primary_object_quaternion_wxyz"]
            gripper_qpos = handle["frame_gripper_qpos"]
            contact = handle["frame_primary_gripper_contact"]
            predicates = handle["frame_task_predicates"]
        for record in by_episode[episode_id]:
            step = int(record.control_step)
            phase = str(phase_array[step])
            _require(
                phase == record.phase,
                f"raw phase disagrees with the cohort at {episode_id} step {step}",
            )
            # theta_rel = wrap(yaw_eef - yaw_object), exactly as PREREG.md:120-122
            # and features.construct_m1_raw_pose define it.
            eef_wxyz = _xyzw_to_wxyz(np.asarray(eef_quaternion_xyzw[step], np.float64))
            relative = _relative_quaternion(
                eef_wxyz, np.asarray(object_quaternion_wxyz[step], np.float64)
            )
            candidates.append(
                CandidateState(
                    candidate_id=f"{episode_id}@{step:04d}",
                    base_init_id=record.base_init_state_id,
                    phase=phase,
                    contact=bool(contact[step]),
                    gripper_opening=float(np.mean(gripper_qpos[step])),
                    eef_position=tuple(
                        float(value) for value in eef_position[step]
                    ),
                    object_position=tuple(
                        float(value) for value in object_position[step]
                    ),
                    normalized_time=step / max_steps,
                    non_primary_predicates=tuple(
                        (f"predicate_{index}", bool(value))
                        for index, value in enumerate(predicates[step])
                    ),
                    orientation_rad=float(_yaw_wxyz(relative)),
                    symmetry_order=symmetry_order,
                )
            )
    return candidates


def _bucket_key(state: CandidateState) -> tuple:
    """The criteria that must match exactly for any pair to be eligible."""

    return (
        state.phase,
        state.contact,
        state.non_primary_predicates,
        state.symmetry_order,
    )


def _eligible_pairs(candidates: list[CandidateState]) -> tuple[list, Counter, int]:
    """Return confirmed eligible pairs, exclusion reasons, and comparisons made."""

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, state in enumerate(candidates):
        buckets[_bucket_key(state)].append(index)

    reasons: Counter[str] = Counter()
    confirmed: list[tuple[int, int, float, float]] = []
    comparisons = 0

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        order = np.asarray(indices, dtype=np.int64)
        gripper = np.asarray(
            [candidates[i].gripper_opening for i in indices], dtype=np.float64
        )
        eef = np.asarray([candidates[i].eef_position for i in indices], np.float64)
        obj = np.asarray([candidates[i].object_position for i in indices], np.float64)
        time = np.asarray(
            [candidates[i].normalized_time for i in indices], dtype=np.float64
        )
        angle = np.asarray(
            [candidates[i].orientation_rad for i in indices], dtype=np.float64
        )
        symmetry = candidates[indices[0]].symmetry_order

        # Vectorised superset prefilter, evaluated row-block by row-block so the
        # pairwise matrices never materialise for the whole bucket at once.
        for start in range(len(indices) - 1):
            tail = slice(start + 1, len(indices))
            gripper_difference = np.abs(gripper[start] - gripper[tail])
            eef_difference = np.linalg.norm(eef[start] - eef[tail], axis=1)
            object_difference = np.linalg.norm(obj[start] - obj[tail], axis=1)
            time_difference = np.abs(time[start] - time[tail])
            # symmetry-aware circular difference, folded into [0, pi/s]
            raw_difference = np.abs(angle[start] - angle[tail]) * symmetry
            folded = np.abs(
                np.arctan2(np.sin(raw_difference), np.cos(raw_difference))
            ) / symmetry
            orientation_deg = np.degrees(folded)
            keep = (
                (gripper_difference <= GRIPPER_TOLERANCE * PREFILTER_SLACK)
                & (eef_difference <= EEF_TOLERANCE * PREFILTER_SLACK)
                & (object_difference <= OBJECT_TOLERANCE * PREFILTER_SLACK)
                & (time_difference <= TIME_TOLERANCE * PREFILTER_SLACK)
                & (orientation_deg >= ORIENTATION_MIN_DEG / PREFILTER_SLACK)
                & (orientation_deg <= ORIENTATION_MAX_DEG * PREFILTER_SLACK)
            )
            comparisons += int(keep.size)
            if not keep.any():
                continue
            for offset in np.flatnonzero(keep):
                left = int(order[start])
                right = int(order[start + 1 + int(offset)])
                # Final authority: the frozen implementation itself.
                result = pair_eligibility(
                    candidates[left], candidates[right], mode="confirmatory"
                )
                if result.eligible:
                    confirmed.append(
                        (
                            left,
                            right,
                            float(result.orientation_difference_deg),
                            float(result.standardized_matching_distance),
                        )
                    )
                else:
                    reasons.update(result.reasons)
    return confirmed, reasons, comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    _require(
        not output_root.exists() and not output_root.is_symlink(),
        "output root must be absent",
    )

    cohort = load_feature_cohort(
        args.cohort_root.resolve() / EXPECTED_COHORT_SHA256, EXPECTED_COHORT_SHA256
    )
    candidates = _build_candidates(cohort, args.raw_root.resolve())
    _require(len(candidates) == 9455, f"expected 9455 states, built {len(candidates)}")

    confirmed, reasons, comparisons = _eligible_pairs(candidates)

    by_id = {index: candidates[index] for index in range(len(candidates))}
    cross_episode = sum(
        1
        for left, right, _, _ in confirmed
        if by_id[left].candidate_id.split("@")[0]
        != by_id[right].candidate_id.split("@")[0]
    )
    cross_cluster = sum(
        1
        for left, right, _, _ in confirmed
        if by_id[left].base_init_id != by_id[right].base_init_id
    )
    participating = {index for pair in confirmed for index in pair[:2]}
    clusters = {by_id[index].base_init_id for index in participating}
    orientations = [entry[2] for entry in confirmed]

    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration_causal_pair_inventory",
        "split": "calibration",
        "locked_test_accessed": False,
        "mode": "confirmatory",
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "tolerances": {
            "gripper_opening": GRIPPER_TOLERANCE,
            "eef_position_m": EEF_TOLERANCE,
            "object_position_m": OBJECT_TOLERANCE,
            "normalized_time": TIME_TOLERANCE,
            "orientation_degrees": [ORIENTATION_MIN_DEG, ORIENTATION_MAX_DEG],
        },
        "candidate_states": len(candidates),
        "prefilter_comparisons": comparisons,
        "eligible_pairs": len(confirmed),
        "eligible_pairs_cross_episode": cross_episode,
        "eligible_pairs_cross_cluster": cross_cluster,
        "participating_states": len(participating),
        "participating_clusters": sorted(clusters),
        "orientation_difference_deg": {
            "min": min(orientations) if orientations else None,
            "median": float(np.median(orientations)) if orientations else None,
            "max": max(orientations) if orientations else None,
        },
        "exclusion_reasons_among_prefiltered": dict(reasons),
        "confirmatory_pair_minimum": CONFIRMATORY_PAIR_MINIMUM,
        "gate": {
            "criterion": "eligible pairs available for alpha calibration",
            "meets_locked_test_minimum": len(confirmed) >= CONFIRMATORY_PAIR_MINIMUM,
        },
    }

    output_root.mkdir(parents=True, exist_ok=False)
    digest = _write_exclusive(
        output_root / "pair-inventory.json", _canonical(summary)
    )
    print(json.dumps({"receipt_sha256": digest, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
