#!/usr/bin/env python3
"""Off-manifold half of the Section 10 alpha calibration, on CPU.

``PREREG.md:345-347`` selects the patch strength alpha from {0.25, 0.5, 1.0} as
the smallest value with the expected target-action sign **and** an off-manifold
rate no greater than 5%.  The two conditions need very different resources: the
sign condition requires a patched forward pass per pair on the GPU, while the
off-manifold condition is a pure geometry check on the patched activation
vector.

This script evaluates only the second condition.  It is deliberately run first,
because an alpha whose off-manifold rate already exceeds 5% is disqualified
regardless of its action effect, and finding that out costs no GPU time.  If
every alpha fails here, patching is unusable on this model and the expensive
half never needs to run.

Off-manifold is defined against the natural Calibration activation distribution:
a patched activation is off-manifold when its mean distance to its five nearest
natural neighbours strictly exceeds the 95th percentile of the same statistic
computed over natural activations themselves.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mech_int_vla.causal import (
    PATCH_ALPHA_GRID,
    CandidateState,
    five_nearest_neighbor_distance,
    off_manifold_flag,
    patch_activation,
    select_pairs_for_three_seeds,
)
from mech_int_vla.feature_artifacts import load_feature_cohort
from mech_int_vla.features import _relative_quaternion, _xyzw_to_wxyz, _yaw_wxyz

SCHEMA_VERSION = 1
OFF_MANIFOLD_MAX_RATE = 0.05
NATURAL_PERCENTILE = 95.0
# PREREG.md:355 fixes 60 attempted pairs balanced over three deterministic seeds.
PAIRING_SEEDS = (260803, 260804, 260805)
NATURAL_QUERY_SAMPLE = 3000


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _build_candidates(cohort, raw_root: Path) -> list[CandidateState]:
    symmetry_order = int(cohort.task_identity.symmetry_order)
    by_episode: dict[str, list] = defaultdict(list)
    for record in cohort.records:
        by_episode[record.episode_id].append(record)
    candidates: list[CandidateState] = []
    for episode_id in sorted(by_episode):
        episode_dir = raw_root / "calibration" / episode_id
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        max_steps = int(metadata["execution"]["max_steps"])
        with np.load(episode_dir / "trajectory.npz") as handle:
            phase = handle["frame_phase"]
            eef_position = handle["frame_eef_position"]
            eef_quat = handle["frame_eef_quaternion_xyzw"]
            object_position = handle["frame_primary_object_position"]
            object_quat = handle["frame_primary_object_quaternion_wxyz"]
            gripper = handle["frame_gripper_qpos"]
            contact = handle["frame_primary_gripper_contact"]
            predicates = handle["frame_task_predicates"]
        for record in by_episode[episode_id]:
            step = int(record.control_step)
            relative = _relative_quaternion(
                _xyzw_to_wxyz(np.asarray(eef_quat[step], np.float64)),
                np.asarray(object_quat[step], np.float64),
            )
            candidates.append(
                CandidateState(
                    candidate_id=f"{episode_id}@{step:04d}",
                    base_init_id=record.base_init_state_id,
                    phase=str(phase[step]),
                    contact=bool(contact[step]),
                    gripper_opening=float(np.mean(gripper[step])),
                    eef_position=tuple(float(v) for v in eef_position[step]),
                    object_position=tuple(float(v) for v in object_position[step]),
                    normalized_time=step / max_steps,
                    non_primary_predicates=tuple(
                        (f"predicate_{i}", bool(v)) for i, v in enumerate(predicates[step])
                    ),
                    orientation_rad=float(_yaw_wxyz(relative)),
                    symmetry_order=symmetry_order,
                )
            )
    return candidates


def _activation_index(score_root: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Map candidate_id -> mean selected activation, plus the natural matrix."""

    by_candidate: dict[str, np.ndarray] = {}
    natural: list[np.ndarray] = []
    for path in sorted(glob.glob(str(score_root / "calibration" / "*" / "primitives.npz"))):
        episode_id = Path(path).parent.name
        with np.load(path) as handle:
            steps = np.asarray(handle["control_step"])
            activation = np.asarray(handle["original_activation"], dtype=np.float64)
        # The frozen probe consumes one activation per state; average the eight
        # noise draws, which is how the factual state is represented.
        per_state = activation.mean(axis=1)
        for index, step in enumerate(steps):
            by_candidate[f"{episode_id}@{int(step):04d}"] = per_state[index]
        natural.append(per_state)
    return by_candidate, np.concatenate(natural, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    _require(not output_root.exists(), "output root must be absent")

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    coefficient = np.asarray(probe["parameters"]["coefficient"], dtype=np.float64)
    _require(coefficient.ndim == 2 and coefficient.shape[0] == 2,
             "probe coefficient must be the 2 x d circular map")

    cohort = load_feature_cohort(
        args.feature_root.resolve() / "cohort" / args.cohort_sha256, args.cohort_sha256
    )
    candidates = _build_candidates(cohort, args.raw_root.resolve())
    by_candidate, natural = _activation_index(args.score_root.resolve())
    _require(len(candidates) == 9455, f"expected 9455 candidates, got {len(candidates)}")
    _require(natural.shape[0] == 9455, f"expected 9455 natural rows, got {natural.shape[0]}")

    selection = select_pairs_for_three_seeds(candidates, seeds=PAIRING_SEEDS)
    pairs = [pair for sel in selection.selections for pair in sel.pairs]
    _require(bool(pairs), "no confirmatory pairs were selected")

    # Natural 5-NN reference distribution.  Each natural query excludes itself by
    # construction, since its own distance is zero and would otherwise dominate.
    rng = np.random.default_rng(260803)
    sample_index = rng.choice(natural.shape[0], size=min(NATURAL_QUERY_SAMPLE, natural.shape[0]), replace=False)
    natural_distances = []
    for i in sample_index:
        others = np.delete(natural, i, axis=0)
        natural_distances.append(five_nearest_neighbor_distance(natural[i], others))
    natural_95 = float(np.percentile(natural_distances, NATURAL_PERCENTILE))

    per_alpha: dict[str, Any] = {}
    for alpha in PATCH_ALPHA_GRID:
        flags = []
        distances = []
        for pair in pairs:
            recipient = by_candidate[pair.recipient_id]
            donor = by_candidate[pair.donor_id]
            patched = patch_activation(recipient, donor, coefficient, alpha=float(alpha))
            distance = five_nearest_neighbor_distance(patched, natural)
            check = off_manifold_flag(
                patched_five_nn_distance=distance, natural_95th_percentile=natural_95
            )
            flags.append(bool(check.off_manifold))
            distances.append(float(distance))
        rate = float(np.mean(flags))
        per_alpha[str(alpha)] = {
            "off_manifold_rate": rate,
            "satisfies_5pct_constraint": rate <= OFF_MANIFOLD_MAX_RATE,
            "pairs": len(flags),
            "patched_5nn_distance": {
                "median": float(np.median(distances)),
                "p95": float(np.percentile(distances, 95)),
                "max": float(np.max(distances)),
            },
        }

    eligible = [a for a in PATCH_ALPHA_GRID if per_alpha[str(a)]["satisfies_5pct_constraint"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration_alpha_off_manifold_report",
        "split": "calibration",
        "locked_test_accessed": False,
        "note": "off-manifold condition only; the sign condition needs GPU forward passes",
        "cohort_sha256": args.cohort_sha256,
        "pairing_seeds": list(PAIRING_SEEDS),
        "attempted_pairs": selection.attempted_pairs,
        "evaluated_pairs": len(pairs),
        "natural_95th_percentile": natural_95,
        "natural_query_sample": int(len(sample_index)),
        "alpha_grid": list(PATCH_ALPHA_GRID),
        "per_alpha": per_alpha,
        "alphas_passing_off_manifold": eligible,
        "smallest_passing_alpha": (min(eligible) if eligible else None),
        "gpu_stage_required": bool(eligible),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    digest = _write_exclusive(output_root / "alpha-off-manifold.json", _canonical(summary))
    print(json.dumps({"receipt_sha256": digest, **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
