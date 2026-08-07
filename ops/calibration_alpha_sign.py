#!/usr/bin/env python3
"""Sign half of the Section 10 alpha calibration: patched forward passes on GPU.

``PREREG.md:345-347`` freezes alpha as the smallest value in {0.25, 0.5, 1.0}
with the expected target-action sign and an off-manifold rate <= 5%.  The
off-manifold half already passed for every alpha
(`calibration_alpha_offmanifold.py`), so this script evaluates the remaining
condition: does patching the recipient toward the donor along the frozen probe
subspace move the yaw action in the donor-aligned direction?

For each selected pair the recipient episode is replayed to the recipient
control step, one unpatched action chunk is taken as the baseline, and one
patched chunk is taken with ``h_r' = h_r + alpha * P(h_d - h_r)`` applied at the
selected activation location.  The donor's factual actions come from the frozen
score sidecar, so no donor replay is needed.

This is Calibration only: it selects the strength and never touches Locked Test.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from mech_int_vla.artifacts import load_rollout_artifact
from mech_int_vla.causal import (
    PATCH_ALPHA_GRID,
    CandidateState,
    probe_patch_shift,
    select_pairs_for_three_seeds,
    summarize_action_effect,
)
from mech_int_vla.config import ConditionSpec, SplitName, load_protocol_config
from mech_int_vla.scoring import CHUNK_ACTIONS
from mech_int_vla.feature_artifacts import load_feature_cohort, load_feature_reference_bundle
from mech_int_vla.features import _relative_quaternion, _xyzw_to_wxyz, _yaw_wxyz
from mech_int_vla.instrumentation import SmolVLAInstrumentation
from mech_int_vla.libero_runtime import RawLiberoEpisode
from mech_int_vla.manifest import reconstruct_episode_manifest
from mech_int_vla.probe_artifacts import load_bound_probe_artifact
from contextlib import nullcontext

from mech_int_vla.scoring_runtime import (
    _clone_batch,
    SmolVLAScoringAdapter,
    candidate_target,
    factual_replay_from_artifact,
)
from mech_int_vla.snapshots import load_locked_smolvla, resolve_snapshot_paths

SCHEMA_VERSION = 1
COLLECTION_COMMIT = "18d64941bc8c899b06306fbec21d1c8d2c08f2ea"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
MANIFEST_SHA256 = "6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f"
PAIRING_SEEDS = (260803, 260804, 260805)
SIGN_MAJORITY = 0.5


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> str:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _split_candidate(candidate_id: str) -> tuple[str, int]:
    episode_id, _, step = candidate_id.rpartition("@")
    return episode_id, int(step)


def _build_candidates(cohort, raw_root: Path) -> list[CandidateState]:
    symmetry = int(cohort.task_identity.symmetry_order)
    by_episode: dict[str, list] = defaultdict(list)
    for record in cohort.records:
        by_episode[record.episode_id].append(record)
    out: list[CandidateState] = []
    for episode_id in sorted(by_episode):
        d = raw_root / "calibration" / episode_id
        max_steps = int(json.loads((d / "metadata.json").read_text())["execution"]["max_steps"])
        with np.load(d / "trajectory.npz") as h:
            ph, ep, eq = h["frame_phase"], h["frame_eef_position"], h["frame_eef_quaternion_xyzw"]
            op, oq = h["frame_primary_object_position"], h["frame_primary_object_quaternion_wxyz"]
            gr, ct, pr = h["frame_gripper_qpos"], h["frame_primary_gripper_contact"], h["frame_task_predicates"]
        for record in by_episode[episode_id]:
            s = int(record.control_step)
            rel = _relative_quaternion(
                _xyzw_to_wxyz(np.asarray(eq[s], np.float64)), np.asarray(oq[s], np.float64)
            )
            out.append(
                CandidateState(
                    candidate_id=f"{episode_id}@{s:04d}",
                    base_init_id=record.base_init_state_id,
                    phase=str(ph[s]),
                    contact=bool(ct[s]),
                    gripper_opening=float(np.mean(gr[s])),
                    eef_position=tuple(float(v) for v in ep[s]),
                    object_position=tuple(float(v) for v in op[s]),
                    normalized_time=s / max_steps,
                    non_primary_predicates=tuple(
                        (f"predicate_{i}", bool(v)) for i, v in enumerate(pr[s])
                    ),
                    orientation_rad=float(_yaw_wxyz(rel)),
                    symmetry_order=symmetry,
                )
            )
    return out


def _sidecar_arrays(score_root: Path, episode_id: str) -> dict[str, np.ndarray]:
    path = score_root / "calibration" / episode_id / "primitives.npz"
    with np.load(path) as handle:
        return {k: np.asarray(handle[k]) for k in ("control_step", "original_actions", "original_activation")}


def _action_chunk(adapter, processed, noise, *, shift=None, location=None,
                  denoising_step=None):
    """Run one action chunk, optionally under an external activation patch.

    ``SmolVLAScoringAdapter.predict_action_chunk`` cannot be used here: it
    validates that its internal ``patched`` flag matches the observed patch
    marker, and that flag is set solely by its frozen probe-shift intervention,
    so a Section 10 donor patch is rejected by design.  This reproduces the
    adapter's inference core exactly — same ``inference_mode``, same
    instrumentation scope, same defensive batch clone, same postprocessor — and
    returns only the action chunk, which is all the sign condition needs.  The
    activation is not read, so the marker check is not needed.
    """

    policy = adapter.policy_runtime.policy
    patch_context = (
        nullcontext()
        if shift is None
        else adapter.instrumentation.patch(
            location, shift, denoising_step=denoising_step
        )
    )
    with torch.inference_mode(), adapter.instrumentation:
        with patch_context:
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
    # The policy emits 50 actions per chunk while the frozen sidecars keep the
    # first CHUNK_ACTIONS, which is also the window summarize_action_effect
    # compares.  Trim here so recipient, donor and patched arrays align.
    return np.asarray(actions[:CHUNK_ACTIONS], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("repo-root", "environment-lock", "cache-dir", "manifest", "raw-root",
                 "score-root", "feature-root", "bound-probe", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--cohort-sha256", required=True)
    p.add_argument("--max-pairs", type=int, default=60)
    # The frozen selection is O(n^2) over 9,455 candidates and takes ~35 minutes.
    # It depends only on factual state, so caching it is a pure speedup with no
    # effect on which pairs are chosen.
    p.add_argument("--pairs-cache", type=Path, default=None)
    args = p.parse_args()

    root = args.repo_root.resolve()
    out = args.output_root.resolve()
    _require(not out.exists(), "output root must be absent")

    protocol = load_protocol_config(root / "configs")
    task = protocol.task_order.tasks[0]
    manifest = reconstruct_episode_manifest(
        SplitName.CALIBRATION, task, protocol,
        policy_revision=POLICY_REVISION, code_commit=COLLECTION_COMMIT,
    )
    spec_by_episode = {e.episode_id: e for e in manifest.episodes}

    bound = load_bound_probe_artifact(
        args.bound_probe.resolve(), protocol=protocol, repo_root=root,
        expected_sha256=args.bound_probe.resolve().name,
    )
    coefficient = np.asarray(bound.probe.model.coefficient, dtype=np.float64)
    # The candidate name encodes hook *and* flow step ("early_expert_t1_0");
    # the instrumentation wants them separately.  Resolve with the same frozen
    # mapping the scoring adapter uses, so the patch lands exactly where the
    # probe was fitted.
    location, denoising_step = candidate_target(bound.probe.candidate)
    candidate_name = bound.probe.candidate

    cohort = load_feature_cohort(
        args.feature_root.resolve() / "cohort" / args.cohort_sha256, args.cohort_sha256
    )
    reference = load_feature_reference_bundle(
        sorted(glob.glob(str(args.feature_root.resolve() / "reference" / "*")))[0]
    )
    action_scale = np.asarray(reference.action_scale.values, dtype=np.float64)

    cache = args.pairs_cache
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        _require(cached["cohort_sha256"] == args.cohort_sha256,
                 "pair cache was built for a different cohort")
        _require(cached["pairing_seeds"] == list(PAIRING_SEEDS),
                 "pair cache was built for different seeds")
        pairs = [
            (int(e["seed"]), SimpleNamespace(
                recipient_id=e["recipient_id"], donor_id=e["donor_id"],
                orientation_difference_deg=float(e["orientation_difference_deg"])))
            for e in cached["pairs"]
        ]
        print(json.dumps({"kind": "pairs_loaded_from_cache", "pairs": len(pairs)}), flush=True)
    else:
        candidates = _build_candidates(cohort, args.raw_root.resolve())
        selection = select_pairs_for_three_seeds(candidates, seeds=PAIRING_SEEDS)
        pairs = [(sel.seed, pr) for sel in selection.selections for pr in sel.pairs]
        if cache is not None:
            payload = {
                "cohort_sha256": args.cohort_sha256,
                "pairing_seeds": list(PAIRING_SEEDS),
                "pairs": [
                    {"seed": int(sd), "recipient_id": pr.recipient_id,
                     "donor_id": pr.donor_id,
                     "orientation_difference_deg": float(pr.orientation_difference_deg)}
                    for sd, pr in pairs
                ],
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            print(json.dumps({"kind": "pairs_cached", "pairs": len(pairs)}), flush=True)
    pairs = pairs[: args.max_pairs]
    _require(bool(pairs), "no pairs selected")

    snapshots = resolve_snapshot_paths(
        args.environment_lock.resolve(), cache_dir=args.cache_dir.resolve(), local_files_only=True
    )
    policy_runtime = load_locked_smolvla(snapshots, device="cuda")

    results: list[dict[str, Any]] = []
    for seed, pair in pairs:
        r_episode, r_step = _split_candidate(pair.recipient_id)
        d_episode, d_step = _split_candidate(pair.donor_id)
        r_side = _sidecar_arrays(args.score_root.resolve(), r_episode)
        d_side = _sidecar_arrays(args.score_root.resolve(), d_episode)
        r_index = int(np.flatnonzero(r_side["control_step"] == r_step)[0])
        d_index = int(np.flatnonzero(d_side["control_step"] == d_step)[0])
        recipient_activation = r_side["original_activation"][r_index].mean(axis=0)
        donor_activation = d_side["original_activation"][d_index].mean(axis=0)
        donor_actions = d_side["original_actions"][d_index, 0]

        spec = spec_by_episode[r_episode]
        artifact = load_rollout_artifact(args.raw_root.resolve() / "calibration" / r_episode,
                                         expected_task=task)
        episode = RawLiberoEpisode.create(
            task, base_init_state_id=spec.base_init_state_id,
            execution=protocol.split.policy_execution, validity=protocol.perturbations.validity,
        )
        instrumentation = SmolVLAInstrumentation(policy_runtime.policy)
        try:
            adapter = SmolVLAScoringAdapter(
                episode, policy_runtime, artifact, bound, instrumentation,
                reset_seed=spec.reset_seed,
                original_condition=ConditionSpec(
                    spec.condition_name, spec.condition_family,
                    spec.condition_index, spec.condition_parameters,
                ),
                protocol=protocol, repo_root=root,
            )
            replay = factual_replay_from_artifact(artifact)
            entry: dict[str, Any] = {
                "seed": int(seed), "recipient_id": pair.recipient_id, "donor_id": pair.donor_id,
                "orientation_difference_deg": float(pair.orientation_difference_deg),
                "alphas": {},
            }
            # NOTE: ``predict_action_chunk`` enters ``self.instrumentation`` as a
            # context manager itself, and its ``__exit__`` calls ``remove()``.
            # Every adapter inference therefore leaves the hooks uninstalled, so
            # an outer ``with instrumentation`` block would be undone by the
            # first forward pass.  ``patch()`` refuses to run uninstalled, so it
            # must be re-installed immediately before each patch context; the
            # patch itself is carried by a context variable that the hooks read
            # during the forward, which the adapter re-installs anyway.
            if True:
                frame = adapter.reset_replay()
                for step_index in range(r_step):
                    frame = adapter.step_replay(replay.actions[step_index]).frame
                adapter.begin_score_state()
                processed = adapter.process_observation(frame)
                noise = adapter.noise_for_seed(int(r_side["control_step"][r_index]))

                baseline = _action_chunk(adapter, processed, noise)
                for alpha in PATCH_ALPHA_GRID:
                    shift = probe_patch_shift(
                        recipient_activation, donor_activation, coefficient, alpha=float(alpha)
                    )
                    tensor = torch.as_tensor(
                        np.array(shift, dtype=np.float32, copy=True), device=adapter.device
                    )
                    patched = _action_chunk(
                        adapter, processed, noise, shift=tensor,
                        location=location, denoising_step=denoising_step,
                    )
                    summary = summarize_action_effect(
                        baseline, donor_actions, patched, action_scale=action_scale
                    )
                    entry["alphas"][str(alpha)] = {
                        "sign_correct": bool(summary.sign_correct),
                        "target_effect": float(summary.target_effect),
                        "donor_aligned_target_effect": float(summary.donor_aligned_target_effect),
                        "off_target_ratio": float(summary.off_target_ratio),
                    }
            results.append(entry)
        finally:
            instrumentation.remove()
            episode.close()

    per_alpha = {}
    for alpha in PATCH_ALPHA_GRID:
        flags = [r["alphas"][str(alpha)]["sign_correct"] for r in results]
        effects = [r["alphas"][str(alpha)]["donor_aligned_target_effect"] for r in results]
        rate = float(np.mean(flags)) if flags else 0.0
        per_alpha[str(alpha)] = {
            "sign_correct_rate": rate,
            "exceeds_half": rate > SIGN_MAJORITY,
            "median_donor_aligned_effect": float(np.median(effects)) if effects else None,
            "pairs": len(flags),
        }
    eligible = [a for a in PATCH_ALPHA_GRID if per_alpha[str(a)]["exceeds_half"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration_alpha_sign_report",
        "split": "calibration",
        "locked_test_accessed": False,
        "selected_candidate": candidate_name,
        "selected_location": location,
        "denoising_step": denoising_step,
        "cohort_sha256": args.cohort_sha256,
        "pairing_seeds": list(PAIRING_SEEDS),
        "evaluated_pairs": len(results),
        "per_alpha": per_alpha,
        "alphas_with_expected_sign": eligible,
        "frozen_alpha": (min(eligible) if eligible else None),
        "pairs": results,
    }
    out.mkdir(parents=True, exist_ok=False)
    digest = _write_exclusive(out / "alpha-sign.json", _canonical(summary))
    print(json.dumps({"receipt_sha256": digest, **{k: v for k, v in summary.items() if k != "pairs"}},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
