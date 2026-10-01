"""Copy every number and image the blog post needs into blog/data/.

Reads only frozen artifacts (no model is fitted here):
  artifacts/locked-test-final-report/report.json
  artifacts/exploratory/internals-only-locked-test/results.json
  <mirror>/research-artifacts/postscore/causal/<sha>/evidence/*.json
  <mirror>/research-artifacts/raw/locked_test/<episode>/trajectory.npz   (camera frames)

<mirror> is the Locked Test rollout mirror, which now lives on the external drive.
Run with any Python that has numpy and Pillow:
  python blog/src/extract_data.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blog" / "data"
FRAMES = ROOT / "blog" / "frames"
MIRROR = Path("/Volumes/VERBATIM HD/mech-int-vla-month1-2026-09-24/artifacts/locked-test-run-51243106")
RAW = MIRROR / "research-artifacts" / "raw" / "locked_test"
CAUSAL = MIRROR / "research-artifacts" / "postscore" / "causal" / \
    "5fc67e65adb46f5b8f201e1d18a9e45e7ffd4444d09dd780468c3001dc6ba5e5" / "evidence"


def metrics():
    rep = json.load(open(ROOT / "artifacts/locked-test-final-report/report.json"))["sections"]
    prim, lead, causal = rep[1]["result"], rep[4]["result"], rep[6]["result"]
    exp = json.load(open(ROOT / "artifacts/exploratory/internals-only-locked-test/results.json"))

    def row(k):
        return {"auroc": exp[k]["auroc"], "auroc_ci": exp[k]["auroc_ci"],
                "log_loss": exp[k]["log_loss"], "log_loss_ci": exp[k]["log_loss_ci"]}

    m1_ll = prim["model1_log_loss"]
    sm = causal["selected_layer_summary"]
    return {
        # Episode-level AUROC (mean of the primary-step probabilities per episode),
        # frozen Platt-calibrated receipt predictions, 90 % cluster bootstrap.
        "frozen": {m: row(f"frozen_{m}") for m in ("M0", "M1", "M2")},
        # Preregistered primary estimand (paired log loss, M2 vs M1).
        "primary": {
            "relative_lift": prim["relative_lift"],
            "lift_ci": [-prim["delta_interval"]["upper"] / m1_ll, -prim["delta_interval"]["lower"] / m1_ll],
            "bar": 0.03,
            "episodes": prim["episodes"], "clusters": prim["clusters"],
            "step_auroc": {"M1": prim["model1_auroc"], "M2": prim["model2_auroc"]},
        },
        # Exploratory re-fits (fit on Calibration, evaluated on the Locked Test).
        "refit": {
            "M0": row("M0"), "M0+act": row("M0+act"), "M1": row("M1"), "M0+M2inc": row("M0+M2inc"),
            "act_vs_M0": {"lift": exp["M0+act_vs_M0"]["relative_lift"], "ci": exp["M0+act_vs_M0"]["relative_lift_ci"]},
            "act_vs_M1": {"lift": exp["M0+act_vs_M1"]["relative_lift"], "ci": exp["M0+act_vs_M1"]["relative_lift_ci"]},
        },
        "lead": {
            "failed": lead["failed_episodes"],
            "detection_M1": lead["model1_detection_rate"], "detection_M2": lead["model2_detection_rate"],
            "median_paired_difference": lead["median_paired_difference"],
            "same_step_pairs": 42,  # report Fig. 7a caption: 42/60 pairs on the diagonal
        },
        "causal_summary": {k: sm[k] for k in (
            "valid_pairs", "sign_correct_count", "median_off_target_ratio",
            "median_donor_aligned_target_effect", "random_control_95th_percentile",
            "matched_control_sign_rate", "off_manifold_rate")},
    }


def causal_null():
    tgt, rc = [], []
    for f in sorted(glob.glob(str(CAUSAL / "*.json"))):
        if Path(f).name.startswith("._"):
            continue
        e = json.load(open(f))
        if not e["pair"]["valid"]:
            continue
        tgt.append(e["selected_patch"]["donor_aligned_target_effect"])
        rc.append([r["donor_aligned_target_effect"] for r in e["random_controls"]])
    tgt, R = np.array(tgt), np.array(rc)                  # (52,), (52, 1000)
    null_medians = np.median(R, axis=0)                     # one value per random direction
    p95_pair = np.percentile(R, 95, axis=1)
    return {
        "pairs": len(tgt),
        "probe_median": float(np.median(tgt)),
        "null_medians": null_medians.tolist(),
        "null_p95": float(np.percentile(null_medians, 95)),
        "pairs_beating_own_p95": int(np.sum(tgt > p95_pair)),
    }


def frames():
    """Agent-view frames, rotated 180 degrees (LIBERO renders them upside down)."""
    FRAMES.mkdir(parents=True, exist_ok=True)
    info = {}

    def load(cell):
        z = np.load(RAW / f"libero_10-task5-locked_test-init30-cell{cell}" / "trajectory.npz")
        return z["frame_agentview_image"][:, ::-1, ::-1], z["frame_control_step"], bool(z["frame_task_success"].any())

    # Figure 1: two runs from init 30, unchanged (cell 0) and book turned +37.5 deg (cell 4).
    for cell, name in ((0, "success"), (4, "failure")):
        imgs, steps, ok = load(cell)
        idx = np.linspace(0, len(imgs) - 1, 5).round().astype(int)
        info[name] = {"cell": cell, "success": ok, "steps": [int(steps[i]) for i in idx]}
        for j, i in enumerate(idx):
            Image.fromarray(np.ascontiguousarray(imgs[i])).save(FRAMES / f"run-{name}-{j}.jpg", quality=88)
    # Figure 2: the four kinds of change, first frame of init 30.
    for cell, name in ((0, "unchanged"), (4, "book-turned"), (5, "book-moved"), (7, "camera-turned")):
        imgs, _, ok = load(cell)
        Image.fromarray(np.ascontiguousarray(imgs[0])).save(FRAMES / f"change-{name}.jpg", quality=90)
        info[name] = {"cell": cell, "success": ok}
    return info


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(metrics(), open(OUT / "metrics.json", "w"), indent=1)
    json.dump(causal_null(), open(OUT / "causal_null.json", "w"))
    json.dump(frames(), open(OUT / "frames.json", "w"), indent=1)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
