#!/usr/bin/env python
"""Publication figures for the Locked Test final report.

Reads ONLY frozen artifacts:
  artifacts/locked-test-final-report/report.json
  artifacts/locked-test-final-report/predictions.json
  artifacts/locked-test-run-51243106/research-artifacts/postscore/causal/<sha>/evidence/*.json
  artifacts/locked-test-final-report/sensitivity-receipt.json
  locks/calibration_frozen.json
and writes PDF (vector) + PNG (300 dpi) versions of every figure into
artifacts/locked-test-final-report/figures/.  Idempotent: re-running overwrites
the same files from the same inputs.  No model is fitted; every number shown is
either copied from the frozen receipts or recomputed from the frozen per-step
predictions with the preregistered rule (each episode has total weight one,
primary steps {0, 50, 100, 150, 200}) and asserted against report.json.

Palette (dataviz skill reference instance, validated with validate_palette.js):
  M0 = neutral grey (de-emphasis hue), M1 = orange slot, M2 = blue slot.
"""
from __future__ import annotations

import collections
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent                 # artifacts/locked-test-final-report
ROOT = HERE.parents[1]                                 # repository root
FIG_DIR = HERE / "figures"
REPORT = HERE / "report.json"
PREDICTIONS = HERE / "predictions.json"
SENSITIVITY = HERE / "sensitivity-receipt.json"
CALIB_LOCK = ROOT / "locks" / "calibration_frozen.json"
POSTSCORE = ROOT / "artifacts" / "locked-test-run-51243106" / "research-artifacts" / "postscore"

# --------------------------------------------------------------------------- palette
C = {"M0": "#898781", "M1": "#eb6834", "M2": "#2a78d6"}       # grey / orange / blue
LIGHT = {"M0": "#c3c2b7", "M1": "#f5b39a", "M2": "#9ec5f4"}   # lighter step of the same hue
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
MODEL_LABEL = {"M0": "M0 (vision baseline)", "M1": "M1 (privileged state)", "M2": "M2 (internals)"}
MODEL_SHORT = {"M0": "M0\nvision", "M1": "M1\nstate", "M2": "M2\ninternals"}

CELL_NAMES = {
    "iid": "iid (no shift)",
    "yaw_neg_37_5": "object yaw −37.5°",
    "yaw_neg_22_5": "object yaw −22.5°",
    "yaw_pos_22_5": "object yaw +22.5°",
    "yaw_pos_37_5": "object yaw +37.5°",
    "planar_diagonal_4_5cm": "planar diagonal 4.5 cm",
    "camera_yaw_neg_5": "camera yaw −5°",
    "camera_yaw_pos_5": "camera yaw +5°",
}
CELL_SHORT = {
    "iid": "iid", "yaw_neg_37_5": "yaw −37.5°", "yaw_neg_22_5": "yaw −22.5°",
    "yaw_pos_22_5": "yaw +22.5°", "yaw_pos_37_5": "yaw +37.5°",
    "planar_diagonal_4_5cm": "planar 4.5 cm", "camera_yaw_neg_5": "cam −5°",
    "camera_yaw_pos_5": "cam +5°",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.unicode_minus": True,
})


# --------------------------------------------------------------------------- helpers
def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def hline(ax, y, label=None, color=INK2, ls=(0, (3, 2)), lw=0.8, **kw):
    ax.axhline(y, color=color, ls=ls, lw=lw, zorder=1, label=label, **kw)


def vline(ax, x, label=None, color=INK2, ls=(0, (3, 2)), lw=0.8, **kw):
    ax.axvline(x, color=color, ls=ls, lw=lw, zorder=1, label=label, **kw)


def panel_label(ax, s, x=-0.02, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK,
            ha="right", va="bottom")


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {name}.pdf / {name}.png")


def spread(values, min_gap):
    """Return label y-positions with at least `min_gap` between neighbours (order preserved)."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_gap:
            pos[i] = pos[i - 1] + min_gap
    # re-centre so the group stays around its original mean
    pos += np.mean(np.array(values, float)[order]) - pos.mean()
    out = np.empty_like(pos)
    out[order] = pos
    return out


# ---- figure-level legend for the three models
def model_legend(fig, y=0.985, models=("M0", "M1", "M2"), **kw):
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[m], markersize=5.5, label=MODEL_LABEL[m])
               for m in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(models),
               fontsize=7, columnspacing=1.6, handletextpad=0.4, **kw)


# --------------------------------------------------------------------------- data
def load_all():
    report = json.load(open(REPORT))
    S = {s["number"]: s for s in report["sections"]}
    preds = json.load(open(PREDICTIONS))
    calib = json.load(open(CALIB_LOCK))
    sens = json.load(open(SENSITIVITY))
    causal_sha = report["inputs"]["causal_receipt_sha256"]
    causal_dir = POSTSCORE / "causal" / causal_sha
    assert (causal_dir / "evidence").is_dir(), causal_dir
    return report, S, preds, calib, sens, causal_dir


def per_episode(preds, primary_steps):
    """Per-episode records grouped, plus per-episode primary-step log loss per model."""
    eps = collections.defaultdict(list)
    for r in preds["records"]:
        eps[r["episode_id"]].append(r)
    for v in eps.values():
        v.sort(key=lambda r: r["control_step"])
    ids = sorted(eps)
    y = np.array([float(eps[e][0]["terminal_failure_label"]) for e in ids])
    P = set(primary_steps)

    def ll(p, yy):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

    loss = {}
    for m in ("M0", "M1", "M2"):
        loss[m] = np.array([
            np.mean([ll(r["probabilities"][m], y[i]) for r in eps[e] if r["control_step"] in P])
            for i, e in enumerate(ids)
        ])
    return eps, ids, y, loss


def causal_pairs(causal_dir):
    rows = []
    for f in sorted(glob.glob(str(causal_dir / "evidence" / "*.json"))):
        e = json.load(open(f))
        if not e["pair"]["valid"]:
            continue
        rc = np.array([r["donor_aligned_target_effect"] for r in e["random_controls"]], float)
        rows.append(dict(
            pair_index=e["pair"]["pair_index"], cell=e["pair"]["condition_index"],
            seed=e["pair"]["seed"],
            target=e["selected_patch"]["donor_aligned_target_effect"],
            sign=bool(e["selected_patch"]["sign_correct"]),
            otr=e["selected_patch"]["off_target_ratio"],
            matched=e["matched_control"]["donor_aligned_target_effect"],
            matched_sign=bool(e["matched_control"]["sign_correct"]),
            rc=rc,
            nn=e["off_manifold"]["patched_five_nn_distance"],
            nat95=e["off_manifold"]["natural_95th_percentile"],
            off=bool(e["off_manifold"]["off_manifold"]),
        ))
    return rows


# --------------------------------------------------------------------------- figure 1
def fig_headline(S, loss, y):
    m3 = S[3]["models"]
    r2 = S[2]["result"]
    delta = loss["M2"] - loss["M1"]
    # cross-checks against the frozen report
    assert abs(delta.mean() - r2["delta_log_loss"]) < 1e-9, (delta.mean(), r2["delta_log_loss"])
    for m in ("M0", "M1", "M2"):
        assert abs(loss[m].mean() - m3[m]["log_loss"]) < 1e-9
    bar_abs = -S[2]["bar"]["minimum_relative_lift"] * r2["model1_log_loss"]   # -0.0156
    lo, hi = r2["delta_interval"]["lower"], r2["delta_interval"]["upper"]

    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.9], wspace=0.6, left=0.065, right=0.99,
                          top=0.8, bottom=0.16)
    metrics = [("auroc", "AUROC (higher better)"), ("brier", "Brier (lower better)"),
               ("log_loss", "Log loss (lower better)")]
    for k, (key, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, k])
        grid(ax)
        vals = [m3[m][key] for m in ("M0", "M1", "M2")]
        rng = max(m3[m][f"{key}_interval"]["upper"] for m in m3) - min(m3[m][f"{key}_interval"]["lower"] for m in m3)
        right = spread([vals[0], vals[2]], 0.075 * rng)
        ypos = [right[0], vals[1], right[1]]
        for i, m in enumerate(("M0", "M1", "M2")):
            v = m3[m][key]
            ci = m3[m][f"{key}_interval"]
            ax.errorbar(i, v, yerr=[[v - ci["lower"]], [ci["upper"] - v]], fmt="none",
                        ecolor=C[m], elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
            ax.scatter([i], [v], s=30, color=C[m], zorder=4, edgecolor=SURF, linewidth=0.8)
            if m == "M1":   # M1 sits between its neighbours: label it on the left
                ax.text(i - 0.2, ypos[i], f"{v:.3f}", fontsize=6.6, color=INK2, va="center", ha="right")
            else:
                ax.text(i + 0.2, ypos[i], f"{v:.3f}", fontsize=6.6, color=INK2, va="center", ha="left")
        ax.set_xticks([0, 1, 2], ["M0", "M1", "M2"])
        ax.set_xlim(-0.6, 2.95)
        ax.set_title(title, fontsize=7.6)
        if key == "auroc":
            ax.set_ylabel("value (90 % bootstrap CI)")
        if k == 0:
            panel_label(ax, "a", x=-0.3, y=1.06)
    model_legend(fig, y=0.995)

    # (b) paired per-episode delta
    ax = fig.add_subplot(gs[0, 3])
    grid(ax)
    bins = np.arange(-0.24, 0.2401, 0.01)
    counts, _, _ = ax.hist(delta, bins=bins, color=LIGHT["M2"], edgecolor=SURF, linewidth=0.8, zorder=2)
    vline(ax, 0, color=AXIS, ls="-", lw=0.8)
    vline(ax, delta.mean(), color=C["M2"], ls="-", lw=1.4, label="mean \u0394")
    ax.axvspan(lo, hi, color=C["M2"], alpha=0.14, lw=0, zorder=1, label="90 % CI")
    vline(ax, bar_abs, color=INK, label="prereg. bar")
    ax.set_xlabel("\u0394 log loss per episode (M2 \u2212 M1); negative favours M2")
    ax.set_ylabel("episodes")
    ax.set_xlim(-0.24, 0.24)
    ax.set_ylim(0, counts.max() * 1.5)
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.2, borderaxespad=0.3, labelspacing=0.35)
    ax.text(0.02, 0.97, f"n = {len(delta)} episodes, 20 clusters\n"
            f"mean \u0394 = {delta.mean():+.4f}\n90 % CI [{lo:+.4f}, {hi:+.4f}]\n"
            f"bar {bar_abs:+.4f} (\u22123 % of M1)\n"
            f"lift +{100 * r2['relative_lift']:.2f} %: bar not met",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=INK2, linespacing=1.3,
            bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"), zorder=6)
    panel_label(ax, "b", x=-0.1, y=1.06)
    ax.set_title("Primary estimand: paired \u0394 log loss, M2 vs M1", fontsize=7.6)

    # zoom inset around the estimate
    ins = ax.inset_axes([0.6, 0.36, 0.38, 0.2])
    ins.set_facecolor(SURF)
    ins.axvline(0, color=AXIS, lw=0.6)
    ins.hlines(0, lo, hi, color=C["M2"], lw=2.2)
    ins.scatter([delta.mean()], [0], s=26, color=C["M2"], zorder=3, edgecolor=SURF, linewidth=0.7)
    ins.axvline(bar_abs, color=INK, ls=(0, (3, 2)), lw=0.8)
    ins.set_xlim(-0.03, 0.015)
    ins.set_ylim(-1, 1)
    ins.set_yticks([])
    ins.set_xticks([-0.03, -0.0156, 0, 0.015], ["\u22120.03", "bar", "0", "0.015"], fontsize=5.8)
    ins.tick_params(length=2, pad=1.5)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)
    ins.set_title("zoom: mean and 90 % CI", fontsize=6, color=INK2, loc="left", pad=2, fontweight="normal")
    save(fig, "fig_headline")
    return delta


# --------------------------------------------------------------------------- figure 2
def fig_cells(S):
    cells = S[6]["cells"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.86, bottom=0.15)
    grid(ax, axis="x")
    ys = np.arange(len(cells))[::-1]
    for yy, c in zip(ys, cells):
        a = {m: c["models"][m]["auroc"] for m in ("M0", "M1", "M2")}
        ax.plot([a["M0"], max(a["M1"], a["M2"])], [yy, yy], color=GRID, lw=1.6, zorder=1,
                solid_capstyle="round")
        for m, dy in (("M0", 0.0), ("M1", 0.16), ("M2", -0.16)):
            ax.scatter([a[m]], [yy + dy], s=30, color=C[m], zorder=3, edgecolor=SURF, linewidth=0.8,
                       label=MODEL_LABEL[m] if yy == ys[0] else None)
    labels = [f"{CELL_NAMES[c['condition_name']]}  (n = {c['models']['M2']['episodes']})" for c in cells]
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0)
    vline(ax, 0.5, color=INK2, label="chance (0.5)")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.7, len(cells) - 0.3)
    ax.set_xlabel("AUROC within condition cell (Locked Test, primary steps)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.32, 1.17), ncol=2, columnspacing=1.0,
              fontsize=6.8, handletextpad=0.3)
    ax.text(0.99, -0.005, "M1 drawn above, M2 below each row line", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color=MUTED)
    save(fig, "fig_cells")


# --------------------------------------------------------------------------- figure 3
def fig_calibration_vs_locked(S, calib):
    m3 = S[3]["models"]
    cm = calib["calibration_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.24, wspace=0.7)
    for ax, key, title, better in zip(axes, ("auroc", "log_loss"), ("AUROC", "Log loss"),
                                      ("higher is better", "lower is better")):
        grid(ax)
        ylim = (0.6, 1.0) if key == "auroc" else (0.15, 0.8)
        v0s = [cm[m.lower()][key] for m in ("M0", "M1", "M2")]
        v1s = [m3[m][key] for m in ("M0", "M1", "M2")]
        gap = 0.06 * (ylim[1] - ylim[0])
        y0s, y1s = spread(v0s, gap), spread(v1s, gap)
        for i, (m, dx) in enumerate((("M0", 0.0), ("M1", -0.035), ("M2", 0.035))):
            v0, v1 = v0s[i], v1s[i]
            ci = m3[m][f"{key}_interval"]
            ax.plot([0 + dx, 1 + dx], [v0, v1], color=C[m], lw=1.8, solid_capstyle="round", zorder=2,
                    label=MODEL_LABEL[m])
            ax.errorbar(1 + dx, v1, yerr=[[v1 - ci["lower"]], [ci["upper"] - v1]], fmt="none",
                        ecolor=C[m], elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
            ax.scatter([0 + dx, 1 + dx], [v0, v1], s=28, color=C[m], zorder=4, edgecolor=SURF, linewidth=0.8)
            ax.text(-0.1, y0s[i], f"{v0:.3f}", ha="right", va="center", fontsize=6.6, color=INK2)
            ax.text(1.12, y1s[i], f"{v1:.3f}", ha="left", va="center", fontsize=6.6, color=INK2)
        ax.set_xticks([0, 1], ["Calibration\n(in-sample fit)\n160 episodes", "Locked Test\n(held out)\n158 episodes"])
        ax.set_xlim(-0.62, 1.62)
        ax.set_ylim(*ylim)
        ax.set_title(f"{title} ({better})", fontsize=7.8)
    axes[0].set_ylabel("value; 90 % bootstrap CI on Locked Test")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.18, 1.22), ncol=3, fontsize=6.8,
                   columnspacing=1.2)
    save(fig, "fig_calibration_vs_locked")


# --------------------------------------------------------------------------- figure 4
def fig_lead_time(S, eps, ids, y):
    thr = S[5]["thresholds"]
    r5 = S[5]["result"]
    failed = [e for e, yy in zip(ids, y) if yy == 1.0]
    succ = [e for e, yy in zip(ids, y) if yy == 0.0]
    assert len(failed) == r5["failed_episodes"]

    def first_alarm(e, m):
        s = [r["control_step"] for r in eps[e] if r["probabilities"][m] >= thr[m]]
        return s[0] if s else np.nan

    a1 = np.array([first_alarm(e, "M1") for e in failed], float)
    a2 = np.array([first_alarm(e, "M2") for e in failed], float)
    both = ~np.isnan(a1) & ~np.isnan(a2)
    d = a2[both] - a1[both]
    n_same = int(np.sum(d == 0))

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.45, left=0.07, right=0.99,
                          top=0.84, bottom=0.2)

    # (a) paired scatter of first-alarm step
    ax = fig.add_subplot(gs[0, 0])
    grid(ax, axis="both")
    lim = (-8, max(np.nanmax(a1), np.nanmax(a2)) + 12)
    ax.plot(lim, lim, color=AXIS, lw=0.8, zorder=1)
    pts = collections.Counter(zip(a1[both], a2[both]))
    xs, ys_, ns = zip(*[(k[0], k[1], n) for k, n in pts.items()])
    ns = np.array(ns)
    ax.scatter(xs, ys_, s=np.minimum(18 + 12 * (ns - 1), 130), color=C["M2"], alpha=0.85, zorder=3,
               edgecolor=SURF, linewidth=0.7)
    imax = int(np.argmax(ns))
    ax.annotate(f"{ns[imax]} pairs at ({xs[imax]:.0f}, {ys_[imax]:.0f})", xy=(xs[imax], ys_[imax]),
                xytext=(xs[imax] + 40, ys_[imax] + 22), fontsize=6, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=6))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("first alarm step, M1 (p ≥ 0.4927)")
    ax.set_ylabel("first alarm step, M2 (p ≥ 0.5040)")
    ax.set_title(f"First alarm, failed episodes (n = {int(both.sum())})", fontsize=7.8)
    ax.text(0.03, 0.97, f"{n_same}/{int(both.sum())} pairs on the diagonal\n"
            f"median paired difference = {np.median(d):.0f} steps",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=INK2)
    panel_label(ax, "a", x=-0.2)

    # (b)/(c) mean P(failure) trajectories
    def traj(group, m, min_n=10):
        by = collections.defaultdict(list)
        for e in group:
            for r in eps[e]:
                by[r["control_step"]].append(r["probabilities"][m])
        ks = sorted(k for k in by if len(by[k]) >= min_n)
        return np.array(ks), np.array([np.mean(by[k]) for k in ks]), np.array([len(by[k]) for k in ks])

    for k, (group, name, lab) in enumerate([(failed, f"Failed episodes (n = {len(failed)})", "b"),
                                            (succ, f"Successful episodes (n = {len(succ)})", "c")]):
        ax = fig.add_subplot(gs[0, k + 1])
        grid(ax)
        for m in ("M1", "M2"):
            xs_, mu, n_ = traj(group, m)
            ax.plot(xs_, mu, color=C[m], lw=1.8, solid_capstyle="round", solid_joinstyle="round",
                    label=MODEL_LABEL[m], zorder=3)
        hline(ax, thr["M1"], color=C["M1"], lw=0.7)
        hline(ax, thr["M2"], color=C["M2"], lw=0.7)
        ax.text(1.0, thr["M2"] + 0.02, "frozen alarm thresholds", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.2, color=INK2)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 520)
        ax.set_xlabel("control step (5-step cadence)")
        if k == 0:
            ax.set_ylabel("mean predicted P(failure)")
            ax.legend(loc="lower right", fontsize=6.6)
        else:
            ax.text(0.98, 0.9, "steps with < 10 episodes\nstill running are omitted",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=INK2)
        ax.set_title(name, fontsize=7.8)
        panel_label(ax, lab, x=-0.16)
    save(fig, "fig_lead_time")
    return a1, a2


# --------------------------------------------------------------------------- figure 5
def fig_causal(S, rows):
    sm = S[7]["result"]["selected_layer_summary"]
    n = len(rows)
    assert n == sm["valid_pairs"]
    tgt = np.array([r["target"] for r in rows])
    mc = np.array([r["matched"] for r in rows])
    R = np.stack([r["rc"] for r in rows])                      # (52, 1000)
    p5, p95 = np.percentile(R, 5, axis=1), np.percentile(R, 95, axis=1)
    null_medians = np.median(R, axis=0)                        # per random direction, median over pairs
    null_p95 = float(np.percentile(null_medians, 95))
    assert abs(null_p95 - sm["random_control_95th_percentile"]) < 1e-12
    assert abs(np.median(tgt) - sm["median_donor_aligned_target_effect"]) < 1e-12
    n_exceed = int(np.sum(tgt > p95))
    nn = np.array([r["nn"] for r in rows])
    nat95 = rows[0]["nat95"]
    n_off = int(np.sum(nn > nat95))
    assert n_off == sum(r["off"] for r in rows) and abs(n_off / n - sm["off_manifold_rate"]) < 1e-12
    n_sign = sum(r["sign"] for r in rows)
    assert n_sign == sm["sign_correct_count"]
    mc_rate = np.mean([r["matched_sign"] for r in rows])
    assert abs(mc_rate - sm["matched_control_sign_rate"]) < 1e-12

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.85], wspace=0.35,
                          hspace=0.62, left=0.08, right=0.99, top=0.93, bottom=0.1)
    sc = 1e3  # show effects in units of 1e-3

    # (a) per-pair: target vs own random-control band
    ax = fig.add_subplot(gs[0, 0])
    grid(ax)
    order = np.argsort(tgt)
    x = np.arange(n)
    ax.vlines(x, p5[order] * sc, p95[order] * sc, color=GRID, lw=2.6, zorder=2,
              label="random directions, 5th–95th pct. (1000 per pair)")
    ax.scatter(x, mc[order] * sc, s=16, facecolor=SURF, edgecolor=MUTED, linewidth=0.9, zorder=3,
               label="matched-donor control (< 5°)")
    ax.scatter(x, tgt[order] * sc, s=22, color=C["M2"], zorder=4, edgecolor=SURF, linewidth=0.7,
               label="probe direction (α = 0.25)")
    hline(ax, 0, color=AXIS, ls="-", lw=0.8)
    ax.set_xlim(-1, n)
    ax.set_ylim(-2.8, 6.0)
    clipped = int(np.sum(p5 * sc < -2.8))
    if clipped:
        ax.text(0.02, 0.03, f"{clipped} band clipped (5th pct. {p5.min() * sc:.1f})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks([0, 10, 20, 30, 40, 51])
    ax.set_xlabel(f"valid pair, sorted by probe effect (n = {n})")
    ax.set_ylabel(r"donor-aligned target effect ($\times 10^{-3}$)")
    ax.set_title("Per-pair effect vs. that pair's random controls", fontsize=7.8)
    ax.legend(loc="upper left", fontsize=6.4, handlelength=1.2, borderaxespad=0.2)
    ax.text(0.03, 0.74, f"{n_exceed} of {n} pairs exceed their own random 95th pct.\n"
            f"({100 * n_exceed / n:.1f} %; 5 % expected by chance)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK2)
    panel_label(ax, "a", x=-0.12)

    # (b) null distribution of the median statistic
    ax = fig.add_subplot(gs[0, 1])
    grid(ax)
    sc4 = 1e4
    cnts, _, _ = ax.hist(null_medians * sc4, bins=30, color=LIGHT["M2"], edgecolor=SURF, linewidth=0.6, zorder=2)
    top = cnts.max() * 1.2
    ax.set_ylim(0, cnts.max() * 1.85)
    ax.vlines(null_p95 * sc4, 0, top, color=INK, ls=(0, (3, 2)), lw=0.8, zorder=3,
              label=f"random 95th pct. = {null_p95 * sc4:+.1f}")
    ax.vlines(np.median(tgt) * sc4, 0, top, color=C["M2"], lw=1.6, zorder=3,
              label=f"probe median = {np.median(tgt) * sc4:+.1f}")
    ax.vlines(np.median(mc) * sc4, 0, top, color=MUTED, lw=1.2, zorder=3,
              label=f"matched-donor median = {np.median(mc) * sc4:+.1f}")
    ax.set_xlabel(r"median effect over 52 pairs ($\times 10^{-4}$)")
    ax.set_ylabel("random directions (of 1000)")
    ax.set_title("Median effect: probe vs. random null", fontsize=7.8)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2, borderaxespad=0.2)
    ax.text(0.98, 0.745, "random_control_passes: false", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.4, color=INK2)
    panel_label(ax, "b", x=-0.12)

    # (c) sign-correct rate
    ax = fig.add_subplot(gs[1, 0])
    grid(ax, axis="x")
    si = sm["sign_interval"]
    rate, lo, hi = 100 * si["estimate"], 100 * si["lower"], 100 * si["upper"]
    ax.errorbar(rate, 1, xerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=C["M2"], elinewidth=1.2,
                capsize=3, capthick=1.2, zorder=3)
    ax.scatter([rate], [1], s=40, color=C["M2"], zorder=4, edgecolor=SURF, linewidth=0.8)
    ax.scatter([100 * mc_rate], [0], s=40, facecolor=SURF, edgecolor=MUTED, linewidth=1.1, zorder=4)
    bb = dict(boxstyle="round,pad=0.15", fc=SURF, ec="none")
    ax.text(rate, 1.3, f"{n_sign}/{n} = {rate:.1f} %  [90 % CI {lo:.1f}, {hi:.1f}]", ha="center",
            va="bottom", fontsize=6.6, color=INK2, bbox=bb, zorder=5)
    ax.text(100 * mc_rate, 0.3, f"{int(round(mc_rate * n))}/{n} = {100 * mc_rate:.1f} % (no CI in receipt)",
            ha="center", va="bottom", fontsize=6.6, color=INK2, bbox=bb, zorder=5)
    vline(ax, 50, color=INK, label="prereg. bar: > 50 %")
    ax.set_yticks([1, 0], ["probe direction\n(α = 0.25)", "matched-donor\ncontrol (< 5°)"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, 1.9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("sign-correct rate (%) over 52 valid pairs, 20 init clusters")
    ax.set_title("Sign-correct rate vs. the 50 % bar", fontsize=7.8)
    ax.legend(loc="upper right", fontsize=6.4, borderaxespad=0.2)
    panel_label(ax, "c", x=-0.12)

    # (d) off-manifold check
    ax = fig.add_subplot(gs[1, 1])
    grid(ax)
    bins = np.arange(3.2, 5.81, 0.1)
    ax.hist(nn, bins=bins, color=LIGHT["M2"], edgecolor=SURF, linewidth=0.6, zorder=2)
    vline(ax, nat95, color=INK, label=f"natural 95th pct. = {nat95:.2f}")
    ax.set_xlabel("5-NN distance of patched state to Calibration reference")
    ax.set_ylabel("valid pairs")
    ax.set_title("Off-manifold rate", fontsize=7.8)
    ax.legend(loc="upper right", fontsize=6.4, borderaxespad=0.2)
    ax.text(0.98, 0.72, f"{n_off}/{n} = {100 * n_off / n:.1f} % beyond threshold",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6, color=INK2)
    panel_label(ax, "d", x=-0.12)
    save(fig, "fig_causal")


# --------------------------------------------------------------------------- figure 6
def fig_dose(S, rows, sens):
    cells = S[6]["cells"]
    names = [CELL_SHORT[c["condition_name"]] for c in cells]
    alphas = [0.25, 0.5, 1.0]
    sign = np.full((3, 8), np.nan); cnt = np.zeros((3, 8), int); nval = np.zeros((3, 8), int)
    otr = np.full((3, 8), np.nan)
    # alpha = 0.25 derived from the section-7 per-pair evidence (selected_patch)
    for ci in range(8):
        sub = [r for r in rows if r["cell"] == ci]
        nval[0, ci] = len(sub); cnt[0, ci] = sum(r["sign"] for r in sub)
        sign[0, ci] = cnt[0, ci] / len(sub); otr[0, ci] = np.median([r["otr"] for r in sub])
    # alpha in {0.5, 1.0} from the frozen sensitivity receipt (9b)
    grid9b = S[9]["subsections"][1]["result"]
    assert grid9b == sens["dose_by_difficulty"]
    for g in grid9b:
        ai = alphas.index(g["alpha"]); ci = g["condition_index"]
        nval[ai, ci] = g["valid_pairs"]; cnt[ai, ci] = g["sign_correct_count"]
        sign[ai, ci] = g["sign_correct_rate"]; otr[ai, ci] = g["median_off_target_ratio"]
        assert g["specificity_passes"] is False and g["median_off_target_ratio_status"] == "finite"
    assert (nval[0] == nval[1]).all() and (nval[0] == nval[2]).all()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.3, wspace=0.28)
    cmap_div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

    def cell_ink(rgba):
        r, g, b = rgba[:3]
        return SURF if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else INK

    # (a) sign-correct rate
    ax = axes[0]
    norm = Normalize(0, 1)
    ax.imshow(sign, cmap=cmap_div, norm=norm, aspect="auto")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, f"{100 * sign[i, j]:.0f} %\n{cnt[i, j]}/{nval[i, j]}", ha="center", va="center",
                    fontsize=6.2, color=cell_ink(cmap_div(norm(sign[i, j]))), linespacing=1.1)
    ax.set_title("Sign-correct rate (bar > 50 %; blue above, red below)", fontsize=7.4)
    # (b) median off-target ratio
    ax = axes[1]
    lnorm = LogNorm(vmin=0.25, vmax=25)
    ax.imshow(otr, cmap=CMAP_BLUE, norm=lnorm, aspect="auto")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, f"{otr[i, j]:.2f}", ha="center", va="center", fontsize=6.4,
                    color=cell_ink(CMAP_BLUE(lnorm(otr[i, j]))))
    ax.set_title(f"Median off-target ratio (bar ≤ 0.25; min {otr.min():.2f}, all 24 cells fail)",
                 fontsize=7.8)
    for k, ax in enumerate(axes):
        ax.set_xticks(range(8), names, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(3), [f"α = {a}" + (" (§7)" if a == 0.25 else "") for a in alphas])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # 2 px surface gap between cells
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color=SURF, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        panel_label(ax, "ab"[k], x=-0.1, y=1.04)
    axes[0].set_ylabel("patch dose")
    fig.text(0.5, 0.01, "condition cell; entries show the rate and k/n valid pairs. The \u03b1 = 0.25 row is derived from "
             "the \u00a77 per-pair evidence, the \u03b1 = 0.5 and 1.0 rows from the frozen 9b grid",
             ha="center", va="bottom", fontsize=6.2, color=INK2)
    save(fig, "fig_dose")
    return sign, otr, nval


# --------------------------------------------------------------------------- figure 7
def fig_pipeline(S):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f1", ec=AXIS, tc=INK, fs=7, bold=False, lw=0.8, r=0.9):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                            fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.25, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK2, lw=0.9, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=8), zorder=4)

    def label(x, y, s, fs=7.2, color=INK, bold=True, ha="left"):
        ax.text(x, y, s, fontsize=fs, color=color, fontweight="bold" if bold else "normal", ha=ha,
                va="center")

    # Row 1: splits
    label(2, 48.5, "Splits (LIBERO-10 task 5; 8 condition cells × 20 init states per split)")
    y1, h1 = 39.5, 7
    box(2, y1, 26, h1, "Discovery\n40 episodes · layer & variable selection", fc="#f4f4f1")
    box(37, y1, 26, h1, "Calibration\n160 episodes · fit M0/M1/M2, freeze\nthresholds, probe, α = 0.25",
        fc="#f4f4f1", fs=6.6)
    box(72, y1, 26, h1, "Locked Test\n160 episodes (158 valid) · evaluated\nexactly once, nothing refit",
        fc="#e8f0fb", ec=C["M2"], fs=6.6, lw=1.0)
    arrow(28, y1 + h1 / 2, 37, y1 + h1 / 2); arrow(63, y1 + h1 / 2, 72, y1 + h1 / 2)
    ax.text(32.5, y1 + h1 / 2 + 1.4, "freeze", ha="center", fontsize=6, color=INK2)
    ax.text(67.5, y1 + h1 / 2 + 1.4, "freeze", ha="center", fontsize=6, color=INK2)

    # Row 2: per-episode prediction pipeline
    label(2, 34.5, "Per episode (Locked Test)")
    yc = 26.0                      # row centre
    h2 = 6
    y2 = yc - h2 / 2
    box(2, y2, 16, h2, "closed-loop\nrollout", fs=6.8)
    box(22, y2, 16, h2, "states every\n5 control steps", fs=6.8)
    arrow(18, yc, 22, yc)
    fx, fw, fh = 43, 22, 3.4
    centres = [yc + 4.2, yc, yc - 4.2]
    specs = [("M0  vision baseline", "#f1f1ee", C["M0"]), ("M1  privileged state", "#fdeee7", C["M1"]),
             ("M2  internals (probe)", "#e8f0fb", C["M2"])]
    for cy, (txt, fc, ec) in zip(centres, specs):
        box(fx, cy - fh / 2, fw, fh, txt, fc=fc, ec=ec, fs=6.4, lw=1.0)
        arrow(38, yc, fx, cy, lw=0.7)
        arrow(fx + fw, cy, 70, yc, lw=0.7)
    box(70, y2, 13, h2, "P(failure)\nper step", fs=6.8)
    arrow(83, yc, 86, yc)
    box(86, yc - 4, 12, 8, "paired log loss\nAUROC \u00b7 Brier\nlead time", fs=6.4)
    ax.text(2, yc - 7.6, "labels: terminal failure; primary steps {0, 50, 100, 150, 200}; "
            "each episode weight one; 10,000 cluster-bootstrap replicates (20 init clusters)",
            fontsize=6, color=INK2, va="center")

    # Row 3: causal patching
    label(2, 14.2, "Causal patching (selected layer early_expert_t1_0, 60 pairs \u2192 52 valid)")
    y3, h3 = 5.5, 6
    box(2, y3, 17, h3, "recipient state\n(cell c, init i)", fs=6.6)
    box(2, y3 - 3.4, 17, 2.6, "donor: same init, yaw-shifted cell", fs=5.8, fc="#fbfbf9")
    box(24, y3, 22, h3, "patch along probe\ndirection, dose \u03b1 = 0.25\n(sensitivity: 0.5, 1.0)",
        fc="#e8f0fb", ec=C["M2"], fs=6.3, lw=1.0)
    arrow(19, y3 + h3 / 2, 24, y3 + h3 / 2)
    box(51, y3, 20, h3, "donor-aligned target\neffect \u00b7 off-target ratio", fs=6.4)
    arrow(46, y3 + h3 / 2, 51, y3 + h3 / 2)
    box(76, y3 - 0.6, 22, h3 + 1.2, "controls per pair:\nmatched donor (< 5\u00b0)\n1000 random directions\n"
        "5-NN off-manifold check", fc="#fbfbf9", fs=6.1)
    arrow(71, y3 + h3 / 2, 76, y3 + h3 / 2, style="-")
    ax.text(61, y3 - 2.2, "bars: sign-correct > 50 %, off-target ratio \u2264 0.25, probe > random 95th pct.",
            fontsize=6, color=INK2, ha="center", va="center")
    save(fig, "fig_pipeline")


# --------------------------------------------------------------------------- main
def main():
    report, S, preds, calib, sens, causal_dir = load_all()
    eps, ids, y, loss = per_episode(preds, report["evaluation_protocol"]["primary_steps"])
    assert len(ids) == 158 and int(y.sum()) == 60
    rows = causal_pairs(causal_dir)
    delta = fig_headline(S, loss, y)
    fig_cells(S)
    fig_calibration_vs_locked(S, calib)
    a1, a2 = fig_lead_time(S, eps, ids, y)
    fig_causal(S, rows)
    sign, otr, nval = fig_dose(S, rows, sens)
    fig_pipeline(S)
    # summary of derived quantities (aggregates only)
    print(f"delta log loss: mean {delta.mean():+.4f}, median {np.median(delta):+.4f}, "
          f"episodes with |delta| < 0.02: {np.mean(np.abs(delta) < 0.02):.0%}")
    d = a2 - a1
    print(f"first-alarm paired difference (M2-M1): median {np.nanmedian(d):.0f}, identical {int(np.sum(d == 0))}/60, "
          f"detection M1 {np.mean(~np.isnan(a1)):.3f} M2 {np.mean(~np.isnan(a2)):.3f}")
    print("dose grid alpha=0.25 sign rates:", np.round(sign[0], 3).tolist(), "median OTR:", np.round(otr[0], 2).tolist())
    print("figures in", FIG_DIR)


if __name__ == "__main__":
    main()
