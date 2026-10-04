"""Build every figure of the blog post from blog/data/ and blog/frames/.

  python blog/src/make_figures.py          # writes blog/figures/*.svg
Charts use only the numbers in blog/data/ (copied from frozen artifacts by
extract_data.py). Sketches (figures 3, 8 and the variants) are illustrative.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from svgkit import (BLUE, BLUE_L, FILL, GRID, INK, MUTED, ORANGE, ORANGE_L, OUT, TEXT2, W, MATH, Svg,
                    nice_ticks)

ROOT = Path(__file__).resolve().parents[2] / "blog"
DATA, FRAMES, FIGS = ROOT / "data", ROOT / "frames", ROOT / "figures"
M = json.loads((DATA / "metrics.json").read_text())
F = json.loads((DATA / "frames.json").read_text())
C = json.loads((DATA / "causal_null.json").read_text())


# =============================================================== 1. two runs
def fig_runs():
    s = Svg(452, "Two runs from the same starting scene, five frames each. In the first, the book lies as "
                 "usual and the arm places it in the caddy by step 157. In the second, the book is turned "
                 "by 37.5 degrees; the arm never gets a grip and the run ends at the step limit of 520.")
    s.header(1, "Two runs from the same start", "Front camera, five evenly spaced moments per run")
    size, gap, x0 = 120, 8, 24
    rows = [("success", 84, [("Book as usual.", {"weight": 700}),
                             ("  The arm places it in the caddy.", {"fill": TEXT2})]),
            ("failure", 270, [("Book turned 37.5°.", {"weight": 700}),
                              ("  The arm never gets a grip. Time runs out.", {"fill": TEXT2})])]
    for name, y, runs in rows:
        s.rich(x0, y, runs, size=13)
        for j, step in enumerate(F[name]["steps"]):
            x = x0 + j * (size + gap)
            s.image(x, y + 12, size, size, FRAMES / f"run-{name}-{j}.jpg", crop=(20, 0, 320))
            s.rect(x, y + 12, size, size, fill="none", stroke=GRID)
            label = f"step {step}"
            if j == 4:
                label += "  ✓ done" if name == "success" else "  time out"
            s.text(x, y + size + 30, label, size=11.5, fill=INK if j == 4 else TEXT2,
                   weight=600 if j == 4 else None)
    s.save(FIGS / "fig1-runs.svg")


# =============================================================== 2. four changes
def fig_changes():
    s = Svg(292, "The same starting scene four times: unchanged, with the book turned by 37.5 degrees, "
                 "with the book moved by 4.5 centimetres, and with the camera turned by 5 degrees.")
    s.header(2, "Small changes before the run starts", "First frame of each run, cropped around the book")
    size, gap, x0, y = 149, 12, 24, 76
    items = [("unchanged", "Unchanged", "for comparison"),
             ("book-turned", "Book turned", "±22.5° or ±37.5°"),
             ("book-moved", "Book moved", "4.5 cm diagonally"),
             ("camera-turned", "Camera turned", "±5°, shifted 2 cm")]
    for j, (key, title, sub) in enumerate(items):
        x = x0 + j * (size + gap)
        s.image(x, y, size, size, FRAMES / f"change-{key}.jpg", crop=(60, 100, 220))
        s.rect(x, y, size, size, fill="none", stroke=GRID)
        s.text(x, y + size + 22, title, size=13, weight=700)
        s.text(x, y + size + 40, sub, size=12, fill=TEXT2)
    s.save(FIGS / "fig2-changes.svg")


# =============================================================== pictograms for the monitors
def pict_plans(s, x, y, color=OUT):
    """Planned path, plus a faint copy showing how it shifts when the input changes a little."""
    s.path(f"M{x+4},{y+34} C{x+14},{y+30} {x+20},{y+12} {x+36},{y+8}", stroke=color, width=1.8, marker="out")
    s.path(f"M{x+4},{y+34} C{x+16},{y+32} {x+26},{y+20} {x+38},{y+20}", stroke=color, width=1.2, dash="3 3")
    s.circle(x + 4, y + 34, 2.6, fill=color)


def pict_scene(s, x, y, color=INK):
    """Top-down scene with exact coordinates."""
    s.rect(x + 1, y + 1, 38, 38, fill="none", stroke=color, width=1.1)
    s.line(x + 1, y + 26, x + 39, y + 26, stroke=MUTED, width=0.8, dash="2 2")
    s.line(x + 15, y + 1, x + 15, y + 39, stroke=MUTED, width=0.8, dash="2 2")
    s.add(f'<rect x="{x+10}" y="{y+21}" width="10" height="10" fill="{color}" '
          f'transform="rotate(-30 {x+15} {y+26})"/>')
    s.rect(x + 24, y + 6, 12, 8, fill="none", stroke=color, width=1.1)
    s.circle(x + 30, y + 30, 2.6, fill=color)


def pict_vector(s, x, y, color=BLUE):
    """A vector of numbers: a strip of cells with varying shade."""
    shades = [0.9, 0.35, 0.6, 0.15, 0.75, 0.45, 0.25, 0.85, 0.5, 0.3]
    for i, a in enumerate(shades):
        cx, cy = x + (i % 5) * 8, y + 8 + (i // 5) * 12
        s.add(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="7" height="10" fill="{color}" fill-opacity="{a}"/>')
    s.text(x + 40, y + 40, "…", size=11, fill=color, anchor="end")


ROWS = [("The robot’s plans", "and how they shift", pict_plans, OUT),
        ("Exact scene state", "poses, contact, stage", pict_scene, INK),
        ("8 internal signals", "from 720 numbers", pict_vector, BLUE)]


# =============================================================== 3. three monitors (variant B, used in the post)
def fig_monitors_b(path=FIGS / "fig3-monitors.svg", num=3):
    s = Svg(392, "Three monitors side by side. The first sees only the robot's own plans. The second also "
                 "sees the exact scene state from the simulator. The third also sees 8 signals computed from "
                 "720 numbers inside the network. The main comparison is the second against the third.")
    s.header(num, "What each monitor gets to see", "Same kind of model, same calibration runs. Only the inputs differ.")
    cw, gap, x0, y0, ch = 200, 16, 24, 74, 262
    names = [("Outputs", OUT, 1), ("+ Simulator", INK, 2), ("+ Internals", BLUE, 3)]
    for k, (name, col, n_rows) in enumerate(names):
        x = x0 + k * (cw + gap)
        s.rect(x, y0, cw, ch, fill="#ffffff", stroke="#d9d9d9", rx=6)
        s.text(x + 16, y0 + 28, name, size=15, weight=700, fill="#6f6f6f" if col == OUT else col)
        s.text(x + cw - 16, y0 + 28, f"M{k}", size=11.5, fill=TEXT2, anchor="end")
        for r, (t1, t2, pict, rc) in enumerate(ROWS):
            ry = y0 + 50 + r * 68
            if r < n_rows:
                s.rect(x + 10, ry, cw - 20, 58, fill=FILL if r < n_rows - 1 or k == 0 else
                       (BLUE_L if rc == BLUE else FILL), rx=4)
                pict(s, x + 18, ry + 9)
                s.text(x + 66, ry + 25, t1, size=12.5, weight=600, fill=INK)
                s.text(x + 66, ry + 42, t2, size=11.5, fill=TEXT2)
            else:
                s.rect(x + 10, ry, cw - 20, 58, fill="none", stroke="#dcdcdc", rx=4, dash="3 3")
    # main comparison
    yb = y0 + ch + 22
    xa, xb = x0 + cw + gap + cw / 2, x0 + 2 * (cw + gap) + cw / 2
    s.path(f"M{xa},{y0+ch+4} L{xa},{yb} L{xb},{yb} L{xb},{y0+ch+8}", stroke=INK, width=1.2, marker="ink")
    s.text((xa + xb) / 2, yb + 20, "Main question: is the third better than the second?", size=12.5,
           anchor="middle", weight=600)
    s.save(path)


# =============================================================== 3, variant A: nested boxes
def fig_monitors_a(path=FIGS / "variants/fig3-variant-A-nested.svg"):
    s = Svg(344, "Three nested boxes. The innermost holds the robot's own plans, the middle one adds the exact "
                 "scene state from the simulator, the outer one adds 8 signals from inside the network.")
    s.header(3, "Each monitor sees everything the previous one sees", "Variant A: nested boxes")
    boxes = [(24, 72, 632, 236, BLUE, "+ Internals", "8 signals from 720 internal numbers", pict_vector),
             (44, 128, 592, 160, INK, "+ Simulator", "exact scene state: poses, contact, stage", pict_scene),
             (64, 184, 552, 84, OUT, "Outputs", "the robot’s own plans and how they shift", pict_plans)]
    for i, (x, y, w, h, col, t, sub, pict) in enumerate(boxes):
        s.rect(x, y, w, h, fill=["#f7f9fc", "#f6f6f6", "#ffffff"][i], stroke=col, width=1.4, rx=8)
        s.rich(x + 16, y + 30, [(t, {"weight": 700, "fill": "#6f6f6f" if col == OUT else col}),
                                ("   " + sub, {"fill": TEXT2})], size=13)
        pict(s, x + w - 60, y + 8)
    s.text(340, 332, "Main question: does the outer box beat the middle one?", size=12.5, weight=600, anchor="middle")
    s.save(path)


# =============================================================== 3, variant C: where each monitor taps in
def fig_monitors_c(path=FIGS / "variants/fig3-variant-C-taps.svg"):
    s = Svg(272, "The control loop: camera images and the instruction go into the network, which plans "
                 "movements; the arm moves and the scene changes. The outputs monitor taps the plans, the "
                 "simulator monitor taps the scene, the internals monitor taps the action expert.")
    s.header(3, "Where each monitor takes its information", "Variant C: taps on the control loop")
    y = 160
    s.rect(24, y - 30, 120, 60, fill=FILL, rx=5)
    s.text(84, y - 6, "Camera images,", size=12, anchor="middle")
    s.text(84, y + 10, "joints, instruction", size=12, anchor="middle")
    s.rect(184, y - 46, 200, 92, fill="#ffffff", stroke=INK, width=1.2, rx=6)
    s.text(284, y - 26, "Network", size=13, weight=700, anchor="middle")
    s.rect(196, y - 12, 80, 44, fill=FILL, rx=4)
    s.text(236, y + 6, "vision and", size=11.5, anchor="middle")
    s.text(236, y + 21, "language", size=11.5, anchor="middle")
    s.rect(292, y - 12, 80, 44, fill=BLUE_L, rx=4)
    s.text(332, y + 6, "action", size=11.5, anchor="middle")
    s.text(332, y + 21, "expert", size=11.5, anchor="middle")
    s.line(276, y + 10, 290, y + 10, stroke=INK, marker="ink")
    s.rect(424, y - 30, 100, 60, fill=FILL, rx=5)
    s.text(474, y - 6, "50 planned", size=12, anchor="middle")
    s.text(474, y + 10, "movements", size=12, anchor="middle")
    s.rect(564, y - 30, 92, 60, fill=FILL, rx=5)
    s.text(610, y - 6, "Arm moves,", size=12, anchor="middle")
    s.text(610, y + 10, "scene changes", size=12, anchor="middle")
    for xa, xb in ((144, 182), (384, 422), (524, 562)):
        s.line(xa, y, xb, y, stroke=INK, marker="ink")
    s.path(f"M610,{y+30} L610,{y+62} L84,{y+62} L84,{y+32}", stroke=MUTED, width=1.1, marker="muted")
    s.text(347, y + 80, "next step: look again", size=11.5, fill=TEXT2, anchor="middle")
    for x, yy, col, t in ((474, y - 30, OUT, "Outputs"), (610, y - 30, INK, "+ Simulator"),
                          (332, y - 12, BLUE, "+ Internals")):
        s.line(x, yy - 2, x, 84, stroke=col, width=1.4, dash="3 2")
        s.circle(x, yy - 2, 3.5, fill=col)
        s.text(x, 78, t, size=12.5, weight=700, fill="#6f6f6f" if col == OUT else col, anchor="middle")
    s.save(path)


# =============================================================== 4. three stages
def lock(s, x, y, color=INK):
    s.path(f"M{x-5},{y} L{x-5},{y-5} A5,5 0 0 1 {x+5},{y-5} L{x+5},{y}", stroke=color, width=1.6)
    s.rect(x - 8, y, 16, 12, fill=color, rx=2)


def fig_stages():
    s = Svg(216, "Three stages on separate starting scenes: discovery on 10 scenes, calibration on 20, and a "
                 "test on 20 new scenes that was collected only after every setting was frozen.")
    s.header(4, "Three stages, kept apart", "Every stage uses its own starting scenes")
    gap, x0, y, h = 26, 24, 82, 56
    unit = (W - 48 - 2 * gap) / 50
    blocks = [("Discovery", "10 scenes · 40 runs", 10, FILL, None,
               ["get the robot running", "check the failure rate"]),
              ("Calibration", "20 scenes · 160 runs", 20, FILL, None,
               ["build the internal signals", "train the monitors", "freeze every setting"]),
              ("Test", "20 new scenes · 160 runs", 20, "#ffffff", INK,
               ["new angles, shifts, camera turns", "collected after the freeze", "main comparison computed once"])]
    x = x0
    for i, (t, sub, n, fill, stroke, notes) in enumerate(blocks):
        w = n * unit
        s.rect(x, y, w, h, fill=fill, stroke=stroke or "none", width=1.6, rx=5)
        s.text(x + 14, y + 24, t, size=14, weight=700)
        s.text(x + 14, y + 42, sub, size=12, fill=TEXT2)
        for k, note in enumerate(notes):
            s.text(x + 2, y + h + 26 + k * 18, note, size=12, fill=TEXT2)
        if i < 2:
            xa = x + w + 4
            s.line(xa, y + h / 2, xa + gap - 8, y + h / 2, stroke=INK, marker="ink")
        x += w + gap
    xl = x0 + 30 * unit + gap + gap / 2
    lock(s, xl, y - 14)
    s.text(xl, y - 32, "freeze", size=12, weight=600, anchor="middle")
    s.save(FIGS / "fig4-stages.svg")


# =============================================================== dot plot helper (figures 5 and 7)
def dot_rows(s, rows, y0, lo=0.5, hi=1.0, xl=232, xr=640, row_h=40):
    """rows: list of (label, color, value, (ci_lo, ci_hi), bold)."""
    X = lambda v: xl + (v - lo) / (hi - lo) * (xr - xl)
    y_axis = y0 + len(rows) * row_h + 4
    for t in nice_ticks(lo, hi, 0.1):
        s.line(X(t), y0 - 18, X(t), y_axis, stroke=GRID, width=1)
        s.text(X(t), y_axis + 18, f"{t:.1f}", size=11.5, fill=TEXT2, anchor="middle")
    s.line(xl, y_axis, xr, y_axis, stroke=MUTED)
    s.text(X(lo), y_axis + 34, "guessing", size=11.5, fill=TEXT2, anchor="middle")
    s.text(X(hi), y_axis + 34, "always right", size=11.5, fill=TEXT2, anchor="end")
    for i, (label, col, v, (a, b), bold) in enumerate(rows):
        y = y0 + i * row_h + row_h / 2 - 6
        s.text(24, y + 4.5, label, size=13, fill=col if col != OUT else "#6f6f6f", weight=700 if bold else 600)
        s.line(X(a), y, X(b), y, stroke=col, width=2.2, cap="round")
        s.circle(X(v), y, 5.5, fill=col, stroke="#ffffff", width=1.5)
        s.text(X(b) + 10, y + 4.5, f"{v:.2f}", size=12.5, fill=INK, weight=600)
    return X, y_axis


# =============================================================== 5. result 1
def fig_result1():
    fz = M["frozen"]
    s = Svg(272, "Dot plot of AUROC on the 158 test runs. Outputs 0.80, plus simulator 0.92, plus internals "
                 "0.92. The last two are almost identical.")
    s.header(5, "How well each monitor predicts failure", "AUROC per run on the 158 test runs, with 90% ranges")
    rows = [("Outputs", OUT, fz["M0"]["auroc"], fz["M0"]["auroc_ci"], False),
            ("+ Simulator", INK, fz["M1"]["auroc"], fz["M1"]["auroc_ci"], False),
            ("+ Internals", BLUE, fz["M2"]["auroc"], fz["M2"]["auroc_ci"], False)]
    dot_rows(s, rows, 92)
    s.save(FIGS / "fig5-result1.svg")


# =============================================================== 6. the pass mark
def fig_passmark():
    pr = M["primary"]
    lo, hi, xl, xr = -0.02, 0.04, 40, 640
    X = lambda v: xl + (v - lo) / (hi - lo) * (xr - xl)
    s = Svg(250, "The measured improvement of the internals monitor over the simulator monitor is 0.5 percent, "
                 "with a 90 percent range from minus 1.2 to plus 2.1 percent. The pass mark at 3 percent lies "
                 "outside the range.")
    s.header(6, "The pass mark", "How much lower the log loss is than the simulator monitor’s, in percent")
    ytop, yax = 82, 190
    s.rect(X(pr["bar"]), ytop, xr - X(pr["bar"]), yax - ytop, fill=FILL)
    s.text(X(pr["bar"]) + 10, ytop + 20, "3% or more", size=12, fill=TEXT2)
    s.line(X(pr["bar"]), ytop, X(pr["bar"]), yax, stroke=INK, width=1.3, dash="5 3")
    s.text(X(pr["bar"]) - 8, ytop + 20, "pass mark: 3%", size=12, weight=600, anchor="end")
    s.text(X(pr["bar"]) - 8, ytop + 36, "set before the test", size=11.5, fill=TEXT2, anchor="end")
    s.line(X(0), ytop, X(0), yax, stroke=MUTED, width=1)
    s.text(X(0) + 8, ytop + 4, "the range must also", size=11.5, fill=TEXT2)
    s.text(X(0) + 8, ytop + 19, "stay right of 0", size=11.5, fill=TEXT2)
    for t in nice_ticks(lo, hi, 0.01):
        s.line(X(t), yax, X(t), yax + 5, stroke=MUTED)
        lab = "0" if abs(t) < 1e-9 else f"{t*100:+.0f}%".replace("-", "−")
        s.text(X(t), yax + 20, lab, size=11.5, fill=TEXT2, anchor="middle")
    s.line(xl, yax, xr, yax, stroke=MUTED)
    y = 138
    a, b = pr["lift_ci"]
    v = pr["relative_lift"]
    s.line(X(a), y, X(b), y, stroke=BLUE, width=2.4, cap="round")
    s.circle(X(v), y, 6, fill=BLUE, stroke="#ffffff", width=1.5)
    s.text(X(v), y - 14, f"measured: +{v*100:.1f}%", size=12.5, weight=700, anchor="middle")
    s.text(X(a), y + 22, f"−{abs(a)*100:.1f}%", size=11.5, fill=TEXT2, anchor="middle")
    s.text(X(b), y + 22, f"+{b*100:.1f}%", size=11.5, fill=TEXT2, anchor="middle")
    s.text(xl, yax + 40, "← internals worse", size=11.5, fill=TEXT2)
    s.text(xr, yax + 40, "internals better →", size=11.5, fill=TEXT2, anchor="end")
    s.save(FIGS / "fig6-passmark.svg")


# =============================================================== 7. result 2: no simulator state
def fig_result2():
    rf = M["refit"]
    s = Svg(318, "Dot plot of AUROC on the test runs, all without simulator state except the last. Outputs 0.80, "
                 "outputs plus the 8 designed signals 0.80, outputs plus the raw internal numbers 0.89, "
                 "outputs plus simulator 0.92.")
    s.header(7, "Without simulator state", "AUROC per run, 158 test runs. Last row: the simulator monitor, for comparison.")
    rows = [("Outputs", OUT, rf["M0"]["auroc"], rf["M0"]["auroc_ci"], False),
            ("Outputs + 8 signals", BLUE, rf["M0+M2inc"]["auroc"], rf["M0+M2inc"]["auroc_ci"], False),
            ("Outputs + raw internals", BLUE, rf["M0+act"]["auroc"], rf["M0+act"]["auroc_ci"], True),
            ("Outputs + simulator", INK, rf["M1"]["auroc"], rf["M1"]["auroc_ci"], False)]
    X, _ = dot_rows(s, rows, 92)
    # hollow marker for the 8 designed signals, so the two blue rows are told apart by more than the label
    y = 92 + 1 * 40 + 20 - 6
    s.circle(X(rf["M0+M2inc"]["auroc"]), y, 5.5, fill="#ffffff", stroke=BLUE, width=2)
    s.save(FIGS / "fig7-result2.svg")


# =============================================================== 8. reading vs editing (variant A, used in the post)
def fig_edit_a(path=FIGS / "fig8-edit.svg", num=9, sub="Sketch. The 720 numbers drawn as two axes."):
    s = Svg(352, "Sketch of the edit. This moment and a partner moment with the book turned are two points. "
                 "The edit moves this moment a quarter of the way toward the partner, only along the probe "
                 "direction. Controls are steps of the same size in random directions.")
    s.header(num, "Reading and editing the same numbers", sub)
    x0, y0, x1, y1 = 64, 300, 640, 84
    s.line(x0, y0, x1, y0, stroke=BLUE, width=1.6, marker="blue")
    s.text(x1, y0 + 22, "probe direction: reads the book’s angle", size=12.5, fill=BLUE, weight=600, anchor="end")
    s.line(x0, y0, x0, y1, stroke=MUTED, width=1.4, marker="muted")
    s.text(x0 + 10, y1 + 6, "the other 718 directions", size=12, fill=TEXT2)
    r, d = (196, 236), (548, 128)
    dp = (d[0], r[1])
    e = (r[0] + 0.25 * (dp[0] - r[0]), r[1])
    # random controls
    L = e[0] - r[0]
    for ang in (62, 118, 158, 206, 248, 318):
        dx, dy = L * math.cos(math.radians(ang)), -L * math.sin(math.radians(ang))
        s.line(r[0], r[1], r[0] + dx, r[1] + dy, stroke=MUTED, width=1.6, marker="muted")
    s.text(78, 124, "controls: 1,000 random steps", size=12, fill=TEXT2)
    s.text(78, 140, "of the same size", size=12, fill=TEXT2)
    # partner and its probe value
    s.line(d[0], d[1] + 7, dp[0], dp[1] - 2, stroke=MUTED, width=1, dash="3 3")
    s.circle(dp[0], dp[1], 3, fill=MUTED)
    s.circle(d[0], d[1], 6.5, fill="#ffffff", stroke=INK, width=1.8)
    s.text(d[0] - 12, d[1] - 20, "partner moment", size=12.5, weight=600, anchor="middle")
    s.text(d[0] - 12, d[1] - 4 - 0, "", size=1)
    s.text(d[0] + 14, d[1] + 5, "", size=1)
    s.text(d[0] - 12, d[1] + 28, "from another run, book", size=11.5, fill=TEXT2, anchor="end")
    s.text(d[0] - 12, d[1] + 43, "angle 30° to 90° different", size=11.5, fill=TEXT2, anchor="end")
    s.text(dp[0], dp[1] + 22, "its probe value", size=11.5, fill=TEXT2, anchor="middle")
    # the edit
    s.line(r[0], r[1], e[0] + 2, e[1], stroke=ORANGE, width=2.6, marker="orange")
    s.line(e[0] + 10, r[1], dp[0] - 8, r[1], stroke=ORANGE, width=1, dash="2 4")
    s.text((r[0] + e[0]) / 2 + 8, r[1] + 26, "edit: a quarter of the way,", size=12, fill=ORANGE, weight=600, anchor="start")
    s.text((r[0] + e[0]) / 2 + 8, r[1] + 41, "along the probe only", size=12, fill=ORANGE, weight=600, anchor="start")
    s.circle(r[0], r[1], 6.5, fill=INK, stroke="#ffffff", width=1.5)
    s.text(r[0] - 12, r[1] + 5, "this moment", size=12.5, weight=600, anchor="end")
    s.save(path)


# =============================================================== 8, variant B: where in the network
def fig_edit_b(path=FIGS / "variants/fig8-variant-B-network.svg"):
    s = Svg(262, "The network as a chain from camera images to the next movement. At the input to layer 4 of "
                 "the action expert, a probe reads the book's angle (blue), and the edit writes the partner's "
                 "angle into the same place (orange).")
    s.header(8, "Reading and editing at the same place", "Variant B: where in the network")
    y = 176
    s.rect(24, y - 28, 112, 56, fill=FILL, rx=5)
    s.text(80, y - 4, "Camera images,", size=12, anchor="middle")
    s.text(80, y + 12, "instruction", size=12, anchor="middle")
    s.rect(164, y - 28, 130, 56, fill=FILL, rx=5)
    s.text(229, y - 4, "Vision and", size=12, anchor="middle")
    s.text(229, y + 12, "language", size=12, anchor="middle")
    s.rect(322, y - 40, 208, 80, fill="#ffffff", stroke=INK, width=1.2, rx=6)
    s.text(518, y - 20, "Action expert", size=12.5, weight=700, anchor="end")
    for i in range(7):
        s.rect(334 + i * 28, y - 4, 22, 30, fill=FILL, rx=3)
    xc = 334 + 3 * 28 - 3
    s.rect(xc - 2, y - 8, 4, 38, fill=BLUE)
    s.text(xc, y + 54, "input to layer 4", size=11.5, fill=BLUE, anchor="middle")
    s.rect(556, y - 28, 100, 56, fill=FILL, rx=5)
    s.text(606, y - 4, "Next", size=12, anchor="middle")
    s.text(606, y + 12, "movement", size=12, anchor="middle")
    for xa, xb in ((136, 162), (294, 320), (530, 554)):
        s.line(xa, y, xb, y, stroke=INK, marker="ink")
    s.line(xc - 7, y - 10, xc - 7, 96, stroke=BLUE, width=1.6, marker="blue")
    s.text(xc - 15, 92, "read: probe gives the book’s angle", size=12, fill=BLUE, weight=600, anchor="end")
    s.line(xc + 7, 100, xc + 7, y - 11, stroke=ORANGE, width=1.6, marker="orange")
    s.text(xc + 15, 92, "write: partner’s angle", size=12, fill=ORANGE, weight=600)
    s.text(606, y + 50, "does it change?", size=12, fill=TEXT2, anchor="middle")
    s.save(path)


# =============================================================== 8, variant C: what the edit should do to the movement
def fig_edit_c(path=FIGS / "variants/fig8-variant-C-movement.svg"):
    s = Svg(342, "Top-down sketch. The gripper approaches the book. If the network used the probe reading, "
                 "writing the partner's angle would turn the planned approach toward the partner's book angle. "
                 "Measured: it changed no more than after a random edit.")
    s.header(8, "What the edit should do if the network used the reading", "Variant C: seen from above (sketch)")
    for cx, title, pred in ((190, "If the network used the reading", True), (500, "What happened", False)):
        cy = 178
        s.text(cx, 88, title, size=13, weight=700, anchor="middle")
        s.add(f'<rect x="{cx-12}" y="{cy-36}" width="24" height="72" fill="{INK}" transform="rotate(10 {cx} {cy})"/>')
        s.add(f'<rect x="{cx-12}" y="{cy-36}" width="24" height="72" fill="none" stroke="{ORANGE}" '
              f'stroke-dasharray="4 3" stroke-width="1.6" transform="rotate(60 {cx} {cy})"/>')
        s.line(cx - 100, cy + 70, cx - 26, cy + 22, stroke=INK, width=1.8, marker="ink")
        if pred:
            s.path(f"M{cx-100},{cy+70} Q{cx-50},{cy+66} {cx-14},{cy+42}", stroke=ORANGE, width=1.8, dash="5 3",
                   marker="orange")
            s.text(cx, cy + 104, "the approach turns toward", size=12, fill=ORANGE, anchor="middle")
            s.text(cx, cy + 120, "the partner’s angle", size=12, fill=ORANGE, anchor="middle")
        else:
            s.text(cx, cy + 104, "the approach changes no more", size=12, fill=TEXT2, anchor="middle")
            s.text(cx, cy + 120, "than after a random edit", size=12, fill=TEXT2, anchor="middle")
    s.text(24, 332, "black: the real book · dashed: the partner’s book angle · arrow: planned approach",
           size=11.5, fill=TEXT2)
    s.save(path)


# =============================================================== 9. the edit vs random edits
def fig_null():
    vals = np.array(C["null_medians"]) * 1e4
    probe, p95 = C["probe_median"] * 1e4, C["null_p95"] * 1e4
    lo, hi, xl, xr = -5.0, 3.0, 40, 640
    X = lambda v: xl + (v - lo) / (hi - lo) * (xr - xl)
    s = Svg(330, "Histogram of 1,000 random edits. The edit along the probe sits inside the pile, below the line "
                 "that 95 percent of random edits stay under.")
    s.header(10, "The probe edit against 1,000 random edits",
             "Change in the gripper’s turn command, toward the partner’s (median over 52 pairs)")
    ytop, yax = 104, 262
    edges = np.arange(lo, hi + 1e-9, 0.25)
    counts, _ = np.histogram(vals, bins=edges)
    scale = (yax - ytop - 20) / counts.max()
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if c:
            s.rect(X(a) + 1, yax - c * scale, X(b) - X(a) - 2, c * scale, fill="#cfcfcf")
    s.line(X(p95), ytop - 18, X(p95), yax, stroke=INK, width=1.3, dash="5 3")
    s.text(X(p95) + 8, ytop - 6, "95% of random edits", size=12, weight=600)
    s.text(X(p95) + 8, ytop + 10, "fall left of this line", size=12, fill=TEXT2)
    s.line(X(probe), ytop - 18, X(probe), yax, stroke=ORANGE, width=2.6)
    s.text(X(probe) - 8, ytop - 6, "edit along the probe", size=12.5, weight=700, fill=ORANGE, anchor="end")
    s.text(X(probe) - 8, ytop + 10, "to pass, it had to be right", size=12, fill=TEXT2, anchor="end")
    s.text(X(probe) - 8, ytop + 26, "of the dashed line", size=12, fill=TEXT2, anchor="end")
    s.line(xl, yax, xr, yax, stroke=MUTED)
    for t in nice_ticks(lo, hi, 1.0):
        s.line(X(t), yax, X(t), yax + 5, stroke=MUTED)
        s.text(X(t), yax + 20, ("0" if t == 0 else f"{t:+.0f}").replace("-", "−"), size=11.5, fill=TEXT2,
               anchor="middle")
    s.text(xl, yax + 44, "← against the partner’s turn", size=11.5, fill=TEXT2)
    s.text(xr, yax + 44, "with it →   (10⁻⁴ of its usual spread)", size=11.5, fill=TEXT2, anchor="end")
    s.save(FIGS / "fig9-null.svg")


# =============================================================== architecture: where the probe reads and the edit writes
def fig_arch(path=FIGS / "fig-arch.svg", num=8):
    s = Svg(448, "SmolVLA during one control step. Camera images, the instruction and the joint readings go into a "
                 "16-layer vision-language model, which runs once. The 16-layer action expert then runs 10 refinement "
                 "steps, from random noise to 50 planned movements, and reads the vision-language context at every step. "
                 "The probe reads, and the edit writes, at one place: after layer 4 of the action expert, in the first "
                 "refinement step, averaged over the 50 movement tokens. Four other measured places are marked in grey.")
    s.header(num, "Where the probe reads and the edit writes",
             "SmolVLA during one control step. Rows are layers, columns are refinement steps.")
    rows, rh, rg, top = 16, 12, 1, 112
    bottom = top + rows * rh + (rows - 1) * rg
    row_y = lambda r: bottom - (r + 1) * rh - r * rg          # top edge of layer r+1 (r = 0 is layer 1)
    yb = lambda k: bottom - k * (rh + rg) + rg / 2              # boundary after k layers
    GREY = "#8f8f8f"
    # vision-language model
    vx, vw = 132, 54
    for r in range(rows):
        s.rect(vx, row_y(r), vw, rh, fill=FILL)
    s.text(vx + vw / 2, top - 30, "Vision-language model", size=12.5, weight=700, anchor="middle")
    s.text(vx + vw / 2, top - 15, "16 layers, runs once", size=11.5, fill=TEXT2, anchor="middle")
    for i, t in enumerate(("joint readings", "instruction", "camera images")):
        y = row_y(i * 2) + rh / 2 + 4
        s.text(112, y, t, size=12, anchor="end")
        s.line(116, y - 4, vx - 4, y - 4, stroke=MUTED, width=1.2, marker="muted")
    # action expert grid
    gx, cw, cg, cols = 290, 22, 4, 10
    gx2 = gx + cols * cw + (cols - 1) * cg
    for c in range(cols):
        x = gx + c * (cw + cg)
        for r in range(rows):
            s.rect(x, row_y(r), cw, rh, fill=FILL)
        s.text(x + cw / 2, bottom + 17, str(c + 1), size=11.5, fill=INK if c == 0 else TEXT2,
               weight=700 if c == 0 else None, anchor="middle")
    s.text((gx + gx2) / 2, top - 30, "Action expert", size=12.5, weight=700, anchor="middle")
    s.text((gx + gx2) / 2, top - 15, "16 layers, 720 numbers per token", size=11.5, fill=TEXT2, anchor="middle")
    s.text(gx - 10, row_y(15) + rh / 2 + 4, "layer 16", size=11, fill=TEXT2, anchor="end")
    s.text(gx - 10, row_y(0) + rh / 2 + 4, "layer 1", size=11, fill=TEXT2, anchor="end")
    s.line(gx, bottom + 29, gx2, bottom + 29, stroke=MUTED, width=1.1, marker="muted")
    s.text(gx, bottom + 45, "random noise", size=11.5, fill=TEXT2)
    s.text(gx2, bottom + 45, "planned movements", size=11.5, fill=TEXT2, anchor="end")
    # context arrow
    ym = row_y(9) + rh / 2
    s.line(vx + vw + 6, ym, gx - 6, ym, stroke=INK, width=1.2, marker="ink")
    s.text((vx + vw + gx) / 2, ym - 10, "context", size=11.5, fill=TEXT2, anchor="middle")
    s.text((vx + vw + gx) / 2, ym + 20, "read at every step", size=11.5, fill=TEXT2, anchor="middle")
    # output
    yo = row_y(15) + rh / 2
    s.line(gx2 + 4, yo, gx2 + 22, yo, stroke=INK, width=1.2, marker="ink")
    for k, t in enumerate(("50 planned", "movements;", "the first one", "is carried out")):
        s.text(gx2 + 27, yo + 4 + k * 15, t, size=12)
    # other measured places: after 4 and 12 layers at steps 1 and 6, and the VLM state token after 12 layers
    def tick(x0, x1, y):
        s.line(x0 + 2, y, x1 - 2, y, stroke=GREY, width=2.2, cap="round")
    for c, k in ((5, 4), (0, 12), (5, 12)):
        x = gx + c * (cw + cg)
        tick(x, x + cw, yb(k))
    tick(vx, vx + vw, yb(12))
    # the site
    y = yb(4)
    s.line(gx - 1, y - 1.4, gx + cw + 1, y - 1.4, stroke=BLUE, width=2.6)
    s.line(gx - 1, y + 1.4, gx + cw + 1, y + 1.4, stroke=ORANGE, width=2.6)
    lx, ly = gx - 4, bottom + 78
    s.path(f"M{gx - 2},{y} L{lx},{y} L{lx},{ly} L{lx + 8},{ly}", stroke=MUTED, width=1)
    s.rich(lx + 12, ly + 4, [("probe reads", {"fill": BLUE, "weight": 700}), (" and ", {"fill": TEXT2}),
                             ("edit writes", {"fill": ORANGE, "weight": 700}), (" here", {"fill": TEXT2})], size=12.5)
    s.text(lx + 12, ly + 20, "after layer 4 of 16, refinement step 1,", size=11.5, fill=TEXT2)
    s.text(lx + 12, ly + 35, "averaged over the 50 movement tokens", size=11.5, fill=TEXT2)
    # legend for the grey marks
    s.line(26, ly, 40, ly, stroke=GREY, width=2.2, cap="round")
    s.text(46, ly + 4, "other places I measured;", size=11.5, fill=TEXT2)
    s.text(46, ly + 19, "a rule fixed in advance", size=11.5, fill=TEXT2)
    s.text(46, ly + 34, "picked the coloured one", size=11.5, fill=TEXT2)
    s.save(path)

# =============================================================== architecture variants (A, B, C)
GREYMARK = "#8f8f8f"


def _site_bar_h(s, x0, x1, y):
    """Blue-over-orange double line: the probe reads and the edit writes at the same place."""
    s.line(x0, y - 1.4, x1, y - 1.4, stroke=BLUE, width=2.6)
    s.line(x0, y + 1.4, x1, y + 1.4, stroke=ORANGE, width=2.6)


def _site_bar_v(s, x, y0, y1):
    s.line(x - 1.4, y0, x - 1.4, y1, stroke=BLUE, width=2.6)
    s.line(x + 1.4, y0, x + 1.4, y1, stroke=ORANGE, width=2.6)


def _site_label(s, x, y, anchor="start", lines=("after layer 4 of 16,", "in the first of 10 passes"), stacked=False):
    if stacked:
        s.text(x, y, "probe reads", size=12.5, weight=700, fill=BLUE, anchor=anchor)
        s.text(x, y + 16, "edit writes", size=12.5, weight=700, fill=ORANGE, anchor=anchor)
        off = 32
    else:
        s.rich(x, y, [("probe reads", {"fill": BLUE, "weight": 700}), (" and ", {"fill": TEXT2}),
                      ("edit writes", {"fill": ORANGE, "weight": 700}), (" here", {"fill": TEXT2})],
               size=12.5, anchor=anchor)
        off = 16
    for k, t in enumerate(lines):
        s.text(x, y + off + 15 * k, t, size=11.5, fill=TEXT2, anchor=anchor)


def _box(s, x, y, w, h, lines, fill=FILL, stroke="none", bold_first=False, size=12):
    s.rect(x, y, w, h, fill=fill, stroke=stroke, rx=5)
    n = len(lines)
    for k, t in enumerate(lines):
        s.text(x + w / 2, y + h / 2 + 4 + (k - (n - 1) / 2) * 15, t, size=size, anchor="middle",
               weight=600 if (bold_first and k == 0) else None)


def fig_arch_a(path=FIGS / "variants/fig-arch-A-paper.svg"):
    """Like the SmolVLA paper: two layer stacks, bottom to top, with the refinement loop."""
    s = Svg(470, "SmolVLA drawn bottom to top. Camera images, instruction and joint readings enter a 16-layer "
                 "vision-language model. Each of the 16 action-expert layers reads the matching vision-language layer. "
                 "Random noise enters the action expert at the bottom and 50 planned movements leave at the top; the plan "
                 "goes round 10 times. The probe reads and the edit writes after layer 4 of the action expert, in the first pass.")
    s.header(8, "Where the probe reads and the edit writes", "Variant A: information flows from bottom to top")
    rows, rh, rg, top = 16, 13, 3, 116
    bottom = top + rows * rh + (rows - 1) * rg
    ry = lambda r: bottom - (r + 1) * rh - r * rg
    yb = lambda k: bottom - k * (rh + rg) + rg / 2
    vx0, vx1, ex0, ex1 = 112, 262, 332, 452
    for r in range(rows):
        s.rect(vx0, ry(r), vx1 - vx0, rh, fill=FILL, rx=2)
        s.rect(ex0, ry(r), ex1 - ex0, rh, fill=FILL, rx=2)
        s.line(vx1 + 4, ry(r) + rh / 2, ex0 - 4, ry(r) + rh / 2, stroke=MUTED, width=0.8, marker="muted")
    s.text((vx0 + vx1) / 2, top - 10, "Vision-language model", size=12.5, weight=700, anchor="middle")
    s.text(ex0, top - 10, "Action expert", size=12.5, weight=700)
    for k, lab in ((0, "layer 1"), (15, "layer 16")):
        s.text(vx0 - 8, ry(k) + rh / 2 + 4, lab, size=11, fill=TEXT2, anchor="end")
    s.text((vx1 + ex0) / 2, bottom + 18, "each layer reads", size=11, fill=TEXT2, anchor="middle")
    s.text((vx1 + ex0) / 2, bottom + 31, "the matching one", size=11, fill=TEXT2, anchor="middle")
    # inputs and noise (bottom)
    s.line((vx0 + vx1) / 2, bottom + 44, (vx0 + vx1) / 2, bottom + 6, stroke=INK, width=1.2, marker="ink")
    s.text((vx0 + vx1) / 2, bottom + 60, "camera images, instruction,", size=12, anchor="middle")
    s.text((vx0 + vx1) / 2, bottom + 75, "joint readings", size=12, anchor="middle")
    s.line((ex0 + ex1) / 2, bottom + 44, (ex0 + ex1) / 2, bottom + 6, stroke=INK, width=1.2, marker="ink")
    s.text((ex0 + ex1) / 2, bottom + 60, "random noise", size=12, anchor="middle")
    # output (top)
    _box(s, 380, 62, 160, 34, ["50 planned movements"], fill="#ffffff", stroke=INK, size=12)
    s.line(444, top - 4, 444, 98, stroke=INK, width=1.2, marker="ink")
    s.text(454, 112, "the first is carried out", size=11, fill=TEXT2)
    # refinement loop on the far right
    lx = 640
    s.path(f"M540,79 L{lx},79 L{lx},{bottom + 40} L{(ex0 + ex1) / 2 + 8},{bottom + 40}", stroke=TEXT2, width=1.2,
           dash="4 3", marker="text2")
    s.text(lx - 8, (top + bottom) / 2 - 8, "plan goes", size=11.5, fill=TEXT2, anchor="end")
    s.text(lx - 8, (top + bottom) / 2 + 7, "round 10 times", size=11.5, fill=TEXT2, anchor="end")
    # other measured places
    for x0, x1 in ((vx0, vx1), (ex0, ex1)):
        s.line(x0 + 3, yb(12), x1 - 3, yb(12), stroke=GREYMARK, width=2.2, cap="round")
    # the site
    _site_bar_h(s, ex0 - 2, ex1 + 2, yb(4))
    s.line(ex1 + 6, yb(4), 470, yb(4), stroke=MUTED, width=1)
    _site_label(s, 476, yb(4) + 4, lines=("after layer 4 of 16,", "in the first of", "10 passes"), stacked=True)
    s.line(476, yb(12) + 1, 490, yb(12) + 1, stroke=GREYMARK, width=2.2, cap="round")
    s.text(496, yb(12) + 5, "other places", size=11, fill=TEXT2)
    s.text(496, yb(12) + 18, "I measured", size=11, fill=TEXT2)
    s.save(path)


def fig_arch_b(path=FIGS / "variants/fig-arch-B-flow.svg"):
    """Left to right: layers along the direction of flow, the refinement loop drawn as a return arrow."""
    s = Svg(330, "SmolVLA drawn left to right. Inputs enter a 16-layer vision-language model. Each layer of the "
                 "16-layer action expert below reads the matching vision-language layer. Random noise enters the action "
                 "expert on the left, goes round 10 times, and leaves on the right as 50 planned movements. The probe reads "
                 "and the edit writes between layers 4 and 5 of the action expert, in the first pass.")
    s.header(8, "Where the probe reads and the edit writes", "Variant B: information flows from left to right")
    n, sw, sg, x0 = 16, 15, 2, 150
    xs = lambda k: x0 + k * (sw + sg)
    xe = xs(n) - sg
    xb = lambda k: x0 + k * (sw + sg) - sg / 2
    vy, ey, h = 96, 196, 34
    for k in range(n):
        s.rect(xs(k), vy, sw, h, fill=FILL, rx=2)
        s.rect(xs(k), ey, sw, h, fill=FILL, rx=2)
    s.text(xe + 10, vy + 14, "Vision-language", size=12.5, weight=700)
    s.text(xe + 10, vy + 29, "model, 16 layers", size=12.5, weight=700)
    s.text(xe + 10, ey - 6, "Action expert,", size=12.5, weight=700)
    s.text(xe + 10, ey + 9, "16 layers", size=12.5, weight=700)
    # context: one arrow per layer would crowd the picture; one arrow and a label say the same
    cx = xs(14)
    s.line(cx, vy + h + 4, cx, ey - 5, stroke=MUTED, width=1.2, marker="muted")
    s.text(cx + 8, (vy + h + ey) / 2 - 2, "each expert layer reads", size=11, fill=TEXT2)
    s.text(cx + 8, (vy + h + ey) / 2 + 11, "the matching layer above", size=11, fill=TEXT2)
    # inputs
    for i, t in enumerate(("camera images,", "instruction,", "joint readings")):
        s.text(126, vy + 5 + i * 13, t, size=11.5, anchor="end")
    s.line(130, vy + h / 2, x0 - 4, vy + h / 2, stroke=INK, width=1.2, marker="ink")
    # noise in, plan out
    s.text(126, ey + h / 2 + 4, "random noise", size=12, anchor="end")
    s.line(130, ey + h / 2, x0 - 4, ey + h / 2, stroke=INK, width=1.2, marker="ink")
    s.line(xe + 4, ey + h / 2 + 10, xe + 30, ey + h / 2 + 10, stroke=INK, width=1.2, marker="ink")
    s.text(xe + 36, ey + h / 2 + 14, "50 planned movements", size=12)
    s.text(xe + 36, ey + h / 2 + 28, "after pass 10", size=11, fill=TEXT2)
    # loop back
    ly = ey + h + 22
    s.path(f"M{xe - 4},{ey + h + 3} L{xe - 4},{ly} L{x0 - 14},{ly} L{x0 - 14},{ey + h / 2 + 6}", stroke=TEXT2,
           width=1.2, dash="4 3", marker="text2")
    s.text(xe - 10, ly + 15, "the plan goes round 10 times", size=11.5, fill=TEXT2, anchor="end")
    # other measured places
    for yy in (vy, ey):
        s.line(xb(12), yy + 3, xb(12), yy + h - 3, stroke=GREYMARK, width=2.2, cap="round")
    s.line(24, ly + 52, 38, ly + 52, stroke=GREYMARK, width=2.2, cap="round")
    s.text(44, ly + 56, "other places I measured", size=11, fill=TEXT2)
    # the site, labelled in the free space between the two models
    _site_bar_v(s, xb(4), ey - 2, ey + h + 2)
    s.path(f"M{xb(4)},{ey - 4} L{xb(4)},{ey - 14} L{xb(4) + 6},{ey - 14}", stroke=MUTED, width=1)
    _site_label(s, xb(4) + 10, ey - 46, lines=("after layer 4 of 16, first pass",), stacked=True)
    s.save(path)


def fig_arch_c(path=FIGS / "variants/fig-arch-C-unrolled.svg"):
    """The 10 refinement passes unrolled from left to right, with a zoom into the first pass."""
    s = Svg(384, "SmolVLA with its 10 refinement passes unrolled from left to right. A vision-language model turns the "
                 "inputs into a context that every pass reads. Random noise goes through pass 1 to pass 10 and comes out as "
                 "50 planned movements. A zoom into pass 1 shows the action expert's 16 layers; the probe reads and the edit "
                 "writes after layer 4.")
    s.header(8, "Where the probe reads and the edit writes", "Variant C: the 10 refinement passes unrolled")
    # vision-language model and context bus (top row)
    vy, vh = 78, 34
    for k, t in enumerate(("camera images,", "instruction,", "joint readings")):
        s.text(118, vy + 5 + k * 13, t, size=11.5, anchor="end")
    s.line(122, vy + vh / 2, 134, vy + vh / 2, stroke=INK, width=1.2, marker="ink")
    _box(s, 138, vy, 118, vh, ["Vision-language model"], fill=FILL, size=11.5)
    s.circle(138 + 118 - 8, vy + vh - 7, 2.4, fill=GREYMARK)
    # passes (track row)
    bx0, pw, pg, n = 268, 24, 8, 10
    px = lambda i: bx0 + i * (pw + pg)
    pend = px(n - 1) + pw
    ty, th = 146, 36
    s.path(f"M256,{vy + vh / 2} L{pend - pw / 2},{vy + vh / 2}", stroke=MUTED, width=1.1)
    s.text(pend - pw / 2, vy + vh / 2 - 7, "context, read by every pass", size=11, fill=TEXT2, anchor="end")
    for i in range(n):
        s.line(px(i) + pw / 2, vy + vh / 2, px(i) + pw / 2, ty - 3, stroke=MUTED, width=0.9, marker="muted")
    s.text(246, ty + th / 2 + 4, "noise", size=12, anchor="end")
    s.line(250, ty + th / 2, pend + 20, ty + th / 2, stroke=INK, width=1.2, marker="ink")
    for i in range(n):
        s.rect(px(i), ty, pw, th, fill=BLUE_L if i == 0 else FILL, stroke=BLUE if i == 0 else "none", width=1.3, rx=3)
        s.text(px(i) + pw / 2, ty + th / 2 + 4, str(i + 1), size=11, anchor="middle", fill=INK if i == 0 else TEXT2,
               weight=700 if i == 0 else None)
    s.circle(px(5) + pw / 2, ty + th - 6, 2.4, fill=GREYMARK)
    s.text(pend + 24, ty + th / 2 - 3, "50 planned", size=12)
    s.text(pend + 24, ty + th / 2 + 12, "movements", size=12)
    s.text(pend, ty + th + 16, "10 refinement passes", size=11, fill=TEXT2, anchor="end")
    # zoom into pass 1
    zy, zh, zx0, zx1, nl = 244, 34, 128, 560, 16
    sw = (zx1 - zx0 - (nl - 1) * 2) / nl
    zs = lambda k: zx0 + k * (sw + 2)
    zb = lambda k: zx0 + k * (sw + 2) - 1
    s.path(f"M{px(0)},{ty + th + 2} L{zx0},{zy - 3}", stroke=BLUE, width=0.9, dash="3 3")
    s.path(f"M{px(0) + pw},{ty + th + 2} L{zx1},{zy - 3}", stroke=BLUE, width=0.9, dash="3 3")
    for k in range(nl):
        s.rect(zs(k), zy, sw, zh, fill=FILL, rx=2)
    s.text(zx0 - 10, zy + zh / 2 - 2, "inside", size=11.5, weight=600, anchor="end")
    s.text(zx0 - 10, zy + zh / 2 + 12, "pass 1", size=11.5, weight=600, anchor="end")
    s.text(zx0, zy + zh + 16, "layer 1", size=11, fill=TEXT2)
    s.text(zx1, zy + zh + 16, "layer 16", size=11, fill=TEXT2, anchor="end")
    s.line(zx1 + 4, zy + zh / 2, zx1 + 22, zy + zh / 2, stroke=INK, width=1.2, marker="ink")
    s.line(zb(12), zy + 3, zb(12), zy + zh - 3, stroke=GREYMARK, width=2.2, cap="round")
    _site_bar_v(s, zb(4), zy - 2, zy + zh + 2)
    s.path(f"M{zb(4)},{zy + zh + 4} L{zb(4)},{zy + zh + 30} L{zb(4) + 6},{zy + zh + 30}", stroke=MUTED, width=1)
    _site_label(s, zb(4) + 10, zy + zh + 34, lines=("after layer 4 of 16, averaged over", "the 50 movement tokens"))
    s.line(zx1 - 116, zy + zh + 34, zx1 - 102, zy + zh + 34, stroke=GREYMARK, width=2.2, cap="round")
    s.text(zx1 - 96, zy + zh + 38, "other places", size=11, fill=TEXT2)
    s.text(zx1 - 96, zy + zh + 51, "I measured (also", size=11, fill=TEXT2)
    s.text(zx1 - 96, zy + zh + 64, "in pass 6)", size=11, fill=TEXT2)
    s.save(path)

if __name__ == "__main__":
    (FIGS / "variants").mkdir(parents=True, exist_ok=True)
    fig_runs()
    fig_changes()
    fig_monitors_b()
    fig_monitors_b(path=FIGS / "variants/fig3-variant-B-cards.svg")
    fig_monitors_a()
    fig_monitors_c()
    fig_stages()
    fig_result1()
    fig_passmark()
    fig_result2()
    fig_edit_a()
    fig_edit_a(path=FIGS / "variants/fig8-variant-A-geometry.svg", sub="Variant A: the geometry of the edit")
    fig_edit_b()
    fig_edit_c()
    fig_null()
    fig_arch()
    fig_arch_a()
    fig_arch_b()
    fig_arch_c()
    print("figures:", sorted(p.name for p in FIGS.glob("*.svg")))
