"""Tiny SVG toolkit shared by every blog figure.

Same visual system as the steering post: white plate, Helvetica Neue, grey/black for
structure, one blue for "read from inside the network", one orange for "write into it".
Figures are drawn at the text-column width (680 px) so they render 1:1.
"""
from __future__ import annotations

import base64
import math
from html import escape
from pathlib import Path

W = 680
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MATH = "'STIX Two Text', 'STIX Two Math', 'Cambria Math', 'Times New Roman', Times, serif"

INK = "#1c1c1c"        # structure, and the simulator monitor
TEXT2 = "#6a6a6a"      # secondary text
MUTED = "#b9b9b9"      # axes, placeholders
GRID = "#e8e8e8"
FILL = "#f4f4f4"
OUT = "#8f8f8f"        # the outputs-only monitor
BLUE = "#2f6db3"       # reading the internals
BLUE_L = "#e3ecf6"
ORANGE = "#d4582b"     # writing into the internals
ORANGE_L = "#f9e3d9"


class Svg:
    def __init__(self, height: int, label: str):
        self.h = height
        self.label = label
        self.parts: list[str] = []

    def add(self, s: str):
        self.parts.append(s)

    # ------------------------------------------------------------------ primitives
    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight=None, italic=False,
             family=SANS, baseline=None):
        w = f' font-weight="{weight}"' if weight else ""
        it = ' font-style="italic"' if italic else ""
        bl = f' dominant-baseline="{baseline}"' if baseline else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
                 f'{w}{it}{bl} font-family="{family}">{escape(s)}</text>')

    def rich(self, x, y, runs, size=13, anchor="start"):
        """runs: list of (text, dict(fill=, weight=, italic=))."""
        spans = []
        for s, st in runs:
            a = f' fill="{st.get("fill", INK)}"'
            if st.get("weight"):
                a += f' font-weight="{st["weight"]}"'
            if st.get("italic"):
                a += ' font-style="italic"'
            if st.get("family"):
                a += f' font-family="{st["family"]}"'
            spans.append(f"<tspan{a}>{escape(s)}</tspan>")
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-family="{SANS}" xml:space="preserve">{"".join(spans)}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0, dash=None, marker=None, cap="butt"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{marker})"' if marker else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
                 f'stroke-width="{width}" stroke-linecap="{cap}"{d}{m}/>')

    def path(self, d, stroke=INK, width=1.0, fill="none", dash=None, marker=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{marker})"' if marker else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{m}/>')

    def rect(self, x, y, w, h, fill=FILL, stroke="none", width=1.0, rx=0, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{width}"{da}/>')

    def circle(self, cx, cy, r, fill=INK, stroke="none", width=1.0):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width}"/>')

    def image(self, x, y, w, h, path: Path, crop=None):
        """Embed a JPEG. crop = (x0, y0, size) in source pixels, drawn via a nested viewBox."""
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        href = f"data:image/jpeg;base64,{b64}"
        if crop:
            cx, cy, cs = crop
            self.add(f'<svg x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                     f'viewBox="{cx} {cy} {cs} {cs}" preserveAspectRatio="xMidYMid slice">'
                     f'<image href="{href}" x="0" y="0" width="360" height="360"/></svg>')
        else:
            self.add(f'<image href="{href}" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"/>')

    def header(self, num, title, sub=None):
        self.text(24, 30, str(num), size=15, weight=700)
        self.text(24 + 9.5 * len(str(num)) + 8, 30, title, size=15, weight=700)
        if sub:
            self.text(24, 52, sub, size=13, fill=TEXT2)

    # ------------------------------------------------------------------ output
    def render(self) -> str:
        defs = "".join(
            f'<marker id="ah-{n}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="8" markerHeight="8" '
            f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0.6,0.8 L9.4,5 L0.6,9.2 Z" '
            f'fill="{c}"/></marker>'
            for n, c in (("ink", INK), ("muted", MUTED), ("blue", BLUE), ("orange", ORANGE),
                         ("text2", TEXT2), ("out", OUT)))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" '
                f'viewBox="0 0 {W} {self.h}" role="img" aria-label="{escape(self.label)}" '
                f'font-family="{SANS}" text-rendering="geometricPrecision">'
                f'<rect width="{W}" height="{self.h}" fill="#ffffff"/><defs>{defs}</defs>'
                + "".join(self.parts) + "</svg>")

    def save(self, path: Path):
        Path(path).write_text(self.render(), encoding="utf-8")


def fmt2(v):
    return f"{v:.2f}"


def nice_ticks(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def angle_vec(deg, length):
    r = math.radians(deg)
    return length * math.cos(r), -length * math.sin(r)
