"""Render SVG figures to 2x PNG previews with headless Chrome.

  python blog/src/render.py blog/figures/fig1-runs.svg [more.svg ...]  -> <name>.png next to each SVG
Chrome sometimes writes the screenshot and then does not exit, so a written PNG counts as success.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render(svg: Path):
    text = svg.read_text()
    w, h = (int(float(v)) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text).groups())
    png = svg.with_suffix(".png")
    png.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "page.html"
        html.write_text(f'<html><body style="margin:0"><img src="{svg.resolve().as_uri()}" '
                        f'width="{w}" height="{h}"></body></html>')
        cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               f"--user-data-dir={tmp}/profile", "--force-device-scale-factor=2",
               f"--window-size={w},{h}", f"--screenshot={png}", "--allow-file-access-from-files",
               html.as_uri()]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
        except subprocess.TimeoutExpired:
            pass
    print(("ok   " if png.exists() else "FAIL ") + str(png))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        render(Path(arg))
