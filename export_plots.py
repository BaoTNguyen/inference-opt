# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""race.html -> docs/plots/*.png: each comparison chart as its own image, plus a still of each race.

uv run export_plots.py          (needs race.html from `uv run race.py build`)
"""
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/plots")
OUT.mkdir(parents=True, exist_ok=True)
slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)  # 2x for crisp images
    pg.goto(Path("race.html").resolve().as_uri())
    pg.add_style_tag(content="html, body { background: #0c0c0c !important; }")  # cards end on a fractional pixel; keep that edge row dark
    for card in pg.query_selector_all(".card"):
        title = card.query_selector("h3").inner_text()
        card.screenshot(path=str(OUT / f"{slug(title)}.png"))
        print("wrote", OUT / f"{slug(title)}.png")
    # race stills: each prompt at the moment the optimized side finishes, baseline still streaming
    pg.goto(Path("race.html").resolve().as_uri() + "?video")
    for i, rid in enumerate(pg.evaluate("DATA.base.requests.map(r => r.id)")):
        t = pg.evaluate(f"DATA.best.requests[{i}].latency") + 0.2
        pg.evaluate(f"window.renderAt({i}, {t})")
        pg.query_selector(".win").screenshot(path=str(OUT / f"race-{rid}.png"))
        print("wrote", OUT / f"race-{rid}.png")
    b.close()
