# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""race.html -> docs/race-replay.mp4: the baseline-vs-optimized replay rendered frame by frame at recorded speed.

uv run render_race.py            (once: uv run --with playwright playwright install chromium)
Each prompt plays from 0 s until both sides finish, then holds the final frame for 2 s. Needs ffmpeg.
"""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

FPS, HOLD, OUT = 20, 2.0, Path("docs/race-replay.mp4")
html = Path("race.html").resolve()
data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html.read_text(), re.S).group(1))
work = Path(tempfile.mkdtemp(prefix="race-frames-"))
n = 0
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    pg.goto(html.as_uri() + "?video")
    for i, req in enumerate(data["base"]["requests"]):
        end = max(req["latency"], data["best"]["requests"][i]["latency"]) + HOLD
        for k in range(int(end * FPS) + 1):
            pg.evaluate(f"window.renderAt({i}, {k / FPS})")
            pg.screenshot(path=str(work / f"f{n:06d}.png"))
            n += 1
        print(f"prompt {req['id']}: {end:.1f} s, {n} frames so far", flush=True)
    b.close()
OUT.parent.mkdir(exist_ok=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(work / "f%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(OUT)], check=True)
shutil.rmtree(work)
print(f"wrote {OUT} ({n} frames, {n / FPS:.1f} s)")
