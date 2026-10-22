#!/usr/bin/env bash
# After the race video: export plots, pull check frames, fill the baseline accuracy into the docs, privacy-scan,
# commit. Push stays manual, after the frames and numbers are looked at.
set -eu
cd "$(dirname "$0")"
CHECK=${CHECK:-/tmp/race-check}; mkdir -p "$CHECK"
for f in race.html docs/race-replay.mp4 results/race/base.json results/race/best.json; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
uv run -q export_plots.py
d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 docs/race-replay.mp4)
for frac in 0.15 0.30 0.50 0.70 0.90; do  # frames across the three races for a visual check
  ffmpeg -v error -y -ss "$(python3 -c "print($d*$frac)")" -i docs/race-replay.mp4 -frames:v 1 -vf scale=1280:-1 "$CHECK/frame-$frac.png"
done
python3 - <<'E'
import glob, json, re
import rescore
from pathlib import Path
def scores(run):
    d = Path("results/eval") / run
    g = sorted(glob.glob(f"{d}/*/samples_gsm8k*_*.jsonl"))[-1]; i = sorted(glob.glob(f"{d}/*/samples_ifeval_*.jsonl"))[-1]
    rows = [json.loads(l) for l in open(g) if '"flexible-extract"' in l]
    gold = lambda r: rescore.number(r["doc"]["answer"].split("####")[-1])
    fa = sum(1 for r in rows if (v := rescore.final_answer(r["resps"][0][0] if r["resps"] else "")) is not None and abs(v - gold(r)) < 1e-6)
    flex = sum(bool(r["exact_match"]) for r in rows)
    ir = [json.loads(l) for l in open(i)]
    ife = sum(bool(r["prompt_level_strict_acc"]) for r in ir); empty = sum(not (r["resps"][0][0] if r["resps"] else "").strip() for r in ir)
    return fa, flex, len(rows), ife, empty, len(ir)
b = scores("base-acc"); o = scores("v032-dflash2w4-k7")
pts = lambda x, y: f"{x - y:+d} pt" + ("s" if abs(x - y) > 1 else "") if x != y else "same"
fill = {"{{BASE_GSM8K}}": f"{b[0]}%", "{{BASE_IFEVAL}}": f"{b[3]}%", "{{D_GSM8K}}": pts(o[0], b[0]), "{{D_IFEVAL}}": pts(o[3], b[3]),
        "{{BASE_GSM8K_ROW}}": f"{b[0]}% final answer (flexible-extract: {b[1]}%)", "{{BASE_IFEVAL_ROW}}": f"{b[3]}% ({b[4]} empty answers)"}
for p in ("README.md", "RUNS.md"):
    t = Path(p).read_text()
    for k, v in fill.items():
        t = t.replace(k, v)
    assert "{{" not in t, p
    Path(p).write_text(t)
print("baseline:", b, "| optimized:", o)
E
uv run -q race.py build >/dev/null  # rebuild so the accuracy chart has the baseline bars
uv run -q export_plots.py >/dev/null
# privacy: nothing personal in any tracked or new file
# patterns live in .privacy-patterns (git-ignored, one extended regex per line) so the list itself stays private
[ -s .privacy-patterns ] || { echo "no .privacy-patterns"; exit 1; }
if grep -r -l -I -E -f .privacy-patterns README.md RUNS.md race.html docs results; then echo "PRIVACY: personal strings found"; exit 1; fi
git add README.md RUNS.md race.html docs results/race/base.json results/race/best.json results/summary.jsonl results/eval/base-acc export_plots.py finalize.sh
git commit -q -F - <<'M'
Baseline vs optimized: accuracy suite on the baseline, race replay video, chart images

README gains a baseline-vs-optimized table (2.0x decode speed, +93% TPM) with both configs' accuracy on the
same suite, the race replay video (docs/race-replay.mp4) and the comparison charts (docs/plots/).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014pJZ1zxc5xJLPw4aDQTbXi
M
git log -1 --format='%h %ae'; git status -sb | head -1
echo "check frames in $CHECK"
