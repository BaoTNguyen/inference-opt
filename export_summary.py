# /// script
# dependencies = ["pillow"]
# ///
"""docs/plots/*.png -> docs/summary.png: every gate metric and accuracy, on one image to send.

uv run export_summary.py          (after export_plots.py)
"""
from PIL import Image

P = "docs/plots/"
top = [Image.open(P + f) for f in ("decode-speed-typical-p50.png", "output-tokens-per-minute.png")]
mid = [Image.open(P + f) for f in ("time-to-first-token-p95.png", "accuracy.png")]
G, BG = 24, (12, 12, 12)  # gap and the charts' own surface color
row = lambda ims, h: [im.resize((round(im.width * h / im.height), h)) for im in ims]
W = 2900  # output width before the outer margin
def fit(ims):  # scale a pair to one shared height so together they span the wide chart's width
    h = round((W - G) / sum(im.width / im.height for im in ims))
    return row(ims, h)
rows = [fit(top), fit(mid)]
out = Image.new("RGB", (W + 2 * G, sum(r[0].height for r in rows) + G * (len(rows) + 1)), BG)
y = G
for r in rows:
    x = G
    for im in r:
        out.paste(im, (x, y)); x += im.width + G
    y += r[0].height + G
out.save("docs/summary.png", optimize=True)
print("wrote docs/summary.png", out.size)
