# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "huggingface_hub"]
# ///
"""data/chat.jsonl: 1000 WildChat first turns (English, non-toxic, non-redacted; 10x the prompts
so a 64-client sweep never runs out). Run: uv run prep.py

data/demo.jsonl (uv run prep.py --demo): 20 prompts for demo.py, evenly spaced across the three benchmarks used:
7 of the 40 speed-benchmark chat prompts, 7 of the 100 GSM8K Platinum and 6 of the 100 IFEval questions the accuracy
suite runs, interleaved chat / GSM8K / IFEval. Needs data/chat.jsonl."""
import sys
import json, random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

def even(rows, k):
    """k rows spread evenly over the list (first row always included)."""
    return [rows[i * len(rows) // k] for i in range(k)]


def hf_rows(dataset, config, split, n=100):
    """First n rows from the Hugging Face datasets server (the same first-n slice lm-eval's --limit takes)."""
    import urllib.request
    url = f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset=0&length={n}"
    return [r["row"] for r in json.load(urllib.request.urlopen(url))["rows"]]


def demo():
    chat = [json.loads(l) for l in open("data/chat.jsonl")][:40]  # the speed benchmark's 40 prompts
    gsm = hf_rows("madrylab/gsm8k-platinum", "main", "test")
    ife = hf_rows("google/IFEval", "default", "train")
    c = [{"id": r["id"], "prompt": r["prompt"]} for r in even(chat, 7)]
    g = [{"id": f"gsm8k-{gsm.index(r)}", "prompt": f"Q: {r['question']}\nA: Let's think step by step."} for r in even(gsm, 7)]  # eval's template
    i = [{"id": f"ifeval-{ife.index(r)}", "prompt": r["prompt"]} for r in even(ife, 6)]
    rows = [x for trio in zip(c, g, i + [None]) for x in trio if x]
    assert len(rows) == 20 and len({r["id"] for r in rows}) == 20
    with open("data/demo.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "max_tokens": 1024}, ensure_ascii=False) + "\n")
    print("wrote data/demo.jsonl:", " ".join(r["id"] for r in rows))


if "--demo" in sys.argv:
    demo()
    sys.exit()

random.seed(0)
p = hf_hub_download("allenai/WildChat-1M", "data/train-00000-of-00014.parquet", repo_type="dataset")
cands = []
for b in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=["conversation", "language", "toxic", "redacted"]):
    for r in b.to_pylist():
        first = r["conversation"][0]
        # ponytail: chars/4 stands in for a tokenizer; the bound only drops empty and giant prompts
        if r["language"] == "English" and not r["toxic"] and not r["redacted"] and first["role"] == "user" \
                and 40 <= len(first["content"]) <= 6000:
            cands.append(first["content"])
    if len(cands) >= 10000:
        break
rows = random.sample(cands, 1000)
Path("data").mkdir(exist_ok=True)
with open("data/chat.jsonl", "w") as f:
    for i, prompt in enumerate(rows):
        f.write(json.dumps({"id": f"chat-{i}", "max_tokens": 1024, "prompt": prompt}, ensure_ascii=False) + "\n")
assert len(rows) == 1000 and len(set(rows)) > 950, "duplicate-heavy sample"
print("wrote data/chat.jsonl", len(rows))
