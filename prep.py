# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "huggingface_hub"]
# ///
"""data/chat.jsonl: 1000 WildChat first turns (English, non-toxic, non-redacted; 10x the prompts
so a 64-client sweep never runs out). Run: uv run prep.py"""
import json, random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

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
