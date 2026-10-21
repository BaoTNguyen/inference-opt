"""Sweep table + knee (highest-TPM run that passes the SLO gate). python3 report.py [label ...]"""
import json, sys

S = [json.loads(l) for l in open("results/summary.jsonl")]
labels = sys.argv[1:] or sorted({r["label"] for r in S})
print("| label | clients | TTFT p50 | TTFT p95 | tok/s/req p50 | p5 | TPM | errors | gate |\n|---|---|---|---|---|---|---|---|---|")
best = []
for lab in labels:
    rows = sorted((r for r in S if r["label"] == lab), key=lambda r: r["concurrency"])
    for r in rows:
        print(f"| {lab} | {r['concurrency']} | {r['ttft_p50']:.2f} | {r['ttft_p95']:.2f} | {r['decode_tps_p50']:.1f} "
              f"| {r['decode_tps_p5']:.1f} | {r['tpm']:,} | {r['n_err']} | {'PASS' if r['pass'] else 'fail'} |")
    ok = [r for r in rows if r["pass"]]
    if ok:
        best.append(max(ok, key=lambda r: r["tpm"]))
for r in sorted(best, key=lambda r: -r["tpm"]):
    print(f"\nknee {r['label']}: {r['tpm']:,} TPM at {r['concurrency']} clients")
