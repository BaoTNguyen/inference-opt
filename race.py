# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Side-by-side baseline-vs-optimized replay.

The two configs each need both GPUs, so they can't stream at the same moment. Instead each one is captured on its
own (every streamed chunk with its time and running token count), and race.html replays both side by side at the
recorded speed.

  uv run race.py capture --side base --url http://localhost:8001     # with the baseline server up
  uv run race.py capture --side best --url http://localhost:8001     # with the optimized server up
  uv run race.py build                                                # -> race.html: replay + comparison charts

race.html is also served by demo.py at /compare.

Prompts: one each from the chat, GSM8K Platinum and IFEval sets (PROMPTS below). Temperature 0, 1024 output tokens,
thinking on, one request at a time, exactly as the benchmark.
"""
import argparse, json, time
from pathlib import Path

import httpx

PROMPTS = [("data/chat.jsonl", "chat-17"), ("data/gsm8k40.jsonl", "gsm8k-0"), ("data/ifeval40.jsonl", "ifeval-16")]
OUT = Path("results/race")
SIDES = {"base": "Baseline: vLLM 0.27.1 defaults, no drafting",
         "best": "Optimized: vLLM 0.32, DFlash2 W4A16 draft, full CUDA graphs"}


def capture(a):
    prompts = {}
    for f, pid in PROMPTS:
        prompts[pid] = next(json.loads(l) for l in open(f) if json.loads(l)["id"] == pid)["prompt"]
    OUT.mkdir(parents=True, exist_ok=True)
    reqs = []
    with httpx.Client(timeout=None) as cl:
        for _ in range(2):  # warmups, not recorded
            cl.post(a.url + "/v1/chat/completions", json={"model": a.model, "max_tokens": 32,
                    "messages": [{"role": "user", "content": "Say hello in one sentence."}]})
        for pid, prompt in prompts.items():
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024,
                    "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True, "continuous_usage_stats": True}}  # exact token count per chunk
            ev, t0, first, last_n = [], time.perf_counter(), None, None
            with cl.stream("POST", a.url + "/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line.endswith("[DONE]"):
                        continue
                    d = json.loads(line[6:])
                    last_n = (d.get("usage") or {}).get("completion_tokens") or last_n  # final usage chunk has no text
                    delta = (d.get("choices") or [{}])[0].get("delta", {})
                    for k, kind in (("reasoning_content", "thinking"), ("reasoning", "thinking"), ("content", "answer")):
                        if delta.get(k):
                            t = time.perf_counter() - t0
                            first = first if first is not None else t
                            ev.append([round(t, 4), kind, delta[k], last_n])
            total = time.perf_counter() - t0
            ntok = last_n
            reqs.append({"id": pid, "prompt": prompt, "ttft": first, "latency": total, "tokens": ntok,
                         "decode_tps": (ntok - 1) / (total - first) if ntok and total > first else None, "events": ev})
            print(f"{a.side} {pid}: ttft {first:.2f} s, {ntok} tokens, {reqs[-1]['decode_tps']:.1f} tok/s")
    (OUT / f"{a.side}.json").write_text(json.dumps({"side": a.side, "label": SIDES[a.side], "requests": reqs}))


def charts(a):
    """Everything the comparison charts need, from recorded results only (summary.jsonl, raw rows, eval samples)."""
    import glob, statistics as st
    import rescore
    S = {}
    for l in open("results/summary.jsonl"):
        r = json.loads(l); S[r["label"]] = r  # last run of a label wins
    def row(lab):  # the label plus its "-r2" repeat if one exists, averaged (same rule as the ladder)
        rs = [S[l] for l in (lab, lab + "-r2") if l in S]
        if not rs:
            return None
        avg = lambda k: sum(r[k] for r in rs) / len(rs)
        return {"label": " + ".join(r["label"] for r in rs), "tpm": round(avg("tpm")), "p50": avg("decode_tps_p50"),
                "p5": avg("decode_tps_p5"), "ttft95": avg("ttft_p95"), "runs": len(rs), "pass": all(r["pass"] for r in rs)}
    bench = {"Chat": {"base": row(a.base_label), "best": row(a.best_label)},
             "GSM8K Platinum": {"base": row("base-gsm8k"), "best": row("best-gsm8k")},
             "IFEval": {"base": row("base-ifeval"), "best": row("best-ifeval")}}
    def dist(lab):
        f = sorted(glob.glob(f"results/raw/{lab}-*-c1-*.jsonl"))
        return [round(r["decode_tps"], 1) for r in (json.loads(l) for l in open(f[-1])) if r.get("decode_tps")] if f else []
    def acc(run):
        d = Path("results/eval") / run
        g = sorted(glob.glob(f"{d}/*/samples_gsm8k*_*.jsonl")); i = sorted(glob.glob(f"{d}/*/samples_ifeval_*.jsonl"))
        if not g or not i:
            return None
        rows = [json.loads(l) for l in open(g[-1]) if '"flexible-extract"' in l]
        gold = lambda r: rescore.number(r["doc"]["answer"].split("####")[-1])
        ok = sum(1 for r in rows if (v := rescore.final_answer(r["resps"][0][0] if r["resps"] else "")) is not None and abs(v - gold(r)) < 1e-6)
        ir = [json.loads(l) for l in open(i[-1])]
        return {"GSM8K Platinum": round(100 * ok / len(rows), 1), "IFEval": round(100 * sum(bool(r["prompt_level_strict_acc"]) for r in ir) / len(ir), 1)}
    def mean_tpm(labels):  # repeats are averaged; the range goes in the tooltip
        v = [S[l]["tpm"] for l in labels if l in S]
        return v and {"tpm": round(sum(v) / len(v)), "lo": min(v), "hi": max(v), "runs": len(v),
                      "p50": round(sum(S[l]["decode_tps_p50"] for l in labels if l in S) / len(v), 1)}
    # (step, runs, note): one change per step, in the order they were made; each gain is "at that point"
    ladder = [("Vanilla", ["q27-tp2-k0"], "vLLM 0.27.1 defaults; ran at 11-15% lower GPU clocks than the next steps"),
              ("+ MTP, 3 tokens", ["q27-tp2-k3"], "part of this gain is the clock difference above"),
              ("+ Triton, full graphs, BF16 KV", ["q27-k3-triton-full-bf16kv", "q27-best-k3"], "three settings changed together"),
              ("MTP 3 → 2", ["q27-best-k2"], "within run-to-run noise (~1%): a tie, kept for its better slow tail"),
              ("+ vLLM 0.32", ["v032-best-k2", "v032-best-k2-r2"], ""),
              ("MTP → DFlash2", ["v032-dflash2-k7", "v032-dflash2-k7-r2"], "also 2 → 7 draft tokens"),
              ("DFlash2 → W4A16 draft", ["v032-dflash2w4-k7", "v032-dflash2w4-k7-r2"], "")]
    lad = []
    for s, labs, note in ladder:
        m = mean_tpm(labs)
        if m:
            lad.append({"step": s, "labels": labs, "note": note, **m})
    def pairs(lab):  # [ttft, decode tok/s] per request
        f = sorted(glob.glob(f"results/raw/{lab}-chat-c1-*.jsonl"))
        return [[round(r["ttft"], 3), round(r["decode_tps"], 1)] for r in (json.loads(l) for l in open(f[-1])) if r.get("decode_tps")] if f else []
    sweep = lambda xs: [{"k": k, **mean_tpm(ls), "acc": S[ls[0]]["accept_rate"]} for k, ls in xs if mean_tpm(ls)]
    sweeps = {"mtp": {"title": "MTP head · vLLM 0.27.1, FlashInfer, piecewise graphs",
                      "pts": sweep([(0, ["q27-tp2-k0"]), (1, ["q27-tp2-k1"]), (2, ["q27-tp2-k2"]), (3, ["q27-tp2-k3"]), (4, ["q27-tp2-k4"])]), "chosen": None},
              "dflash": {"title": "DFlash2 draft model (BF16) · vLLM 0.32, Triton, full graphs",
                         "pts": sweep([(4, ["v032-dflash2-k4"]), (5, ["v032-dflash2-k5"]), (7, ["v032-dflash2-k7", "v032-dflash2-k7-r2"])]), "chosen": 7}}
    return {"bench": bench, "dist": {"base": dist(a.base_label), "best": dist(a.best_label)},
            "scatter": {"base": pairs(a.base_label), "best": pairs(a.best_label)},
            "acc": {"base": acc("base-acc"), "best": acc("v032-dflash2w4-k7")},
            "ladder": lad, "sweeps": sweeps}


def build(a):
    data = {s: json.loads((OUT / f"{s}.json").read_text()) for s in SIDES}
    S = {json.loads(l)["label"]: json.loads(l) for l in open("results/summary.jsonl")}
    data["charts"] = charts(a)
    for s in ("base", "best"):  # header line: the full 40-request benchmark, same numbers as the charts
        r = data["charts"]["bench"]["Chat"][s]
        data[s]["bench"] = {"label": r["label"], "tpm": r["tpm"], "tps_p50": r["p50"], "ttft_p95": r["ttft95"]}
        for q in data[s]["requests"]:  # if the server didn't stream running token counts, estimate them from text length
            ev = q["events"]
            if ev and any(e[3] is None for e in ev) and q["tokens"]:
                total = sum(len(e[2]) for e in ev); seen = 0
                for e in ev:
                    seen += len(e[2]); e[3] = e[3] or max(1, round(seen / total * q["tokens"]))
                q["tokens_estimated"] = True
    html = Path("race.template.html").read_text().replace("/*DATA*/null", json.dumps(data))
    Path("race.html").write_text(html)
    print("wrote race.html")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("capture"); c.add_argument("--side", choices=SIDES, required=True)
c.add_argument("--url", default="http://localhost:8001"); c.add_argument("--model", default="llm")
b = sub.add_parser("build")
b.add_argument("--base-label", default="q27-tp2-k0")  # the clean vanilla run; base-chat ran with OBS open (~6.5% slower)
b.add_argument("--best-label", default="v032-dflash2w4-k7")
a = p.parse_args()
capture(a) if a.cmd == "capture" else build(a)
