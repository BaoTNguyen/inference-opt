# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Closed-loop load generator for an OpenAI-compatible server.

uv run bench.py data/chat.jsonl -c 16 --label nemo-dp2 --url http://localhost:8001,http://localhost:8002 [--reset-cache]

Several --url values (comma-separated) are independent replicas; workers are spread round-robin over them.
Per-request rows go to results/raw/, one summary line per run to results/summary.jsonl.
SLO gate (pass): TTFT p95 < 1 s and per-request decode tok/s p50 >= 60. TPM = output tokens per minute.
Leave --ignore-eos off when measuring speculation: forced text past the natural end is
repetitive junk and inflates acceptance. Use it only for pure throughput comparisons.
"""
import argparse, asyncio, json, re, time
from datetime import datetime
from pathlib import Path

import httpx


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p / 100 * len(v)))], 4) if v else None


async def scrape(cl, url):
    """Prometheus counters worth diffing (vLLM: spec_decode / prefix_cache; llama.cpp: --metrics)."""
    try:
        text = (await cl.get(url + "/metrics")).text
    except httpx.HTTPError:
        return {}
    out = {}
    for m in re.finditer(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([-0-9.e+]+)$", text, re.M):
        if re.search(r"spec_decode|prefix_cache|tokens_total", m[1]) and not m[1].endswith("_created"):
            out[m[1] + (m[2] or "")] = float(m[3])
    return out


GPU_Q = "index,temperature.gpu,clocks.sm,power.draw,power.limit,utilization.gpu,memory.used,clocks_event_reasons.active,pcie.link.gen.current"
THROTTLE = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}


async def gpu_poll(samples, stop):
    """Sample nvidia-smi once a second while the timed section runs."""
    while not stop.is_set():
        try:
            p = await asyncio.create_subprocess_exec(
                "nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            for line in (await p.communicate())[0].decode().strip().splitlines():
                f = [x.strip() for x in line.split(",")]
                num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else 0.0
                samples.append({"i": int(f[0]), "temp": num(f[1]), "sm": num(f[2]), "power": num(f[3]), "limit": num(f[4]),
                                "util": num(f[5]), "mem": num(f[6]), "why": int(f[7], 16) if f[7].startswith("0x") else 0,
                                "gen": num(f[8])})
        except Exception:  # health logging must never break a benchmark run
            pass
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


def gpu_summary(samples):
    out = {}
    for i in sorted({s["i"] for s in samples}):
        g = [s for s in samples if s["i"] == i]
        busy = sorted(s["sm"] for s in g if s["util"] >= 50) or sorted(s["sm"] for s in g)
        why = 0
        for s in g:
            why |= s["why"]
        out[i] = {"samples": len(g), "temp_max": max(s["temp"] for s in g), "power_max_w": max(s["power"] for s in g),
                  "power_limit_w": g[0]["limit"], "util_mean": round(sum(s["util"] for s in g) / len(g), 1),
                  "sm_clock_median_busy": busy[len(busy) // 2], "sm_clock_min_busy": busy[0],
                  "mem_used_mib_min": min(s["mem"] for s in g), "mem_used_mib_max": max(s["mem"] for s in g),
                  "pcie_gen_max": max(s["gen"] for s in g),
                  "throttle": [n for b, n in THROTTLE.items() if why & b] or None}
    return out


async def one(cl, a, r, url):
    body = {"model": a.model, "messages": [{"role": "user", "content": r["prompt"]}],
            "max_tokens": a.max_tokens or r["max_tokens"], "temperature": a.temperature,
            "stream": True, "stream_options": {"include_usage": True}}
    if a.ignore_eos:
        body["ignore_eos"] = True
    if a.no_cache_prompt:  # llama.cpp slot prompt cache; vLLM uses --reset-cache instead
        body["cache_prompt"] = False
    rec = {"id": r["id"], "category": r.get("category"), "doc": r.get("doc")}
    t0 = time.perf_counter()
    first = usage = timings = None
    try:
        async with cl.stream("POST", url + "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                timings = d.get("timings") or timings
                ch = d.get("choices") or []
                delta = ch[0].get("delta", {}) if ch else {}
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.perf_counter()
    except Exception as e:  # keep going; errors are counted in the summary
        return {**rec, "error": repr(e)[:200]}
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens", 0)
    rec.update(ttft=(first or end) - t0, latency=end - t0, out_tokens=n,
               prompt_tokens=(usage or {}).get("prompt_tokens"),
               cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               decode_tps=(n - 1) / (end - first) if first and end > first and n > 1 else None)
    if timings:  # llama.cpp reports draft acceptance and prompt-cache reuse per request
        rec.update({k: timings[k] for k in ("draft_n", "draft_n_accepted", "cache_n") if k in timings})
    return rec


async def main(a):
    rows = [json.loads(l) for l in open(a.data)]
    if a.category:
        rows = [r for r in rows if r.get("category") == a.category]
    rows = rows[: a.n]
    urls = a.url.split(",")

    async def scrape_all(cl):  # counters summed over replicas
        out = {}
        for u in urls:
            for k, v in (await scrape(cl, u)).items():
                out[k] = out.get(k, 0) + v
        return out

    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=a.c + 4)) as cl:
        for u in urls:
            if a.reset_cache:
                print("reset_prefix_cache:", (await cl.post(u + "/reset_prefix_cache")).status_code)
            for _ in range(a.warmup):
                await one(cl, a, {"id": "warmup", "prompt": "Say hello in one sentence.", "max_tokens": 32}, u)
        m0 = await scrape_all(cl)
        it, res = iter(rows), []

        async def worker(url):
            for r in it:  # shared iterator: each worker pulls the next request
                res.append(await one(cl, a, r, url))

        gpu, stop = [], asyncio.Event()
        poller = asyncio.create_task(gpu_poll(gpu, stop))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(urls[i % len(urls)]) for i in range(a.c)])
        wall = time.perf_counter() - t0
        stop.set()
        await poller
        m1 = await scrape_all(cl)

    ok = [r for r in res if "error" not in r]
    dm = {k: round(m1[k] - m0.get(k, 0), 3) for k in m1 if m1[k] != m0.get(k, 0)}
    acc = sum(v for k, v in dm.items() if "spec_decode_num_accepted_tokens" in k and "per_pos" not in k)
    drf = sum(v for k, v in dm.items() if "spec_decode_num_draft_tokens" in k)
    if not drf:  # llama.cpp path
        acc, drf = sum(r.get("draft_n_accepted", 0) for r in ok), sum(r.get("draft_n", 0) for r in ok)
    out_tok = sum(r["out_tokens"] for r in ok)
    cached = [r["cached_tokens"] for r in ok if r.get("cached_tokens") is not None]
    ttft, dtps = [r["ttft"] for r in ok], [r["decode_tps"] for r in ok if r["decode_tps"]]
    summary = {
        "label": a.label, "data": Path(a.data).stem, "category": a.category, "concurrency": a.c, "temperature": a.temperature,
        "replicas": len(urls), "max_tokens": a.max_tokens,
        "ignore_eos": a.ignore_eos, "n_ok": len(ok), "n_err": len(res) - len(ok), "wall_s": round(wall, 2),
        "out_tokens": out_tok, "agg_tps": round(out_tok / wall, 2), "tpm": round(out_tok / wall * 60),
        "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95), "ttft_p99": pct(ttft, 99),
        "decode_tps_p50": pct(dtps, 50), "decode_tps_p5": pct(dtps, 5),
        "pass": bool(ok) and len(ok) == len(res) and pct(ttft, 95) < 1.0 and (pct(dtps, 50) or 0) >= 60,
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
        "cached_tokens": sum(cached) if cached else None,
        "accept_rate": round(acc / drf, 3) if drf else None,
        "gpu": gpu_summary(gpu),
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    stem = f"{a.label}-{summary['data']}-c{a.c}-{datetime.now():%H%M%S}"
    with open(f"results/raw/{stem}.jsonl", "w") as f:
        for r in res + [{"summary": summary, "metrics_delta": dm}]:
            f.write(json.dumps(r) + "\n")
    with open("results/summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))
    for r in res:
        if "error" in r:
            print("ERR", r["id"], r["error"])
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--label", required=True)
    p.add_argument("--url", default="http://localhost:8001", help="comma-separated for several replicas")
    p.add_argument("--model", default="llm")
    p.add_argument("-c", type=int, default=1, help="concurrent clients")
    p.add_argument("-n", type=int, default=10**9, help="limit number of requests")
    p.add_argument("--max-tokens", type=int, default=0, help="override per-request max_tokens")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--category", help="only requests with this category (Spec-Bench)")
    p.add_argument("--no-cache-prompt", action="store_true", help="send cache_prompt=false (llama.cpp)")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--reset-cache", action="store_true", help="POST /reset_prefix_cache first (vLLM dev mode)")
    asyncio.run(main(p.parse_args()))
