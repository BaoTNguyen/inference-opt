# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Live demo page: bench.py's run, streamed to a browser as it happens.

uv run demo.py data/chat.jsonl --label demo-q27-k3 -n 10 --max-tokens 1024 [--url ... --model ...] [--port 8090]
Open http://localhost:8090; the run starts when the page connects (so a recording catches request 1).

It takes bench.py's flags and runs bench.py's own main(), so every number on the page comes from the same timing
code and the run lands in results/summary.jsonl like any other. Works with any OpenAI-compatible server.
"""
import asyncio, json, queue, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bench

PAGE = (Path(__file__).parent / "demo.html").read_bytes()
clients, lock = [], threading.Lock()
state = {"done": [], "current": None, "gpu": [], "args": {}, "stats": None}  # replayed to a page that connects mid-run
connected = threading.Event()


def send(ev):
    with lock:
        for q in clients:
            q.put(ev)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]  # ignore query strings (?shot=...)
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue()
            with lock:
                q.put({"type": "snapshot", **state})
                clients.append(q)
            connected.set()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(q.get())}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                with lock:
                    clients.remove(q)
        elif path == "/snapshot":  # current state as JSON (demo.html ?shot mode renders from it before load completes)
            body = json.dumps({"type": "snapshot", **state}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        else:
            self.send_error(404)


def gpu_loop():
    """1 Hz GPU line for the page (bench.py logs its own samples for the record)."""
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,clocks.sm",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            g = [dict(zip(("i", "temp", "power", "util", "sm"), (float(x) for x in l.split(",")))) for l in out.strip().splitlines()]
            state["gpu"] = g
            send({"type": "gpu", "gpu": g})
        except Exception:  # a missing or busy nvidia-smi must not stop the demo
            pass
        time.sleep(1)


def stats(rows, t0):
    """Running numbers, computed exactly as bench.py's summary computes them."""
    ok = [r for r in rows if "error" not in r]
    ttft, dtps = [r["ttft"] for r in ok], [r["decode_tps"] for r in ok if r["decode_tps"]]
    out_tok = sum(r["out_tokens"] for r in ok)
    s = {"n": len(rows), "n_err": len(rows) - len(ok), "ttft_p50": bench.pct(ttft, 50), "ttft_p95": bench.pct(ttft, 95),
         "decode_tps_p50": bench.pct(dtps, 50), "decode_tps_p5": bench.pct(dtps, 5),
         "tpm": round(out_tok / (time.perf_counter() - t0) * 60) if out_tok else 0}
    s["pass"] = bool(ok) and not s["n_err"] and s["ttft_p95"] < 1.0 and (s["decode_tps_p50"] or 0) >= 60
    return s


def main():
    p = bench.parser()
    p.add_argument("--port", type=int, default=8090)
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=gpu_loop, daemon=True).start()
    state["args"] = {"label": a.label, "model": a.model, "url": a.url, "n": a.n, "max_tokens": a.max_tokens}
    print(f"open http://localhost:{a.port} - the run starts when the page connects")
    connected.wait()
    t0 = None

    def on_start(r):
        nonlocal t0
        t0 = t0 or time.perf_counter()  # TPM clock starts at the first timed request, like bench.py's wall clock
        state["current"] = {"id": r["id"], "prompt": r["prompt"], "i": len(state["done"]) + 1, "thinking": "", "answer": "",
                            "t": time.perf_counter()}
        send({"type": "start", "current": state["current"]})

    def on_delta(kind, text):
        cur = state["current"]
        if cur is None:  # warmup requests run before the first on_start: not shown
            return
        cur[kind] += text
        send({"type": "delta", "kind": kind, "text": text, "t": time.perf_counter()})

    def on_done(rec, rows):
        state["done"].append(rec)
        state["stats"] = stats(rows, t0)
        send({"type": "done", "rec": rec, "stats": state["stats"]})
        state["current"] = None

    a.on_start, a.on_delta, a.on_done = on_start, on_delta, on_done
    asyncio.run(bench.main(a))
    send({"type": "finished"})
    print(f"finished; recorded in results/summary.jsonl as {a.label}. Ctrl-C to stop the page")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
