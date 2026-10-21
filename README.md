# Local LLM inference on 2x RTX 3090

The goal is the highest output tokens per minute (TPM) for the smartest model that runs locally, under a per-request gate: TTFT p95 < 1 s and decode speed p50 >= 60 tok/s. Accuracy is checked with an eval, not assumed. Current scope: one request at a time, chat workload (WildChat first turns), thinking on, up to 1024 output tokens per request.

## Best result

| Model | Engine | Parallelism | Speculation | tok/s p50 | tok/s p5 | TTFT p50 | TTFT p95 | TPM | Errors | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B INT4 | vLLM 0.27.1 | TP=2 | MTP k=3 | **84.1** | 72.6 | 0.145 s | 0.599 s | **4,968** | 0 / 40 | PASS |

Label `q27-tp2-k3` in `results/summary.jsonl`. 40 requests produced 35,613 output tokens in 430 s. Accuracy for this config is still pending (see [Pending](#pending)).

## Hardware

| Part | Detail |
|---|---|
| GPUs | 2x NVIDIA GeForce RTX 3090, 24 GB each, power-capped to 225 W (default 420 W) |
| GPU link | No NVLink. No P2P: `nvidia-smi topo -p2p r` reports CNS (chipset not supported), topology PHB |
| PCIe | GPU0 negotiated 4.0 x4, GPU1 4.0 x8 (both cards support x16) |
| CPU | AMD Ryzen 7 9800X3D, 12 vCPUs visible to the OS |
| RAM | 48 GiB, plus an 8 GiB swap file |
| Platform | The OS runs as a QEMU/KVM guest (`systemd-detect-virt`) with both GPUs passed through. This is the likely reason P2P is unavailable |
| Software | Ubuntu 24.04.5 LTS, Linux kernel 6.17.0-1033-oem, NVIDIA driver 590.44.01 (CUDA 13.1), PyTorch 2.12.0+cu130, system nvcc 12.0 |

Every benchmark run hit the 225 W cap (`sw_power_cap` throttle flag in all runs). GPU1 runs about 18-20 °C hotter than GPU0 (max 75-76 °C against 56-57 °C). No thermal throttling was recorded.

## Engines and settings

### vLLM 0.27.1 (every result so far)

Set in `serve.sh`:

| Setting | Value |
|---|---|
| Model | RedHatAI Qwen3.8-27B-INT4, local path `~/models/qwen3.8-27b/int4` |
| `--tensor-parallel-size` | 2 |
| `--speculative-config` | `{"method": "mtp", "num_speculative_tokens": K}`, K defaults to 3 |
| `--max-num-seqs` | 4 |
| `--max-num-batched-tokens` | 4096 |
| `--max-model-len` | 16384 |
| `--gpu-memory-utilization` | 0.86 |
| Chat template | `enable_thinking: true`, `--reasoning-parser qwen3`, `--language-model-only` (vision tower skipped) |
| Env | `VLLM_USE_FLASHINFER_SAMPLER=0` (the FlashInfer sampler JIT fails with the system nvcc 12.0) |

Resolved by vLLM itself, from `results/server-q27-tp2-k3-0.log`:

- Quantization: compressed-tensors, `MarlinLinearKernel` for the W4A16 layers.
- KV cache: FP8 (`float8_e4m3fn`), from the checkpoint's own KV scheme. 262,856 tokens of KV capacity.
- Attention: FlashInfer.
- CUDA graphs: PIECEWISE. vLLM logs: `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE`.
- Chunked prefill on, prefix caching off.
- All-reduce: PyNCCL. Custom all-reduce is disabled because the platform lacks P2P.

Other knobs, all env vars on `serve.sh`: `SEQS`, `BATCH`, `UTIL`, `LEN`, `K`, `THINK`, plus:

- `ATTN`: attention backend, e.g. `TRITON_ATTN`. Unset lets vLLM pick (FlashInfer here).
- `CG`: `cudagraph_mode`, one of `NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE`. Unset lets vLLM pick.
- `KV`: KV cache dtype, passed as `--kv-cache-dtype`. Unset keeps the checkpoint's FP8. Triton attention can't read FP8 KV on sm_86, so `ATTN=TRITON_ATTN` needs `KV=bfloat16`.

MTP k is kept to 1-3 for one-client tests. k=0 and k=4 both lost to k=3 (see [Other results](#other-results)).

### SGLang EXL3 fork (pending, not yet run)

`exl3.sh` runs the local-ai-registry recipe for Qwen3.8-27B on an RTX 3090: turboderp's EXL3 3.0 bpw weights, one GPU, NEXTN speculative decoding (3 steps, 4 draft tokens, top-k 1, the `qwen38_hot32k_v2` token map), FP8 KV, one running request. The launch arguments are the registry's, verbatim. Only the host port and GPU index are ours. The registry reports 138.8 tok/s on a rented 3090 and 135.0 tok/s on its owner's 3090, from its own speed test. Those are the registry's claims, not measurements from this repo.

### Strata (explored, shelved)

Strata serves Qwen3.8-Flash-Next at 1-3 bits, with experts split across RAM and VRAM. Its load locks about 30 GB of RAM. On this 48 GB machine that pushed memory pressure past systemd-oomd's limit, and oomd killed the terminal's whole cgroup. `strata.sh` now runs the server in its own systemd scope. It was deprioritized in favour of EXL3 and has no results.

## Benchmark method

`bench.py` is a closed-loop load generator for any OpenAI-compatible server.

- One client sends 40 requests back to back, after 2 warmup requests. The prefix cache is reset first, so runs don't reuse each other's prompts.
- Prompts: English WildChat first turns (`data/chat.jsonl`, built by `prep.py`). Temperature 0, up to 1024 output tokens, natural EOS.
- TTFT is measured on the client, from sending the request to the first streamed reasoning or content token.
- Decode tok/s per request = (output tokens - 1) / (time after the first token).
- TPM = total output tokens / wall time x 60. It includes TTFT and gaps between requests.
- GPU temperature, SM clock, power, utilization, memory, PCIe gen and throttle flags are polled from `nvidia-smi` every second.
- Draft acceptance (accepted per drafted token) comes from diffing vLLM's Prometheus counters before and after the run.

Pass rule: every request succeeds, TTFT p95 < 1.0 s, and decode tok/s p50 >= 60.

Percentiles are taken over the 40 requests. p50 is the median. For speed, p5 is the slow tail (95% of requests were faster). For TTFT, p95 is the slow tail (95% of requests started sooner).

## Reproduce

```bash
uv run prep.py                                    # data/chat.jsonl (1000 WildChat prompts)
# the sweep needs empty GPUs: stop llama-server.service / llama-swap first
./sweep.sh q27-k3                                 # start server, bench one client, stop; defaults = best config
K=2 ./sweep.sh q27-k2                             # vary one knob; keep K in 1-3
ATTN=TRITON_ATTN CG=FULL_AND_PIECEWISE KV=bfloat16 ./sweep.sh q27-k3-triton-full-bf16kv
./serve.sh & ./eval.sh q27-k3                     # accuracy: GSM8K CoT zero-shot + IFEval, 250 each -> results/eval/q27-k3/
./exl3.sh exl3-k3                                 # SGLang EXL3 recipe, one GPU (GPU=1 for the other card)
python3 report.py                                 # table of every config in results/summary.jsonl
docker compose -f grafana/docker-compose.yaml up -d   # live dashboard on http://localhost:3001
```

The dashboard's top row shows the gate (TTFT p95, decode tok/s per request, output TPM) in red or green against the thresholds, with vLLM's stock dashboard below. Those are server-side numbers from Prometheus. `bench.py`'s client-side numbers are the record.

## Other results

### MTP k sweep

Qwen3.8-27B INT4, vLLM 0.27.1, TP=2, one client, 40 requests each.

| MTP k | TTFT p50 | TTFT p95 | tok/s p50 | tok/s p5 | Accepted per drafted | TPM | Gate |
|---|---|---|---|---|---|---|---|
| 0 | 0.114 | 0.555 | 62.5 | 61.1 | - | 3,703 | PASS |
| 1 | 0.123 | 0.588 | 75.5 | 71.6 | 0.773 | 4,468 | PASS |
| 2 | 0.130 | 0.587 | 81.4 | 72.5 | 0.641 | 4,725 | PASS |
| **3** | 0.145 | 0.599 | **84.1** | 72.6 | 0.531 | **4,968** | PASS |
| 4 | 0.139 | 0.592 | 80.2 | 70.2 | 0.460 | 4,766 | PASS |

k=4 lost to k=3 on decode speed and TPM: acceptance drops faster than the extra draft token pays. The k=0 run had median busy SM clocks of 1365 / 1230 MHz (GPU0 / GPU1), against 1530-1575 / 1395-1455 MHz for k=1-4, about 11-15% lower. Part of the k=0 to k=1 gain may be clock. Decode at one client is bandwidth-bound, so probably a small part.

### Per-request spread, k=3

From `results/raw/q27-tp2-k3-chat-c1-113621.jsonl` (raw files are gitignored):

- TTFT: 0.078 s to 0.615 s. The longest TTFTs belong to the longest prompts (0.615 s at 750 prompt tokens, 0.599 s at 721), so the TTFT tail is prefill.
- Decode: 70.6 to 117.0 tok/s. The slowest requests have short prompts (66-117 tokens) and ran to the 1024-token limit, so the speed tail is not prefill. It is most likely draft acceptance, which depends on the text. Acceptance is not logged per request.
- 32 of 40 requests hit the 1024-token limit. Thinking uses most of the budget.

### Dashboard check

`bench.py`'s diff of vLLM's `generation_tokens_total` counter matched its own token count exactly in all five runs (k=3: 35,613 both). The Grafana/Prometheus dashboard matched `bench.py` within 0.1%: 35,382 against 35,412 at k=1, and 36,179 against 36,151 at k=4.

### Prior work

In the sibling project [inference-explainer](../inference-explainer), the same dense 27B on vLLM TP=2 fell to 31.4 tok/s per request at 16 clients without MTP, and 30.1 with MTP k=3. That misses the 60 tok/s gate by half, which is why concurrency is out of scope for now.

## Pending

- **Accuracy eval of the k=3 config.** The first attempt crashed before scoring (IFEval needed `langdetect`). `eval.sh` now installs `lm-eval[api,ifeval]`, and the eval is rerunning as `q27-k3` after a 2-question smoke test. Scores will land in `results/eval/q27-k3/`.
- **Triton attention with full CUDA graphs.** The first run (`q27-k3-triton-full`) failed at startup: `FP8 KV cache is not supported by the Triton attention backend on NVIDIA GeForce RTX 3090 (compute capability 8.6); native FP8 (fp8e4nv) requires SM89+. Re-run with --kv-cache-dtype bfloat16.` A retry with `KV=bfloat16` (`q27-k3-triton-full-bf16kv`) is queued after the eval. It changes three things at once (attention backend, CUDA graph mode, KV dtype), so any difference from the best result can't be pinned on one of them.
- **SGLang EXL3 run** (`exl3-k3`), queued behind the weight download and image pull.

## Files

| File | Purpose |
|---|---|
| `bench.py` | Closed-loop benchmark client; writes `results/raw/` and appends to `results/summary.jsonl` |
| `serve.sh` | Starts one vLLM server; every setting is an env var |
| `sweep.sh` | Starts the server(s) for one config, runs the bench, stops them, prints the result |
| `eval.sh` | GSM8K CoT + IFEval accuracy run against a live server (lm-eval) |
| `exl3.sh` | Runs the registry's SGLang EXL3 recipe in Docker, then the same bench |
| `strata.sh` | Runs Strata in its own systemd scope, then the same bench |
| `prep.py` | Builds `data/chat.jsonl` from WildChat (gitignored) |
| `report.py` | Prints the results table and the best passing run per label |
| `results/summary.jsonl` | One line per benchmark run, the source for every table here |
| `grafana/` | Prometheus + Grafana compose file, scrape config and dashboard |
