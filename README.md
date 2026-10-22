# Qwen3.8-27B at 122 tok/s per request on 2x RTX 3090

The goal was the highest output tokens per minute (TPM) from the smartest model this machine can run, with every request meeting a latency gate: **TTFT p95 under 1 s and decode speed p50 of at least 60 tok/s**. Answers also have to stay correct, so every claimed setup is checked against a fixed accuracy suite. Scope: one request at a time, chat prompts, thinking on.

## Best result

| Model | Engine | Speculation | tok/s p50 | tok/s p5 | TTFT p50 | TTFT p95 | TPM | Errors | Gate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B INT4 | vLLM 0.32.0, TP=2 | DFlash2 draft model, 7 tokens per block | **121.6** | 96.4 | 0.128 s | 0.579 s | **6,967** | 0 / 40 | PASS |

That's **+88% TPM** over the same model and engine with no tuning (62.5 tok/s, 3,703 TPM). A back-to-back repeat gave 117.5 tok/s and 6,793 TPM. Across both pairs of runs, DFlash2 beats the model's built-in MTP drafting head on the same vLLM by **24-27%** (MTP: 5,470 and 5,483 TPM). DFlash2 varies more between runs (2.5%) than MTP (under 1%).

**Faster, accuracy not yet validated:** swapping in a 4-bit draft model (`syvai/Qwen3.8-27B-DFlash2-W4A16`, 1.28 GB instead of 3.85 GB) reached **125.3 tok/s p50 and 7,151-7,158 TPM** in two runs (TTFT p95 0.57 s, 40/40), about **+4%**. It changes nothing in the main model, but speculative decoding isn't bit-exact (output token totals shifted by under 1%), so it needs its own accuracy suite before it replaces the headline. Run it with `DRAFT=~/models/qwen3.8-27b/dflash2-w4a16 ./serve.sh`.

**Accuracy** on this exact config (fixed suite, first 100 questions each, one request at a time):

| Benchmark | Score | MTP config (vLLM 0.27.1) | RedHat, same INT4 checkpoint |
|---|---|---|---|
| GSM8K Platinum | **99%** (99/100) | 99% | 96.77% |
| IFEval, prompt-level strict | **91%** (91/100) | 89% | 91.93% |

- **GSM8K:** scored by the final stated answer (`rescore.py`). lm-eval's flexible-extract reports 92%, because it takes the last number in the response: in "**160 minutes**, or 2 hours and 40 minutes" it reads 40. The one real miss is the same question in both configs.
- **IFEval:** 2 of the 9 misses are empty answers, where thinking used the whole 32,768-token budget on a constraint-heavy prompt. 91 of the 98 prompts it answered pass.
- Speculative decoding is lossless: the full model checks every drafted token. The suite confirms the faster drafter didn't cost accuracy.

Where each gain came from is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md). Every run, including the ones that lost or failed, is in [RUNS.md](RUNS.md).

## Hardware

| Part | Detail |
|---|---|
| GPUs | 2x RTX 3090 24 GB, each power-capped to **225 W** (stock 350 W); every run hit the cap |
| GPU link | No NVLink, no P2P (`nvidia-smi topo -p2p r`: CNS). Cross-GPU traffic goes through host memory |
| PCIe | GPU0 4.0 **x4**, GPU1 4.0 x8 |
| CPU / RAM | Ryzen 7 9800X3D (12 vCPUs visible), 48 GiB RAM + 8 GiB swap |
| Platform | Ubuntu 24.04.5 guest on QEMU/KVM, both GPUs passed through; kernel 6.17, driver 590.44.01, CUDA 13.1 toolkit at `/usr/local/cuda-13.1` (system `nvcc` is 12.0) |

## Engine and settings

vLLM 0.32.0 (PyTorch 2.13.0+cu130) in its own environment, `~/venvs/vllm032`. `./serve.sh` with no arguments builds exactly this launch:

```bash
CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm032/bin/vllm serve ~/models/qwen3.8-27b/int4 \
  --tensor-parallel-size 2 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$HOME/models/qwen3.8-27b/dflash2\",\"num_speculative_tokens\":7}" \
  --attention-backend TRITON_ATTN \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-cache-dtype bfloat16 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 --max-model-len 16384 --gpu-memory-utilization 0.86 \
  --default-chat-template-kwargs '{"enable_thinking": true}' --reasoning-parser qwen3 \
  --language-model-only --enable-prompt-tokens-details --served-model-name llm
```

| Setting | Value | Why |
|---|---|---|
| Weights | `RedHatAI/Qwen3.8-27B-INT4` (W4A16, Marlin kernels) | ~19 GB; RedHat measures 98.5-101% of BF16 accuracy |
| Parallelism | Tensor parallel across both cards | One request gets both cards' memory bandwidth and power budget |
| Speculation | `z-lab/Qwen3.8-27B-DFlash2` (5-layer, 3.85 GB draft model, revision `50307d4`), 7 tokens per block | Drafts a whole block in one pass: +24-27% TPM over the model's own MTP head on the same vLLM |
| vLLM version | 0.32.0 | DFlash2 support (merged Aug 2026); also +3% over 0.27.1 for the same MTP config |
| Attention / CUDA graphs / KV cache | Triton, full + piecewise graphs, BF16 | +6.5% over vLLM's own pick (FlashInfer, piecewise only, FP8 KV), which can't capture full graphs with speculation |
| `CUDA_HOME` = CUDA 13.1 | for FlashInfer's JIT compiler | DFlash2's top-k kernel compiles on first start and fails with the system nvcc 12.0 |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | torch sampler | Kept from the 0.27.1 runs, where the FlashInfer sampler failed to compile with nvcc 12.0 |

Resolved by vLLM itself (from the run's log): Marlin INT4 kernels, Triton attention for the main model and FlashAttention 2 for the draft model, full CUDA graphs captured, chunked prefill on, prefix caching off, PyNCCL all-reduce (custom all-reduce off: no P2P), 93,762 tokens of KV capacity (the draft model takes the rest).

## Method

- **Speed:** `bench.py` sends the first 40 prompts of `data/chat.jsonl` (English WildChat first turns) one at a time, after 2 warmups. Temperature 0, up to 1,024 output tokens, natural stop, thinking on. TTFT is measured on the client, to the first streamed reasoning or answer token. Decode tok/s per request = (output tokens - 1) / time after the first token. TPM = total output tokens / wall time x 60. GPU clocks, power, temperature and throttle flags are logged every second.
- **Gate:** all requests succeed, TTFT p95 < 1.0 s, decode tok/s p50 >= 60. p5 is the slow tail for speed (95% of requests were faster), p95 the slow tail for TTFT.
- **Accuracy:** `suite.sh` runs `eval.sh` (lm-eval 0.4.13) then `rescore.py`: GSM8K Platinum zero-shot CoT, scored by final answer, and IFEval prompt-level strict, the first 100 questions of each. 32,768-token budget including thinking, one request at a time. These are the benchmarks RedHat reports for this checkpoint; with 100 questions expect about ±3-4 points.

## Reproduce

```bash
uv run prep.py                        # data/chat.jsonl (1,000 WildChat prompts; the bench uses the first 40)
./sweep.sh my-run                     # start the best config, bench it, stop it; row lands in results/summary.jsonl
./suite.sh my-run                     # accuracy suite on the same config -> results/eval/my-run/
SPEC=mtp ./sweep.sh my-mtp-run        # the MTP-head alternative (5,470 TPM on vLLM 0.32)
python3 report.py                     # results table for every label
```

Needs `~/venvs/vllm032` (`uv venv ~/venvs/vllm032 --python 3.12 && uv pip install --python ~/venvs/vllm032/bin/python vllm==0.32.0`) and the draft model in `~/models/qwen3.8-27b/dflash2` (`z-lab/Qwen3.8-27B-DFlash2` at revision `50307d4`).

Live demo: `uv run prep.py --demo` builds `data/demo.jsonl`, 20 prompts spread evenly over the three benchmarks used (7 chat, 7 GSM8K Platinum, 6 IFEval, interleaved). Then start `./serve.sh` and `uv run demo.py data/demo.jsonl --label demo -n 20 --max-tokens 1024`, open http://localhost:8090 and press **Run**. Each press records a new run (`demo-1`, `demo-2`, …) while the page streams the thinking, the answer and the gate metrics. With 20 requests, p5 is the second-slowest request: exactly one is slower. Math and instruction prompts draft more easily than chat, so the demo's speed runs higher than the chat-only table above and isn't directly comparable. Streaming each chunk to the page also costs about 4%. `docker compose -f grafana/docker-compose.yaml up -d` gives a Grafana dashboard of vLLM's own metrics at http://localhost:3001. Its TPM panel averages over a minute, so it reads low for the first ~60 s of a run.

| File | Purpose |
|---|---|
| `serve.sh`, `sweep.sh` | Start vLLM with every setting as an env var; run one benchmarked config |
| `bench.py`, `report.py` | Benchmark client and results table |
| `suite.sh`, `eval.sh`, `tasks/`, `rescore.py` | Fixed accuracy suite and final-answer GSM8K scoring |
| `exl3.sh`, `tabby.sh`, `dflash.sh`, `strata.sh` | The other engines and drafters tried (see RUNS.md) |
| `demo.py`, `demo.html`, `grafana/` | Live demo page and dashboard |
| `results/summary.jsonl` | One line per benchmark run, the source of every number here |
