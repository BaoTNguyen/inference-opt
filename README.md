# Qwen3.8-27B at 89 tok/s per request on 2x RTX 3090

The goal was the highest output tokens per minute (TPM) from the smartest model this machine can run, with every request meeting a latency gate: **TTFT p95 under 1 s and decode speed p50 of at least 60 tok/s**. Answers also have to stay correct, so every claimed setup is checked against a fixed accuracy suite. Scope: one request at a time, chat prompts, thinking on.

## Best result

| Model | Engine | Speculation | tok/s p50 | tok/s p5 | TTFT p50 | TTFT p95 | TPM | Errors | Gate |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B INT4 | vLLM 0.27.1, TP=2 | MTP, 2 draft tokens | **89.2** | 82.0 | 0.137 s | 0.586 s | **5,294** | 0 / 40 | PASS |

That's **+43% decode speed and TPM** over the same model and engine without any of the tuning: 62.5 tok/s and 3,703 TPM. A repeat of the closest configuration (MTP 3 draft tokens) came within 0.2% at p50 and 0.9% on TPM, so differences under about 1% are noise. MTP with 2 and 3 draft tokens tie on TPM; 2 has the steadier slow tail.

**Accuracy:** pending. The fixed suite (GSM8K Platinum + IFEval, 100 each) is running on this exact config. For reference, RedHat's model card for this INT4 checkpoint reports GSM8K Platinum 96.77% and IFEval 91.93%.

Where each gain came from is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md). Every run, including the ones that lost or failed, is in [RUNS.md](RUNS.md).

## Hardware

| Part | Detail |
|---|---|
| GPUs | 2x RTX 3090 24 GB, each power-capped to **225 W** (stock 350 W); every run hit the cap |
| GPU link | No NVLink, no P2P (`nvidia-smi topo -p2p r`: CNS). Cross-GPU traffic goes through host memory |
| PCIe | GPU0 4.0 **x4**, GPU1 4.0 x8 |
| CPU / RAM | Ryzen 7 9800X3D (12 vCPUs visible), 48 GiB RAM + 8 GiB swap |
| Platform | Ubuntu 24.04.5 guest on QEMU/KVM, both GPUs passed through; kernel 6.17, driver 590.44.01 (CUDA 13.1) |

## Engine and settings

vLLM 0.27.1 (PyTorch 2.12.0+cu130). The exact launch, as `./serve.sh` builds it by default (K=2 shown):

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ~/models/qwen3.8-27b/int4 \
  --tensor-parallel-size 2 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
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
| Speculation | The model's built-in MTP head, 2 draft tokens | The biggest single gain; 1-3 draft tokens all beat none, 4 lost |
| Attention / CUDA graphs / KV cache | Triton, full + piecewise graphs, BF16 | +6.5% over vLLM's own pick (FlashInfer, piecewise only, FP8 KV), which can't capture full graphs with speculation |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | torch sampler | The FlashInfer sampler fails to JIT-compile with the system nvcc 12.0 |

Resolved by vLLM itself (from the run's log): Marlin INT4 kernels, Triton attention for the main model and FlashAttention 2 for the MTP draft layer, full decode CUDA graphs captured, chunked prefill on, prefix caching off, async scheduling on (the 0.27.1 default with MTP), PyNCCL all-reduce (custom all-reduce off: no P2P), 201,523 tokens of KV capacity.

## Method

- **Speed:** `bench.py` sends the first 40 prompts of `data/chat.jsonl` (English WildChat first turns) one at a time, after 2 warmups. Temperature 0, up to 1,024 output tokens, natural stop, thinking on. TTFT is measured on the client, to the first streamed reasoning or answer token. Decode tok/s per request = (output tokens - 1) / time after the first token. TPM = total output tokens / wall time x 60. GPU clocks, power, temperature and throttle flags are logged every second.
- **Gate:** all requests succeed, TTFT p95 < 1.0 s, decode tok/s p50 >= 60. p5 is the slow tail for speed (95% of requests were faster), p95 the slow tail for TTFT.
- **Accuracy:** `eval.sh` (lm-eval 0.4.13): GSM8K Platinum zero-shot CoT (flexible-extract) and IFEval (prompt-level strict), the first 100 questions of each. 32,768-token budget including thinking, one request at a time. These are the benchmarks RedHat reports for this checkpoint; with 100 questions expect about ±3-4 points.

## Reproduce

```bash
uv run prep.py                        # data/chat.jsonl (1,000 WildChat prompts; the bench uses the first 40)
K=2 ./sweep.sh my-run                 # start the best config, bench it, stop it; row lands in results/summary.jsonl
LEN=36864 K=2 ./serve.sh &            # then:
./eval.sh my-run                      # accuracy suite -> results/eval/my-run/
python3 report.py                     # results table for every label
```

Live views for a demo: `uv run demo.py data/chat.jsonl --label demo -n 10 --max-tokens 1024` streams the run (thinking, answer and the gate metrics) at http://localhost:8090. `docker compose -f grafana/docker-compose.yaml up -d` gives a Grafana dashboard of vLLM's own metrics at http://localhost:3001.

## Status

- Accuracy suite for the best config: running.
- EXL3 on both GPUs through TabbyAPI: queued. EXL3 on one GPU lost (67 tok/s; the card is power-bound at 225 W).
- DFlash2 block drafting on vLLM 0.32: queued. The draft model's authors report +36% over MTP on chat.

| File | Purpose |
|---|---|
| `serve.sh`, `sweep.sh` | Start vLLM with every setting as an env var; run one benchmarked config |
| `bench.py`, `report.py` | Benchmark client and results table |
| `eval.sh`, `tasks/` | Fixed accuracy suite |
| `exl3.sh`, `tabby.sh`, `dflash.sh`, `strata.sh` | The other engines and drafters tried (see RUNS.md) |
| `demo.py`, `demo.html`, `grafana/` | Live demo page and dashboard |
| `results/summary.jsonl` | One line per benchmark run, the source of every number here |
