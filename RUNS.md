# Runs

Every benchmark, eval and failed attempt from 2026-10-21, in order. Speed numbers come from `results/summary.jsonl`; the label in each table is the row's `label` there. Hardware, method and the gate are in the [README](README.md). Explanations of why each change helped are in [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

All speed runs used the same workload: the first 40 prompts of `data/chat.jsonl`, one client, 2 warmups, temperature 0, up to 1,024 output tokens, thinking on, 225 W power cap on both cards. Columns: TTFT in seconds; tok/s is decode speed per request; "acc." is MTP accepted tokens per drafted token, from vLLM's counters; clocks are median busy SM MHz for GPU0/GPU1.

## Speed runs

### 1. MTP draft-token sweep (vLLM's own attention and graph picks)

Qwen3.8-27B INT4, vLLM 0.27.1, TP=2, FlashInfer attention, piecewise CUDA graphs, FP8 KV cache (the checkpoint's scheme).

| Label | Draft tokens | TTFT p50 | TTFT p95 | tok/s p50 | tok/s p5 | acc. | TPM | Clocks | Gate |
|---|---|---|---|---|---|---|---|---|---|
| `q27-tp2-k0` | 0 (none) | 0.114 | 0.555 | 62.5 | 61.1 | - | 3,703 | 1365/1230 | PASS |
| `q27-tp2-k1` | 1 | 0.123 | 0.588 | 75.5 | 71.6 | 0.773 | 4,468 | 1575/1455 | PASS |
| `q27-tp2-k2` | 2 | 0.130 | 0.587 | 81.4 | 72.5 | 0.641 | 4,725 | 1575/1455 | PASS |
| `q27-tp2-k3` | 3 | 0.145 | 0.599 | 84.1 | 72.6 | 0.531 | 4,968 | 1530/1395 | PASS |
| `q27-tp2-k4` | 4 | 0.139 | 0.592 | 80.2 | 70.2 | 0.460 | 4,766 | 1530/1395 | PASS |

The k=0 run had 11-15% lower clocks than the others, so part of the jump to k=1 may be clock. From here on, tests use 1-3 draft tokens only.

### 2. Attention backend and CUDA graphs (MTP, 3 draft tokens)

| Label | Attention / graphs / KV | TTFT p95 | tok/s p50 | tok/s p5 | acc. | TPM | Clocks | Gate |
|---|---|---|---|---|---|---|---|---|
| `q27-tp2-k3` | FlashInfer / piecewise / FP8 | 0.599 | 84.1 | 72.6 | 0.531 | 4,968 | 1530/1395 | PASS |
| `q27-k3-triton-full` | Triton / full + piecewise / FP8 | - | - | - | - | - | - | failed to start |
| `q27-k3-triton-full-bf16kv` | Triton / full + piecewise / BF16 | 0.584 | 89.6 | 76.4 | 0.523 | 5,229 | 1470/1350 | PASS |

The FP8 attempt failed at startup: `FP8 KV cache is not supported by the Triton attention backend on NVIDIA GeForce RTX 3090 (compute capability 8.6); native FP8 (fp8e4nv) requires SM89+`.

### 3. Draft-token re-sweep on the new settings, plus a repeat

Triton, full + piecewise graphs, BF16 KV (the `serve.sh` default for this model from here on).

| Label | Draft tokens | TTFT p50 | TTFT p95 | tok/s p50 | tok/s p5 | acc. | TPM | Clocks | Gate |
|---|---|---|---|---|---|---|---|---|---|
| `q27-k3-triton-full-bf16kv` | 3 | 0.145 | 0.584 | 89.6 | 76.4 | 0.523 | 5,229 | 1470/1350 | PASS |
| `q27-best-k3` (repeat) | 3 | 0.144 | 0.576 | 89.7 | 77.1 | 0.523 | 5,278 | 1455/1335 | PASS |
| **`q27-best-k2`** (best on 0.27.1) | **2** | 0.137 | 0.586 | 89.2 | **82.0** | 0.635 | **5,294** | 1485/1365 | PASS |

Run-to-run variation, from the repeat: 0.2% at p50, 0.9% on TPM.

### 4. Other engines

| Label | Engine | GPUs | TTFT p95 | tok/s p50 | tok/s p5 | TPM | Clocks | Gate |
|---|---|---|---|---|---|---|---|---|
| `exl3-k3` | SGLang EXL3 fork, Qwen3.8-27B EXL3 3.0 bpw, NEXTN 3 steps / 4 draft tokens (local-ai-registry RTX 3090 recipe, verbatim) | 1 | 0.963 | 67.3 | 53.1 | 3,944 | 915/idle | PASS, barely |
| `exl3-tp2-k3` | Same, `--tp-size 2` | 2 | - | - | - | - | - | failed to start |
| `tabby-tp2-k3` | TabbyAPI (ExLlamaV3), same EXL3 weights, tensor parallel, MTP | 2 | 1.118 | 72.6 | 59.2 | 4,110 | 1515/1440 | **FAIL** (TTFT) |

`exl3-k3`: the server logged 2.42 accepted tokens per verify step (accept rate 0.47), and GPU0 sat at the 225 W cap at 915 MHz. The registry reports 138.8 / 135.0 tok/s for this recipe on its own speed test, presumably at stock power. `exl3-tp2-k3` failed with `NotImplementedError: tensor-parallel EXL3 is not implemented (single GPU only)`.

### 5. vLLM 0.32.0 and DFlash2 block drafting

Same settings as section 3 (Triton, full + piecewise graphs, BF16 KV), on vLLM 0.32.0 in `~/venvs/vllm032`. DFlash2 is `z-lab/Qwen3.8-27B-DFlash2` (revision `50307d4`), a 5-layer draft model that proposes a block of tokens in one pass. "acc." for DFlash2 is accepted per drafted token out of 7.

| Label | Speculation | TTFT p50 | TTFT p95 | tok/s p50 | tok/s p5 | acc. | TPM | Clocks | Gate |
|---|---|---|---|---|---|---|---|---|---|
| `v032-best-k2` | MTP, 2 draft tokens | 0.125 | 0.571 | 92.2 | 85.1 | 0.630 | 5,470 | 1440/1305 | PASS |
| **`v032-dflash2-k7`** (best) | **DFlash2, 7 per block** | 0.128 | 0.579 | **121.6** | **96.4** | 0.335 | **6,967** | 1575/1470 | PASS |

MTP on 0.32 beat the same config on 0.27.1 by 3.3% at slightly lower clocks. The DFlash2 run's clocks were 9-12% higher than the 0.32 baseline's. Decode is bandwidth-bound, so that's a small share of the 27% gap, but a repeat at matched clocks would tighten it. The first DFlash2 start failed: FlashInfer JIT-compiles a top-k kernel for DFlash2's candidate selector and picked up the system `nvcc` 12.0 (`Ninja build failed`). Pointing `CUDA_HOME` at the installed CUDA 13.1 fixed it.

## Accuracy runs

Fixed suite: GSM8K Platinum (`tasks/gsm8k_platinum_think.yaml`, re-scored by final answer with `rescore.py`) + IFEval (prompt-level strict), first 100 each, 32,768-token budget, one request at a time.

| Label | Config | GSM8K Platinum | IFEval | Status |
|---|---|---|---|---|
| `q27-best-k2` | MTP 2 draft tokens, vLLM 0.27.1 | **99%** final answer (lm-eval flexible-extract: 91%) | **89%** (6 empty answers: thinking hit 32k) | done, 14:37-15:50 |
| **`v032-dflash2-k7`** | **DFlash2, vLLM 0.32 (best)** | **99%** final answer (flexible-extract: 92%) | **91%** (2 empty answers) | done, 16:26-16:54 |
| reference | RedHat model card, INT4, full test sets | 96.77% | 91.93% | published |

The 9 flexible-extract misses on GSM8K: 8 were correct answers that the last-number rule misread ("**$75.00** on yogurt over 30 days" read as 30; "$26.00" not matched to 26), and 1 was wrong (40% for 60%). IFEval's 11 misses: 6 empty answers on constraint-heavy prompts (forbidden words, lowercase, JSON, quoting, Marathi), 3 letter-frequency counts, 1 highlighted-sections count, 1 repeat-the-prompt with no commas.

Earlier attempts, none of which produced scores:

| Attempt | What happened | Fix |
|---|---|---|
| 1 | IFEval import error: `No module named 'langdetect'` | `lm-eval[api,ifeval]` |
| 2 | GSM8K's stock stop sequence `"Q:"` fired inside the model's thinking, cutting answers off. At 4 requests in flight, vLLM crashed: `cudaErrorIllegalAddress` on GPU1 in FlashInfer attention metadata under MTP (kernel Xid 31) | Task copy without stop sequences; evals run one request at a time |
| 3 | Smoke test: one IFEval prompt ("300+ word summary", no commas) spent the whole 8,192-token budget counting words in its reasoning; the answer came back empty | Budget raised to 32,768 (Qwen's suggested thinking budget), server context 36,864 |
| 4 | Crashed at 157/200: one prompt thought for longer than lm-eval's 300 s request timeout | `timeout=3600` |

## Incidents

| Time | What happened | Cause | Fix |
|---|---|---|---|
| 11:07 | k=1-4 sweep runs aborted at start | `sweep.sh` checked GPU memory before the previous server had released it | Wait up to 90 s for release |
| 12:31 | systemd-oomd killed the terminal and everything in it (19 processes) | Strata locking ~30 GB of RAM pushed memory pressure to 63% (> 50% for 20 s) | Long jobs run as systemd user units; Strata shelved |
| 12:40-13:45 | EXL3 weight download stalled for ~50 min while its process stayed alive | `hf download` hang | Resumed with `aria2c --lowest-speed-limit`, SHA-256 checked |
| 12:52 | vLLM crash during an eval (attempt 2 above) | Batched MTP path in vLLM 0.27.1 (`cudaErrorIllegalAddress`) | One request at a time; batched runs untested since |
| 16:11 | DFlash2 server failed to start, twice | FlashInfer JIT compile of a top-k kernel used the system nvcc 12.0 | `CUDA_HOME=/usr/local/cuda-13.1` (`serve.sh` sets it for `SPEC=dflash`) |
| 16:22 | DFlash2 accuracy suite didn't start (`env: '': No such file or directory`) | systemd expanded `${...}` inside an inline `bash -c` script | Suite steps moved to `suite.sh`, run as a file |

## Considered and not run



| Option | Why not |
|---|---|
| Qwen3.8-Flash-Next on Strata (Coder IQ1_M, IQ2_XS) | Smarter at full precision, but loading crashed the session through systemd-oomd on 48 GB of RAM; the 2-bit quality is unmeasured. IQ2_XS is 18.6 of 39.2 GB downloaded, paused |
| GLM-5.3-Flash / DeepSeek-V4.1-Flash on project-maya | Comparable 2x3090 rigs report 44-53 tok/s, under the gate; the user chose to skip |
| Google Colab (free T4, v5e-1 TPU) | 16 GB memory can't hold the 27B or the MoE candidates; lower bandwidth than one 3090 |
| 350 W power limit | User rule: keep the 225 W cap |
| Concurrency (batching, two replicas) | Out of scope for now: one client only. Also MTP lost 16% aggregate throughput at 16 clients in an earlier, unpublished study on this machine (`inference-explainer`) |
| `--async-scheduling` | Already on by default in vLLM 0.27.1 with MTP (`config/vllm.py`); nothing to test |
