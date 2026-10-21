# What each setting contributed

The best run decodes at 89.2 tok/s per request and 5,294 TPM. The same model and engine with no tuning did 62.5 tok/s and 3,703 TPM. This file explains where the +43% came from, what limits the rest, and which ideas didn't pay off. The numbers are from [RUNS.md](RUNS.md).

## The frame: one request, so speed per step is everything

With one request at a time, TPM is decode speed × 60, minus about 1% for time-to-first-token and the gaps between requests. So the only way to raise TPM is to make each decode step faster or have it produce more tokens.

Decoding one request is **memory-bandwidth bound**: each step reads every weight once to produce one token. For this model on two 3090s, the bytes-per-token arithmetic in the sibling project `inference-explainer` puts the ceiling at 123.6 tok/s. Without drafting we measured **62.5 tok/s, about 51% of it**: 16.0 ms per step against a ~8.1 ms ideal.

Most of that gap is **communication between the two GPUs**. Tensor parallelism needs two all-reduces per layer, about 128 per step for 64 layers. Without P2P they go through host memory, and `inference-explainer` measured about 57 µs each for this message size (an upper bound, measured outside CUDA graphs). That's ~7.3 ms per step. Adding it to the ~8.1 ms ideal gives ~15.4 ms, close to the 16.0 ms measured. This is the biggest remaining cost, and no software setting here removes it. See [What limits further gains](#what-limits-further-gains).

## 1. Model and weights: Qwen3.8-27B, INT4 (W4A16)

- **Why this model:** the smartest one tested that fits the gate. Its model card beats the previous generation by wide margins (SWE-bench Pro 61.7 vs 53.5, LiveCodeBench 90.3 vs 83.9). The larger Qwen3.8-Flash-Next (125B MoE) is smarter at full precision, but on this machine it needs Strata at ~2 bits, which crashed the session on 48 GB of RAM and has no published quality numbers at that size.
- **Why INT4:** fewer bytes per step is directly more speed. `RedHatAI/Qwen3.8-27B-INT4` is about 19 GB, and RedHat measures 98.5-101% of BF16 accuracy (IFEval 91.93 vs 92.24, GSM8K Platinum 96.77 vs 95.75). vLLM runs it with Marlin kernels, the fast INT4 path on Ampere.
- **The output layer and the MTP head stay BF16** (the checkpoint's quantization skips them). That matters for drafting cost, below.

## 2. Tensor parallel across both cards (TP=2)

For one request, splitting the model across both GPUs doubles the memory bandwidth and the power budget behind each token. The price is the all-reduce traffic described above. Running a full copy on each card (data parallel) would only help with more than one request in flight, which is out of scope.

The EXL3 run shows why the second card matters under a 225 W cap: alone on one card, EXL3's decode was **power-bound**, with the GPU at 915 MHz and 67 tok/s. vLLM's TP=2 runs held 1,335-1,575 MHz median busy clocks, depending on card and run.

## 3. MTP speculative decoding: the biggest single gain

Qwen3.8-27B ships a multi-token-prediction (MTP) head: one extra layer that guesses the next tokens. Each step, vLLM runs the head *k* times to draft *k* tokens, then checks all of them with one pass of the full model. Accepted drafts are free tokens. A rejected draft costs nothing in correctness, because the full model's own token is used instead. At temperature 0 the output matches decoding without drafts.

What a step costs and yields, from the measured speed and acceptance (vLLM's default attention settings):

| Draft tokens | Tokens per step | ms per step | Step cost vs no drafting | tok/s |
|---|---|---|---|---|
| 0 | 1.00 | 16.0 | 1.00× | 62.5 |
| 1 | 1.77 | 23.5 | 1.47× | 75.5 |
| 2 | 2.28 | 28.0 | 1.75× | 81.4 |
| 3 | 2.59 | 30.8 | 1.93× | 84.1 |
| 4 | 2.84 | 35.4 | 2.21× | 80.2 |

Tokens per step = 1 + *k* × acceptance. Step time = tokens per step / tok/s. Both use per-run medians, so treat them as estimates.

- **Why it wins:** a step gets 47-93% more expensive, but it yields 77-159% more tokens.
- **Why it stops at 3-4:** each extra draft token costs roughly 3-5 ms. That's mostly the head's pass plus a read of the BF16 output layer (248,320 × 5,120, about 1.27 GB per GPU). Meanwhile acceptance per drafted token falls (0.77, 0.64, 0.53, 0.46), because each later guess builds on earlier ones. At 4, the cost outruns the extra tokens.
- **It depends on the text.** On chat prompts 52-64% of drafted tokens are accepted. During the accuracy runs (math and instruction-following) it was 76%, and decode ran at a median of ~99 tok/s. The reported number uses chat, the harder case.

## 4. Triton attention + full CUDA graphs + BF16 KV cache: +6.5%

- **CUDA graphs** record a whole decode step once and replay it with a single launch, instead of the CPU launching hundreds of kernels every step. That matters most at one request, where steps are short.
- vLLM's own pick, **FlashInfer**, can't capture full graphs together with speculative decoding. The log says `FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend; setting cudagraph_mode=PIECEWISE`, so attention ran outside the graphs.
- **Triton attention** can, so the decode step becomes one full graph (`Capturing CUDA graphs (decode, FULL)` in the log).
- **The cost:** Triton can't read an FP8 KV cache on a 3090 (it needs compute capability 8.9+), so the cache is BF16. At chat context lengths the KV cache is small next to the weights, and BF16 is the more precise format anyway.

Measured effect at the same draft count: step time fell from 28.0 to 25.5 ms (k=2) and from 30.8 to 28.6 ms (k=3), about 2.2-2.5 ms per step. That's 89.6 against 84.1 tok/s at k=3. The GPUs ran slightly *lower* clocks in the new runs, so the gain isn't clock. Three settings changed together, so the gain can't be split between them; removing per-step launch overhead is the likely main cause. The MTP draft layer itself uses FlashAttention 2 (vLLM's pick for it).

## 5. Two draft tokens instead of three

With full graphs, 2 and 3 draft tokens tie: 89.2 vs 89.7 tok/s, and 5,294 vs 5,278 TPM, within the ~1% run-to-run noise. Two has the **better slow tail**: 82.0 vs 77.1 tok/s at p5. On text where drafts are mostly rejected, a third draft token is pure cost. The gate is about the slow requests, so the claimed config uses 2.

## Settings that don't move the number

- **Async scheduling:** already on in every run. It's vLLM 0.27.1's default with MTP.
- **`--max-num-seqs 4`, `--max-model-len 16384`, `--max-num-batched-tokens 4096`:** sized for one client and short chat. Not measured one by one. At one request they mainly set memory reservations, and prompts here (up to ~1,500 tokens) prefill in one chunk.
- **Prefix caching:** off, as vLLM resolves it for this hybrid model with MTP. Chat prompts don't share prefixes, so it wouldn't help.
- **Thinking mode:** thinking tokens count toward TPM like any other. 32 of 40 requests used the full 1,024-token budget, and output totals were within 2% across runs, so TPM compares fairly between configs. One cost of thinking shows up in accuracy, not speed: on 6 of 100 IFEval prompts, all heavy on constraints, the model kept checking itself until the 32,768-token budget ran out and never answered. That's 5+ minutes of tokens with nothing to show for it. A smaller budget wouldn't fix it. A thinking-off mode or a fine-tune that thinks shorter would.

## TTFT

TTFT p95 is 0.58-0.60 s in every vLLM config, with p50 at 0.11-0.15 s. The tail comes from the longest prompts, which need the most prefill. MTP adds 30-40 ms at p95, since the draft head runs during prefill too. EXL3 on one power-capped card prefilled at roughly 700 tok/s and reached 0.96 s at p95, which is why it only just passed.

## What limits further gains

| Limit | Size | What would remove it |
|---|---|---|
| All-reduce through host memory (no P2P) | ~7 ms of the 25.5 ms step (estimate) | An NVLink bridge between the 3090s (stock drivers block GeForce P2P over PCIe) |
| Drafting one token at a time | ~3-5 ms per draft token | A block drafter such as DFlash2, which drafts 7-8 tokens in one pass (queued, vLLM 0.32) |
| 225 W cap | Bandwidth-bound decode loses little; compute-bound engines lose a lot (EXL3) | Out of scope by choice |
| One request at a time | TPM = one stream's speed | Concurrency (batching or two replicas), out of scope |

## What didn't pay off

- **EXL3 3.0 bpw on SGLang, one GPU:** 67.3 tok/s against the registry's ~135-139. Unpacking 3-bit EXL3 weights is compute-heavy, so on one 225 W card the clock fell to 915 MHz. The SGLang fork can't split EXL3 across GPUs.
- **4 draft tokens:** lower acceptance, higher cost; see section 3.
- **Triton with the FP8 KV cache:** not supported on sm_86; BF16 was required.
- **Strata / Flash-Next:** loading locks ~30 GB of RAM and triggered systemd-oomd on this 48 GB VM.
