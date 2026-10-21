# What each setting contributed

The best run decodes at 121.6 tok/s per request and 6,967 TPM. The same model and engine with no tuning did 62.5 tok/s and 3,703 TPM. This file explains where the +88% came from, what limits the rest, and which ideas didn't pay off. The numbers are from [RUNS.md](RUNS.md).

| Step | tok/s p50 | TPM | Change in TPM |
|---|---|---|---|
| Untuned: vLLM 0.27.1, TP=2, no drafting | 62.5 | 3,703 | |
| + MTP drafting, 2-3 tokens | 81.4-84.1 | 4,725-4,968 | +28-34% |
| + Triton attention, full CUDA graphs, BF16 KV | 89.2 | 5,294 | +7% |
| + vLLM 0.32.0 | 92.2 | 5,470 | +3% |
| + DFlash2 block drafting instead of MTP | **121.6** | **6,967** | **+27%** |

## The frame: one request, so speed per step is everything

With one request at a time, TPM is decode speed × 60, minus about 1% for time-to-first-token and the gaps between requests. So the only way to raise TPM is to make each decode step faster or have it produce more tokens.

Decoding one request is **memory-bandwidth bound**: each step reads every weight once to produce one token. For this model on two 3090s, bytes-per-token arithmetic puts the ceiling at 123.6 tok/s. That figure, and the all-reduce timing below, come from `inference-explainer`, a separate (unpublished) study on this same machine; they're quoted here, not reproduced by this repo's scripts. Without drafting we measured **62.5 tok/s, about 51% of it**: 16.0 ms per step against a ~8.1 ms ideal.

Most of that gap is **communication between the two GPUs**. Tensor parallelism needs two all-reduces per layer, about 128 per step for 64 layers. Without P2P they go through host memory, and that study measured about 57 µs each for this message size (an upper bound, measured outside CUDA graphs). That's ~7.3 ms per step. Adding it to the ~8.1 ms ideal gives ~15.4 ms, close to the 16.0 ms measured. This is the biggest remaining cost, and no software setting here removes it. See [What limits further gains](#what-limits-further-gains).

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

With full graphs, 2 and 3 draft tokens tie: 89.2 vs 89.7 tok/s, and 5,294 vs 5,278 TPM, within the ~1% run-to-run noise. Two has the **better slow tail**: 82.0 vs 77.1 tok/s at p5. On text where drafts are mostly rejected, a third draft token is pure cost. The gate is about the slow requests, so the MTP configuration uses 2. The best result replaced MTP with DFlash2 (section 7).

## 6. vLLM 0.32: +3%

The same MTP configuration went from 89.2 to 92.2 tok/s (5,294 to 5,470 TPM) on vLLM 0.32.0, at slightly lower clocks. That's above the ~1% run-to-run noise. The gain likely comes from kernel and scheduler improvements between versions. It wasn't broken down further, because the main reason for upgrading was DFlash2.

## 7. DFlash2 block drafting: +27%, the second-biggest gain

MTP's limit (section 3) is that it drafts **one token per pass**, and each pass costs 3-5 ms, so drafting more than 2-3 tokens stops paying. DFlash2 is a separate 5-layer draft model (3.85 GB, BF16) trained for this exact model. It reads hidden states from five of the main model's layers and proposes a **block of 7 tokens in a single pass**. The full model then checks all 7 at once, as with MTP.

| Drafter (vLLM 0.32) | Accepted per drafted | Tokens per step | ms per step | tok/s |
|---|---|---|---|---|
| MTP, 2 tokens | 0.630 | 2.26 | 24.5 | 92.2 |
| DFlash2, 7 tokens | 0.335 | 3.35 | 27.5 | 121.6 |

DFlash2 guesses worse per token (a third of its drafts are accepted, against nearly two thirds for MTP), but it makes 7 guesses for about +3 ms of step time instead of 2. So each step yields **48% more tokens for 13% more time**. Because the full model verifies every token, output at temperature 0 is the same as without drafting. The accuracy suite agrees: GSM8K 99% and IFEval 91%, against 99% and 89% with MTP.

The draft model's memory leaves 93,762 tokens of KV cache, against 201,523 with MTP, still far more than a 16k chat needs. The DFlash2 run happened at 9-12% higher clocks than its baseline. Decode is bandwidth-bound, so that's a small share of the 27%, but it's worth a repeat at matched clocks before quoting the number to one decimal.

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
| All-reduce through host memory (no P2P) | ~7 ms of the 27.5 ms step (estimate) | An NVLink bridge between the 3090s (stock drivers block GeForce P2P over PCIe) |
| Draft acceptance | DFlash2 accepts 33.5% of its 7 drafts on chat (about 3.3 tokens per step) | Draft models trained on chat-like data, or tuning the block size (7 was the model card's default, not swept here) |
| 225 W cap | Bandwidth-bound decode loses little; compute-bound engines lose a lot (EXL3) | Out of scope by choice |
| One request at a time | TPM = one stream's speed | Concurrency (batching or two replicas), out of scope |

## What didn't pay off

- **EXL3 3.0 bpw on SGLang, one GPU:** 67.3 tok/s against the registry's ~135-139. Unpacking 3-bit EXL3 weights is compute-heavy, so on one 225 W card the clock fell to 915 MHz. The SGLang fork can't split EXL3 across GPUs.
- **4 MTP draft tokens:** lower acceptance, higher cost; see section 3.
- **EXL3 on two GPUs through TabbyAPI:** 72.6 tok/s, but TTFT p95 of 1.12 s failed the gate. The SGLang fork can't run EXL3 across GPUs at all.
- **Triton with the FP8 KV cache:** not supported on sm_86; BF16 was required.
- **Strata / Flash-Next:** loading locks ~30 GB of RAM and triggered systemd-oomd on this 48 GB VM.
