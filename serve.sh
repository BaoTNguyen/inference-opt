#!/usr/bin/env bash
# One vLLM server; every knob is an env var so sweep.sh can A/B them.
#   ./serve.sh                          the measured best: Qwen3.8-27B, TP=2, MTP k=3, Triton + full graphs + BF16 KV
#   ATTN= CG= KV= ./serve.sh            the earlier best: vLLM's picks (FlashInfer, piecewise graphs, FP8 KV)
#   K=1 ./serve.sh                      keep K in 1-3: each beat k=0 on every speed metric at one client (k=4 lost to k=3)
#   M=nemo TP=1 K=0 GPU=1 PORT=8002 ./serve.sh   one replica per card (sweep.sh DP=2); no MTP head in that checkpoint
#   ATTN=TRITON_ATTN CG=FULL_AND_PIECEWISE KV=bfloat16 ./serve.sh   full CUDA graphs with MTP (FlashInfer can't, so
#                                       vLLM drops to PIECEWISE; Triton can't read FP8 KV on a 3090, so BF16 KV)
# SEQS = max concurrent sequences, BATCH = chunked-prefill token budget per step (TTFT vs decode trade)
# ATTN = attention backend (unset: vLLM picks, FlashInfer here), CG = cudagraph_mode NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE (unset: vLLM picks)
# ponytail: K defaults to 3 for one client only; MTP cost 16% of throughput at 16 clients (inference-explainer), re-sweep K with concurrency
M=${M:-q27}; TP=${TP:-2}; GPU=${GPU:-0}; PORT=${PORT:-8001}; K=${K:-3}; THINK=${THINK:-true}
SEQS=${SEQS:-4}; BATCH=${BATCH:-4096}; UTIL=${UTIL:-0.86}; LEN=${LEN:-16384}
case $M in
  # 17.8 GB, fits one card; 6 attention layers x 2 KV heads, so KV is tiny. Checkpoint has no MTP head.
  nemo) W=useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16; EXTRA=(--reasoning-parser nemotron_v3) ;;
  # 25 GB, needs TP=2; skip the vision tower
  qwen) W=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit; EXTRA=(--reasoning-parser qwen3 --language-model-only) ;;
  # smartest that fits: dense hybrid, 19 GB INT4 (ships an FP8 KV scheme), has MTP. Already downloaded.
  # measured best (2026-10-21): Triton + full CUDA graphs + BF16 KV beat FlashInfer + piecewise + FP8 KV, 89.6 vs 84.1 tok/s.
  # Set ATTN= CG= KV= (empty) to get vLLM's own picks back.
  q27) W=$HOME/models/qwen3.8-27b/int4; EXTRA=(--reasoning-parser qwen3 --language-model-only)
       : "${ATTN=TRITON_ATTN}" "${CG=FULL_AND_PIECEWISE}" "${KV=bfloat16}" ;;
  *) echo "unknown M=$M"; exit 1 ;;
esac
[ "$TP" -gt 1 ] && GPU=$(seq -s, 0 $((TP - 1)))
# SPEC=mtp (default, the model's own head) or SPEC=dflash with DRAFT=<draft model dir> (block drafter, K tokens per
# pass; z-lab DFlash2 suggests K=7). DFlash2 needs vLLM >= 0.32 (VLLM=~/venvs/vllm032/bin/vllm).
SPEC=${SPEC:-mtp}; DRAFT=${DRAFT:-}
if [ "$K" -gt 0 ]; then
  if [ "$SPEC" = dflash ]; then
    [ -n "$DRAFT" ] || { echo "SPEC=dflash needs DRAFT=<dir>"; exit 1; }
    EXTRA+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K}")
  else
    EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K}")
  fi
fi
[ -n "${ATTN:-}" ] && EXTRA+=(--attention-backend "$ATTN")
[ -n "${CG:-}" ] && EXTRA+=(--compilation-config "{\"cudagraph_mode\":\"$CG\"}")
# KV = KV cache dtype (unset: the checkpoint's FP8). Triton attention can't read FP8 KV on sm_86, so it needs KV=bfloat16
[ -n "${KV:-}" ] && EXTRA+=(--kv-cache-dtype "$KV")
# ASYNC=0: --no-async-scheduling. vLLM 0.27.1 already turns async scheduling on by default with MTP (config/vllm.py),
# so every run here had it; this only measures what it is worth
[ "${ASYNC:-}" = 0 ] && EXTRA+=(--no-async-scheduling)
# FlashInfer sampler JIT fails with the system nvcc 12.0 (found in inference-explainer); DEV_MODE exposes /reset_prefix_cache
CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SERVER_DEV_MODE=1 exec "${VLLM:-vllm}" serve "$W" --served-model-name llm \
  --tensor-parallel-size "$TP" --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH" --max-model-len "$LEN" \
  --gpu-memory-utilization "$UTIL" --default-chat-template-kwargs "{\"enable_thinking\": $THINK}" \
  --enable-prompt-tokens-details "${EXTRA[@]}" --port "$PORT"
