#!/usr/bin/env bash
# One vLLM server; every knob is an env var so sweep.sh can A/B them.
#   ./serve.sh                          the measured best: Qwen3.8-27B, TP=2, DFlash2 block drafting (7 tokens),
#                                       Triton + full CUDA graphs + BF16 KV, on vLLM 0.32 (6,967 TPM)
#   SPEC=mtp ./serve.sh                 the model's own MTP head instead (2 draft tokens; keep K in 1-3 for MTP)
#   ATTN= CG= KV= ./serve.sh            vLLM's own picks (FlashInfer, piecewise graphs, FP8 KV)
#   VLLM=vllm SPEC=mtp ./serve.sh       the system vLLM 0.27.1 (DFlash2 needs >= 0.32)
#   M=nemo TP=1 K=0 GPU=1 PORT=8002 ./serve.sh   one replica per card (sweep.sh DP=2); no MTP head in that checkpoint
# SEQS = max concurrent sequences, BATCH = chunked-prefill token budget per step (TTFT vs decode trade)
# ATTN = attention backend, CG = cudagraph_mode NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE (empty: vLLM picks)
# ponytail: speculation defaults are for one client only; MTP cost 16% of throughput at 16 clients (inference-explainer),
# and batched MTP crashed vLLM 0.27.1 (cudaErrorIllegalAddress). Re-measure before adding concurrency.
M=${M:-q27}; TP=${TP:-2}; GPU=${GPU:-0}; PORT=${PORT:-8001}; K=${K:-}; THINK=${THINK:-true}
SEQS=${SEQS:-4}; BATCH=${BATCH:-4096}; UTIL=${UTIL:-0.86}; LEN=${LEN:-16384}
case $M in
  # 17.8 GB, fits one card; 6 attention layers x 2 KV heads, so KV is tiny. Checkpoint has no MTP head.
  nemo) W=useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16; EXTRA=(--reasoning-parser nemotron_v3) ;;
  # 25 GB, needs TP=2; skip the vision tower
  qwen) W=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit; EXTRA=(--reasoning-parser qwen3 --language-model-only) ;;
  # smartest that fits: dense hybrid, 19 GB INT4 (ships an FP8 KV scheme), has MTP. Measured best (2026-10-21):
  # DFlash2 on vLLM 0.32 (6,967 TPM) > MTP k=2 on 0.32 (5,470) > MTP k=2 on 0.27.1 (5,294); Triton + full CUDA
  # graphs + BF16 KV beat vLLM's picks by 6.5%. Set any of these to empty to drop it.
  q27) W=$HOME/models/qwen3.8-27b/int4; EXTRA=(--reasoning-parser qwen3 --language-model-only)
       : "${ATTN=TRITON_ATTN}" "${CG=FULL_AND_PIECEWISE}" "${KV=bfloat16}" "${SPEC=dflash}"
       : "${DRAFT=$HOME/models/qwen3.8-27b/dflash2}" "${VLLM=$HOME/venvs/vllm032/bin/vllm}" ;;
  *) echo "unknown M=$M"; exit 1 ;;
esac
[ "$TP" -gt 1 ] && GPU=$(seq -s, 0 $((TP - 1)))
# SPEC=mtp (the model's own head, K=2 default) or SPEC=dflash with DRAFT=<draft model dir> (block drafter, K=7 default,
# z-lab DFlash2). DFlash2 needs vLLM >= 0.32 and JIT-compiles a top-k kernel on first start, which fails with the
# system nvcc 12.0, so it uses CUDA 13.1 when installed.
SPEC=${SPEC:-mtp}; DRAFT=${DRAFT:-}
[ -z "$K" ] && { [ "$SPEC" = dflash ] && K=7 || K=2; }
[ "$SPEC" = dflash ] && [ -d /usr/local/cuda-13.1 ] && export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
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
# FI_SAMPLER=1 turns the FlashInfer sampler back on (it needs CUDA 13.1 nvcc; the system nvcc 12.0 failed to JIT it); DEV_MODE exposes /reset_prefix_cache
CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=${FI_SAMPLER:-0} VLLM_SERVER_DEV_MODE=1 exec "${VLLM:-vllm}" serve "$W" --served-model-name llm \
  --tensor-parallel-size "$TP" --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH" --max-model-len "$LEN" \
  --gpu-memory-utilization "$UTIL" --default-chat-template-kwargs "{\"enable_thinking\": $THINK}" \
  --enable-prompt-tokens-details "${EXTRA[@]}" --port "$PORT"
