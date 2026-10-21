#!/usr/bin/env bash
# Accuracy gate against a running server (start it with serve.sh, same knobs as the sweep).
#   ./eval.sh q27-k3 [LIMIT]   -> results/eval/<label>/        CONC=4 sends 4 requests at once
# Thinking is on, so allow long generations; the reasoning parser keeps the thoughts out of the graded answer.
# CONC defaults to 1, the same one-at-a-time load as the benchmark: at 4 in flight, vLLM 0.27.1 with MTP crashed
# (cudaErrorIllegalAddress in FlashInfer metadata, 2026-10-21). gsm8k_cot_zeroshot_think is in tasks/ (no "Q:" stop).
L=$1; LIMIT=${2:-250}; URL=${URL:-http://localhost:8001}; CONC=${CONC:-1}; MODEL=${MODEL:-llm}
uvx --from 'lm-eval[api,ifeval]' lm_eval --model local-chat-completions --include_path tasks \
  --model_args "model=$MODEL,base_url=$URL/v1/chat/completions,num_concurrent=$CONC,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k_cot_zeroshot_think,ifeval --apply_chat_template --limit "$LIMIT" \
  --gen_kwargs "max_gen_toks=8192,temperature=0" --output_path "results/eval/$L" --log_samples
