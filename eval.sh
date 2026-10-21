#!/usr/bin/env bash
# Accuracy gate against a running server (start it with serve.sh, same knobs as the sweep).
#   ./eval.sh nemo-dp2 [LIMIT]   -> results/eval/<label>/
# Thinking is on, so allow long generations; the reasoning parser keeps the thoughts out of the graded answer.
L=$1; LIMIT=${2:-250}; URL=${URL:-http://localhost:8001}
uvx --from 'lm-eval[api,ifeval]' lm_eval --model local-chat-completions \
  --model_args "model=llm,base_url=$URL/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k_cot_zeroshot,ifeval --apply_chat_template --limit "$LIMIT" \
  --gen_kwargs "max_gen_toks=8192,temperature=0" --output_path "results/eval/$L" --log_samples
