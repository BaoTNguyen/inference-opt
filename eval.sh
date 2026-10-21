#!/usr/bin/env bash
# Accuracy gate against a running server (start it with serve.sh, same knobs as the sweep).
#   ./eval.sh q27-best [LIMIT]   -> results/eval/<label>/        CONC=4 sends 4 requests at once
# Thinking is on, so allow long generations; the reasoning parser keeps the thoughts out of the graded answer.
# GEN = generation budget, thinking included: at 8192 the model ran out mid-reasoning (word-counting for a "300+ words"
# prompt) and returned an empty answer, so serve with LEN >= prompt + GEN (LEN=36864 for the default 32768, Qwen's suggested thinking budget).
# CONC defaults to 1, the same one-at-a-time load as the benchmark: at 4 in flight, vLLM 0.27.1 with MTP crashed
# (cudaErrorIllegalAddress in FlashInfer metadata, 2026-10-21).
# Fixed suite for every config: GSM8K Platinum (tasks/gsm8k_platinum_think.yaml, no "Q:" stop) + IFEval, 100 each,
# matching RedHat's model card. Compare prompt_level_strict_acc (IFEval) and flexible-extract (GSM8K) across configs.
L=$1; LIMIT=${2:-100}; URL=${URL:-http://localhost:8001}; CONC=${CONC:-1}; MODEL=${MODEL:-llm}; GEN=${GEN:-32768}; TASKS=${TASKS:-gsm8k_platinum_think,ifeval}
uvx --from 'lm-eval[api,ifeval]' lm_eval --model local-chat-completions --include_path tasks \
  --model_args "model=$MODEL,base_url=$URL/v1/chat/completions,num_concurrent=$CONC,max_retries=3,tokenized_requests=False" \
  --tasks "$TASKS" --apply_chat_template --limit "$LIMIT" \
  --gen_kwargs "max_gen_toks=$GEN,temperature=0" --output_path "results/eval/$L" --log_samples
