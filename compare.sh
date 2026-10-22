#!/usr/bin/env bash
# Baseline vs best on all three benchmarks: speed on chat, GSM8K Platinum and IFEval (40 prompts each), plus the
# baseline's accuracy suite (the best config's already exists: v032-dflash2w4-k7).
#   Baseline = README ladder row 0: vLLM 0.27.1, TP=2, INT4, no drafting, vLLM's own attention/graph/KV picks.
set -u
cd "$(dirname "$0")"
BASE=(VLLM=vllm SPEC=mtp K=0 ATTN= CG= KV=)  # empty ATTN/CG/KV = vLLM decides, as in the untuned run
run() { echo "$(date +%T) $1"; shift; env "$@" ./sweep.sh "$L" > "results/$L.log" 2>&1; grep -E "^\| $L " "results/$L.log"; }
L=base-chat;    run "baseline, chat"   "${BASE[@]}" DATA=data/chat.jsonl
L=base-gsm8k;   run "baseline, GSM8K"  "${BASE[@]}" DATA=data/gsm8k40.jsonl
L=base-ifeval;  run "baseline, IFEval" "${BASE[@]}" DATA=data/ifeval40.jsonl
L=best-gsm8k;   run "best, GSM8K"      DATA=data/gsm8k40.jsonl
L=best-ifeval;  run "best, IFEval"     DATA=data/ifeval40.jsonl
echo "$(date +%T) baseline accuracy suite"
env "${BASE[@]}" ./suite.sh base-acc
python3 report.py base-chat v032-dflash2w4-k7 v032-dflash2w4-k7-r2 base-gsm8k best-gsm8k base-ifeval best-ifeval
echo "$(date +%T) done"
