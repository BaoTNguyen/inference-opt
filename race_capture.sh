#!/usr/bin/env bash
# Capture the race traces (baseline, then optimized) on the same 3 prompts, then build race.html.
set -u
cd "$(dirname "$0")"
gpu_free() { for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 4000 ] && return; sleep 3; done; }
cap() {  # cap SIDE [serve.sh env...]
  local side=$1; shift; gpu_free
  env "$@" setsid ./serve.sh > "results/server-race-$side.log" 2>&1 & local sp=$!
  until curl -sf localhost:8001/health >/dev/null; do kill -0 $sp 2>/dev/null || { echo "$side server died"; return 1; }; sleep 3; done
  uv run -q race.py capture --side "$side"; kill -- -$sp; wait $sp 2>/dev/null
}
cap base VLLM=vllm SPEC=mtp K=0 ATTN= CG= KV=
cap best
uv run -q race.py build
