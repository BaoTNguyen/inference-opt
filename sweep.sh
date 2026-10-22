#!/usr/bin/env bash
# Start the server(s) for one config, sweep concurrency, stop them, print the knee.
#   ./sweep.sh q27-k3                     defaults: Qwen3.8-27B, TP=2, MTP k=3, one client
#   K=2 ./sweep.sh q27-k2                 vary one knob; keep K in 1-3
#   DP=2 M=nemo TP=1 K=0 ./sweep.sh nemo-dp2   two independent replicas, one per card
# Other env: everything serve.sh takes, CS="1 4 8" (total concurrency, default 1), MAXTOK=1024 per request
set -u
cd "$(dirname "$0")"
L=$1; DP=${DP:-1}; CS=${CS:-1}; MAXTOK=${MAXTOK:-1024}; DATA=${DATA:-data/chat.jsonl}  # DATA=data/gsm8k40.jsonl etc.
# a server killed by the previous sweep takes a while to free its memory, so wait up to 90 s before giving up
for _ in $(seq 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 4000 ] && break
  sleep 3
done
[ "$used" -lt 4000 ] || { echo "a GPU has $used MiB in use: stop other GPU jobs first"; exit 1; }
mkdir -p results; pids=(); urls=()
trap 'for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null; done' EXIT
for ((i = 0; i < DP; i++)); do
  GPU=$i PORT=$((8001 + i)) setsid ./serve.sh >"results/server-$L-$i.log" 2>&1 &
  pids+=($!); urls+=("http://localhost:$((8001 + i))")
done
for u in "${urls[@]}"; do
  until curl -sf "$u/health" >/dev/null; do
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "server died, see results/server-$L-*.log"; exit 1; }; done
    sleep 3
  done
done
URLS=$(IFS=,; echo "${urls[*]}")
# 8 requests per client (40 minimum, so a single-client p95 is not just 1 sample) so the steady state outweighs ramp-up and the tail; cold cache so runs don't reuse each other's prompts
for c in $CS; do uv run -q bench.py "$DATA" --label "$L" -c "$c" -n $((c * 8 > 40 ? c * 8 : 40)) --max-tokens "$MAXTOK" --url "$URLS" --reset-cache >/dev/null; done
python3 report.py "$L"
