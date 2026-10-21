#!/usr/bin/env bash
# Accuracy suite for one config: start serve.sh with the caller's env (plus LEN for the 32k thinking budget),
# run eval.sh, re-score GSM8K by final answer, stop the server.
#   SPEC=dflash K=7 VLLM=~/venvs/vllm032/bin/vllm ./suite.sh v032-dflash2-k7
set -u
cd "$(dirname "$0")"
L=$1
for _ in $(seq 60); do  # wait for a previous server to release the GPUs
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 4000 ] && break
  sleep 3
done
LEN=${LEN:-36864} setsid ./serve.sh > "results/server-$L-suite.log" 2>&1 &
sp=$!
trap 'kill -- -"$sp" 2>/dev/null' EXIT
until curl -sf localhost:8001/health >/dev/null; do
  kill -0 "$sp" 2>/dev/null || { echo "server died, see results/server-$L-suite.log"; exit 1; }
  sleep 3
done
echo "$(date +%T) $L: fixed suite, one request at a time"
./eval.sh "$L" > "results/eval-$L.log" 2>&1
python3 rescore.py "results/eval/$L"
echo "$(date +%T) $L: done"
