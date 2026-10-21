#!/usr/bin/env bash
# Qwen3.8-27B EXL3 on TabbyAPI (ExLlamaV3's server), tensor-parallel across both cards, then the same bench as sweep.sh.
#   ./tabby.sh tabby-tp2           config: tabby/config.yml (the registry's 3090 Ti recipe, our lines marked)
#   BEST=5294 ./tabby.sh ...       run the accuracy suite only if TPM beats BEST (same rule as exl3.sh)
# The SGLang EXL3 fork is single-GPU only ("tensor-parallel EXL3 is not implemented"), so two cards need this engine.
# Needs native Docker (docker --context default, nvidia runtime).
set -u
cd "$(dirname "$0")"
L=$1; PORT=${PORT:-8001}; MAXTOK=${MAXTOK:-1024}; URL=http://localhost:$PORT
IMAGE=ghcr.io/0xsero/tabbyapi-exl3@sha256:0f83e6198dc3be2561652d8df2525a7d1a69733e12fc8e25cbc5bc319a8a3ad1
NAME=turboderp-Qwen3.8-27B-exl3-3.00bpw
W=$HOME/models/qwen3.8-27b/exl3-3.00bpw
DK="docker --context default"
[ -f "$W/.verified" ] || { echo "weights not verified in $W"; exit 1; }
for _ in $(seq 1200); do  # up to 2 h, like exl3.sh
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 4000 ] && break
  sleep 6
done
[ "$used" -lt 4000 ] || { echo "GPUs busy ($used MiB)"; exit 1; }
mkdir -p results
$DK rm -f tabby-bench >/dev/null 2>&1
$DK run -d --name tabby-bench --gpus all --shm-size 8g -p 127.0.0.1:$PORT:5000 \
  -v "$W:/workspace/models/$NAME:ro" -v "$PWD/tabby/config.yml:/app/config.yml:ro" -e NVIDIA_VISIBLE_DEVICES=all \
  --entrypoint /opt/venv/bin/python3 "$IMAGE" main.py --config /app/config.yml >/dev/null || exit 1
trap '$DK logs tabby-bench >"results/server-$L.log" 2>&1; $DK rm -f tabby-bench >/dev/null' EXIT
for _ in $(seq 200); do  # up to 10 min
  curl -sf $URL/health >/dev/null && break
  [ "$($DK inspect -f '{{.State.Running}}' tabby-bench 2>/dev/null)" = true ] || { echo "container exited, see results/server-$L.log"; exit 1; }
  sleep 3
done
curl -sf $URL/health >/dev/null || { echo "not healthy after 10 min"; exit 1; }
uv run -q bench.py data/chat.jsonl --label "$L" -c 1 -n 40 --max-tokens "$MAXTOK" --url $URL --model $NAME >/dev/null
python3 report.py "$L"
if tail -1 results/summary.jsonl | BEST=${BEST:-0} python3 -c "import json,os,sys; s=json.loads(sys.stdin.read()); sys.exit(0 if s['pass'] and s['tpm'] > int(os.environ['BEST']) else 1)"; then
  MODEL=$NAME URL=$URL ./eval.sh "$L" > "results/eval-$L.log" 2>&1
  grep -E "^\|" "results/eval-$L.log"
fi
