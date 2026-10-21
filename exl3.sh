#!/usr/bin/env bash
# The local-ai-registry's RTX 3090 recipe for Qwen3.8-27B (EXL3 3.0 bpw, SGLang EXL3 fork, NEXTN MTP, one GPU),
# launched exactly as the registry pins it, then the same one-client bench as sweep.sh.
#   ./exl3.sh exl3-k3              GPU=1 picks the other card (GPU 0 ran ~20 C cooler in the vLLM sweep)
#   TP=2 ./exl3.sh exl3-tp2-k3     both cards (--tp-size 2). At one GPU and the 225 W cap the card ran at ~915 MHz:
#                                  EXL3 decode is compute/power-bound there, so a second card adds power, not just bandwidth
# Needs native Docker (docker --context default, which has the nvidia runtime); Docker Desktop can't pass GPUs through.
set -u
cd "$(dirname "$0")"
L=$1; TP=${TP:-1}; GPU=${GPU:-0}; [ "$TP" -gt 1 ] && GPU=$(seq -s, 0 $((TP - 1))); PORT=${PORT:-8001}; MAXTOK=${MAXTOK:-1024}; URL=http://localhost:$PORT
IMAGE=ghcr.io/0xsero/sglang-exl3@sha256:84f75f3424c99a3d63392a4e0348292bdeba9cf7efba2ff1aa9b85c8fc0131e8
NAME=turboderp-Qwen3.8-27B-exl3-3.00bpw
W=$HOME/models/qwen3.8-27b/exl3-3.00bpw
DK="docker --context default"
[ -f "$W/model-00002-of-00002.safetensors" ] || { echo "weights not complete in $W"; exit 1; }
for _ in $(seq 1200); do  # up to 2 h: it is queued behind downloads and may land while another test holds the GPU
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 4000 ] && break
  sleep 6
done
[ "$used" -lt 4000 ] || { echo "GPU $GPU has $used MiB in use"; exit 1; }
mkdir -p results
$DK rm -f exl3-bench >/dev/null 2>&1
# registry launch args verbatim (app/core/recipes.json, rtx-3090-24gb); only the host port, GPU index and --tp-size are ours
$DK run -d --name exl3-bench --gpus "\"device=$GPU\"" --shm-size 16g -p 127.0.0.1:$PORT:30000 \
  -v "$W:/models/$NAME:ro" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e SGLANG_EXL3_EMBED_HOST=1 -e SGLANG_EXL3_KERNEL=auto \
  -e SGLANG_EXL3_MODEL_PATH=/models/$NAME \
  --entrypoint /opt/entrypoint.sh "$IMAGE" \
  python3 -m sglang.launch_server --model-path /models/$NAME --quantization exl3 --trust-remote-code \
  --host 0.0.0.0 --port 30000 --served-model-name $NAME --context-length 204800 --mem-fraction-static 0.80 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --max-running-requests 1 --max-mamba-cache-size 5 \
  --cuda-graph-max-bs-decode 1 --chunked-prefill-size 1024 --max-prefill-tokens 1024 --tp-size "$TP" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-token-map /opt/sglang-exl3/tokenmaps/qwen38_hot32k_v2.pt >/dev/null || exit 1
trap '$DK logs exl3-bench >"results/server-$L.log" 2>&1; $DK rm -f exl3-bench >/dev/null' EXIT
for _ in $(seq 200); do  # up to 10 min: model load plus CUDA graph capture
  curl -sf $URL/health >/dev/null && break
  [ "$($DK inspect -f '{{.State.Running}}' exl3-bench 2>/dev/null)" = true ] || { echo "container exited, see results/server-$L.log"; exit 1; }
  sleep 3
done
curl -sf $URL/health >/dev/null || { echo "not healthy after 10 min"; exit 1; }
uv run -q bench.py data/chat.jsonl --label "$L" -c 1 -n 40 --max-tokens "$MAXTOK" --url $URL --model $NAME >/dev/null
python3 report.py "$L"
# same accuracy suite as the vLLM configs, only if the gate passed and TPM beats BEST (the current best TPM, default 0)
if tail -1 results/summary.jsonl | BEST=${BEST:-0} python3 -c "import json,os,sys; s=json.loads(sys.stdin.read()); sys.exit(0 if s['pass'] and s['tpm'] > int(os.environ['BEST']) else 1)"; then
  MODEL=$NAME URL=$URL ./eval.sh "$L" > "results/eval-$L.log" 2>&1
  grep -E "^\|" "results/eval-$L.log"
fi
