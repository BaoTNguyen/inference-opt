#!/usr/bin/env bash
# One-hour lever sweep on the best config (vLLM 0.32, DFlash2, Triton + full graphs + BF16 KV), one change per run.
# Baselines: v032-dflash2-k7 (6,967 TPM) and v032-dflash2-k7-r2 (6,793 TPM).
set -u
cd "$(dirname "$0")"
run() { echo "$(date +%T) $1"; shift; env "$@" ./sweep.sh "$L" > "results/$L.log" 2>&1; tail -3 "results/$L.log" | head -1; }
L=v032-dflash2-k5;       run "draft length 5" K=5
L=v032-dflash2-k4;       run "draft length 4" K=4
L=v032-dflash2-k7-fa;    run "FlashAttention backend" ATTN=FLASH_ATTN
L=v032-dflash2-k7-fisam; run "FlashInfer sampler on" FI_SAMPLER=1
W=$HOME/models/qwen3.8-27b/dflash2-w4a16
for _ in $(seq 90); do systemctl --user is-active -q w4a16-download || break; sleep 10; done
if [ -f "$W/model.safetensors" ] && ! systemctl --user is-active -q w4a16-download; then
  L=v032-dflash2w4-k7; run "W4A16 draft model, 7 tokens" DRAFT="$W"
else
  echo "$(date +%T) W4A16 draft model not ready; skipped"
fi
python3 report.py v032-dflash2-k7 v032-dflash2-k7-r2 v032-dflash2-k5 v032-dflash2-k4 v032-dflash2-k7-fa v032-dflash2-k7-fisam v032-dflash2w4-k7
for l in v032-dflash2-k7-fa v032-dflash2-k7-fisam v032-dflash2w4-k7; do  # did the change actually take effect?
  f=results/server-$l-0.log; [ -f "$f" ] || continue
  echo "== $l: $(grep -o -E "Capturing CUDA graphs \(decode, FULL\)|cudagraph_mode=[A-Z_]+|not supported with spec[^;]*|FlashInfer.*sampl[^.]*|Ninja build failed|quantization=[a-z_-]+" "$f" | sort | uniq -c | tr '\n' ' ')"
done
echo "$(date +%T) done"
