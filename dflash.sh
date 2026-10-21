#!/usr/bin/env bash
# DFlash2 block drafting (z-lab/Qwen3.8-27B-DFlash2) on vLLM 0.32 against the MTP baseline on the same vLLM.
#   ./dflash.sh                     baseline (MTP k=2), DFlash2 (K=7), then the accuracy suite if DFlash2 wins
# A new vLLM version is a new environment, so DFlash2 is compared to MTP on 0.32, not to the 0.27.1 numbers.
set -u
cd "$(dirname "$0")"
export VLLM=${VLLM:-$HOME/venvs/vllm032/bin/vllm}
DRAFT=${DRAFT:-$HOME/models/qwen3.8-27b/dflash2}; DK=${DK:-7}; BEST=${BEST:-5294}
[ -x "$VLLM" ] && [ -f "$DRAFT/.verified" ] || { echo "need $VLLM and a verified $DRAFT"; exit 1; }
tpm() { python3 -c "import json,sys; r=[json.loads(l) for l in open('results/summary.jsonl') if json.loads(l)['label']==sys.argv[1]]; print(r[-1]['tpm'] if r and r[-1]['pass'] else 0)" "$1"; }

echo "$(date +%T) baseline: MTP k=2 on vLLM 0.32"
K=2 ./sweep.sh v032-best-k2 > results/v032-best-k2.log 2>&1; tail -3 results/v032-best-k2.log
echo "$(date +%T) DFlash2 K=$DK"
L=v032-dflash2-k$DK
SPEC=dflash DRAFT="$DRAFT" K=$DK ./sweep.sh $L > results/$L.log 2>&1
if grep -q "server died" results/$L.log; then  # Triton + full graphs may not support the draft model: vLLM's own picks
  echo "$(date +%T) server died with the Triton/full-graph defaults; retrying with ATTN= CG= KV="
  L=v032-dflash2-k$DK-auto
  ATTN= CG= KV= SPEC=dflash DRAFT="$DRAFT" K=$DK ./sweep.sh $L > results/$L.log 2>&1
fi
tail -3 results/$L.log
base=$(tpm v032-best-k2); new=$(tpm $L); echo "TPM: baseline $base, DFlash2 $new, best so far $BEST"
if [ "$new" -gt "$base" ] && [ "$new" -gt "$BEST" ]; then
  echo "$(date +%T) DFlash2 wins: accuracy suite"
  extra=(); [[ $L == *-auto ]] && extra=(ATTN= CG= KV=)
  env "${extra[@]}" SPEC=dflash DRAFT="$DRAFT" K=$DK LEN=36864 setsid ./serve.sh > results/server-$L-suite.log 2>&1 & sp=$!
  trap 'kill -- -$sp 2>/dev/null' EXIT
  until curl -sf localhost:8001/health >/dev/null; do kill -0 $sp 2>/dev/null || { echo "suite server died"; exit 1; }; sleep 3; done
  ./eval.sh $L > results/eval-$L.log 2>&1
  f=$(ls -t results/eval/$L/*/results_*.json 2>/dev/null | head -1); [ -n "$f" ] || { echo "no results file"; exit 1; }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['results']; [print(t, {k:round(v,3) for k,v in m.items() if isinstance(v,(int,float)) and 'stderr' not in k}) for t,m in d.items()]" "$f"
fi
echo "$(date +%T) done"
