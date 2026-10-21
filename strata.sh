#!/usr/bin/env bash
# Strata (Qwen3.8-Flash-Next at 1-3 bits, experts in RAM/VRAM) through the same bench as sweep.sh, one client.
#   ./strata.sh strata-coder          CONFIG=<strata json> picks the model size (default: the Coder IQ1_M)
set -u
cd "$(dirname "$0")"
L=$1; MAXTOK=${MAXTOK:-1024}; PORT=8080; URL=http://localhost:$PORT
S="$HOME/Coding/Projects/Local Models/Qwen-3.8-Flash-Next/Strata"
CONFIG=${CONFIG:-$S/strata-coder-iq1_m.json}
NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" "$CONFIG")
for _ in $(seq 30); do  # same wait as sweep.sh: the last vLLM server may still be freeing memory
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 4000 ] && break
  sleep 3
done
[ "$used" -lt 4000 ] || { echo "a GPU has $used MiB in use"; exit 1; }
mkdir -p results
# Own systemd scope: Strata locks ~30 GB of RAM while loading, and systemd-oomd kills the whole cgroup that holds it.
# Launched from a terminal, that cgroup was the terminal (2026-10-21: oomd killed it and everything in it).
UNIT=strata-bench-$L
(cd "$S" && exec systemd-run --user --scope --quiet --unit="$UNIT" .venv/bin/python serve/server.py \
  --engine strata --config "$CONFIG" --port $PORT) >"results/server-$L.log" 2>&1 &
pid=$!
trap 'systemctl --user stop "$UNIT.scope" 2>/dev/null' EXIT
# loads ~55 GB from disk on a cold page cache, so allow 15 minutes
for _ in $(seq 300); do curl -sf $URL/health >/dev/null && break; kill -0 $pid 2>/dev/null || { echo "server died, see results/server-$L.log"; exit 1; }; sleep 3; done
curl -sf $URL/health >/dev/null || { echo "server not healthy after 15 min"; exit 1; }
# smoke test: bench.py needs streamed usage (completion_tokens) and a reasoning or content delta
curl -sN $URL/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}],\"max_tokens\":200,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  | python3 -c "
import json, sys
keys, usage = set(), None
for line in sys.stdin:
    if line.startswith('data: ') and '[DONE]' not in line:
        d = json.loads(line[6:]); usage = d.get('usage') or usage
        for c in d.get('choices') or []: keys |= set(c.get('delta', {}))
print('delta keys:', sorted(keys), '| usage:', usage)
sys.exit(0 if usage and usage.get('completion_tokens') else 1)" || { echo "no streamed usage: bench.py can't count tokens"; exit 1; }
uv run -q bench.py data/chat.jsonl --label "$L" -c 1 -n 40 --max-tokens "$MAXTOK" --url $URL --model "$NAME" >/dev/null
python3 report.py "$L"
