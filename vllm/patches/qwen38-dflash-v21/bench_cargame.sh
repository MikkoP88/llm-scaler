#!/bin/bash
# bench_cargame.sh <TAG> - canonical car-game benchmark cell driver (host side).
# Assumes a serving container is already up and healthy on 127.0.0.1:8000
# and its serve log path is given via LOG=... (default /root/telemetry/serve_v19.log).
#
#   TAG=c3 LOG=/root/telemetry/serve_v19.log bash bench_cargame.sh
#
# Canonical test (user-specified): prompt "Write a html car game.",
# temp 0.3 / top_k 20 / top_p 0.95 / min_p 0 / presence 0 / repetition 1.0,
# max_tokens 4096, streaming. One warmup completion first absorbs the
# first-request JIT (v18 trade-off) so cells compare steady serving.
set -u
TAG="${1:?usage: bench_cargame.sh <TAG>}"
PORT="${PORT:-8000}"
LOG="${LOG:-/root/telemetry/serve_v19.log}"
OUT="${OUT:-/root/telemetry}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/cargame_client.py"

echo "=== [$TAG] wait for health ==="
C=000
for i in $(seq 1 90); do
  C=$(timeout 4 curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$C" = "200" ] && break
  sleep 8
done
[ "$C" != "200" ] && { echo "NEVER HEALTHY"; tail -5 "$LOG" | cut -c1-160; exit 1; }

echo "=== [$TAG] warmup (absorb first-request JIT) ==="
LOGBEFORE=$(wc -l < "$LOG" 2>/dev/null || echo 0)
timeout 300 curl -s "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"Count from 1 to 60: 1, 2,","max_tokens":96,"temperature":0}' \
  | head -c 120; echo
sleep 3

echo "=== [$TAG] canonical car-game run ==="
python3 "$PY" --port "$PORT" --tag "$TAG" --json "$OUT/cargame_results.jsonl"

echo "=== [$TAG] engine-side stats since warmup ==="
tail -n +"$LOGBEFORE" "$LOG" 2>/dev/null | grep -a \
  "Mean acceptance length\|Draft acceptance rate\|Accepted throughput\|Avg generation throughput\|Speculative" \
  | tail -6
NOW=$(date +%s); LOGLINE=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
RESETS=$(dmesg 2>/dev/null | grep -ac "Engine reset")
echo "[$TAG] log_age=$((NOW-LOGLINE))s engine_resets=$RESETS"
echo "=== [$TAG] done ==="
