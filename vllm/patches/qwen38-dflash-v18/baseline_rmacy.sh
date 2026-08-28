#!/bin/bash
# baseline_rmacy.sh - ghcr.io/rmacy/qwen38-fp8-dspark:v16 baseline gates.
# Uses the OLD patches/qwen38-dflash/serve.sh default case (the rmacy-
# validated recipe, verbatim): graphs OFF, gmu 0.90, maxlen 8192,
# maxseqs 1, port 8003, 127.0.0.1, container qwen38-dspark, always-spec.
# longctx/deep cells are structurally impossible (maxlen 8192) - that
# IS the comparison cell. conc4 run shows the maxseqs=1 serialization.
set -u
TAG="${1:-rmacy_v16}"
PORT=8003
LOG=/root/telemetry/battery.log
ts() { date '+%F %T'; }

RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); RESETS=${RESETS:-0}
if [ "$RESETS" -ne 0 ]; then
  echo "$(ts) BASELINE rmacy ABORT: $RESETS engine resets this boot" | tee -a "$LOG"
  exit 3
fi

if docker ps -a --format '{{.Names}}' | grep -q '^lsv-test$\|^qwen38-dspark$'; then
  docker stop -t 30 lsv-test qwen38-dspark >/dev/null 2>&1
  sleep 5
fi

cd /root/build/qwen38-dflash || exit 4   # old serve.sh (rmacy default case)
echo "$(ts) BASELINE rmacy $TAG LAUNCH (own recipe: eager gmu0.90 maxlen8192 maxseqs1 spec)" >> "$LOG"
pkill -9 -f "[t]elemetry/monitor3.sh" 2>/dev/null
sleep 1
env IMAGE=ghcr.io/rmacy/qwen38-fp8-dspark:v16 \
    TARGET_DIR=/models/qwen3.8-27b-fp8 DRAFTER_DIR=/models/drafter-fp8-v5 \
    setsid nohup ./serve.sh > /root/telemetry/serve_rmacy.log 2>&1 < /dev/null &
sleep 3
ARM="rmacy_$TAG" SERVELOG=/root/telemetry/serve_rmacy.log \
  setsid nohup /root/telemetry/monitor3.sh > "/root/telemetry/monitor3_rmacy_$TAG.out" 2>&1 < /dev/null &

UP=0
for i in $(seq 1 144); do
  if curl -s -m 3 -o /dev/null http://127.0.0.1:$PORT/health; then UP=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^qwen38-dspark$'; then
    echo "$(ts) BASELINE rmacy $TAG CONTAINER_EXITED" | tee -a "$LOG"
    tail -8 /root/telemetry/serve_rmacy.log
    exit 6
  fi
  sleep 5
done
[ "$UP" = "1" ] || { echo "$(ts) BASELINE rmacy $TAG HEALTH_TIMEOUT" | tee -a "$LOG"; tail -5 /root/telemetry/serve_rmacy.log; exit 5; }
echo "$(ts) BASELINE rmacy $TAG HEALTH_UP (~$((i*5))s)" >> "$LOG"

python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_cold512" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_warm512a" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_warm512b" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_1536" --model qwen3.8-27b-fp8 --max-tokens 1536 --depth-step 1536 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_conc4_512" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 --concurrency 4 2>&1 | tail -1
curl -s -m 3 http://127.0.0.1:$PORT/metrics | grep -aE '^vllm:spec_decode_(num_accepted_tokens_total|num_draft_tokens_total)' | grep -av '#'
echo "$(ts) BASELINE rmacy $TAG GATES_DONE" >> "$LOG"
