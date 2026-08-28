#!/bin/bash
# baseline_b31.sh - intel/llm-scaler-vllm:0.21.0-b3.1 baseline gates.
# Recipe parity with the adv:v17 default arm (same v17 serve.sh, IMAGE
# override only). GENCFG=auto: upstream-default value (the "vllm" choice
# is fork-newer than b3.1); bench sampling is client-pinned so the
# generation-config default is immaterial for gates.
# Usage: baseline_b31.sh <TAG> [SPEC] [GMU] [KV_DTYPE]
set -u
TAG="${1:?tag}"
SPEC="${2:-1}"
GMU="${3:-0.8}"
KV="${4:-}"
PORT=8000
LOG=/root/telemetry/battery.log
ts() { date '+%F %T'; }

RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); RESETS=${RESETS:-0}
if [ "$RESETS" -ne 0 ]; then
  echo "$(ts) BASELINE b31 $TAG ABORT: $RESETS engine resets this boot" | tee -a "$LOG"
  exit 3
fi

if docker ps -a --format '{{.Names}}' | grep -q '^lsv-test$\|^qwen38-dspark$'; then
  docker stop -t 30 lsv-test qwen38-dspark >/dev/null 2>&1
  sleep 5
fi

cd /root/build/qwen38-dflash-v17 || exit 4
echo "$(ts) BASELINE b31 $TAG LAUNCH spec=$SPEC gmu=$GMU kv=[${KV:-default}]" >> "$LOG"
pkill -9 -f "[t]elemetry/monitor3.sh" 2>/dev/null
sleep 1
env IMAGE=intel/llm-scaler-vllm:0.21.0-b3.1 GENCFG=auto \
    TARGET_DIR=/models/qwen3.8-27b-fp8 DRAFTER_DIR=/models/drafter-fp8-v5 \
    ${SPEC:+SPEC=$SPEC} ${GMU:+GMU=$GMU} KV_DTYPE="$KV" \
    setsid nohup ./serve.sh > /root/telemetry/serve_b31.log 2>&1 < /dev/null &
sleep 3
ARM="b31_$TAG" SERVELOG=/root/telemetry/serve_b31.log \
  setsid nohup /root/telemetry/monitor3.sh > "/root/telemetry/monitor3_b31_$TAG.out" 2>&1 < /dev/null &

UP=0; WEDGE=0
for i in $(seq 1 144); do
  if curl -s -m 3 -o /dev/null http://127.0.0.1:$PORT/health; then UP=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "$(ts) BASELINE b31 $TAG CONTAINER_EXITED" | tee -a "$LOG"
    tail -8 /root/telemetry/serve_b31.log
    exit 6
  fi
  # #05e wedge signature on pre-fix builds: repeated shm_broadcast 60s lines
  if [ $((i % 12)) -eq 0 ]; then
    W=$(grep -ac "No available shared memory broadcast block" /root/telemetry/serve_b31.log 2>/dev/null || true); W=${W:-0}
    if [ "$W" -ge 3 ]; then WEDGE=1; break; fi
  fi
  sleep 5
done
if [ "$UP" != "1" ]; then
  if [ "$WEDGE" = "1" ]; then
    echo "$(ts) BASELINE b31 $TAG BOOT_WEDGE(#05e pre-fix): shm_broadcast x$W at gmu $GMU" | tee -a "$LOG"
  else
    echo "$(ts) BASELINE b31 $TAG HEALTH_TIMEOUT" | tee -a "$LOG"
    tail -5 /root/telemetry/serve_b31.log
  fi
  exit 5
fi
echo "$(ts) BASELINE b31 $TAG HEALTH_UP (~$((i*5))s)" >> "$LOG"

python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_cold512" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_warm512a" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_warm512b" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --url http://127.0.0.1:$PORT --tag "${TAG}_1536" --model qwen3.8-27b-fp8 --max-tokens 1536 --depth-step 1536 2>&1 | tail -1
curl -s -m 3 http://127.0.0.1:$PORT/metrics | grep -aE '^vllm:spec_decode_(num_accepted_tokens_total|num_draft_tokens_total)' | grep -av '#'
echo "$(ts) BASELINE b31 $TAG GATES_DONE" >> "$LOG"
