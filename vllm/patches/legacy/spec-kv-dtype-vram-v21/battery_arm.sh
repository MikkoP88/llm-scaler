#!/bin/bash
# battery_arm.sh - one battery arm: graceful teardown, relaunch, monitor, gates.
# Usage: battery_arm.sh <ARM_NAME> [KV_DTYPE] [SPEC] [GMU] [MAXLEN] [EXTRA_SERVE_ENV]
#   KV_DTYPE: "" (default bf16) | fp8 | turboquant_4bit_nc | turboquant_k8v4
#   SPEC: 1 (dflash k=4) | 0 (target only)
#   GMU: gpu-memory-utilization (default 0.8)
# Aborts BEFORE relaunch if dmesg shows any engine reset this boot (#03 protocol:
# reboot required - never chain arms on a reset host).
set -u
ARM_NAME="${1:?arm name}"
KV_DTYPE="${2:-}"
SPEC="${3:-1}"
GMU="${4:-0.8}"
MAXLEN="${5:-64000}"
EXTRA_ENV="${6:-}"

LOG=/root/telemetry/battery.log
ts() { date '+%F %T'; }

RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); RESETS=${RESETS:-0}
if [ "$RESETS" -ne 0 ]; then
  echo "$(ts) ABORT arm=$ARM_NAME: $RESETS engine resets this boot - host reboot required (#03)" | tee -a "$LOG"
  exit 3
fi

# graceful teardown of previous serve (SIGTERM path verified zero engine events)
if docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
  echo "$(ts) arm=$ARM_NAME stopping previous serve (graceful)" >> "$LOG"
  docker stop -t 30 lsv-test >/dev/null 2>&1
  sleep 5
fi
# keep prior serve log per arm
[ -f /root/telemetry/serve_tel.log ] && mv /root/telemetry/serve_tel.log "/root/telemetry/serve_tel.${ARM_NAME}.prev.log"

cd /root/build/qwen38-dflash-v17 || exit 4
echo "$(ts) arm=$ARM_NAME LAUNCH kv=[${KV_DTYPE:-bf16}] spec=$SPEC gmu=$GMU maxlen=$MAXLEN extra=[$EXTRA_ENV]" >> "$LOG"
# stop the PREVIOUS arm's monitor3 before starting this arm's
# NOTE: -9 required — monitor3 traps TERM but blocks in `tail -F | while
# read`, so a trapped TERM is deferred forever (5 stale monitors observed
# after a3-a5 with plain pkill; they replayed the a5 reset into capture
# dirs of finished arms).
pkill -9 -f "[t]elemetry/monitor3.sh" 2>/dev/null
sleep 1
env TARGET_DIR=/models/qwen3.8-27b-fp8 \
    DRAFTER_DIR=/models/drafter-fp8-v5 \
    ${SPEC:+SPEC=$SPEC} KV_DTYPE="$KV_DTYPE" GMU="$GMU" MAXLEN="$MAXLEN" \
    ${EXTRA_ENV:+$EXTRA_ENV} \
    setsid nohup ./serve.sh > /root/telemetry/serve_tel.log 2>&1 < /dev/null &
sleep 3
ARM="$ARM_NAME" SERVELOG=/root/telemetry/serve_tel.log \
  setsid nohup /root/telemetry/monitor3.sh > "/root/telemetry/monitor3_${ARM_NAME}.out" 2>&1 < /dev/null &
echo "monitor3 started for $ARM_NAME"

# wait for health (max 12 min; deep TQ arms boot slower)
UP=0
for i in $(seq 1 144); do
  if curl -s -m 3 -o /dev/null http://127.0.0.1:8000/health; then UP=1; break; fi
  # fast-fail: serve process died (config error, UR39 crash)
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "$(ts) arm=$ARM_NAME CONTAINER_EXITED (serve died before health)" | tee -a "$LOG"
    tail -8 /root/telemetry/serve_tel.log
    exit 6
  fi
  sleep 5
done
if [ "$UP" != "1" ]; then
  echo "$(ts) arm=$ARM_NAME HEALTH_TIMEOUT" | tee -a "$LOG"
  tail -5 /root/telemetry/serve_tel.log
  exit 5
fi
echo "$(ts) arm=$ARM_NAME HEALTH_UP (boot took ~$((i*5))s)" >> "$LOG"

# gates: cold 512 (JIT warm, discarded), warm 512 x2, 1536
python3 /root/bench_gen.py --tag "${ARM_NAME}_cold512" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --tag "${ARM_NAME}_warm512a" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --tag "${ARM_NAME}_warm512b" --model qwen3.8-27b-fp8 --max-tokens 512 --depth-step 512 2>&1 | tail -1
python3 /root/bench_gen.py --tag "${ARM_NAME}_1536"  --model qwen3.8-27b-fp8 --max-tokens 1536 --depth-step 1536 2>&1 | tail -1
# acceptance snapshot (spec arms)
curl -s -m 3 http://127.0.0.1:8000/metrics | grep -aE '^vllm:spec_decode_(num_accepted_tokens_total|num_draft_tokens_total)' | grep -av '#'
echo "$(ts) arm=$ARM_NAME GATES_DONE" >> "$LOG"
