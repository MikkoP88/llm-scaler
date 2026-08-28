#!/bin/bash
# cargame_matrix.sh <CELL> - one car-game matrix cell on llm-scaler-vllm-adv:v19
# (host 10.20.3.58). One boot per cell; monitor3 telemetry; graceful teardown.
# Cells (bar-shape: dtype float16, block 128, mnbt 8192, cudagraph
# FULL_DECODE_ONLY; maxlen 262144 nospec / per-cell fit for spec — the
# drafter's uncompressed KV pool dominates spec memory, see case block):
#   bar   turboquant_4bit_nc nospec   (= the user's superior config, number to beat)
#   nbf16 default-KV(bf16)   nospec
#   nfp8  fp8_e4m3            nospec
#   nk8v4 turboquant_k8v4     nospec
#   c1    default-KV(bf16)    spec k4
#   c2    fp8_e4m3            spec k4
#   c3    turboquant_4bit_nc  spec k4   (headline: c3 >= bar)
#   c4    turboquant_k8v4     spec k4
# ACCEPT per KV: spec cell >= matching nospec cell steady tok/s.
set -u
CELL="${1:?usage: cargame_matrix.sh <bar|nbf16|nfp8|nk8v4|c1|c2|c3|c4>}"
LOG=/root/telemetry/cargame_matrix.log
SERVELOG=/root/telemetry/serve_v19_${CELL}.log
ts() { date '+%F %T'; }

# MAXLEN: nospec cells fit 262144 easily (measured bar: 9.69 GiB free KV =
# 1,210,665 tokens = 4.62x concurrency @262k; nbf16 also booted @262144).
# Spec cells hit a DIFFERENT wall: the dflash drafter's KV pool is forced
# UNCOMPRESSED (auto/bf16) and sized by the same max_model_len — ~11.3 GiB
# per GPU @262144 on top of the target's ~2.1 GiB (TQ4nc) = 13.48 GiB needed
# vs 6.67 GiB available after drafter fp8 weights (~3.0 GiB). So spec cells
# run the largest maxlen that fits (drafter pool dominates; target dtype
# barely moves it). Decode speed at the bench depth (~4k) is independent of
# KV pool size, so nospec-vs-spec speed comparison stays valid.
case "$CELL" in
  bar)   KV=turboquant_4bit_nc; SPEC=0; MAXLEN=262144 ;;
  nbf16) KV="";                  SPEC=0; MAXLEN=262144 ;;
  nfp8)  KV=fp8_e4m3;            SPEC=0; MAXLEN=262144 ;;
  nk8v4) KV=turboquant_k8v4;     SPEC=0; MAXLEN=262144 ;;
  c1)    KV="";                  SPEC=1; MAXLEN=73728 ;;  # bf16 tgt ~8.4GiB + drafter
  c2)    KV=fp8_e4m3;            SPEC=1; MAXLEN=98304 ;;
  c3)    KV=turboquant_4bit_nc;  SPEC=1; MAXLEN=98304 ;;  # vLLM est. max 118784
  c4)    KV=turboquant_k8v4;     SPEC=1; MAXLEN=98304 ;;
  *) echo "unknown cell $CELL"; exit 2 ;;
esac

# #03 protocol: never chain cells on a host with engine resets this boot
RESETS=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); RESETS=${RESETS:-0}
if [ "$RESETS" -ne 0 ]; then
  echo "$(ts) ABORT cell=$CELL: $RESETS engine resets this boot - host reboot required (#03)" | tee -a "$LOG"
  exit 3
fi

# graceful teardown of any previous serve
if docker ps -a --format '{{.Names}}' | grep -q '^lsv-test$'; then
  echo "$(ts) cell=$CELL stopping previous serve (graceful)" >> "$LOG"
  docker stop -t 30 lsv-test >/dev/null 2>&1
  docker rm -f lsv-test >/dev/null 2>&1
  sleep 5
fi
pkill -9 -f "[t]elemetry/monitor3.sh" 2>/dev/null
sleep 1

cd /root/build/v19 || exit 4
echo "$(ts) cell=$CELL LAUNCH kv=[${KV:-bf16}] spec=$SPEC" >> "$LOG"
env TARGET_DIR=/models/qwen3.8-27b-fp8 \
    DRAFTER_DIR=/models/drafter-fp8-v5 \
    SPEC=$SPEC KV_DTYPE="$KV" DTYPES=float16 BLOCKSIZE=128 MNBT=8192 MAXLEN=$MAXLEN \
    EXTRA_ARGS='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    setsid nohup ./serve.sh > "$SERVELOG" 2>&1 < /dev/null &
sleep 3
ARM="$CELL" SERVELOG="$SERVELOG" \
  setsid nohup /root/telemetry/monitor3.sh > "/root/telemetry/monitor3_${CELL}.out" 2>&1 < /dev/null &

# wait for health (max 14 min; deep maxlen 262144 arms boot slower)
UP=0
for i in $(seq 1 168); do
  if curl -s -m 3 -o /dev/null http://127.0.0.1:8000/health; then UP=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "$(ts) cell=$CELL CONTAINER_EXITED before health" | tee -a "$LOG"
    tail -10 "$SERVELOG" | cut -c1-170
    exit 6
  fi
  sleep 5
done
[ "$UP" != "1" ] && { echo "$(ts) cell=$CELL HEALTH_TIMEOUT" | tee -a "$LOG"; tail -5 "$SERVELOG" | cut -c1-170; exit 5; }
echo "$(ts) cell=$CELL HEALTH_UP (boot ~$((i*5))s)" >> "$LOG"

# canonical car-game benchmark (warmup + measured run + engine stats)
LOG="$SERVELOG" TAG="$CELL" bash /root/telemetry/bench_cargame.sh "$CELL" 2>&1 | tee "/root/telemetry/cargame_${CELL}.out"

# spec acceptance counters (spec cells)
if [ "$SPEC" = "1" ]; then
  curl -s -m 3 http://127.0.0.1:8000/metrics 2>/dev/null | \
    grep -aE '^vllm:spec_decode_(num_accepted_tokens_total|num_draft_tokens_total)' | grep -av '#'
fi

# graceful teardown
docker stop -t 30 lsv-test >/dev/null 2>&1
sleep 5
RESETS2=$(dmesg 2>/dev/null | grep -ac 'Engine reset' || true); RESETS2=${RESETS2:-0}
echo "$(ts) cell=$CELL DONE resets_before=$RESETS resets_after=$RESETS2" >> "$LOG"
[ "$RESETS2" -gt "$RESETS" ] && { echo "$(ts) cell=$CELL ENGINE_RESET_DURING_CELL" | tee -a "$LOG"; exit 7; }
exit 0
