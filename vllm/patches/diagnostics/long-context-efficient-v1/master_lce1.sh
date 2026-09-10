#!/bin/bash
# master_lce1.sh — LCE1 round 1 driver: PLAN.md 'Next' item 1 (+ item 3 A/B slice).
#
# Blocks (each tears down and reboots lsv-test on turboquant_4bit_nc @0.9/262144):
#   nospec   llm-scaler-exp:v1.2.5          spec=nospec
#   mtp4     llm-scaler-exp:v1.2.5          spec={"method":"mtp","num_speculative_tokens":4}
#   dflash7  llm-scaler-exp:v1.2.5          spec={"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}
#   cand-df7 llm-scaler-exp:lce1-fp32-cache same dflash7 spec + VLLM_LCE1_FP32_CACHE=1
#
# Matrix per block (clients outer for early single-client falloff curves):
#   for C in 1 2 4 8; for LEN in 65536 131072 261888; 5 repeats.
#   Prompts are seed-functions of (LEN, C, repeat, client) only -> identical
#   across the four blocks; every block reboot resets the prefix cache.
# Each block: teardown -> boot -> health -> dt_warmup -> KV snapshot ->
#   suite -> cancel probe -> log preserve. Prod lane restored at the end.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master.log
BASESEED=20260908
REPEATS=5
mkdir -p "$ROOT"

log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

wait_health() { # up to 15 min
  local i
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -o /dev/null -m 4 http://localhost:8000/health; then return 0; fi
  done
  return 1
}

wait_warmup() { # dt_warmup_v51 writes WARMUP_DONE to /root/dt_warmup.log
  local ND0 ND i
  ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
  for i in $(seq 1 90); do
    sleep 10
    ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
    if [ "$ND" -gt "$ND0" ]; then return 0; fi
  done
  return 1
}

run_block() { # MODE IMG SPEC EXTRAENV
  local MODE="$1" IMG="$2" SPEC="$3" EE="$4"
  local LOGNAME="b_lce1_${MODE}"
  log "== BLOCK $MODE start (image $IMG, spec $SPEC, extraenv '$EE')"
  docker rm -f lsv-test >/dev/null 2>&1
  sleep 5
  if ! bash /root/build/serve_bench.sh turboquant_4bit_nc "$SPEC" 0.9 262144 \
      "$LOGNAME" "$EE" '' "$IMG" > "$ROOT/boot_${MODE}.out" 2>&1; then
    log "BOOT_FAIL $MODE — skipping block"
    tail -15 "$ROOT/boot_${MODE}.out" | tee -a "$M"
    return 0
  fi
  log "BOOT_OK $MODE"
  if wait_health; then log "HEALTH_OK $MODE"; else
    log "HEALTH_TIMEOUT $MODE — skipping block"; return 0; fi
  if wait_warmup; then log "WARMUP_OK $MODE"; else
    log "WARMUP_TIMEOUT $MODE — continuing"; fi
  docker exec lsv-test grep -a -m2 -E 'GPU KV cache size|Available KV cache memory' \
    "/root/${LOGNAME}.log" > "$ROOT/kv_${MODE}.txt" 2>/dev/null || true
  log "KV_SNAPSHOT $MODE: $(head -2 "$ROOT/kv_${MODE}.txt" | tr '\n' ' ')"

  local C LEN SEED RC ABORTED=0
  for C in 1 2 4 8; do
    for LEN in 65536 131072 261888; do
      SEED=$((BASESEED + LEN + C))
      log "CELL $MODE len=$LEN c=$C seed=$SEED start"
      ENGINE_LOG="/root/${LOGNAME}.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
        "$ROOT" "$MODE" "$LEN" "$C" "$REPEATS" "$SEED" \
        >> "$ROOT/suite_${MODE}.jsonl" 2>> "$ROOT/suite_${MODE}.err"
      RC=$?
      if [ "$RC" -eq 42 ]; then
        log "CELL $MODE len=$LEN c=$C rc=42 ENGINE_UNHEALTHY — aborting block"
        ABORTED=1; break 2
      elif [ "$RC" -eq 43 ]; then
        log "CELL $MODE len=$LEN c=$C rc=43 CELL_INVALID — skipping length"
        continue
      fi
      log "CELL $MODE len=$LEN c=$C done rc=$RC"
    done
  done
  [ "$ABORTED" = "1" ] && log "BLOCK $MODE aborted unhealthy (engine left for inspection)"

  log "CANCEL_PROBE $MODE start"
  ENGINE_LOG="/root/${LOGNAME}.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
    "$ROOT" cancel $((BASESEED + 999)) >> "$ROOT/cancel_${MODE}.jsonl" 2>&1
  log "CANCEL_PROBE $MODE rc=$?"

  docker cp "lsv-test:/root/${LOGNAME}.log" "$ROOT/${LOGNAME}.log" >/dev/null 2>&1 || \
    cp "/root/${LOGNAME}.log" "$ROOT/${LOGNAME}.log" 2>/dev/null || true
  gzip -f "$ROOT/${LOGNAME}.log" 2>/dev/null || true
  log "== BLOCK $MODE end"
}

run_block nospec  'llm-scaler-exp:v1.2.5'          'nospec' ''
run_block mtp4    'llm-scaler-exp:v1.2.5'          '{"method":"mtp","num_speculative_tokens":4}' ''
run_block dflash7 'llm-scaler-exp:v1.2.5'          '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' ''
run_block cand-df7 'llm-scaler-exp:lce1-fp32-cache' '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' 'VLLM_LCE1_FP32_CACHE=1'

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore.out and /root/build/bench127/prod_restore/"
fi
log "MASTER_LCE1_DONE"
