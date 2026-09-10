#!/bin/bash
# master_lce1_ext2.sh — second chained extension (user add 09-09):
#   PF pf8m4 : MTP4 + fp8_e4m3 KV on llm-scaler-exp:spec-prefill-phase-v1-v125
#              (= v1.2.5 + spec-prefill-phase.patch, image 563f7362bb35, built
#              from v1.2.5 a522bf15be2b; scheduler phase-race fix SPF1).
#              C 1/2 x LEN(65536 131072 261888) @ max-model-len 262144.
#              Direct A/B vs ext block f8e4m4 (same lane on unpatched v1.2.5:
#              c1 11.02/7.59/5.09 mean tps) — isolates the prefill-fix effect
#              on FP8-KV speculative long-context.
# Waits for MASTER_LCE1_EXT_DONE in master_ext.log (up to 20h), then runs the
# block, then restores prod lane again. Logs to /root/build/lce1/master_ext2.log.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_ext2.log
BASESEED=20260908
REPEATS=5
mkdir -p "$ROOT"

log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== EXT2 start (pid $$) — waiting for MASTER_LCE1_EXT_DONE"
DONE=0
for i in $(seq 1 1200); do
  if grep -q MASTER_LCE1_EXT_DONE "$ROOT/master_ext.log" 2>/dev/null; then DONE=1; break; fi
  sleep 60
done
if [ "$DONE" != "1" ]; then
  log "EXT2_WAIT_TIMEOUT (20h, no MASTER_LCE1_EXT_DONE) — aborting, leaving lane as-is"
  exit 1
fi
log "== ext done; EXT2 begins"

wait_health() { # up to 15 min
  local i
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -o /dev/null -m 4 http://localhost:8000/health; then return 0; fi
  done
  return 1
}

wait_warmup() { # dt_warmup_v51.py writes WARMUP_DONE lines to /root/dt_warmup.log
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

run_block_ext2() {
  local MODE="$1" IMG="$2" SPEC="$3" EE="$4" KVDTYPE="$5" MAXLEN="$6" CLIST="$7" LENLIST="$8"
  local LOGNAME="b_lce1_${MODE}"
  log "== BLOCK $MODE start (image $IMG, kv $KVDTYPE, maxlen $MAXLEN, spec $SPEC, extraenv '$EE', C='$CLIST', LEN='$LENLIST')"
  docker rm -f lsv-test >/dev/null 2>&1
  sleep 5
  if ! bash /root/build/serve_bench.sh "$KVDTYPE" "$SPEC" 0.9 "$MAXLEN" \
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
  for C in $CLIST; do
    for LEN in $LENLIST; do
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

# PF: prefill-fix v1.2.5, mtp4 + fp8_e4m3 KV, C 1/2
run_block_ext2 pf8m4 'llm-scaler-exp:spec-prefill-phase-v1-v125' \
  '{"method":"mtp","num_speculative_tokens":4}' '' fp8_e4m3 262144 '1 2' '65536 131072 261888'

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_ext2.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_ext2.out"
fi
log "MASTER_LCE1_EXT2_DONE"
