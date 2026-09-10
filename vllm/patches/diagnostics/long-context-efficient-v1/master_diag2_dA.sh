#!/bin/bash
# master_diag2_dA.sh — chained re-run of the dA baseline knee sweep.
# dA wedged 2/2 at the first spec warmup request on 09-09 evening (identical
# config ran fine as master f8e4m4 on 09-08); dB-sp64 survived p1 and is
# running. If the wedge was transient host/device state, this re-run collects
# the missing baseline; if dA wedges again while dB/dD/dE/dC all ran, that is
# itself a finding (default-SPLITS first-request capture wedge).
# Waits for MASTER_DIAG_DONE in master_diag.log (up to 8h), re-runs dA,
# restores prod, marks MASTER_DIAG2_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=3
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG2 waiting for main diag completion"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 60
done
[ "$DONE" = "1" ] || { log "DIAG2_WAIT_TIMEOUT — aborting"; exit 1; }

wait_health() {
  local i
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -o /dev/null -m 4 http://localhost:8000/health; then return 0; fi
  done
  return 1
}

wait_warmup() {
  local ND0 ND i
  ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
  for i in $(seq 1 90); do
    sleep 10
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      log "WARMUP_ABORT — container died during warmup"; return 1
    fi
    ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
    if [ "$ND" -gt "$ND0" ]; then return 0; fi
  done
  return 1
}

MODE=dA2-base
LOGNAME="b_lce1_${MODE}"
log "== BLOCK $MODE start (re-run of dA baseline)"
docker rm -f lsv-test >/dev/null 2>&1
sleep 20
if ! bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 "$LOGNAME" '' '' 'llm-scaler-exp:v1.2.5' > "$ROOT/boot_${MODE}.out" 2>&1; then
  log "BOOT_FAIL $MODE"; tail -15 "$ROOT/boot_${MODE}.out" | tee -a "$M"
else
  log "BOOT_OK $MODE"
  if wait_health; then log "HEALTH_OK $MODE"; else log "HEALTH_TIMEOUT $MODE"; exit 0; fi
  if wait_warmup; then log "WARMUP_OK $MODE"; else log "WARMUP_TIMEOUT $MODE — continuing"; fi
  ABORTED=0
  for LEN in 2048 8192 16384 32768 65536 131072 261888; do
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      log "CELL $MODE len=$LEN — container dead, aborting"; ABORTED=1; break
    fi
    SEED=$((BASESEED + LEN + 1))
    log "CELL $MODE len=$LEN c=1 seed=$SEED start"
    ENGINE_LOG="/root/${LOGNAME}.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
      "$ROOT" "$MODE" "$LEN" 1 "$REPEATS" "$SEED" \
      >> "$ROOT/suite_${MODE}.jsonl" 2>> "$ROOT/suite_${MODE}.err"
    RC=$?
    if [ "$RC" -eq 42 ]; then log "CELL $MODE len=$LEN rc=42 — aborting"; ABORTED=1; break; fi
    [ "$RC" -eq 43 ] && log "CELL $MODE len=$LEN rc=43 skip" || log "CELL $MODE len=$LEN done rc=$RC"
  done
  [ "$ABORTED" = "1" ] && log "BLOCK $MODE aborted (engine left for inspection)" && exit 0
  docker cp "lsv-test:/root/${LOGNAME}.log" "$ROOT/${LOGNAME}.log" >/dev/null 2>&1 || true
  gzip -f "$ROOT/${LOGNAME}.log" 2>/dev/null || true
  log "== BLOCK $MODE end"
fi

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL"
fi
log "MASTER_DIAG2_DONE"
