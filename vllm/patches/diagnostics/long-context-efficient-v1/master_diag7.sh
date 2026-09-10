#!/bin/bash
# master_diag7.sh — v4 flash fan-out A/B (the fix candidate).
#
# Measured slope ladder for fp8 paged KV on this stack (µs/KVtok):
#   flash q=1 (nospec lane)       0.075   <- only healthy shape
#   v51 Triton vector kernel q5   1.98    (current prod path)
#   stock flash q5 (branch1)      3.35    (dR2)
#   tl.dot stage-1 q5             4.33    (dO, convicted)
# v4 decomposes uniform q_len verify into q_len flash q=1 calls
# (position slices, shared paged KV/descales, device-side per-position
# causal limits). Expected step slope ~ 5x0.075 + small -> gates
# (64k > 29.79, 262k > 20.69 tps) projected PASS.
#
# Block (image fp8-mtp4-v4 baked in-script, kv fp8_e4m3, spec mtp4, C1):
#   dS-fanout — VLLM_XPU_FP8_FANOUT=1 @8192/65536/261888, reps 2
# Ends with prod_restore127.sh + MASTER_DIAG7_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=2
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG7 queued (flash fan-out A/B)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG6_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG7_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20

log "== baking v4 (base v1 + flash fan-out route, knob default off)"
if ! docker build -t llm-scaler-exp:fp8-mtp4-v4 /root/build/fp8m4bake \
    -f /root/build/fp8m4bake/Dockerfile.v4 > "$ROOT/bake_v4.out" 2>&1; then
  log "BAKE_V4_FAIL — see $ROOT/bake_v4.out"; tail -15 "$ROOT/bake_v4.out" | tee -a "$M"
  log "MASTER_DIAG7_DONE"; exit 1
fi
grep -a "PATCHED\|naming to" "$ROOT/bake_v4.out" | tail -3 | while read -r l; do log "BAKE_V4: $l"; done
log "BAKE_V4_OK"

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

run_block_diag() {
  local MODE="$1" IMG="$2" SPEC="$3" EE="$4" KVDTYPE="$5" MAXLEN="$6" CLIST="$7" LENLIST="$8"
  local REPS="${9:-$REPEATS}"
  local XF="${10:-}"
  local LOGNAME="b_lce1_${MODE}"
  log "== BLOCK $MODE start (image $IMG, kv $KVDTYPE, maxlen $MAXLEN, spec $SPEC, extraenv '$EE', extraflags '$XF', C='$CLIST', LEN='$LENLIST', reps=$REPS)"
  docker rm -f lsv-test >/dev/null 2>&1
  sleep 20
  if ! bash /root/build/serve_bench.sh "$KVDTYPE" "$SPEC" 0.9 "$MAXLEN" \
      "$LOGNAME" "$EE" "$XF" "$IMG" > "$ROOT/boot_${MODE}.out" 2>&1; then
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
      if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
        log "CELL $MODE len=$LEN — container dead, aborting block"
        ABORTED=1; break 2
      fi
      SEED=$((BASESEED + LEN + C))
      log "CELL $MODE len=$LEN c=$C seed=$SEED start"
      ENGINE_LOG="/root/${LOGNAME}.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
        "$ROOT" "$MODE" "$LEN" "$C" "$REPS" "$SEED" \
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
  [ "$ABORTED" = "1" ] && { log "BLOCK $MODE aborted unhealthy (engine left for inspection)"; log "SETTLE 90s after abort (protect driver state)"; sleep 90; }

  log "CANCEL_PROBE $MODE start"
  ENGINE_LOG="/root/${LOGNAME}.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
    "$ROOT" cancel $((BASESEED + 999)) >> "$ROOT/cancel_${MODE}.jsonl" 2>&1
  log "CANCEL_PROBE $MODE rc=$?"

  docker cp "lsv-test:/root/${LOGNAME}.log" "$ROOT/${LOGNAME}.log" >/dev/null 2>&1 || \
    cp "/root/${LOGNAME}.log" "$ROOT/${LOGNAME}.log" 2>/dev/null || true
  gzip -f "$ROOT/${LOGNAME}.log" 2>/dev/null || true
  log "== BLOCK $MODE end"
}

run_block_diag dS-fanout 'llm-scaler-exp:fp8-mtp4-v4' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_XPU_FP8_FANOUT=1' \
  fp8_e4m3 262144 '1' '8192 65536 261888' 2

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag7.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag7.out"
fi
log "MASTER_DIAG7_DONE"
