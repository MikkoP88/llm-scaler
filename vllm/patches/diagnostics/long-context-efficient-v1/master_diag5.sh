#!/bin/bash
# master_diag5.sh — validate the SPLITS=16 bake + attack the residual
# ctx-linear spec-machinery cost (~2.2-2.7 µs/KVtok, identical across
# SPLITS and across KV backends incl tq4nc+mtp4 — see REPORT §3.6).
#
# Blocks (all image llm-scaler-exp:fp8-mtp4-v1, kv fp8_e4m3, C1):
#   dV-sp16v  — baked SPLITS=16, full knee 2k..262k, reps 3  (BAKE GATE)
#   dH-q1sp16 — +VLLM_XPU_FP8_MQ_Q1=1 @64k/262k: route DRAFT q=1
#               attention through the now-tuned v51 kernel (dE tested q1
#               only at sp32 where kernel==C++ in cost; at sp16 the kernel
#               got 3.8x cheaper — if the draft loop pays the same
#               inefficiency class, this cuts it).
#   dM-k2     — num_speculative_tokens=2 @64k/262k: step-cost vs
#               #draft-forwards discriminator (fewer draft steps, q=3
#               verify) — separates draft-loop-scaled vs once-per-step
#               O(ctx) cost.
# Ends with prod_restore127.sh + MASTER_DIAG5_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=3
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG5 queued (bake validation + residual discriminators)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG4_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG5_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20   # settle after diag4 teardown/restore churn

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

# 1) BAKE GATE — full knee on the baked SPLITS=16 image
run_block_diag dV-sp16v 'llm-scaler-exp:fp8-mtp4-v1' \
  '{"method":"mtp","num_speculative_tokens":4}' '' \
  fp8_e4m3 262144 '1' '2048 8192 32768 65536 131072 261888' 3

# 2) drafter q=1 through the tuned kernel
run_block_diag dH-q1sp16 'llm-scaler-exp:fp8-mtp4-v1' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_XPU_FP8_MQ_Q1=1' \
  fp8_e4m3 262144 '1' '65536 261888' 2

# 3) draft-forward-count / Q_LEN-scaling discriminators.
#    Step-time series q=5 (dV) -> q=3 (dM-k2) -> q=2 (dN-k1) separates:
#    verify-kernel ALU (scales ~2*Q_LEN passes/tile), draft loop (scales
#    with #forwards: 4 -> 2 -> 1), host per-step O(ctx) (scales not at all).
#    Compare STEP MS (=1000*acceptance/tps), not tps.
run_block_diag dM-k2 'llm-scaler-exp:fp8-mtp4-v1' \
  '{"method":"mtp","num_speculative_tokens":2}' '' \
  fp8_e4m3 262144 '1' '65536 261888' 2

run_block_diag dN-k1 'llm-scaler-exp:fp8-mtp4-v1' \
  '{"method":"mtp","num_speculative_tokens":1}' '' \
  fp8_e4m3 262144 '1' '65536 261888' 2

# 4) FIX CANDIDATE A/B — tl.dot stage-1 (matrix units) vs the shipped
#    masked-reduction vector passes. v2 image = SPLITS=16 defaults +
#    VLLM_FP8MQ_DOT knob (default off; this block turns it ON).
#    Also runs 8192 to watch for a short-ctx regression.
run_block_diag dO-dot 'llm-scaler-exp:fp8-mtp4-v2' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_FP8MQ_DOT=1' \
  fp8_e4m3 262144 '1' '8192 65536 261888' 2

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag.out"
fi
log "MASTER_DIAG5_DONE"
