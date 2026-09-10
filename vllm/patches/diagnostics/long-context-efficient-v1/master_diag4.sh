#!/bin/bash
# master_diag4.sh — SPLITS ladder + eager probe: localize the per-step
# SPLITS-scaled cost that floors fp8+mtp4 at ~330-460 ms/step.
#
# Evidence so far (fp8-mtp4-v1/REPORT.md §3.2-3.5): dA (SPLITS=32) has a
# flat 330-380 ms/step floor 8k-32k; dB (SPLITS=64) = flat ~5.8 s/step at
# ALL ctx incl 2k (superlinear in SPLITS, ctx-independent); dF-t8 (BKV=8)
# made 64k WORSE (tile theory dead). Kernel arithmetic cannot account for
# the floor at any SPLITS -> per-step launch/graph/allocator-scaled cost.
# Nobody tested SPLITS < 32: this sweep walks 16/8/4 + an --enforce-eager
# probe (graph-interaction discriminator).
# Ends with prod_restore127.sh (real lane restore owed: the diag3 chain
# skip). Marks MASTER_DIAG4_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=3
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG4 queued (SPLITS ladder + eager probe)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'CHAIN4_GO$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG4_WAIT_TIMEOUT — aborting"; exit 1; }

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

# SPLITS ladder (BLOCK_KV/warps/stages at shipped 32/4/2 = measured best)
run_block_diag dJ-sp16 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_FP8MQ_SPLITS=16' \
  fp8_e4m3 262144 '1' '2048 8192 65536 261888' 2

run_block_diag dK-sp8 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_FP8MQ_SPLITS=8' \
  fp8_e4m3 262144 '1' '8192 65536' 2

run_block_diag dL-sp4 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_FP8MQ_SPLITS=4' \
  fp8_e4m3 262144 '1' '8192 65536' 2

# eager probe — graph-replay interaction discriminator
run_block_diag dG-eager 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' '' \
  fp8_e4m3 262144 '1' '8192 65536' 2 '' '--enforce-eager'

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag.out"
fi
log "MASTER_DIAG4_DONE"
