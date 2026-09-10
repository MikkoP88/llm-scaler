#!/bin/bash
# master_diag3_dF.sh — LIVE FIX A/B: retuned fp8mq kernel tile config.
#
# Root cause candidate (fp8-mtp4-v1/REPORT.md §3.3 + TQ kernel tuning notes
# measured on THIS hardware): the v51 fp8mq kernel ships
# BLOCK_KV=32 / warps=4 / stages=2 (code defaults; its own docstring says
# 16/32/1/2 — drifted mid-experiment). The TQ kernel's measured conviction
# on 2x Arc Pro B70: BLOCK_KV=16 -> -72% deep, 32 -> -83% deep, 16+2warps
# -20%, 32+2warps FATAL; TQ ships 4/1/1. fp8mq ships exactly the convicted
# spill regime. All fp8mq knobs are env-read at import, so the fix is
# testable WITHOUT baking:
#   dF-t4: VLLM_FP8MQ_BLOCK_KV=4  WARPS=1 STAGES=1  (TQ-shipped config)
#   dF-t8: VLLM_FP8MQ_BLOCK_KV=8  WARPS=1 STAGES=1  (upper safe bound per
#          TQ note "do not raise past 8 without re-validating")
# Gates: beat dA at EVERY length; target = beat fp8+nospec
# (29.79/25.91/20.69 tps c1 @64k/128k/262k).
# Waits for MASTER_DIAG_DONE (v3 sweep), runs both blocks, restores prod,
# marks MASTER_DIAG3_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=3
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG3 queued (retuned fp8mq tile config A/B)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 60
done
[ "$DONE" = "1" ] || { log "DIAG3_WAIT_TIMEOUT — aborting"; exit 1; }

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
  local LOGNAME="b_lce1_${MODE}"
  log "== BLOCK $MODE start (image $IMG, kv $KVDTYPE, maxlen $MAXLEN, spec $SPEC, extraenv '$EE', C='$CLIST', LEN='$LENLIST', reps=$REPS)"
  docker rm -f lsv-test >/dev/null 2>&1
  sleep 20
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

# dF-t4: TQ-shipped tile config (BLOCK_KV=4, 1 warp, 1 stage)
run_block_diag dF-t4 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' \
  'VLLM_FP8MQ_BLOCK_KV=4 VLLM_FP8MQ_STAGE1_WARPS=1 VLLM_FP8MQ_STAGE1_STAGES=1' \
  fp8_e4m3 262144 '1' '2048 8192 16384 32768 65536 131072 261888'

# dF-t8: upper safe bound (BLOCK_KV=8, same warps/stages), reps 2
run_block_diag dF-t8 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' \
  'VLLM_FP8MQ_BLOCK_KV=8 VLLM_FP8MQ_STAGE1_WARPS=1 VLLM_FP8MQ_STAGE1_STAGES=1' \
  fp8_e4m3 262144 '1' '8192 65536 261888' 2

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag.out"
fi
log "MASTER_DIAG3_DONE"
