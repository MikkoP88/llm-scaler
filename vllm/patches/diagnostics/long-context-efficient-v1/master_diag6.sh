#!/bin/bash
# master_diag6.sh — route-family discriminators after the dO-dot rejection.
#
# diag5 verdicts feeding this chain:
#   * dV/dM/dN Q_LEN series: slope = c + a*Q_LEN + b*k with a+b ~= 0.28
#     µs/KVtok, k-independent floor c in [0.56, 0.84] paid even at k=1.
#   * dO-dot (tl.dot stage-1): 2.2x WORSE everywhere (slope 4.33) ->
#     tl.dot at [16,256]x[256,32] on this Triton-XPU stack is convicted;
#     the planned GQA-dot kernel is moot.
#   * Route map (flash_attn.py): with BOTH VLLM_XPU_FP8_MQ=0 and
#     VLLM_XPU_TRITON_MQ3D=0 the uniform q5 fp8 paged verify falls
#     through to the stock flash_attn_varlen_func call (paged KV +
#     seqused_k + descales) — the standard upstream spec-verify shape,
#     NEVER measured on this stack (v33 intercepted it since v33; the
#     v50 F6 conviction was q8/dflash, not this shape).
#
# Blocks (image fp8-mtp4-v1 everywhere, kv fp8_e4m3, C1):
#   dR2-flash — mtp4 + BOTH routes OFF @8k/64k/262k: the zero-code fix
#               candidate (stock flash q5 paged fp8).
#   dQ-kq1    — nospec + VLLM_XPU_FP8_MQ_Q1=1 @64k/262k: the fp8mq
#               kernel forced onto the q=1 shape vs flash nospec
#               (29.79/20.69) — isolates kernel fixed cost (c+a).
# Ends with prod_restore127.sh + MASTER_DIAG6_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=2
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG6 queued (route-family discriminators)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG5_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG6_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20

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

# 1) zero-code fix candidate — stock flash q5 paged fp8 (both routes off)
run_block_diag dR2-flash 'llm-scaler-exp:fp8-mtp4-v1' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_XPU_FP8_MQ=0 VLLM_XPU_TRITON_MQ3D=0' \
  fp8_e4m3 262144 '1' '8192 65536 261888' 2

# 2) kernel-vs-flash on the identical q=1 shape (isolates c+a)
run_block_diag dQ-kq1 'llm-scaler-exp:fp8-mtp4-v1' \
  'nospec' 'VLLM_XPU_FP8_MQ_Q1=1' \
  fp8_e4m3 262144 '1' '65536 261888' 2

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag6.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag6.out"
fi
log "MASTER_DIAG6_DONE"
