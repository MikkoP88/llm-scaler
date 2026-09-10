#!/bin/bash
# master_diag9.sh — v5 case-axis closure: concurrency + e5m2 + same-boot
# nospec references. Mandate: fp8 KV + MTP k=4 superior on ANY case.
#
# Proven so far (C1, dV5-cert): 77.05/65.2/61.5/64.1/52.67/33.21/23.47
# @2k/8k/16k/32k/64k/128k/262k vs nospec 29.79/25.91/20.69 @64k/128k/262k.
# Unproven axes:
#   * CONCURRENCY — v50-era history: conc8 spec loss fork-generic, MTP
#     acceptance ~0.65x at conc8. The fan-out is batch-friendly by
#     construction (per-position slice spans B rows -> 5 flash q1 calls
#     of B seqs each, the exact healthy decode shape), but it must be
#     MEASURED vs nospec on the same image/boot discipline.
#   * e5m2 KV parity (v51 certified e5m2 lanes; fan-out gate matches any
#     fp8* str dtype -> fix should carry for free).
#   * Same-boot nospec references at 2k/64k/262k x C1/C2 (kills any
#     harness-drift doubt vs the older 29.79/25.91/20.69 numbers).
# KV pool 707,980 tok -> C8@262k impossible (8x262k > pool), C2@262k
# fits (524k). C8@64k fits (524k).
#
# Blocks (all image fp8-mtp4-v5, C-list per block, reps 2):
#   dW-m4c2  — mtp4  C2   @2048/65536/261888
#   dW-m4c8  — mtp4  C8   @2048/65536
#   dW-ns-lo — nospec C1,C2 @2048
#   dW-ns64  — nospec C2,C8 @65536
#   dW-ns262 — nospec C2   @261888
#   dX-e5m2  — mtp4 kv fp8_e5m2 C1 @8192/65536/261888
# Ends with prod_restore_v5.sh + MASTER_DIAG9_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=2
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG9 queued (v5 conc + e5m2 case axes)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG8_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG9_WAIT_TIMEOUT — aborting"; exit 1; }
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

M4='{"method":"mtp","num_speculative_tokens":4}'

# 1) mtp4 concurrency: C2 across the knee incl 262k (2x262k=524k fits pool)
run_block_diag dW-m4c2 'llm-scaler-exp:fp8-mtp4-v5' \
  "$M4" '' fp8_e4m3 262144 '2' '2048 65536 261888' 2

# 2) mtp4 concurrency: C8 short+mid (8x262k impossible; 8x64k=524k fits)
run_block_diag dW-m4c8 'llm-scaler-exp:fp8-mtp4-v5' \
  "$M4" '' fp8_e4m3 262144 '8' '2048 65536' 2

# 3) nospec same-boot refs: short ctx C1+C2
run_block_diag dW-ns-lo 'llm-scaler-exp:fp8-mtp4-v5' \
  'nospec' '' fp8_e4m3 262144 '1 2' '2048' 2

# 4) nospec same-boot refs: 64k C2+C8
run_block_diag dW-ns64 'llm-scaler-exp:fp8-mtp4-v5' \
  'nospec' '' fp8_e4m3 262144 '2 8' '65536' 2

# 5) nospec same-boot refs: 262k C2
run_block_diag dW-ns262 'llm-scaler-exp:fp8-mtp4-v5' \
  'nospec' '' fp8_e4m3 262144 '2' '261888' 2

# 6) e5m2 KV parity with the fan-out (gate matches any fp8* str dtype)
run_block_diag dX-e5m2 'llm-scaler-exp:fp8-mtp4-v5' \
  "$M4" '' fp8_e5m2 262144 '1' '8192 65536 261888' 2

log "== restoring prod lane on v5 (prod_restore_v5.sh)"
if bash /root/build/prod_restore_v5.sh >> "$ROOT/prod_restore_v5_diag9.out" 2>&1; then
  log "PROD_RESTORE_V5_OK"
else
  log "PROD_RESTORE_V5_FAIL — see $ROOT/prod_restore_v5_diag9.out"
fi
log "MASTER_DIAG9_DONE"
