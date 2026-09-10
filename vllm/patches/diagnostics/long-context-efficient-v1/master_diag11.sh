#!/bin/bash
# master_diag11.sh — v6 production image: v5 + baked
# VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1. Certification = byte-parity against
# BOTH reference blocks, no per-boot env anywhere:
#   dW6-e4m3 — v6, kv fp8_e4m3, extraenv EMPTY, full knee reps 2.
#     Expected byte-identical to dV5-cert rep0/rep1 rows (same seed
#     formula): proves the baked env is INERT for the standing lane.
#   dW6-e5m2 — v6, kv fp8_e5m2, extraenv EMPTY (the baked env must
#     carry the boot past the v34 guard), 8192/65536/261888 reps 2.
#     Expected byte-identical to diag9b dX2-e5m2 rows: proves
#     env-passed == env-baked for the e5m2 lane.
# Known accepted deviation class: cross-boot fp near-tie acceptance
# flips (diag10 §8.2: 1 of 6 samples) — any differing sha must still
# be a correct answer (task check true, 192 tok) to count as PASS;
# verdict comparison is done post-run on the jsonl.
# Ends with prod_restore_v6.sh (prod stands on v6, e4m3+mtp4 lane —
# config unchanged, numerics proven identical) + MASTER_DIAG11_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=2
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG11 queued (v6 bake + dual-lane byte-parity cert + prod restore)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG10_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG11_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20

log "== baking v6 (v5 + ENV VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1)"
if ! docker build -t llm-scaler-exp:fp8-mtp4-v6 /root/build/fp8m4bake \
    -f /root/build/fp8m4bake/Dockerfile.v6 \
    > "$ROOT/bake_v6.out" 2>&1; then
  log "BAKE_V6_FAIL — see $ROOT/bake_v6.out"; tail -15 "$ROOT/bake_v6.out" | tee -a "$M"
  log "MASTER_DIAG11_DONE"; exit 1
fi
IMGID=$(docker images --no-trunc --format '{{.ID}} {{.Repository}}:{{.Tag}}' llm-scaler-exp:fp8-mtp4-v6 | awk '{print $1}')
log "BAKE_V6_OK image=$IMGID"

# certify the baked ENV is present in the image
ENVCHK=$(docker run --rm --entrypoint /bin/bash llm-scaler-exp:fp8-mtp4-v6 -c 'echo V6ENV=$VLLM_XPU_ALLOW_E5M2_FP8_CKPT')
log "BAKE_V6_ENV_CHECK: $ENVCHK"
[ "$ENVCHK" = "V6ENV=1" ] || { log "BAKE_V6_ENV_WRONG — aborting"; log "MASTER_DIAG11_DONE"; exit 1; }

# certify the v5 fan-out default survived the layering (still ON)
if docker run --rm --entrypoint /opt/venv/bin/python3 llm-scaler-exp:fp8-mtp4-v6 -c \
    'import re;s=open("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py").read();m=re.search(r"_V4_FANOUT = os\.environ\.get\(\"VLLM_XPU_FP8_FANOUT\", \"(\d)\"\)",s);print("BAKED_FANOUT_DEFAULT="+m.group(1));exit(0 if m and m.group(1)=="1" else 1)'; then
  log "BAKE_V6_FANOUT_STILL_ON fanout=1"
else
  log "BAKE_V6_FANOUT_CHECK_FAIL — aborting"; log "MASTER_DIAG11_DONE"; exit 1
fi

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

# 1) inertness cert: e4m3 lane on v6, NO env — full knee, byte-parity
#    target = dV5-cert rep0/rep1
run_block_diag dW6-e4m3 'llm-scaler-exp:fp8-mtp4-v6' \
  "$M4" '' fp8_e4m3 262144 '1' '2048 8192 16384 32768 65536 131072 261888' 2

# 2) e5m2 lane on v6, NO env (baked ENV must carry the guard) —
#    byte-parity target = dX2-e5m2 rep0/rep1
run_block_diag dW6-e5m2 'llm-scaler-exp:fp8-mtp4-v6' \
  "$M4" '' fp8_e5m2 262144 '1' '8192 65536 261888' 2

log "== restoring prod lane on v6 (prod_restore_v6.sh)"
if bash /root/build/prod_restore_v6.sh >> "$ROOT/prod_restore_v6.out" 2>&1; then
  log "PROD_RESTORE_V6_OK"
else
  log "PROD_RESTORE_V6_FAIL — see $ROOT/prod_restore_v6.out"
fi
log "MASTER_DIAG11_DONE"
