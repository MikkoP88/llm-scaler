#!/bin/bash
# master_diag8.sh — v5 certification: fan-out default-on bake + full knee +
# prod restore standing on the fix.
#
# dS-fanout A/B verdict (diag7, v4 image, VLLM_XPU_FP8_FANOUT=1, reps 2):
#   8192   71.6 tps  (54.6 ms step; kernel lane 64.62)
#   65536  52.63 tps (79.3 ms step; kernel lane 22.26; gate 29.79 PASS +77%)
#   261888 23.47 tps (160.4 ms step; kernel lane 5.80; gate 20.69 PASS +13%)
#   slope 0.413 µs/KVtok vs kernel 1.976 (4.8x) — on projection (0.375).
# Gates at 64k/262k PASSED with the env knob. This chain bakes the knob
# DEFAULT-ON (v5 = v1 + fan-out, --build-arg FANOUT=1), validates the full
# knee reps 3 with NO env (proves the baked default), adds a 2048 short-ctx
# guard (v51 cert fp8+mtp4 @2k = 78.7-100.4 tps), then restores PROD on v5.
#
# Block (image fp8-mtp4-v5, kv fp8_e4m3, spec mtp4, C1, extraenv EMPTY):
#   dV5-cert — @2048/8192/16384/32768/65536/131072/261888, reps 3
# Ends with prod_restore_v5.sh (fp8_e4m3 + mtp4 @0.9/262144 on v5) +
# MASTER_DIAG8_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG8 queued (v5 cert + prod restore)"
DONE=0
for i in $(seq 1 480); do
  if grep -q 'MASTER_DIAG7_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG8_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20

log "== baking v5 (v1 + flash fan-out route, knob default ON)"
if ! docker build -t llm-scaler-exp:fp8-mtp4-v5 /root/build/fp8m4bake \
    -f /root/build/fp8m4bake/Dockerfile.v4 --build-arg FANOUT=1 \
    > "$ROOT/bake_v5.out" 2>&1; then
  log "BAKE_V5_FAIL — see $ROOT/bake_v5.out"; tail -15 "$ROOT/bake_v5.out" | tee -a "$M"
  log "MASTER_DIAG8_DONE"; exit 1
fi
grep -a "PATCHED\|naming to" "$ROOT/bake_v5.out" | tail -3 | while read -r l; do log "BAKE_V5: $l"; done
IMGID=$(docker images --no-trunc --format '{{.ID}} {{.Repository}}:{{.Tag}}' llm-scaler-exp:fp8-mtp4-v5 | awk '{print $1}')
log "BAKE_V5_OK image=$IMGID"

# certify the baked default is ON (grep the knob line inside the image)
if docker run --rm --entrypoint /opt/venv/bin/python3 llm-scaler-exp:fp8-mtp4-v5 -c \
    'import re;s=open("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py").read();m=re.search(r"_V4_FANOUT = os\.environ\.get\(\"VLLM_XPU_FP8_FANOUT\", \"(\d)\"\)",s);print("BAKED_FANOUT_DEFAULT="+m.group(1));exit(0 if m and m.group(1)=="1" else 1)'; then
  log "BAKE_V5_DEFAULT_CONFIRMED fanout=1"
else
  log "BAKE_V5_DEFAULT_WRONG — aborting"; log "MASTER_DIAG8_DONE"; exit 1
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

# certification knee: NO extraenv — the baked default must carry the route
run_block_diag dV5-cert 'llm-scaler-exp:fp8-mtp4-v5' \
  '{"method":"mtp","num_speculative_tokens":4}' '' \
  fp8_e4m3 262144 '1' '2048 8192 16384 32768 65536 131072 261888' 3

log "== restoring prod lane on v5 (prod_restore_v5.sh)"
if bash /root/build/prod_restore_v5.sh >> "$ROOT/prod_restore_v5.out" 2>&1; then
  log "PROD_RESTORE_V5_OK"
else
  log "PROD_RESTORE_V5_FAIL — see $ROOT/prod_restore_v5.out"
fi
log "MASTER_DIAG8_DONE"
