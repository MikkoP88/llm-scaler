#!/bin/bash
# master_lce1_ext.sh — chained extension to master_lce1.sh.
# Waits for MASTER_LCE1_DONE, then runs (c=8 AND c=4 dropped from ALL testing
# per user 09-08; remaining concurrency = C 1/2):
#   E1 oldns   : nospec on llm-scaler-prod:v1 (old prod image), tq4nc @0.9/262144,
#                C(1 2) x LEN(65536 131072 261888) — alternative baseline +
#                P0 wedge pre-existence check.
#   E2 f8e4ns  : nospec + fp8_e4m3 KV on v1.2.5 — C 1/2 (fp8 KV pool ~half of
#                tq4nc's 1.57M; c2/262k=524k fits, higher C risks the known wedge).
#   E3 f8e4m4  : mtp4 + fp8_e4m3 KV on v1.2.5 — C 1/2 (v51 certified spec x fp8
#                at 2k ctx; this is the long-context leg).
#   E4 f8e4df7 : dflash7 + fp8_e4m3 KV on v1.2.5 @ max-model-len 245760 (v51
#                constraint for dflash lanes) — C 1/2; 261888 prompts exceed
#                context -> rc43 recorded.
#   (fp8_e5m2 lane DROPPED per user decision 09-08.)
#   Secondary M1 on llm-scaler-exp:v1.2.6t2 (tq4nc, C 1/2):
#     S1 s2ns S2 s2m4 S3 s2df7 — mirrors master blocks 1-3 on the v53 test image.
#   Secondary M2 on v1.2.6t2 (fp8_e4m3, C 1/2):
#     S4 s2f8ns S5 s2f8m4 S6 s2f8df7 (dflash @245760).
#   E5 df7r    : dflash7+tq4nc retry @ 245760, C 1, LEN 65536/131072 — ONLY if
#                master's dflash7 block BOOT_FAILed or HEALTH_TIMEOUTed.
# Ends with prod_restore127.sh again. Logs to /root/build/lce1/master_ext.log.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_ext.log
BASESEED=20260908
REPEATS=5
mkdir -p "$ROOT"

log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== EXT start (pid $$) — waiting for MASTER_LCE1_DONE"
DONE=0
for i in $(seq 1 1200); do
  if grep -q MASTER_LCE1_DONE "$ROOT/master.log" 2>/dev/null; then DONE=1; break; fi
  sleep 60
done
if [ "$DONE" != "1" ]; then
  log "EXT_WAIT_TIMEOUT (20h, no MASTER_LCE1_DONE) — aborting extension, leaving lane as-is"
  exit 1
fi
log "== master done; extension begins"

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

# run_block_ext MODE IMG SPEC EE KVDTYPE MAXLEN CLIST LENLIST
run_block_ext() {
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

# E1: old prod image, C 1/2 (same seeds/prompts as nospec block)
run_block_ext oldns 'llm-scaler-prod:v1' 'nospec' '' turboquant_4bit_nc 262144 '1 2' '65536 131072 261888'

# E2/E3/E4: fp8_e4m3 KV lanes on v1.2.5 (nospec + MTP4 + DFlash7), C 1/2
run_block_ext f8e4ns  'llm-scaler-exp:v1.2.5' 'nospec' '' fp8_e4m3 262144 '1 2' '65536 131072 261888'
run_block_ext f8e4m4  'llm-scaler-exp:v1.2.5' '{"method":"mtp","num_speculative_tokens":4}' '' fp8_e4m3 262144 '1 2' '65536 131072 261888'
run_block_ext f8e4df7 'llm-scaler-exp:v1.2.5' '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' '' fp8_e4m3 245760 '1 2' '65536 131072 261888'

# Secondary M1 on llm-scaler-exp:v1.2.6t2 (tq4nc), C 1/2
run_block_ext s2ns  'llm-scaler-exp:v1.2.6t2' 'nospec' '' turboquant_4bit_nc 262144 '1 2' '65536 131072 261888'
run_block_ext s2m4  'llm-scaler-exp:v1.2.6t2' '{"method":"mtp","num_speculative_tokens":4}' '' turboquant_4bit_nc 262144 '1 2' '65536 131072 261888'
run_block_ext s2df7 'llm-scaler-exp:v1.2.6t2' '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' '' turboquant_4bit_nc 262144 '1 2' '65536 131072 261888'

# Secondary M2 on v1.2.6t2 (fp8_e4m3 KV), C 1/2
run_block_ext s2f8ns  'llm-scaler-exp:v1.2.6t2' 'nospec' '' fp8_e4m3 262144 '1 2' '65536 131072 261888'
run_block_ext s2f8m4  'llm-scaler-exp:v1.2.6t2' '{"method":"mtp","num_speculative_tokens":4}' '' fp8_e4m3 262144 '1 2' '65536 131072 261888'
run_block_ext s2f8df7 'llm-scaler-exp:v1.2.6t2' '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' '' fp8_e4m3 245760 '1 2' '65536 131072 261888'

# E5: dflash7+tq4nc retry at 245760 ONLY if master's block failed to boot/health
if grep -qE 'BOOT_FAIL dflash7|HEALTH_TIMEOUT dflash7' "$ROOT/master.log" 2>/dev/null; then
  log "E5 trigger: master dflash7 boot/health failure detected"
  run_block_ext df7r 'llm-scaler-exp:v1.2.5' \
    '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' '' \
    turboquant_4bit_nc 245760 '1' '65536 131072'
else
  log "E5 skipped: master dflash7 block booted"
fi

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_ext.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_ext.out"
fi
log "MASTER_LCE1_EXT_DONE"
