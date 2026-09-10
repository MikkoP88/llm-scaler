#!/bin/bash
# master_diag_fp8m4.sh — fp8_e4m3 KV + MTP4 collapse localization sweep (09-09).
#
# Context: campaign lanes f8e4m4/pf8m4/s2f8m4 all collapse at long ctx
# (c1 mean tps 11.02/7.59/5.09 at 64k/128k/262k vs fp8+nospec 29.79/25.91/20.69)
# while acceptance is healthy (~4.9 tok/step) and warmup-window (2k ctx) hits
# ~74 tok/s. Step-time excess ~linear in ctx -> O(ctx) per-step op somewhere in
# the spec-on-fp8 path. Source reading narrows suspects to:
#   S1 target verify (q=5 multi-row) via v51 triton_fp8_mq kernel (fixed SPLITS)
#   S2 drafter loop (3 extra q=1 forwards/step) attention on fp8 draft KV
#      (v51 kernel does NOT cover q=1 by default -> C++ FA2 / ESIMD path)
#   S3 something falling back off the v51 route at long ctx (v33/C++ branch1,
#      convicted at 249.5-4976 ms/step in v50 lineage)
#
# Five boots on llm-scaler-exp:v1.2.5, kv fp8_e4m3, maxlen 262144, C1 only.
#
# NOTE v2 (18:0x): VLLM_SPEC_TIMING=1 was dropped — first attempt (dA, 17:30)
# killed the worker SILENTLY on the very first spec request (58-token warmup
# prefill, step_counter=0; EngineCore died of shm-broadcast TimeoutError, no
# worker traceback = hard device-level abort, presumably torch.xpu.Event
# timing inside/around XPU-graph capture). The v20 A2 instrumentation is
# unusable on this lineage; attribution relies on the knob A/Bs instead.
#
#   dA-base : mtp4 baseline, lens 2048..262144  -> knee + per-segment attribution
#   dB-sp64 : mtp4 + VLLM_FP8MQ_SPLITS=64        -> knob sensitivity: if tps
#             changes vs dA, the v51 kernel IS the active verify path
#   dC-mq0  : mtp4 + VLLM_XPU_FP8_MQ=0 (v33 route), len 16384 R1 only
#             -> if far worse than dA@16k, v51 kernel was active (and is the
#                bottleneck); if identical, v51 was NOT the active path
#   dD-mtp1 : mtp k=1 (no draft-loop iterations, verify q=2), 64k/131k
#             -> isolates draft-loop cost vs multi-row-verify cost
#   dE-q1   : mtp4 + VLLM_XPU_FP8_MQ_Q1=1 (route q=1 through v51 kernel too)
#             -> tests the C++/ESIMD q=1 draft-attention hypothesis
#
# Ends with prod_restore127.sh. Logs to /root/build/lce1/master_diag.log.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
REPEATS=3
mkdir -p "$ROOT"

log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG start (pid $$)"

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
    if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
      log "WARMUP_ABORT $MODE — container died during warmup"
      return 1
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

EE_TIMING=''

# dA: baseline knee sweep (261888 = validated campaign sentinel; shorter lens
# are new — if the suite rejects any it just rc43-skips that length)
run_block_diag dA-base 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' "$EE_TIMING" fp8_e4m3 262144 '1' \
  '2048 8192 16384 32768 65536 131072 261888'

# dB: SPLITS=64 knob A/B (v52 measured +34% @32k with this knob)
run_block_diag dB-sp64 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_FP8MQ_SPLITS=64' \
  fp8_e4m3 262144 '1' '16384 32768 65536 131072'

# dD: mtp k=1 — no draft-loop iterations, verify q=2 only
run_block_diag dD-mtp1 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":1}' '' fp8_e4m3 262144 '1' \
  '65536 131072'

# dE: route q=1 through the v51 kernel as well (draft steps + any q1 forward)
run_block_diag dE-q1 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_XPU_FP8_MQ_Q1=1' \
  fp8_e4m3 262144 '1' '65536 131072'

# dC: kill the v51 route (v33 fallback) — 16k only, R1 (may be very slow)
run_block_diag dC-mq0 'llm-scaler-exp:v1.2.5' \
  '{"method":"mtp","num_speculative_tokens":4}' 'VLLM_XPU_FP8_MQ=0' \
  fp8_e4m3 262144 '1' '16384' 1

log "== restoring prod lane (prod_restore127.sh)"
if bash /root/build/prod_restore127.sh >> "$ROOT/prod_restore_diag.out" 2>&1; then
  log "PROD_RESTORE_OK"
else
  log "PROD_RESTORE_FAIL — see $ROOT/prod_restore_diag.out"
fi
log "MASTER_DIAG_DONE"
