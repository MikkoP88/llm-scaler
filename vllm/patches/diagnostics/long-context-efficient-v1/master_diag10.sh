#!/bin/bash
# master_diag10.sh — v5 soak + cross-boot determinism. Final case axes for
# "superior on ANY case, no degradation".
#
# diag9 verdicts feeding this chain:
#   * conc: mtp4 >= nospec at every measured case (+119% 2k C1, +47% 64k
#     C2, +44% 64k C8, +6-20% 262k C2); acceptance HOLDS at C8 (4.4).
#   * e5m2 (bypass knob): 64.8-81.1 / 58.8-61.5 / 31.3-32.7 @8k/64k/262k —
#     FASTER than e4m3 (slope 0.283 vs 0.413; no descale path), det.,
#     answers validated. e4m3 stays the standing prod lane (conservative,
#     no env knob); e5m2 recorded as the validated faster variant.
# This chain:
#   Phase 1 SOAK — 10 cycles against the STANDING prod lane (no reboot):
#     per cycle: health poll + engine-log fault scan + cells 2k C2,
#     64k C2, 262k C1 (mixed load, ~10 min/cycle ≈ 1h45m). Abort +
#     preserve on health fail or new asserts.
#   Phase 2 DETERMINISM — reboot (dZ-det) + 64k C1 reps 2 SAME SEED as
#     dV5-cert; compare completion tokens / step-ms / answers cross-boot.
# Ends with prod_restore_v5.sh + MASTER_DIAG10_DONE.
set -u
ROOT=/root/build/lce1
M=/root/build/lce1/master_diag.log
BASESEED=20260908
log() { echo "[$(date +%m-%d' '%H:%M:%S)] $*" | tee -a "$M"; }

log "== DIAG10 queued (v5 soak + cross-boot determinism)"
DONE=0
for i in $(seq 1 240); do
  if grep -q 'MASTER_DIAG9B_DONE$' "$M" 2>/dev/null; then DONE=1; break; fi
  sleep 20
done
[ "$DONE" = "1" ] || { log "DIAG10_WAIT_TIMEOUT — aborting"; exit 1; }
sleep 20

# ensure the lane is standing on v5 (diag9b restored it; verify)
CUR=$(docker ps --format '{{.Names}} {{.Image}}' | grep '^lsv-test ' | awk '{print $2}')
if [ "$CUR" != "llm-scaler-exp:fp8-mtp4-v5" ] || ! curl -s -o /dev/null -m 4 http://localhost:8000/health; then
  log "lane not standing on v5 (cur='$CUR') — restoring"
  bash /root/build/prod_restore_v5.sh >> "$ROOT/prod_restore_v5_diag10.out" 2>&1 \
    || { log "PROD_RESTORE_V5_FAIL — aborting"; log "MASTER_DIAG10_DONE"; exit 1; }
fi
log "SOAK lane confirmed: $(docker ps --format '{{.Names}} {{.Image}}' | grep '^lsv-test ')"

ENGRUN="/root/b_prod_v5.log"
fault_scan() {
  docker exec lsv-test grep -a -c -E "AssertionError|CRITICAL|Traceback \(most recent" "$ENGRUN" 2>/dev/null || echo 0
}

ABORT=0
for CYC in $(seq 1 10); do
  if ! curl -s -o /dev/null -m 4 http://localhost:8000/health; then
    log "SOAK cycle $CYC HEALTH_FAIL — aborting, engine left for inspection"; ABORT=1; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    log "SOAK cycle $CYC CONTAINER_DEAD — aborting"; ABORT=1; break
  fi
  F0=$(fault_scan)
  log "SOAK cycle $CYC start (faults_so_far=$F0)"
  T0=$(date +%s)
  for SPEC in "soakA 2048 2" "soakA 65536 2" "soakA 261888 1"; do
    set -- $SPEC
    MODE="$1"; LEN="$2"; C="$3"
    SEED=$((BASESEED + LEN + C + CYC))
    ENGINE_LOG="$ENGRUN" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
      "$ROOT" "$MODE" "$LEN" "$C" 1 "$SEED" \
      >> "$ROOT/suite_${MODE}.jsonl" 2>> "$ROOT/suite_${MODE}.err"
    RC=$?
    if [ "$RC" -ne 0 ]; then
      log "SOAK cycle $CYC cell $MODE len=$LEN rc=$RC — aborting"; ABORT=1; break
    fi
  done
  [ "$ABORT" = "1" ] && break
  F1=$(fault_scan)
  T1=$(date +%s)
  log "SOAK cycle $CYC done in $((T1-T0))s (faults_now=$F1)"
  if [ "$F1" != "$F0" ]; then
    log "SOAK cycle $CYC NEW_FAULTS detected ($F0 -> $F1) — aborting, engine left for inspection"; ABORT=1; break
  fi
done

if [ "$ABORT" = "0" ]; then
  log "SOAK COMPLETE 10/10 cycles, zero new faults"
  log "SOAK_CANCEL_PROBE start"
  ENGINE_LOG="$ENGRUN" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
    "$ROOT" soakCancel $((BASESEED + 998)) >> "$ROOT/cancel_soak.jsonl" 2>&1
  log "SOAK_CANCEL_PROBE rc=$?"
else
  log "SETTLE 90s after soak abort (protect driver state)"; sleep 90
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

# Phase 2: cross-boot determinism — same seed formula as dV5-cert 64k C1
log "== BLOCK dZ-det start (cross-boot determinism, seed formula = dV5-cert)"
docker rm -f lsv-test >/dev/null 2>&1
sleep 20
if ! bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_lce1_dZ-det '' '' llm-scaler-exp:fp8-mtp4-v5 > "$ROOT/boot_dZ-det.out" 2>&1; then
  log "BOOT_FAIL dZ-det"
else
  log "BOOT_OK dZ-det"
  if wait_health; then log "HEALTH_OK dZ-det"; else log "HEALTH_TIMEOUT dZ-det"; fi
  if wait_warmup; then log "WARMUP_OK dZ-det"; else log "WARMUP_TIMEOUT dZ-det"; fi
  ENGINE_LOG="/root/b_lce1_dZ-det.log" LCE1_DOCKER=lsv-test python3 /root/build/nlp_suite.py \
    "$ROOT" dZ-det 65536 1 2 $((BASESEED + 65536 + 1)) \
    >> "$ROOT/suite_dZ-det.jsonl" 2>> "$ROOT/suite_dZ-det.err"
  log "DZ_DET rc=$?"
  docker cp lsv-test:/root/b_lce1_dZ-det.log "$ROOT/b_lce1_dZ-det.log" >/dev/null 2>&1 || true
  gzip -f "$ROOT/b_lce1_dZ-det.log" 2>/dev/null || true
fi
log "== BLOCK dZ-det end"

log "== restoring prod lane on v5 (prod_restore_v5.sh)"
if bash /root/build/prod_restore_v5.sh >> "$ROOT/prod_restore_v5_diag10.out" 2>&1; then
  log "PROD_RESTORE_V5_OK"
else
  log "PROD_RESTORE_V5_FAIL — see $ROOT/prod_restore_v5_diag10.out"
fi
log "MASTER_DIAG10_DONE"
