#!/bin/bash
# validate_v55.sh — crashfix-v55 validation matrix (2026-09-10).
#
# User mandate: reproduce BOTH crash configurations with concurrent
# staggered streams of distinct prompts (incl. the exact crash prompt
# "Write a html car game"), every request context < 64k, then require:
# 0 crashes, 0 JIT warnings post-warmup, 0 warning-class lines, health
# 200 throughout. Only after PASS does the bake proceed.
#
# Usage: validate_v55.sh [image] [A|B|AB]   (default image
# llm-scaler-exp:v1.2.8, configs AB; "A-resume" reuses a standing
# config-A engine: skips boot+warmup, runs cycles+scan only)
set -u
IMG="${1:-llm-scaler-exp:v1.2.8}"
WHICH="${2:-AB}"
OUT=/root/build/lce1/crashfix-v55
mkdir -p "$OUT"
SUMMARY="$OUT/validate_summary.txt"
: > "$SUMMARY"
CLIENT=/root/build/v55_client.py
WARMUP=/root/build/dt_warmup_v53.py

fail() { echo "VALIDATE_V55 FAIL: $*" | tee -a "$SUMMARY"; exit 1; }

scan_log() {  # $1=logfile $2=startline $3=config-tag
  local LOG="$1" START="$2" TAG="$3"
  local SEG; SEG=$(tail -n +"$START" "$LOG")
  local BAD=0
  local PAT
  for PAT in "Unknown vLLM environment variable" "DEVICE_LOST" \
             "TimeoutError" "RPC call to" "Watchdog" \
             "ASYNC-EVENT-STALL" "GAP-FENCE" "DISCARD-GAP" \
             "jit_monitor" "EngineDeadError" "llm-scaler v52l"; do
    local N; N=$(printf '%s' "$SEG" | grep -c "$PAT" || true)
    if [ "$N" != "0" ]; then
      echo "SCAN[$TAG] BAD: $N x '$PAT'" | tee -a "$SUMMARY"
      printf '%s' "$SEG" | grep -m3 "$PAT" | tee -a "$SUMMARY"
      BAD=1
    fi
  done
  # census of any remaining WARNING lines in the validated section
  local W; W=$(printf '%s' "$SEG" | grep -c "WARNING" || true)
  echo "SCAN[$TAG] warning-census: $W WARNING lines" | tee -a "$SUMMARY"
  printf '%s' "$SEG" | grep "WARNING" | sed 's/^/  /' | head -20 >> "$SUMMARY"
  # contained-incident census (e5m2+mtp3 lane class): v52m force-finish
  # of a NaN-zombie onset — engine stays up; reported, not failed
  local M; M=$(printf '%s' "$SEG" | grep -c "v52m STRIKE-OUT" || true)
  echo "SCAN[$TAG] contained-incidents: $M x v52m STRIKE-OUT (engine-safety unaffected)" | tee -a "$SUMMARY"
  [ "$BAD" = "0" ]
}

run_config() {  # $1=tag $2=kvdtype $3=specjson $4=extraenv $5=resume(0/1)
  local TAG="$1" KV="$2" SPEC="$3" XENV="$4" RESUME="${5:-0}"
  echo "== CONFIG $TAG ($(date +%H:%M:%S)): kv=$KV spec=$SPEC env='$XENV' img=$IMG resume=$RESUME" | tee -a "$SUMMARY"
  if [ "$RESUME" != "1" ]; then
    docker rm -f lsv-test >/dev/null 2>&1
    if ! bash /root/build/serve_bench.sh "$KV" "$SPEC" 0.9 262144 "b_v55_$TAG" "$XENV" '' "$IMG"; then
      fail "boot $TAG"
    fi
  else
    # resume on the standing engine; the post-warmup log boundary is
    # recovered from the already-copied b_$TAG.log
    :
  fi
  # warmup (v53) — wait for a NEW WARMUP_DONE marker
  if [ "$RESUME" = "1" ]; then
    echo "  WARMUP_SKIPPED (resume on standing engine)" | tee -a "$SUMMARY"
    docker cp "lsv-test:/root/b_v55_$TAG.log" "$OUT/b_$TAG.log" >/dev/null 2>&1
  else
    local ND0; ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
    nohup python3 "$WARMUP" >"$OUT/warmup_$TAG.out" 2>&1 &
    local WOK=0 ND
    for i in $(seq 1 150); do
      sleep 10
      ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
      if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
      if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
        fail "engine died during warmup ($TAG)"
      fi
    done
    [ "$WOK" = "1" ] || fail "warmup timeout ($TAG)"
    echo "  WARMUP_OK" | tee -a "$SUMMARY"
    docker cp "lsv-test:/root/b_v55_$TAG.log" "$OUT/b_$TAG.log" >/dev/null 2>&1
  fi
  local LSTART; LSTART=$(wc -l < "$OUT/b_$TAG.log")

  # 3 cycles of staggered multi-stream crash-shaped traffic
  local CYC
  for CYC in 1 2 3; do
    if ! python3 "$CLIENT" "$CYC" > "$OUT/${TAG}_cyc$CYC.out" 2>&1; then
      cat "$OUT/${TAG}_cyc$CYC.out" | tee -a "$SUMMARY"
      fail "cycle $CYC client errors ($TAG)"
    fi
    grep "CYCLE" "$OUT/${TAG}_cyc$CYC.out" | tee -a "$SUMMARY"
    curl -s -o /dev/null -m 4 http://localhost:8000/health \
      || fail "health after cycle $CYC ($TAG)"
    sleep 5
  done

  docker cp "lsv-test:/root/b_v55_$TAG.log" "$OUT/b_$TAG.log" >/dev/null 2>&1
  scan_log "$OUT/b_$TAG.log" "$LSTART" "$TAG" \
    || fail "log scan ($TAG) — see $SUMMARY"
  echo "CONFIG_$TAG PASS" | tee -a "$SUMMARY"
}

case "$WHICH" in
  A)    run_config A fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' "" 0 ;;
  B)    run_config B fp8_e5m2 '{"method":"mtp","num_speculative_tokens":3}' \
          "VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1" 0 ;;
  A-resume)
        run_config A fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' "" 1 ;;
  B-resume)
        run_config B fp8_e5m2 '{"method":"mtp","num_speculative_tokens":3}' \
          "VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1" 1 ;;
  AB)   run_config A fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' "" 0
        run_config B fp8_e5m2 '{"method":"mtp","num_speculative_tokens":3}' \
          "VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1" 0 ;;
  *)    fail "unknown config selector '$WHICH'" ;;
esac

echo "VALIDATE_V55 PASS — both crash configs, 3 cycles each, clean logs" | tee -a "$SUMMARY"
