#!/bin/bash
# prod_restore_v7.sh — stand production on llm-scaler-exp:v1.2.8
# (crashfix-v55) after the 2026-09-10 crash pair.
#
# Lane: fp8_e4m3 KV + MTP k=4 @ temp 0.9 / maxlen 262144 — the config-A
# lane of VALIDATE_V55 (fully clean: 3 cycles 5/5, 0 WARNING lines).
# e5m2 must be served with mtp4 (v51 pairing), NEVER mtp3 (crash-2 boot
# combo; NaN-onset lane, see crashfix-v55/README.md).
#
# Warmup: dt_warmup_v53.py (p1-p14 incl. dress-rehearsal cycle) —
# skipping warmup is the root cause of BOTH 2026-09-10 crashes.
set -u
OUT=/root/build/lce1/crashfix-v55/prod_v128
R=/root/build/prod_restore_v7.out
M=/root/build/lce1/prod_marker.txt
mkdir -p "$OUT"
: > "$R"

docker rm -f lsv-test >/dev/null 2>&1
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_prod_v128 '' '' llm-scaler-exp:v1.2.8 > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi
docker images llm-scaler-exp:v1.2.8 --format '{{.ID}}' | head -1 | tee -a "$R"
docker exec lsv-test grep -m2 -E "GPU KV cache size|cache_config" /root/b_prod_v128.log 2>/dev/null | tee -a "$R"

HOK=0
for i in $(seq 1 60); do
  sleep 10
  if curl -s -o /dev/null -m 4 http://localhost:8000/health; then HOK=1; break; fi
done
[ "$HOK" = "1" ] && echo "HEALTH_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo "HEALTH_TIMEOUT" | tee -a "$R"; exit 1; }

# warmup v53 (p1-p14; p14 dress-rehearsal guarantees the multi-stream
# crash-traffic shape class is JIT-compiled before any user traffic)
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v53.py >"$OUT/warmup.out" 2>&1 &
WOK=0
for i in $(seq 1 150); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "ENGINE_DIED_DURING_WARMUP" | tee -a "$R"; exit 1
  fi
done
[ "$WOK" = "1" ] && echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo "WARMUP_TIMEOUT" | tee -a "$R"; exit 1; }
grep -E "p12-htmlgame|p14:" "$OUT/warmup.out" | tail -2 | tee -a "$R"

# smoke: ctxscan ladder on the standing lane (decode health check)
python3 /root/build/dt_ctxscan.py prodsmoke_v128 > "$OUT/smoke.out" 2>&1 || true
grep CTXSCAN "$OUT/smoke.out" | head -4 | tee -a "$R"

docker cp lsv-test:/root/b_prod_v128.log "$OUT/b_prod_v128.log" >/dev/null 2>&1 || true
W=$(grep -c WARNING "$OUT/b_prod_v128.log" || true)
echo "prod-boot warning-census: $W WARNING lines (expect 0)" | tee -a "$R"

echo "PROD_V128_STANDING $(date +%H:%M:%S) — image llm-scaler-exp:v1.2.8 (crashfix-v55), fp8_e4m3 + mtp4 @0.9/262144, warmup v53 complete" | tee -a "$R"
