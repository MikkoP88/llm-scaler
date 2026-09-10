#!/bin/bash
# prod_restore_v5.sh — stand PROD on the fan-out fix: fp8_e4m3 KV + MTP k=4
# @0.9/262144 on llm-scaler-exp:fp8-mtp4-v5 (v1.2.5 lineage + SPLITS=16 +
# flash q=1 fan-out verify, BAKED DEFAULT ON — no env knob needed).
# Gates this lane now beats at every measured length (dS-fanout A/B):
#   fp8+nospec 29.79/25.91/20.69 -> fan-out 52.63 @64k, 23.47 @262k
#   tq4nc+mtp4 25.06/15.31/7.50  -> fan-out 52.63 @64k, 23.47 @262k
# tq4nc lane untouched by the route (gate requires fp8 KV dtype).
set -u
OUT=/root/build/lce1/prod_restore_v5
mkdir -p "$OUT"
R="$OUT/report.txt"
M=/root/build/lce1/master_diag.log

echo "== PROD_V5 restore teardown+boot $(date +%H:%M:%S)" | tee -a "$R" "$M"
docker rm -f lsv-test >/dev/null 2>&1
sleep 5
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_prod_v5 '' '' llm-scaler-exp:fp8-mtp4-v5 > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R" "$M"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi
docker exec lsv-test grep -m2 -E "GPU KV cache size|cache_config" /root/b_prod_v5.log 2>/dev/null | tee -a "$R"

HOK=0
for i in $(seq 1 60); do
  sleep 10
  if curl -s -o /dev/null -m 4 http://localhost:8000/health; then HOK=1; break; fi
done
[ "$HOK" = "1" ] && echo "HEALTH_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo "HEALTH_TIMEOUT" | tee -a "$R" "$M"; exit 1; }

ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 90); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
done
[ "$WOK" = "1" ] && echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$R" || echo "WARMUP_TIMEOUT (lane stands, investigate)" | tee -a "$R" "$M"

# smoke: ctxscan ladder on the standing lane (decode health check)
python3 /root/build/dt_ctxscan.py prodsmoke_v5 > "$OUT/smoke.out" 2>&1 || true
grep CTXSCAN "$OUT/smoke.out" | head -4 | tee -a "$R"

docker cp lsv-test:/root/b_prod_v5.log "$OUT/b_prod_v5.log" >/dev/null 2>&1 || true
echo "PROD_V5_STANDING $(date +%H:%M:%S) — image llm-scaler-exp:fp8-mtp4-v5, fp8_e4m3 + mtp4 @0.9/262144" | tee -a "$R" "$M"
