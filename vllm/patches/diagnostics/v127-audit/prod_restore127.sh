#!/bin/bash
# prod_restore127.sh — v127-audit closure: restore PROD lane after matrix +
# discriminators. Prod = mtp4 + tq4nc @0.9/262144 on llm-scaler-exp:v1.2.5.
# Battery gate ALREADY PASSED this round on the lane-4 attempt-2 boot
# (6/6 SHAs bit-exact prod refs, ACCEPT 0.614 — REPORT §4.6), so restore =
# teardown p1b + boot + health + warmup + smoke ctxscan only, per §8 posture.
set -u
OUT=/root/build/bench127/prod_restore
mkdir -p "$OUT"
R="$OUT/report.txt"
M=/root/build/bench127/master.log

echo "== PROD_RESTORE teardown+boot $(date +%H:%M:%S)" | tee -a "$R" "$M"
docker rm -f lsv-test >/dev/null 2>&1
sleep 5
if bash /root/build/serve_bench.sh turboquant_4bit_nc '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_prod '' '' > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R" "$M"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi
docker exec lsv-test grep -m2 -E "GPU KV cache size|cache_config" /root/b_prod.log 2>/dev/null | tee -a "$R"

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
python3 /root/build/dt_ctxscan.py prodsmoke > "$OUT/smoke.out" 2>&1 || true
grep CTXSCAN "$OUT/smoke.out" | head -4 | tee -a "$R"

docker cp lsv-test:/root/b_prod.log /root/build/bench127/logs/b_prod.log >/dev/null 2>&1 || true
echo "PROD_RESTORE_DONE $(date +%H:%M:%S) — lane STANDING" | tee -a "$R" "$M"
