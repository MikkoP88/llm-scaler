#!/bin/bash
# prod_restore_v6.sh — stand PROD on the v6 production image: fp8_e4m3 KV
# + MTP k=4 @0.9/262144 on llm-scaler-exp:fp8-mtp4-v6 (v5 lineage + baked
# VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 — provably inert on this lane, certified
# byte-identical to dV5-cert by master_diag11 dW6-e4m3; the e5m2 lane is
# selectable at serve time with --kv-cache-dtype fp8_e5m2, NO env needed).
# Rollback: prod_restore_v5.sh (v5 image) or prod_restore127.sh (tq4nc).
# PRODUCTION TAG: llm-scaler-exp:v1.2.7 (dual-tag alias of
# fp8-mtp4-v6, same image ID 7cf3d51cfc60…).
set -u
OUT=/root/build/lce1/prod_restore_v6
mkdir -p "$OUT"
R="$OUT/report.txt"
M=/root/build/lce1/master_diag.log

echo "== PROD_V6 restore teardown+boot $(date +%H:%M:%S)" | tee -a "$R" "$M"
docker rm -f lsv-test >/dev/null 2>&1
sleep 5
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_prod_v6 '' '' llm-scaler-exp:fp8-mtp4-v6 > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R" "$M"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi
docker exec lsv-test grep -m2 -E "GPU KV cache size|cache_config" /root/b_prod_v6.log 2>/dev/null | tee -a "$R"

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
python3 /root/build/dt_ctxscan.py prodsmoke_v6 > "$OUT/smoke.out" 2>&1 || true
grep CTXSCAN "$OUT/smoke.out" | head -4 | tee -a "$R"

docker cp lsv-test:/root/b_prod_v6.log "$OUT/b_prod_v6.log" >/dev/null 2>&1 || true
echo "PROD_V6_STANDING $(date +%H:%M:%S) — image llm-scaler-exp:fp8-mtp4-v6, fp8_e4m3 + mtp4 @0.9/262144 (e5m2 lane: --kv-cache-dtype fp8_e5m2, no env)" | tee -a "$R" "$M"
