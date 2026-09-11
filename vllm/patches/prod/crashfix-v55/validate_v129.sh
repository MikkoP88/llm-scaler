#!/bin/bash
# validate_v129.sh — validate baked v1.2.9 (crashfix-v55.2 progressive-
# wait) on the prod lane (config A: fp8_e4m3 + mtp4 @0.9/262144):
# boot -> FULL warmup v53 -> 2 v55_client cycles -> ctxscan -> genspeed
# -> engine-log census. PASS = zero crash classes, zero fence trips,
# v1.2.7-class throughput (ctxscan ~60+ @2k, genspeed ~130+ agg).
set -u
D=/root/build/lce1
R=$D/validate_v129.out
: > "$R"

docker rm -f lsv-test >/dev/null 2>&1
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_val_v129 '' '' llm-scaler-exp:v1.2.9 > "$D/validate_v129_boot.out" 2>&1; then
  echo "BOOT_OK $(date +%H:%M:%S)" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R"; tail -20 "$D/validate_v129_boot.out" | tee -a "$R"; exit 1
fi
HOK=0
for i in $(seq 1 60); do sleep 10; curl -s -o /dev/null -m 4 http://localhost:8000/health && { HOK=1; break; }; done
[ "$HOK" = 1 ] && echo "HEALTH_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo HEALTH_TIMEOUT | tee -a "$R"; exit 1; }

ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
python3 /root/build/dt_warmup_v53.py > "$D/validate_v129_warmup.out" 2>&1 || true
ND1=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
echo "warmup markers $ND0 -> $ND1 $(date +%H:%M:%S)" | tee -a "$R"
grep -E "p13-longdecode|p14:" /root/dt_warmup.log | tail -2 | tee -a "$R"
docker ps --format '{{.Names}}' | grep -q '^lsv-test$' || { echo ENGINE_DEAD_DURING_WARMUP | tee -a "$R"; exit 1; }

for SEED in 101 102; do
  if python3 /root/build/v55_client.py "$SEED" >> "$R" 2>&1; then
    echo "CYCLE $SEED PASS $(date +%H:%M:%S)" | tee -a "$R"
  else
    echo "CYCLE $SEED FAIL $(date +%H:%M:%S)" | tee -a "$R"
  fi
done

python3 /root/build/dt_ctxscan.py valv129 >> "$R" 2>&1 || true
python3 /root/build/bench_genspeed.py 4 1024 3 >> "$R" 2>&1 || true

docker cp lsv-test:/root/b_val_v129.log "$D/validate_v129_engine.log" >/dev/null 2>&1
BAD=$(grep -cE "Unknown vLLM environment variable|DEVICE_LOST|TimeoutError|RPC call to|Watchdog|GAP-FENCE|DISCARD-GAP|jit_monitor|EngineDeadError|llm-scaler v52l" "$D/validate_v129_engine.log" || true)
STALL=$(grep -c "ASYNC-EVENT-STALL" "$D/validate_v129_engine.log" || true)
W=$(grep -c WARNING "$D/validate_v129_engine.log" || true)
echo "census: crash-classes=$BAD fence-stalls=$STALL total-WARNING=$W" | tee -a "$R"
echo "VALIDATE_V129_DONE $(date +%H:%M:%S)" | tee -a "$R"
