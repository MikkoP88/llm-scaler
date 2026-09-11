#!/bin/bash
# scratch_v56.sh — scratch-test the v56 progressive-wait fix on a live
# v1.2.8 boot: boot -> health -> patch gmr in-container -> docker restart
# -> health -> FULL warmup v53 -> ctxscan -> genspeed. Success criterion:
# ctxscan ~70 tps @2k / genspeed ~140 agg (v1.2.7-class), zero deaths.
set -u
D=/root/build/lce1
R=$D/scratch_v56.out
: > "$R"

docker rm -f lsv-test >/dev/null 2>&1
if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
    0.9 262144 b_scr_v56 '' '' llm-scaler-exp:v1.2.8 > "$D/scratch_v56_boot.out" 2>&1; then
  echo "BOOT_OK $(date +%H:%M:%S)" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R"; tail -20 "$D/scratch_v56_boot.out" | tee -a "$R"; exit 1
fi
HOK=0
for i in $(seq 1 60); do sleep 10; curl -s -o /dev/null -m 4 http://localhost:8000/health && { HOK=1; break; }; done
[ "$HOK" = 1 ] && echo "HEALTH1_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo HEALTH1_TIMEOUT | tee -a "$R"; exit 1; }

# v56 patch in-container + verify + restart
docker cp /root/build/v56/patch_progressive_wait.py lsv-test:/tmp/patch_progressive_wait.py
docker exec lsv-test python3 /tmp/patch_progressive_wait.py --sp /opt/venv/lib/python3.12/site-packages 2>&1 | tee -a "$R"
docker exec lsv-test grep -c "_el < 0.02" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py | tee -a "$R"
docker restart lsv-test >/dev/null
HOK=0
for i in $(seq 1 90); do sleep 10; curl -s -o /dev/null -m 4 http://localhost:8000/health && { HOK=1; break; }; done
[ "$HOK" = 1 ] && echo "HEALTH2_OK (patched boot) $(date +%H:%M:%S)" | tee -a "$R" || { echo HEALTH2_TIMEOUT | tee -a "$R"; exit 1; }
docker exec lsv-test grep -m1 "_el < 0.02" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py >/dev/null && echo "PATCH_PERSISTED" | tee -a "$R"

ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
python3 /root/build/dt_warmup_v53.py > "$D/scratch_v56_warmup.out" 2>&1 || true
ND1=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
echo "warmup markers $ND0 -> $ND1 $(date +%H:%M:%S)" | tee -a "$R"
grep -E "p13-longdecode|p14:" /root/dt_warmup.log | tail -2 | tee -a "$R"

python3 /root/build/dt_ctxscan.py scrv56 >> "$R" 2>&1 || true
python3 /root/build/bench_genspeed.py 4 1024 3 >> "$R" 2>&1 || true
docker cp lsv-test:/root/b_scr_v56.log "$D/scratch_v56_engine.log" >/dev/null 2>&1
echo "SCRATCH_V56_DONE $(date +%H:%M:%S)" | tee -a "$R"
