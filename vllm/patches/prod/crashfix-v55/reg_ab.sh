#!/bin/bash
# reg_ab.sh — generation-speed A/B: v1.2.8 vs v1.2.7, identical lane
# (fp8_e4m3 + mtp k=4 @ 0.9 / 262144 via serve_bench.sh), each leg:
# boot -> health -> FULL warmup v53 (p1-p14) -> ctxscan ladder ->
# genspeed 4x1024x3. Answers the 2026-09-11 "v1.2.8 speed drop" report
# with warm-vs-warm numbers (the 04:00 v1.2.8 boot was cold: no warmup,
# manual /bin/bash launch, env soup -> not a valid comparison).
set -u
D=/root/build/lce1

leg() {
  local VER=$1 TAG=$2
  local R=$D/reg_ab_${VER}.out
  : > "$R"
  docker rm -f lsv-test >/dev/null 2>&1
  if bash /root/build/serve_bench.sh fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' \
      0.9 262144 "$TAG" '' '' "llm-scaler-exp:${VER}" > "$D/reg_ab_${VER}_boot.out" 2>&1; then
    echo "BOOT_OK ${VER} $(date +%H:%M:%S)" | tee -a "$R"
  else
    echo "BOOT_FAIL ${VER}" | tee -a "$R"
    tail -20 "$D/reg_ab_${VER}_boot.out" | tee -a "$R"
    return 1
  fi
  local HOK=0 i
  for i in $(seq 1 60); do
    sleep 10
    curl -s -o /dev/null -m 4 http://localhost:8000/health && { HOK=1; break; }
  done
  if [ "$HOK" = 1 ]; then
    echo "HEALTH_OK ${VER} $(date +%H:%M:%S)" | tee -a "$R"
  else
    echo "HEALTH_TIMEOUT ${VER}" | tee -a "$R"
    return 1
  fi
  local ND0 ND1
  ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  echo "warmup_markers_before=${ND0}" | tee -a "$R"
  python3 /root/build/dt_warmup_v53.py > "$D/reg_ab_${VER}_warmup.out" 2>&1 || true
  ND1=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  echo "warmup_markers_after=${ND1} ${VER} $(date +%H:%M:%S)" | tee -a "$R"
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "ENGINE_DEAD_AFTER_WARMUP ${VER}" | tee -a "$R"
  fi
  python3 /root/build/dt_ctxscan.py "reg${VER}" >> "$R" 2>&1 || true
  python3 /root/build/bench_genspeed.py 4 1024 3 >> "$R" 2>&1 || true
  docker cp "lsv-test:/root/${TAG}.log" "$D/reg_ab_${VER}_engine.log" >/dev/null 2>&1
  echo "LEG_DONE ${VER} $(date +%H:%M:%S)" | tee -a "$R"
}

leg v1.2.8 b_reg_v128
leg v1.2.7 b_reg_v127
echo "REG_AB_ALL_DONE $(date +%H:%M:%S)" | tee -a "$D/reg_ab_v128.out"
