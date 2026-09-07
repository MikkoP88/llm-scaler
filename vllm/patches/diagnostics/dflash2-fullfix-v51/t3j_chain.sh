#!/bin/bash
# t3j_chain.sh — boot df7t3j lane (v1.2.5t3j = t3f + v52k scheduler
# boundary protocol v5; v52h worker trim stays reverted).
# Validation targets:
#   - "v52k BOUNDARY-CLAMP" / "v52k BOUNDARY-SKIP" / "v52k BOUNDARY-SPAN"
#     lines fire at boundary zones
#   - ZERO "v52d EMPTY-ROW" / ATTR all-NaN / F8v2 FORCE-FINISH after
#     boundary crossings
#   - ZERO worker deaths
#   - all P1-P8 finish normally (streams crossing 2048 AND 4096 AND the
#     88064 zone)
LOG=/root/build/df7t3j_boot.out
{
bash /root/build/serve_boot_var.sh \
  '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' \
  '' df7t3j \
  '--max-model-len 245760 --gpu-memory-utilization 0.85 --kv-cache-dtype turboquant_4bit_nc' \
  512 llm-scaler-exp:v1.2.5t3j
OK=0
for i in $(seq 1 60); do
  sleep 10
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "HEALTH OK after ~$((i*10))s"; OK=1; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOT_EXIT container died"; break
  fi
done
[ "$OK" = "1" ] || { echo "BOOT_FAILED"; exit 9; }
python3 /root/build/dt_warmup_v51.py > /root/dt_warmup.log 2>&1
echo "WARMUP rc=$? lines=$(wc -l < /root/dt_warmup.log)"
nohup python3 /root/build/dt_loop8t07.py > /root/build/loop8t07_df7t3j.out 2>&1 &
echo "LOOP_LAUNCHED $(date +%H:%M:%S)"
} >> "$LOG" 2>&1