#!/bin/bash
# t3m_chain.sh — boot df7t3m lane (v1.2.5t3m = t3f + crash-4 R1 root
# fix v6'': v52l scheduler geometric clamp/skip; core = v52 R2 overlay
# (EMPTY-SKIP reverted after t3k/t3l request-pause wedge).
# Validation targets:
#   - warmup COMPLETES (t3k/t3l wedged at ~47 tokens; wedge must be gone)
#   - "v52l BOUNDARY-CLAMP"/"BOUNDARY-SKIP"/"BOUNDARY-SPAN-CORNER" fire
#     at 2048/4096/88064 zones only (no 512-zone noise)
#   - ZERO unreaped "v52d EMPTY-ROW" zombies; ZERO worker deaths
#   - all P1-P8 finish normally or client-visible abort; engine SURVIVES
#     the full battery (streams crossing 2048 AND 4096)
LOG=/root/build/df7t3m_boot.out
{
bash /root/build/serve_boot_var.sh \
  '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' \
  '' df7t3m \
  '--max-model-len 245760 --gpu-memory-utilization 0.85 --kv-cache-dtype turboquant_4bit_nc' \
  512 llm-scaler-exp:v1.2.5t3m
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
nohup python3 /root/build/dt_loop8t07.py > /root/build/loop8t07_df7t3m.out 2>&1 &
echo "LOOP_LAUNCHED $(date +%H:%M:%S)"
} >> "$LOG" 2>&1