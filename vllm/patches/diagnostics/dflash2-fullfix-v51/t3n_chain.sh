#!/bin/bash
# t3n_chain.sh — boot df7t3n lane (v1.2.5t3n = t3m + crash-4 ESCALATION
# closure: GMR v52n draft-input clamp + ASCH v52m zero-emission strike-out
# + skipped-zombie watchdog).
# Validation targets:
#   - warmup COMPLETES (wedge still gone)
#   - "v52l BOUNDARY-*" telemetry at 2048/4096/88064 zones only
#   - boot log shows "v52m zero-emission strikes=3" F8 guard line
#   - ZERO worker deaths; ZERO EngineDeadError; engine SURVIVES full
#     battery even if a NaN-zombie forms (t3m died at P5/4096)
#   - any zombie -> "v52m STRIKE-OUT (strike-out)" or "(watchdog)" line
#     + client-visible abort; P6 (and later probes) still complete
#   - P2/P4 tok/s within noise of t3m (clamp cost negligible)
LOG=/root/build/df7t3n_boot.out
{
bash /root/build/serve_boot_var.sh \
  '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' \
  '' df7t3n \
  '--max-model-len 245760 --gpu-memory-utilization 0.85 --kv-cache-dtype turboquant_4bit_nc' \
  512 llm-scaler-exp:v1.2.5t3n
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
nohup python3 /root/build/dt_loop8t07.py > /root/build/loop8t07_df7t3n.out 2>&1 &
echo "LOOP_LAUNCHED $(date +%H:%M:%S)"
} >> "$LOG" 2>&1
