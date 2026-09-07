#!/bin/bash
# repro_t3g.sh — reboot v1.2.5t3g lane, health-wait, fire warmup probes,
# then immediately docker-cp the engine log (container may die).
bash /root/build/serve_boot_var.sh \
  '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' \
  '' df7t3g \
  '--max-model-len 245760 --gpu-memory-utilization 0.85 --kv-cache-dtype turboquant_4bit_nc' \
  512 llm-scaler-exp:v1.2.5t3g > /root/build/repro_boot.out 2>&1
OK=0
for i in $(seq 1 45); do
  sleep 10
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "HEALTH OK after ~$((i*10))s"; OK=1; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOT_EXIT container died"; break
  fi
done
if [ "$OK" = "1" ]; then
  python3 /root/build/dt_warmup_v51.py > /root/build/repro_warmup.out 2>&1
  echo "WARMUP rc=$?"
  # capture engine state right after (alive or dead)
  sleep 3
fi
docker cp lsv-test:/root/df7t3g.log /root/build/repro_engine.log 2>/dev/null \
  && echo "ENGINE_LOG_SAVED $(wc -l < /root/build/repro_engine.log) lines" \
  || echo "ENGINE_LOG_UNAVAILABLE"
docker ps -a --format '{{.Names}} {{.Status}}' | head -2
