#!/bin/bash
# v24r_screen_run.sh — §24 R: block-table diagnostic probe.
# Lane G85NSMB4KV24R: PC ON + patch_v24r (read-only HIT/FREE table dumps).
# Protocol: pause watchdog -> teardown -> boot -> 15min boosted settle ->
# bs64_screen_x (f8ref + q17 + bench3x cold/warm) -> dt_halfprobe (65k
# prompt sharing only first ~32k with cached prefix, x2) -> export [v24r]
# events from serve_full.log.
cd /root/build
set -u
LBL=G85NSMB4KV24R
BOOT=repro_bootBS64_E4M3_G85NS_MB4K_V24R.sh

echo "[V24R] pause watchdog + teardown $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[V24R] boot $BOOT $(date +%H:%M:%S)"
bash "$BOOT" "$LBL" > "lce1/boot_$LBL.out" 2>&1 || true
if ! grep -q HEALTH_OK "lce1/boot_$LBL.out"; then
  echo "BOOT_FAIL [$LBL]"; tail -25 "lce1/boot_$LBL.out"; exit 1
fi
grep -m1 HEALTH_OK "lce1/boot_$LBL.out"
docker exec lsv-test sh -c "grep -oE -- '--block-size [0-9]+|--gpu-memory-utilization [0-9.]+|--max-num-batched-tokens [0-9]+' /root/serve_user.sh; echo pc_count=\$(grep -c 'enable-prefix-caching' /root/serve_user.sh); echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh)"
grep -E 'V24R_INSTALLED|patched|installed' lce1/v24r_apply_bootV24R.log
docker exec lsv-test grep -c "llm-scaler v24r" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py

echo "[V24R] warm-settle ~15min boosted $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[V24R] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"

bash bs64_screen_x.sh "$LBL" 52f598e7d38a
python3 dt_halfprobe.py > "lce1/halfprobe_$LBL.out" 2>&1
tail -2 "lce1/halfprobe_$LBL.out"
docker exec lsv-test grep -E '\[v24r\] (HIT|FREE)' /root/serve_full.log > "lce1/v24r_events_$LBL.out" || true
echo "v24r events exported: $(wc -l < lce1/v24r_events_$LBL.out)"
echo "V24R_SCREEN_DONE $LBL $(date +%H:%M:%S)"
