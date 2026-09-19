#!/bin/bash
# coh2_screen_run.sh — §24 S: arena copy-on-hit (COH v2) validation screen.
# Lane G85NSMB4KCOH2: PC ON + patch_v24pc2 (arena + SchedulerOutput IPC +
# worker drain). Protocol: pause watchdog -> teardown -> boot -> 15min
# boosted settle -> bs64_screen_x (f8ref + q17 + bench3x cold/warm) ->
# dt_halfprobe (x2) -> coh2_sanity (65k cold-vs-warm greedy text equality)
# -> export [v24pc2]/[v24pc2w] events + arena leak check.
cd /root/build
set -u
LBL=G85NSMB4KCOH2
BOOT=repro_bootBS64_E4M3_G85NS_MB4K_COH2.sh

echo "[COH2] pause watchdog + teardown $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[COH2] boot $BOOT $(date +%H:%M:%S)"
bash "$BOOT" "$LBL" > "lce1/boot_$LBL.out" 2>&1 || true
if ! grep -q HEALTH_OK "lce1/boot_$LBL.out"; then
  echo "BOOT_FAIL [$LBL]"; tail -25 "lce1/boot_$LBL.out"; exit 1
fi
grep -m1 HEALTH_OK "lce1/boot_$LBL.out"
docker exec lsv-test sh -c "grep -oE -- '--block-size [0-9]+|--gpu-memory-utilization [0-9.]+|--max-num-batched-tokens [0-9]+' /root/serve_user.sh; echo pc_count=\$(grep -c 'enable-prefix-caching' /root/serve_user.sh); echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh)"
grep -E 'V24PC2_INSTALLED|patched|installed' lce1/v24pc2_apply_bootCOH2.log
docker exec lsv-test grep -c "llm-scaler v24pc2" /opt/venv/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py

echo "[COH2] warm-settle ~15min boosted $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[COH2] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"
grep -m2 '\[v24pc2\] ARENA' lce1/boot_$LBL.out 2>/dev/null || docker exec lsv-test grep -m2 '\[v24pc2\] ARENA' /root/serve_full.log 2>/dev/null || true

bash bs64_screen_x.sh "$LBL" 52f598e7d38a
python3 dt_halfprobe.py > "lce1/halfprobe_$LBL.out" 2>&1
tail -2 "lce1/halfprobe_$LBL.out"
python3 coh2_sanity.py > "lce1/sanity_$LBL.out" 2>&1
tail -6 "lce1/sanity_$LBL.out"
docker exec lsv-test grep -E '\[v24pc2\] |\[v24pc2w\] ' /root/serve_full.log > "lce1/v24pc2_events_$LBL.out" || true
echo "v24pc2 events exported: $(wc -l < lce1/v24pc2_events_$LBL.out)"
echo "arena hits=$(grep -c 'HIT2' lce1/v24pc2_events_$LBL.out) frees=$(grep -c 'FREE2' lce1/v24pc2_events_$LBL.out) drains=$(grep -c 'drained' lce1/v24pc2_events_$LBL.out)"
echo "arena_free tail: $(grep 'FREE2' lce1/v24pc2_events_$LBL.out | tail -3 | grep -oE 'af=[0-9]+' | tr '\n' ' ')"
echo "COH2_SCREEN_DONE $LBL $(date +%H:%M:%S)"
