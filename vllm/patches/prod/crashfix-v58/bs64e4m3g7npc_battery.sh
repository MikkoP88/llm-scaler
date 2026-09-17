#!/bin/bash
# bs64e4m3g7npc_battery.sh — §20 F-bar crash battery, arm 5 (user-directed,
# CONDITIONAL: run only if arm 4 G7NS crashed): e4m3 KV + --block-size 64
# + --gpu-memory-utilization 0.7 + async-scheduling REMOVED +
# --enable-prefix-caching REMOVED (5th standing-flag lever; prefix-cache
# KV block reuse interacts with the bs64 block-table walk). Host MUST be
# freshly rebooted first. Protocol identical: 18-round repro_sustain +
# q33 auto-arm watcher. Verdict: event = 16th §14 datum (prefix-cache
# inert); 18/18 clean = candidate crash-avoidance lane (needs perf
# screen before any promotion).
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.7 no-async-sched no-prefix-cache $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G7NPC.sh BS64E4M3G7NPCB > lce1/boot_BS64E4M3G7NPCB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G7NPCB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G7NPCB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G7NPCB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh); echo pcache_count=\$(grep -c 'enable-prefix-caching' /root/serve_user.sh); grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 18-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g7npcb_watch.out 2>&1 &
nohup bash repro_sustain.sh 18 > lce1/sustain_BS64E4M3G7NPCB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
