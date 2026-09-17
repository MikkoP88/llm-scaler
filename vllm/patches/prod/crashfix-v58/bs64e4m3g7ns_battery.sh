#!/bin/bash
# bs64e4m3g7ns_battery.sh — §20 F-bar crash battery, arm 4 (user-directed,
# CONDITIONAL: run only if arm 3 GMU7 crashed): e4m3 KV + --block-size 64
# + --gpu-memory-utilization 0.7 + async-scheduling REMOVED (standing
# configs ran --async-scheduling since lineage start; nosched = first
# lever aimed at the convicted trigger profile itself — the V1 overlap
# scheduler is what sustains max-rate tiny-kernel dispatch). Host MUST
# be freshly rebooted first. Protocol identical: 18-round repro_sustain
# + q33 auto-arm watcher. Verdict: event = 15th §14 datum (nosched
# inert); 18/18 clean = first crash-free battery since Q31 + the first
# candidate crash-avoidance lane ever (then needs perf screen).
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.7 no-async-sched $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G7NS.sh BS64E4M3G7NSB > lce1/boot_BS64E4M3G7NSB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G7NSB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G7NSB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G7NSB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; grep -c 'async-scheduling' /root/serve_user.sh; grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 18-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g7nsb_watch.out 2>&1 &
nohup bash repro_sustain.sh 18 > lce1/sustain_BS64E4M3G7NSB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
