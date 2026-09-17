#!/bin/bash
# bs64e4m3g9ns_battery.sh — §20 F-bar crash battery, arm 6 (user-directed,
# CONDITIONAL: run only if arm 4 G7NS finished 18/18 crash-free):
# CONFIRMATION ARM — identical to the first crash-free config (e4m3 KV +
# --block-size 64 + async-scheduling REMOVED) EXCEPT gpu-memory-utilization
# back to standing 0.9 (isolate which variable carried the clean run), and
# LONGER: 30 rounds (~4.5h at no-async pacing > Q31's 4h25m longest clean
# run ever). Host MUST be freshly rebooted first. Verdict: event =
# gmu-interaction finding (0.7 protective OR 18/18 was luck); 30/30 clean
# = nosched confirmed crash-avoidance candidate at standing memory.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.9 no-async-sched LONG $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G9NS.sh BS64E4M3G9NSB > lce1/boot_BS64E4M3G9NSB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G9NSB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G9NSB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G9NSB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh); grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 30-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g9nsb_watch.out 2>&1 &
nohup bash repro_sustain.sh 30 > lce1/sustain_BS64E4M3G9NSB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
