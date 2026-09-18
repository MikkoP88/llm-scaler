#!/bin/bash
# bs64e4m3g85ns_battery.sh — §20 F-bar crash battery, arm 7 (user-directed):
# crash-free lineage (e4m3 KV + --block-size 64 + async-scheduling REMOVED)
# with the ONLY delta gmu 0.9 -> 0.85 — the midpoint probe between clean
# 0.7 (G7NS 18/18 2h36m) and crash 0.9 (G9NS event #15 36m58s). LONGER:
# 30 rounds (~4.5h). Host MUST be freshly rebooted first. Verdict:
# event = threshold sits below 0.85; 30/30 clean = threshold in
# [0.85,0.9) and the no-async family extends upward.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown standing $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.85 no-async-sched LONG $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G85NS.sh BS64E4M3G85NSB > lce1/boot_BS64E4M3G85NSB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G85NSB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G85NSB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G85NSB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh); grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 30-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g85nsb_watch.out 2>&1 &
nohup bash repro_sustain.sh 30 > lce1/sustain_BS64E4M3G85NSB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
