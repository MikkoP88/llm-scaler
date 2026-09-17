#!/bin/bash
# bs64e4m3gmu7_battery.sh — §20 F-bar crash battery, arm 3 (user-directed):
# e4m3 KV + --block-size 64 + --gpu-memory-utilization 0.7, on a CLEAN
# host (rebooted 16:44:51 after events #12 e4m3-bs64 1h13m and #13
# e5m2-bs64 12m18s, both 'LR job cleanup'). GMU7 = new lever: standing
# value 0.9; 0.7 shrinks KV pool / changes allocator layout. Protocol
# identical: 18-round repro_sustain + q33 auto-arm watcher. Verdict:
# event = 14th §14 datum (mem-util joins the inert class); 18/18 clean
# = first crash-free battery since Q31.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.7 $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_GMU7.sh BS64E4M3G7B > lce1/boot_BS64E4M3G7B.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G7B.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G7B.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G7B.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 18-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g7b_watch.out 2>&1 &
nohup bash repro_sustain.sh 18 > lce1/sustain_BS64E4M3G7B.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
