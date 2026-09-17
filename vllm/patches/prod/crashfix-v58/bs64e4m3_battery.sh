#!/bin/bash
# bs64e4m3_battery.sh — §20 F-bar crash battery on the §24-J pick:
# e4m3 KV + --block-size 64 (fastest clean lane measured, 54.6 solo).
# Protocol = the one every §14 event died under: 18-round repro_sustain
# + q33 auto-arm watcher (reset-count / health-dead trigger, §18 capture
# then teardown). Verdict classes: round 13+ event = 12th §14 datum;
# 18/18 clean = first crash-free battery since Q31, at 54-class speed.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown standing $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3.sh BS64E4M3B > lce1/boot_BS64E4M3B.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3B.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3B.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3B.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 18-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3b_watch.out 2>&1 &
nohup bash repro_sustain.sh 18 > lce1/sustain_BS64E4M3B.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
