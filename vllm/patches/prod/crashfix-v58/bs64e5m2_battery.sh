#!/bin/bash
# bs64e5m2_battery.sh — §20 F-bar crash battery, arm 2: e5m2 KV +
# --block-size 64 (bs64 dtype-parity lane from §24 J; e4m3 bs64 died
# in 1h13m as event #12). Protocol identical to bs64e4m3_battery.sh:
# 18-round repro_sustain + q33 auto-arm watcher (reset-count /
# health-dead trigger, §18 capture then teardown). Same-host sequential
# after event #12, no intervening reboot (matches chained-battery
# precedent events #5-#7). Verdict classes: round 13+ event = 13th
# §14 datum; 18/18 clean = first crash-free battery since Q31.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e5m2 bs64 $(date +%H:%M:%S)"
bash repro_bootBS64_E5M2.sh BS64E5M2B > lce1/boot_BS64E5M2B.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E5M2B.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E5M2B.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E5M2B.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 18-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e5m2b_watch.out 2>&1 &
nohup bash repro_sustain.sh 18 > lce1/sustain_BS64E5M2B.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
