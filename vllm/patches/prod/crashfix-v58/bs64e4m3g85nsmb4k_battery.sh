#!/bin/bash
# bs64e4m3g85nsmb4k_battery.sh — §20 F-bar crash battery, arm 9
# (user-directed, CONDITIONAL: run only if arm 8 G8NS crashed): e4m3 KV +
# --block-size 64 + async OFF + gmu 0.85 + --max-num-batched-tokens 4096
# (halved per-step batch budget; vs G85NS event #16 at 3h59m21s the ONLY
# delta is mnbt 8192->4096). Host MUST be freshly rebooted first.
# Protocol: 30-round repro_sustain + q33 auto-arm watcher.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.85 no-async mnbt4096 $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G85NS_MB4K.sh BS64E4M3G85NSMB4KB > lce1/boot_BS64E4M3G85NSMB4KB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G85NSMB4KB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G85NSMB4KB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G85NSMB4KB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*\|--gpu-memory-utilization [0-9.]*\|--max-num-batched-tokens [0-9]*' /root/serve_user.sh; echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh); grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 30-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g85nsmb4kb_watch.out 2>&1 &
nohup bash repro_sustain.sh 30 > lce1/sustain_BS64E4M3G85NSMB4KB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
