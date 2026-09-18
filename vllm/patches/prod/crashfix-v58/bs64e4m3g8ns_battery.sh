#!/bin/bash
# bs64e4m3g8ns_battery.sh — §20 F-bar crash battery, arm 8 (user-directed,
# fires because arm 7 G85NS crashed at 3h59m21s = event #16): crash-free
# lineage (e4m3 KV + --block-size 64 + async-scheduling REMOVED) with the
# ONLY delta gmu -> 0.8. Ladder so far (no-async): 0.9 crash 36m58s (#15),
# 0.85 crash 3h59m21s (#16), 0.7 clean 2h36m 18/18 (G7NS). LONGER: 30
# rounds. Host MUST be freshly rebooted first. Verdict: 30/30 clean =>
# BUILD llm-scaler-exp image async-OFF (user directive) + push + commit;
# event = threshold below 0.8, 0.7-family remains sole clean class.
cd /root/build
set -u
echo "[BATT] pause watchdog + teardown leftovers $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[BATT] boot e4m3 bs64 gmu0.8 no-async-sched LONG $(date +%H:%M:%S)"
bash repro_bootBS64_E4M3_G8NS.sh BS64E4M3G8NSB > lce1/boot_BS64E4M3G8NSB.out 2>&1 || true
if ! grep -q HEALTH_OK lce1/boot_BS64E4M3G8NSB.out; then
  echo "BOOT_FAIL"; tail -20 lce1/boot_BS64E4M3G8NSB.out; exit 1
fi
grep -m1 HEALTH_OK lce1/boot_BS64E4M3G8NSB.out
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -o -- '--gpu-memory-utilization [0-9.]*' /root/serve_user.sh; echo async_count=\$(grep -c 'async-scheduling' /root/serve_user.sh); grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"
sleep 30
echo "[BATT] ARM watcher + 30-round sustain $(date +%H:%M:%S)"
nohup bash q33_watch.sh > lce1/bs64e4m3g8nsb_watch.out 2>&1 &
nohup bash repro_sustain.sh 30 > lce1/sustain_BS64E4M3G8NSB.out 2>&1 &
echo "[BATT] ARMED $(date +%H:%M:%S)"
