#!/bin/bash
# restore26m_cert.sh — post-battery standing-lane re-cert after event #15:
# 15-min boosted warm-settle (f8ref-WARM rule I6), f8ref cert (e5m2 set
# cb8c/6833/05c8), q17 spot, then re-arm lane watchdog (rm pause flag).
cd /root/build
set -u
echo "[CERT] warm-settle ~15min boosted $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[CERT] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"
echo "[CERT] f8ref"
timeout 360 python3 -u f8ref.py RESTORE26N 2>&1 | tee lce1/f8ref_RESTORE26N.out | grep -o 'hashes=.*' || true
grep -q 'distinct=OK' lce1/f8ref_RESTORE26N.out && echo F8REF_DETERMINISTIC || echo F8REF_NONDET
grep -q 'cb8c3851b897' lce1/f8ref_RESTORE26N.out && echo F8REF_EXACT || echo F8REF_NOT_EXACT
echo "[CERT] q17 spot"
bash q17_perf.sh > lce1/q17_RESTORE26N.out 2>&1
tail -4 lce1/q17_RESTORE26N.out
echo "[CERT] re-arm watchdog"
rm -f /root/build/lane_watchdog.paused
systemctl is-active lane-watchdog
echo "engine_resets=$(dmesg | grep -ac 'Engine reset')"
echo "[CERT] DONE $(date +%H:%M:%S)"
