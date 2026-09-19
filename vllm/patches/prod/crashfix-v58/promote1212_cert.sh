#!/bin/bash
# promote1212_cert.sh — §24 U promotion certification for the v1.2.12
# standing lane: 15-min boosted warm-settle (I6 rule), f8ref cert
# (e4m3 bs64 set: 52f598e7d38a / 6b1c26403bfc / 95e24129958b), q17
# spot probe. NOTE: watchdog re-arm is intentionally NOT done here —
# promote1212_run.sh re-points the relaunch lineage FIRST, then arms.
cd /root/build
set -u
echo "[PCERT] warm-settle ~15min boosted $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[PCERT] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"
echo "[PCERT] f8ref (e4m3 bs64 set)"
timeout 360 python3 -u f8ref.py PROMOTE1212 2>&1 | tee lce1/f8ref_PROMOTE1212.out | grep -o 'hashes=.*' || true
grep -q 'distinct=OK' lce1/f8ref_PROMOTE1212.out && echo F8REF_DETERMINISTIC || echo F8REF_NONDET
grep -q '52f598e7d38a' lce1/f8ref_PROMOTE1212.out && echo F8REF_EXACT || echo F8REF_NOT_EXACT
echo "[PCERT] q17 spot"
bash q17_perf.sh > lce1/q17_PROMOTE1212.out 2>&1
tail -4 lce1/q17_PROMOTE1212.out
echo "[PCERT] engine_resets=$(dmesg | grep -ac 'Engine reset')"
echo "[PCERT] DONE $(date +%H:%M:%S)"
