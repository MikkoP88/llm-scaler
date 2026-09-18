#!/bin/bash
# g8ns_mtp3_screen_run.sh — §24 N bench3: Lane A config (e4m3 bs64
# gmu0.8 no-async, §24-L G8NS / §24-M lane A) with MTP=3 (single delta
# vs §24 M lane A: num_speculative_tokens 4 -> 3).
# Protocol: boot -> 15min boosted settle (I6 f8ref-WARM rule) ->
# bs64_screen (f8ref e4m3 cert 52f598e7d38a + bench3 cold/warm + q17).
# NOTE: MTP depth changes verification batching; f8ref EXACT is
# expected (greedy spec-decode is lossless) but NOT gated — record only.
cd /root/build
set -u
LBL=G8NSMTP3B3
BOOT=repro_bootBS64_E4M3_G8NS_MTP3.sh

echo "[$LBL] pause watchdog + teardown $(date +%H:%M:%S)"
touch /root/build/lane_watchdog.paused
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[$LBL] boot $BOOT $(date +%H:%M:%S)"
bash "$BOOT" "$LBL" > "lce1/boot_$LBL.out" 2>&1 || true
if ! grep -q HEALTH_OK "lce1/boot_$LBL.out"; then
  echo "BOOT_FAIL [$LBL]"; tail -25 "lce1/boot_$LBL.out"; exit 1
fi
grep -m1 HEALTH_OK "lce1/boot_$LBL.out"
docker exec lsv-test sh -c "grep -oE -- '--block-size [0-9]+|--gpu-memory-utilization [0-9.]+|--max-num-batched-tokens [0-9]+|\"num_speculative_tokens\":[0-9]+' /root/serve_user.sh; grep -o \"'num_speculative_tokens': [0-9]*\" /root/serve_full.log | head -1; grep -o \"'kv_cache_dtype': '[a-z0-9_]*'\" /root/serve_full.log | head -1"

echo "[$LBL] warm-settle ~15min boosted $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[$LBL] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"

bash bs64_screen.sh "$LBL" 52f598e7d38a
echo "MTP3_SCREEN_DONE $LBL $(date +%H:%M:%S)"
