#!/bin/bash
# bs64_run.sh — boot one bs64 lane + warm-settle + full screen.
# usage: bs64_run.sh <e5m2|e4m3|tq4nc> <CERT_PROBE1>
# Settle = 7x (gen 256 tok + sleep 120) ~ 15 min: keeps clocks boosted
# (mode-flip lesson: idle parks at 950MHz; cold-settle lesson I6: f8ref
# on a ramping lane drifts all 3 probes).
cd /root/build
set -u
D="${1:?e5m2|e4m3|tq4nc}"
CERT="${2:?cert probe1}"
LBL="BS64_$(echo "$D" | tr 'a-z' 'A-Z')"

case "$D" in
  e5m2)  BOOT=repro_bootBS64_E5M2.sh ;;
  e4m3)  BOOT=repro_bootBS64_E4M3.sh ;;
  tq4nc) BOOT=repro_bootBS64_TQ4NC.sh ;;
  *) echo "bad dtype: $D"; exit 2 ;;
esac

echo "[$LBL] teardown standing lane $(date +%H:%M:%S)"
docker rm -f lsv-test >/dev/null 2>&1 || true
sleep 10
echo "[$LBL] boot $BOOT $(date +%H:%M:%S)"
bash "$BOOT" "$LBL" > "lce1/boot_$LBL.out" 2>&1 || true
if ! grep -q HEALTH_OK "lce1/boot_$LBL.out"; then
  echo "BOOT_FAIL [$LBL]"; tail -25 "lce1/boot_$LBL.out"; exit 1
fi
grep -m1 HEALTH_OK "lce1/boot_$LBL.out"
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh"

echo "[$LBL] warm-settle ~15min (boosted) $(date +%H:%M:%S)"
for i in 1 2 3 4 5 6 7; do
  curl -s -o /dev/null -m 120 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Write a detailed essay about the history of lighthouse construction on the Baltic coast, covering materials, weather challenges, and automation.","max_tokens":256,"temperature":0}'
  sleep 120
done
echo "[$LBL] settle done $(date +%H:%M:%S); health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://localhost:8000/health)"

bash bs64_screen.sh "$LBL" "$CERT"
echo "BS64_LANE_DONE $D $(date +%H:%M:%S)"
