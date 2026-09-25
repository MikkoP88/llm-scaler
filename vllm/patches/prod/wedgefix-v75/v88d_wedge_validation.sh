#!/bin/bash
# v88d — WEDGE VALIDATION on the fixed wheel (engine already up from v88c):
# serialized 24 (historically DEAD at cycle 17) then burst_harsh x3
# (historically dead early). Expect: serialized SURVIVED ok=24 and all three
# bursts SURVIVED with zero engine resets in dmesg.
set -u
TS=$(date +%H%M%S)
L=/root/build/lce1/v88d_$TS.out
echo "v88d wedge validation start $TS" | tee -a $L
R0=$(dmesg | grep -c 'Engine reset')
echo "resets_before=$R0" | tee -a $L

# 1. serialized 24 (the deterministic killer)
OK=0; FAIL=0; H="200"
for i in $(seq 1 24); do
  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"Count from 1 to 5.","max_tokens":24,"temperature":0}' \
    -o /dev/null -w '%{http_code}')
  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
  sleep 6
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:8000/health)
  [ "$H" != "200" ] && { echo "[v88d] SERIALIZED DEAD cycle $i ok=$OK fail=$FAIL" | tee -a $L; break; }
done
[ "$H" = "200" ] && echo "[v88d] SERIALIZED SURVIVED ok=$OK fail=$FAIL" | tee -a $L

# 2. burst_harsh x3 (concurrent boundary storm)
for b in 1 2 3; do
  bash /root/build/burst_harsh.sh v88d_$b 2>&1 | tail -2 | tee -a $L
  H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://localhost:8000/health)
  [ "$H" != "200" ] && { echo "[v88d] DEAD after burst $b" | tee -a $L; break; }
done

R1=$(dmesg | grep -c 'Engine reset')
echo "resets_after=$R1" | tee -a $L
echo V88D_DONE | tee -a $L
