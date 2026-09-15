#!/bin/bash
# poll_bootO2.sh — Boot O poller v2: 36 x 300 s (180 min).
# Watch5 disabled as trigger (metrics-endpoint startup false-positive
# on this boot; engine health tracked by resets/EngineDeadError/
# container/loop progress instead).
set -u
BASE=51
L=/root/build/lce1
for i in $(seq 1 36); do
  sleep 300
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE" ]; then
    echo "BOOTO_RESETS at check $i: $R (base $BASE) $(date -u)"
    dmesg | grep -a 'Engine reset' | tail -n 4
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 5" 2>/dev/null
    tail -n 5 "$L/bootO_pressure.out" "$L/bootO_loop2.out" "$L/bootO_loop3.out" 2>/dev/null
    exit 1
  fi
  if docker exec lsv-test sh -c "grep -aq 'EngineDeadError' /root/serve_full.log" 2>/dev/null; then
    echo "BOOTO_ENGINE_DEAD at check $i $(date -u)"
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 6" 2>/dev/null
    exit 3
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOTO_CONTAINER_GONE at check $i $(date -u)"
    tail -n 20 "$L/bootO.log" 2>/dev/null
    exit 4
  fi
done
echo "BOOTO_NO_FIRE_180MIN resets=$(dmesg | grep -ac 'Engine reset') $(date -u)"
echo "--- pressure tail:"; tail -n 4 "$L/bootO_pressure.out" 2>/dev/null
echo "--- loop2 tail:"; tail -n 4 "$L/bootO_loop2.out" 2>/dev/null
echo "--- loop3 tail:"; tail -n 4 "$L/bootO_loop3.out" 2>/dev/null
