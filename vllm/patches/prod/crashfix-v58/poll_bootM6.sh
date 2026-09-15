#!/bin/bash
# poll_bootM6.sh — host-side verdict poller for Boot M6 (e5m2+mtp4 @131072).
# 24 x 60s checks: reset delta over baseline 35, EngineDeadError in serve
# log, watch5 FLAG/WEDGE, container alive. Exits early on any firing.
set -u
BASE=35
L=/root/build/lce1
for i in $(seq 1 24); do
  sleep 60
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE" ]; then
    echo "M6_RESETS at check $i: $R (base $BASE) $(date -u)"
    dmesg | grep -a 'Engine reset' | tail -n 4
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|timed out\|TimeoutError' /root/serve_full.log | tail -n 5" 2>/dev/null
    tail -n 5 "$L/bootM6_pressure.out" 2>/dev/null
    exit 1
  fi
  if grep -aq 'FLAG\|WEDGE' "$L/bootM6_watch.out" 2>/dev/null; then
    echo "M6_WATCH_FLAG at check $i $(date -u)"
    tail -n 30 "$L/bootM6_watch.out"
    exit 2
  fi
  if docker exec lsv-test sh -c "grep -aq 'EngineDeadError' /root/serve_full.log" 2>/dev/null; then
    echo "M6_ENGINE_DEAD at check $i $(date -u)"
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 6" 2>/dev/null
    exit 3
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "M6_CONTAINER_GONE at check $i $(date -u)"
    tail -n 20 "$L/bootM6.log" 2>/dev/null
    exit 4
  fi
done
echo "M6_24MIN_NO_FIRE resets=$(dmesg | grep -ac 'Engine reset') $(date -u)"
echo "--- pressure tail:"; tail -n 4 "$L/bootM6_pressure.out" 2>/dev/null
echo "--- watch tail:"; tail -n 3 "$L/bootM6_watch.out" 2>/dev/null
