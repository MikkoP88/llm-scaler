#!/bin/bash
# poll_bootN.sh — Boot N certification poller: 40 x 300 s (200 min).
# Fire on: reset delta over 47, watch5 FLAG/WEDGE, EngineDeadError,
# container gone. Else NO_FIRE_200MIN summary.
set -u
BASE=47
L=/root/build/lce1
for i in $(seq 1 40); do
  sleep 300
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE" ]; then
    echo "BOOTN_RESETS at check $i: $R (base $BASE) $(date -u)"
    dmesg | grep -a 'Engine reset' | tail -n 4
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 5" 2>/dev/null
    tail -n 5 "$L/bootN_pressure.out" "$L/bootN_loop2.out" "$L/bootN_loop3.out" 2>/dev/null
    exit 1
  fi
  if grep -aq 'FLAG\|WEDGE' "$L/bootN_watch.out" 2>/dev/null; then
    echo "BOOTN_WATCH_FLAG at check $i $(date -u)"
    tail -n 30 "$L/bootN_watch.out"
    exit 2
  fi
  if docker exec lsv-test sh -c "grep -aq 'EngineDeadError' /root/serve_full.log" 2>/dev/null; then
    echo "BOOTN_ENGINE_DEAD at check $i $(date -u)"
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 6" 2>/dev/null
    exit 3
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOTN_CONTAINER_GONE at check $i $(date -u)"
    tail -n 20 "$L/bootN.log" 2>/dev/null
    exit 4
  fi
done
echo "BOOTN_NO_FIRE_200MIN resets=$(dmesg | grep -ac 'Engine reset') $(date -u)"
echo "--- pressure tail:"; tail -n 4 "$L/bootN_pressure.out" 2>/dev/null
echo "--- loop2 tail:"; tail -n 4 "$L/bootN_loop2.out" 2>/dev/null
echo "--- loop3 tail:"; tail -n 4 "$L/bootN_loop3.out" 2>/dev/null
echo "--- watch tail:"; tail -n 3 "$L/bootN_watch.out" 2>/dev/null
