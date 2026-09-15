#!/bin/bash
# poll_laneP.sh — Lane P poller: 40 x 300 s (200 min).
# Fires on: reset delta over 4, EngineDeadError, container gone.
# Else LANEP_NO_FIRE_200MIN summary.
set -u
BASE=4
for i in $(seq 1 40); do
  sleep 300
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE" ]; then
    echo "LANEP_RESETS at check $i: $R (base $BASE) $(date -u)"
    dmesg | grep -a 'Engine reset' | tail -n 4
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 5" 2>/dev/null
    exit 1
  fi
  if docker exec lsv-test sh -c "grep -aq 'EngineDeadError' /root/serve_full.log" 2>/dev/null; then
    echo "LANEP_ENGINE_DEAD at check $i $(date -u)"
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 6" 2>/dev/null
    exit 3
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "LANEP_CONTAINER_GONE at check $i $(date -u)"
    exit 4
  fi
done
echo "LANEP_NO_FIRE_200MIN resets=$(dmesg | grep -ac 'Engine reset') $(date -u)"
docker exec lsv-test sh -c "tail -n 3 /root/serve_full.log" 2>/dev/null
