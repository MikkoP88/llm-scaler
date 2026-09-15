#!/bin/bash
# poll_bootQ.sh — Boot Q diag poller: 30 x 300 s (150 min).
# Fires on: reset delta over BASE (set at deploy), EngineDeadError,
# container gone. Also snapshots any F15b dumps on fire. Else
# BOOTQ_NO_FIRE summary.
set -u
BASE=4
L=/root/build/lce1
for i in $(seq 1 30); do
  sleep 300
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE" ]; then
    echo "BOOTQ_RESETS at check $i: $R (base $BASE) $(date -u)"
    dmesg | grep -a 'Engine reset' | tail -n 4
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 5" 2>/dev/null
    docker exec lsv-test sh -c "ls -la /root/f15b_dump_*.log /root/f15b_pyspy_*.log /root/f15b_xpusmi.log 2>/dev/null" 2>/dev/null
    exit 1
  fi
  if docker exec lsv-test sh -c "grep -aq 'EngineDeadError' /root/serve_full.log" 2>/dev/null; then
    echo "BOOTQ_ENGINE_DEAD at check $i $(date -u)"
    docker exec lsv-test sh -c "grep -a 'EngineDeadError\|TimeoutError\|timed out' /root/serve_full.log | tail -n 6" 2>/dev/null
    docker exec lsv-test sh -c "ls -la /root/f15b_dump_*.log 2>/dev/null" 2>/dev/null
    exit 3
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOTQ_CONTAINER_GONE at check $i $(date -u)"
    exit 4
  fi
done
echo "BOOTQ_NO_FIRE_150MIN resets=$(dmesg | grep -ac 'Engine reset') $(date -u)"
docker exec lsv-test sh -c "ls -la /root/f15b_dump_*.log 2>/dev/null; tail -n 3 /root/serve_full.log" 2>/dev/null
