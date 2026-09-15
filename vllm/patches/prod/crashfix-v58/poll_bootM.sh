#!/bin/bash
# poll_bootM.sh — 100-min verdict poller for Boot M (user lane on v1.2.10).
# Every 5 min check watch5 output for FLAG/WEDGE. Alert + exit on flag;
# NOFLAG_100MIN after 20 clean intervals.
set -u
W=/root/build/lce1/bootM_watch.out
for i in $(seq 1 20); do
  sleep 300
  if grep -q 'FLAG\|WEDGE' "$W" 2>/dev/null; then
    echo "BOOTM_FLAGGED at interval $i ($(date -u))"
    tail -n 40 "$W"
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "BOOTM_CONTAINER_GONE at interval $i ($(date -u))"
    tail -n 20 /root/build/lce1/bootM.log 2>/dev/null
    exit 2
  fi
done
echo "NOFLAG_100MIN $(date -u)"
echo "--- watch:"; tail -n 5 "$W" 2>/dev/null
echo "--- pressure tail:"; tail -n 3 /root/build/lce1/bootM_pressure.out 2>/dev/null
echo "--- loop tail:"; tail -n 3 /root/build/lce1/bootM_loop2.out 2>/dev/null
echo "--- fence trips in serve:"; docker exec lsv-test sh -c 'grep -c "v55 event timeout\|V55_FENCE" /root/serve_full.log' 2>/dev/null
