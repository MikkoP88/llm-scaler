#!/bin/bash
# wedge_watch.sh <tag> — background fence watcher. Polls serve log every
# 20s; on first ASYNC-EVENT-STALL (or container death mid-run) runs
# wedge_capture.sh, kills any stuck sustain/warmup loops, writes flag.
set -u
TAG="${1:?usage: wedge_watch.sh <tag>}"
FLAG=/root/build/lce1/${TAG}.flag
while :; do
  F=$(docker exec lsv-test sh -c 'grep -c ASYNC-EVENT-STALL /root/serve_full.log' 2>/dev/null)
  if [ -n "$F" ] && [ "$F" != "0" ] && [ "$F" != "?" ]; then
    echo "WEDGE detected fence=$F at $(date +%H:%M:%S) — capturing"
    bash /root/build/wedge_capture.sh "$TAG" > /root/build/lce1/${TAG}_capture.out 2>&1
    pkill -f repro_sustain 2>/dev/null
    pkill -f dt_warmup_v53 2>/dev/null
    echo "$F $(date)" > "$FLAG"
    echo "CAPTURED $FLAG"
    exit 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER GONE at $(date +%H:%M:%S)"
    bash /root/build/wedge_capture.sh "${TAG}_death" > /root/build/lce1/${TAG}_death_capture.out 2>&1
    echo "dead $(date)" > "$FLAG"
    exit 3
  fi
  sleep 20
done
