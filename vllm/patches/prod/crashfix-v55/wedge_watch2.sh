#!/bin/bash
# wedge_watch2.sh <tag> — fence watcher using enhanced capture v2.
set -u
TAG="${1:?usage: wedge_watch2.sh <tag>}"
FLAG=/root/build/lce1/${TAG}.flag
while :; do
  F=$(docker exec lsv-test sh -c 'grep -c ASYNC-EVENT-STALL /root/serve_full.log' 2>/dev/null)
  if [ -n "$F" ] && [ "$F" != "0" ] && [ "$F" != "?" ]; then
    echo "WEDGE detected fence=$F at $(date +%H:%M:%S) — deep capture"
    bash /root/build/wedge_capture2.sh "$TAG" > /root/build/lce1/${TAG}_capture2.out 2>&1
    pkill -f repro_sustain 2>/dev/null
    pkill -f dt_warmup_v53 2>/dev/null
    pkill -f repro_loop 2>/dev/null
    echo "$F $(date)" > "$FLAG"
    echo "CAPTURED2 $FLAG"
    exit 2
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^lsv-test$'; then
    echo "CONTAINER GONE at $(date +%H:%M:%S)"
    echo "dead $(date)" > "$FLAG"
    exit 3
  fi
  sleep 15
done
