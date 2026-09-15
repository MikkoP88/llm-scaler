#!/bin/bash
# q22_watch.sh — §14-class AUTO-ARM watcher for the Q22a battery:
# polls health every 60 s AND dmesg for fresh 'Engine reset' lines
# (reset precedes EngineDeadError by ~5 min = earlier capture, live
# container). On trigger: run q22_crash_capture.sh ONCE, then exit.
# Log: lce1/q22_watch.out
cd /root/build
BASE_RESETS=$(dmesg | grep -ac 'Engine reset')
echo "watch armed $(date +%H:%M:%S) base_resets=$BASE_RESETS"
FAILS=0
while true; do
  sleep 60
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://localhost:8000/health 2>/dev/null || echo 000)
  R=$(dmesg | grep -ac 'Engine reset')
  if [ "$R" -gt "$BASE_RESETS" ]; then
    echo "TRIGGER dmesg reset count $BASE_RESETS -> $R at $(date +%H:%M:%S) health=$code"
    bash /root/build/q22_crash_capture.sh
    echo "Q22_WATCH_TRIGGERED_RESET $(date +%H:%M:%S)"
    exit 0
  fi
  if [ "$code" = "200" ]; then
    FAILS=0
  else
    FAILS=$((FAILS+1))
    echo "health=$code fails=$FAILS at $(date +%H:%M:%S)"
    if [ "$FAILS" -ge 2 ]; then
      echo "TRIGGER health-dead x2 at $(date +%H:%M:%S)"
      bash /root/build/q22_crash_capture.sh
      echo "Q22_WATCH_TRIGGERED_HEALTH $(date +%H:%M:%S)"
      exit 0
    fi
  fi
done
