#!/bin/bash
# sustain_wait.sh — block until the exp:v1.2.11 sustain finishes (18 rounds
# done or crash capture), then print the summary. Runs ON THE HOST.
S=/root/build/lce1/sustain_EXPV1211.out
W=/root/build/lce1/expv1211_watch.out
while true; do
  R=$(grep -c 'SUSTAIN ROUND' "$S" 2>/dev/null || echo 0)
  F=$(grep -c 'fence-hits=0' "$S" 2>/dev/null || echo 0)
  E=$(dmesg 2>/dev/null | grep -c 'Engine reset')
  if [ "$R" -gt 18 ] || grep -q 'SUSTAIN DONE\|WEDGE\|CAPTURED' "$S" "$W" 2>/dev/null; then
    echo "FINAL rounds=$R clean=$F resets=$E"
    tail -5 "$S"
    tail -3 "$W"
    break
  fi
  sleep 60
done
