#!/bin/sh
# FN8 d9 snapshotter (DEC40): preserve the serve log across boot-retry
# truncations (5s cadence).  If the current log shrinks below 50000
# bytes (a new boot try truncated it) while the previous snapshot
# exceeds 100000 bytes (a full death log), bank the snapshot first,
# THEN refresh.  Companion of /stage/bank/d9b_snap.log logic.
LOG=/stage/serve_d9.log
SNAP=/stage/bank/d9_snap.log
DEAD=/stage/bank/d9_dead.log
while true; do
  if [ -f "$LOG" ]; then
    CUR=$(wc -c < "$LOG")
    if [ "$CUR" -lt 50000 ] && [ -f "$SNAP" ]; then
      S=$(wc -c < "$SNAP")
      if [ "$S" -gt 100000 ]; then
        cp "$SNAP" "$DEAD"
      fi
    fi
    cp "$LOG" "$SNAP" 2>/dev/null
  fi
  sleep 5
done
