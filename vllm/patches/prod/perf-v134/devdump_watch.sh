#!/bin/bash
# devdump_watch.sh v2 — poll sysfs devcoredump every 2 s and harvest by
# READ-PROBE. v1 law (P75 crash #5): the sysfs devcoredump data node
# ALWAYS stats 0 bytes (bin_attr size unknown) — [ -s ] / stat-size gates
# are structurally blind; readability exists only ON READ. v2 cats the
# node, requires >=100 KB (observed dumps ~512 KB; guards against saving
# a torn partial read and dismissing a still-generating dump), then:
#   save + dismiss (dismiss frees the slot for the NEXT reset) +
#   snapshot the owning process /proc/<pid> maps+cmdline — pairs the
#   dump's ACTHD/RING_BBADDR host VA with the live allocation map
#   (P73 kernel-naming input; v1 missed this on every prior crash).
# Run under setsid nohup for the duration of a storm era; kill after.
OUT=/root/build/lce1
mkdir -p "$OUT"
TMP=$(mktemp)
while :; do
  for c in 1 2; do
    d=/sys/class/drm/card$c/device/devcoredump
    if [ -e "$d/data" ]; then
      sz=$(cat "$d/data" 2>/dev/null | tee "$TMP" | wc -c)
      if [ "$sz" -ge 100000 ]; then
        ts=$(date +%H%M%S)
        f="$OUT/LIVE_crash_${ts}_card$c.devcoredump"
        cp "$TMP" "$f"
        pid=$(grep -a -m1 '^Process:' "$f" | sed 's/.*\[\([0-9]*\)\].*/\1/')
        if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
          cat "/proc/$pid/maps" > "$OUT/LIVE_crash_${ts}_pid${pid}.maps" 2>/dev/null
          tr '\0' ' ' < "/proc/$pid/cmdline" > "$OUT/LIVE_crash_${ts}_pid${pid}.cmdline" 2>/dev/null
        fi
        echo 1 > "$d/dismiss" 2>/dev/null
        echo "HARVESTED card$c pid=${pid:-gone} at $(date +%H:%M:%S) sz=$sz -> $f" >> "$OUT/devdump_watch.log"
      fi
    fi
  done
  sleep 2
done
rm -f "$TMP"
