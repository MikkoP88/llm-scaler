#!/bin/bash
# devdump_watch.sh v3 — v2 + HARVEST-DEDUPE LAW (P76 B1 leg): `echo 1 >
# dismiss` does NOT stop the slot re-arming readable — v2 saved an
# identical copy every 2 s for ~70 s (35+ duplicates). v3 fingerprints the
# probe content (md5) per card into lce1/.devdump_seen_cardN and harvests
# ONLY on a fingerprint change: exactly one copy per unique dump, new
# crashes on the same card still captured. Everything else is v2 law:
# read-probe >=100 KB (sysfs stats 0 always), save + dismiss + snapshot
# the owning pid maps/cmdline for ACTHD/BBADDR VA naming.
OUT=/root/build/lce1
mkdir -p "$OUT"
TMP=$(mktemp)
while :; do
  for c in 1 2; do
    d=/sys/class/drm/card$c/device/devcoredump
    if [ -e "$d/data" ]; then
      sz=$(cat "$d/data" 2>/dev/null | tee "$TMP" | wc -c)
      if [ "$sz" -ge 100000 ]; then
        fp=$(md5sum "$TMP" | cut -d' ' -f1)
        last=$(cat "$OUT/.devdump_seen_card$c" 2>/dev/null || true)
        if [ "$fp" != "$last" ]; then
          ts=$(date +%H%M%S)
          f="$OUT/LIVE_crash_${ts}_card$c.devcoredump"
          cp "$TMP" "$f"
          pid=$(grep -a -m1 '^Process:' "$f" | sed 's/.*\[\([0-9]*\)\].*/\1/')
          if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
            cat "/proc/$pid/maps" > "$OUT/LIVE_crash_${ts}_pid${pid}.maps" 2>/dev/null
            tr '\0' ' ' < "/proc/$pid/cmdline" > "$OUT/LIVE_crash_${ts}_pid${pid}.cmdline" 2>/dev/null
          fi
          echo "$fp" > "$OUT/.devdump_seen_card$c"
          echo "HARVESTED card$c pid=${pid:-gone} at $(date +%H:%M:%S) sz=$sz fp=$fp -> $f" >> "$OUT/devdump_watch.log"
        fi
        echo 1 > "$d/dismiss" 2>/dev/null
      fi
    fi
  done
  sleep 2
done
rm -f "$TMP"
