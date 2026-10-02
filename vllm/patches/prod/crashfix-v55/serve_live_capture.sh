#!/bin/bash
# serve_live_capture.sh — continuous HOST-side capture of lsv-test serve log.
# Why: on hard deaths (XPU engine reset class) the container vanishes from docker
# before the watchdog Q22 capture runs, so /root/serve_full.log (in-container) is
# lost. This streamer tails it out to the host continuously; the host file survives
# any container death. Re-attaches to any new lsv-test container within ~3-6s.
OUT=/root/build/serve_live.log
V55RELAY=/root/build/v55_3_crash_relay.log
MAXBYTES=$((200*1024*1024))
KEEPBYTES=$((20*1024*1024))
mkdir -p /root/build
while true; do
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^lsv-test$'; then
    # relay any fast-kill crash marker from a (possibly dying) container
    docker exec lsv-test cat /root/v55_3_crash.log >> "$V55RELAY" 2>/dev/null
    echo "===ATTACH $(date -u '+%F %T')===" >> "$OUT"
    docker exec lsv-test tail -F -n 0 /root/serve_full.log >> "$OUT" 2>&1
    echo "===DETACH $(date -u '+%F %T') rc=$?===" >> "$OUT"
  fi
  # size guard: rotate at MAXBYTES, keep tail KEEPBYTES
  SZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$SZ" -gt "$MAXBYTES" ]; then
    tail -c "$KEEPBYTES" "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  fi
  sleep 3
done
