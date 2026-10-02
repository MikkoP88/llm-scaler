#!/bin/bash
# capture_stall.sh — full forensics capture of the p14 e5m2+mtp4 stall
# (zombie engine: health 200 shallow, EngineCore wedged on shm broadcast).
set -u
D=/root/build/lce1/stall_p14_e5m2
mkdir -p "$D"

docker exec lsv-test sh -c 'tail -80 /tmp/fr_542.log' > "$D/fr_542_tail.log" 2>&1
docker exec lsv-test sh -c 'tail -80 /tmp/fr_562.log' > "$D/fr_562_tail.log" 2>&1
docker exec lsv-test sh -c 'tail -250 /root/serve_full.log' > "$D/serve_tail.log" 2>&1
docker exec lsv-test sh -c 'grep -nE "ASYNC-EVENT-STALL|v52d|v52e|v52m|DISCARD-GAP|GAP-FENCE|Traceback|ERROR" /root/serve_full.log | tail -40' > "$D/class_lines.log" 2>&1
docker exec lsv-test sh -c 'ps aux | grep -E "EngineCore|Worker_TP" | grep -v grep' > "$D/procs.log" 2>&1
docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f $(stat -c %Y $f) $(stat -c %s $f)"; done' > "$D/fr_sizes.log" 2>&1
dmesg -T > "$D/dmesg_full.log" 2>&1
date +%H:%M:%S > "$D/captured_at.txt"

echo "=== captured in $D ==="
ls -la "$D"
echo "=== FR542 last 8 ==="
tail -8 "$D/fr_542_tail.log"
echo "=== FR562 last 8 ==="
tail -8 "$D/fr_562_tail.log"
echo "=== procs ==="
cat "$D/procs.log"
echo "=== serve class lines (last 10) ==="
tail -10 "$D/class_lines.log"
