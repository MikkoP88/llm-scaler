#!/bin/bash
# wedge_capture.sh — full instrumented capture at wedge time.
# Usage: wedge_capture.sh <tag>
set -u
TAG="${1:-wedge}"
D=/root/build/lce1/${TAG}_$(date +%H%M%S)
mkdir -p "$D"

echo "=== capture at $(date +%H:%M:%S) -> $D ==="

# GPU utilization / engine state DURING the wedge (spinning vs idle!)
{ xpu-smi discovery; xpu-smi dump -m 0 2>&1 | head -60;
  xpu-smi dump -m 1 2>&1 | head -60; } > "$D/xpu_state.txt" 2>&1

# xe sysfs engine busy if present
for c in /sys/class/drm/card*/device; do
  echo "== $c" >> "$D/xe_sysfs.txt"
  ls "$c" | grep -iE 'gt|engine|busy' >> "$D/xe_sysfs.txt" 2>&1
  for f in "$c"/gt*/gpu_busy "$c"/gt*/*busy*; do
    [ -f "$f" ] && echo "$f: $(cat "$f")" >> "$D/xe_sysfs.txt"
  done
done 2>/dev/null

# engine log: last 400 lines incl. DEBUG v52f boundary bookkeeping
docker exec lsv-test sh -c 'tail -400 /root/serve_full.log' > "$D/serve_tail.txt" 2>&1

# v52f boundary lines (the DEF-PP / PRE-COPY / POST-COPY trail)
docker exec lsv-test sh -c \
  'grep -E "v52f|v52d|v52e|v52m|DISCARD-GAP|GAP-FENCE|ASYNC-EVENT-STALL" /root/serve_full.log | tail -60' \
  > "$D/class_lines.txt" 2>&1

# flight recorders
docker exec lsv-test sh -c 'for f in /tmp/fr_*.log; do echo "== $f"; tail -50 "$f"; done' \
  > "$D/fr_tails.txt" 2>&1

# procs + dmesg
docker exec lsv-test sh -c 'ps aux | grep -E "EngineCore|Worker" | grep -v grep' \
  > "$D/procs.txt" 2>&1
dmesg -T | tail -30 > "$D/dmesg_tail.txt" 2>&1

echo "=== xpu_state (first 40) ==="
head -40 "$D/xpu_state.txt"
echo "=== class_lines (last 25) ==="
tail -25 "$D/class_lines.txt"
echo "=== serve tail last 12 non-HTTP ==="
grep -v "GET /health" "$D/serve_tail.txt" | tail -12
