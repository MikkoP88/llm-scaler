#!/bin/bash
# wedge_capture2.sh <tag> — enhanced wedge-time capture v2.
# Adds: GPU engine utilization (spin vs idle classification), EU array
# activity, worker kernel wait channels + thread states (via container
# /proc), devcoredump check. Runs at fence detection while workers are
# still alive (60s+ window before fail-fast completes).
set -u
TAG="${1:?tag}"
D=/root/build/lce1/${TAG}_$(date +%H%M%S)
mkdir -p "$D"
echo "=== capture2 at $(date +%H:%M:%S) -> $D"

# GPU engine utilization: 3 samples x 1s per device (UTILIZATION group
# has per-engine-class rows; EU_ARRAY = execution unit activity).
for dev in 0 1; do
  xpu-smi dump --device $dev --metrics UTILIZATION,EU_ARRAY,MEMORY \
    --interval 1 --number 3 > "$D/xpu_dump_d${dev}.txt" 2>&1
done

# xe sysfs engine/busy probes + devcoredump check
for c in /sys/class/drm/card*/device; do
  echo "== $c" >> "$D/xe_sysfs.txt"
  ls "$c" >> "$D/xe_sysfs.txt" 2>&1
  for f in "$c"/gt*/gpu_busy "$c"/gt*/*busy*; do
    [ -f "$f" ] && echo "$f: $(cat "$f" 2>/dev/null)" >> "$D/xe_sysfs.txt"
  done
  ls "$c/devcoredump/" >> "$D/xe_sysfs.txt" 2>&1
done 2>/dev/null

# worker/engine pids (container namespace) + kernel stacks + wchan +
# per-thread state — INSIDE the privileged container.
docker exec lsv-test sh -c '
  for pid in $(ps aux | grep -E "VLLM::" | grep -v grep | awk "{print \$2}"); do
    echo "== pid $pid $(cat /proc/$pid/comm 2>/dev/null)"
    echo "   wchan: $(cat /proc/$pid/wchan 2>/dev/null)"
    echo "   kernel-stack:"; cat /proc/$pid/stack 2>/dev/null | head -8
    echo "   threads:"
    for t in /proc/$pid/task/*; do
      echo "     $(basename $t) state=$(awk "{print \$3}" $t/stat 2>/dev/null) wchan=$(cat $t/wchan 2>/dev/null)"
    done
  done' > "$D/worker_stacks.txt" 2>&1

# fr tails (live workers only — by current pids)
docker exec lsv-test sh -c '
  for pid in $(ps aux | grep -E "VLLM::Worker" | grep -v grep | awk "{print \$2}"); do
    f=/tmp/fr_${pid}.log
    [ -f "$f" ] && { echo "== $f"; tail -25 "$f"; }
  done' > "$D/fr_live.txt" 2>&1

# serve log: stall lines + last 300 lines
docker exec lsv-test sh -c 'grep -n "ASYNC-EVENT-STALL" /root/serve_full.log | head -20' \
  > "$D/stall_lines.txt" 2>&1
docker exec lsv-test sh -c 'tail -300 /root/serve_full.log' > "$D/serve_tail.txt" 2>&1

# host dmesg tail (engine resets?)
dmesg -T | tail -20 > "$D/dmesg_tail.txt" 2>&1

echo "=== xpu_dump_d0 (head 30)"
head -30 "$D/xpu_dump_d0.txt"
echo "=== worker_stacks (head 40)"
head -40 "$D/worker_stacks.txt"
