#!/bin/bash
# mem_growth_logger.sh <out.log> — every 5 min: per-worker VM-mapping count
# (total + /dev/dri device maps), VmRSS; per-device xpu-smi MEMORY used/total.
# Detects unaccounted growing consumers (IPC peer mappings, allocator
# segments, event/driver objects) OUTSIDE gpu-memory-utilization.
set -u
OUT="${1:?usage: mem_growth_logger.sh <out.log>}"
while :; do
  TS=$(date -u +%H:%M:%S)
  PIDS=$(docker exec lsv-test sh -c 'ps aux | grep VLLM::Worker | grep -v grep' 2>/dev/null | awk '{print $2}')
  [ -z "$PIDS" ] && { echo "$TS no-workers" >> "$OUT"; sleep 300; continue; }
  LINE="$TS"
  for pid in $PIDS; do
    MAPS=$(docker exec lsv-test sh -c "wc -l < /proc/$pid/maps 2>/dev/null" 2>/dev/null)
    DRI=$(docker exec lsv-test sh -c "grep -c '/dev/dri' /proc/$pid/maps 2>/dev/null" 2>/dev/null)
    RSS=$(docker exec lsv-test sh -c "grep VmRSS /proc/$pid/status 2>/dev/null" 2>/dev/null | awk '{print $2}')
    LINE="$LINE | w$pid maps=${MAPS:-?} dri=${DRI:-?} rss=${RSS:-?}kB"
  done
  for dev in 0 1; do
    VR=$(xpu-smi dump --device $dev --metrics MEMORY --interval 1 --number 1 2>/dev/null | tail -1)
    LINE="$LINE | d$dev [${VR}]"
  done
  echo "$LINE" >> "$OUT"
  sleep 300
done
