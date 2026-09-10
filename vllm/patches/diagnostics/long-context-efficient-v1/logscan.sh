#!/bin/bash
# logscan.sh — scan preserved lce1 engine logs for kernel/path/step markers
cd /root/build/lce1
for m in f8e4m4 f8e4ns mtp4 nospec; do
  echo "=== $m"
  zcat b_lce1_${m}.log.gz 2>/dev/null \
    | grep -a -o -E 'fp8[_a-zA-Z0-9]*|FP8[A-Z_]*|SPLIT[A-Z_]*|split_k[_a-z0-9]*|route[_a-zA-Z]*|flash[_a-zA-Z0-9]*|turboquant[_a-zA-Z0-9]*|fallback[ _a-zA-Z]*|eager|draft[_a-zA-Z]*|JIT[_a-zA-Z]*|jit[_a-zA-Z]*|WARMUP[A-Z_]*|recompile[_a-zA-Z]*' \
    | sort | uniq -c | sort -rn | head -30
  echo "--- step-time samples (last 6 metric lines during 64k decode)"
  zcat b_lce1_${m}.log.gz 2>/dev/null | grep -a -E 'throughput|Running:' | tail -6
done
