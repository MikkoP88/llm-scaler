#!/bin/bash
# logscan2.sh — decode-window spec metrics, JIT events with context, route/SPLITS lines
cd /root/build/lce1
for m in f8e4m4 mtp4; do
  echo "=== $m : SpecDecoding during active decode (Accepted throughput > 1)"
  zcat b_lce1_${m}.log.gz 2>/dev/null \
    | grep -a 'SpecDecoding metrics' | grep -a -v 'Accepted throughput: 0.0' \
    | awk -F'Mean acceptance length: ' '{print $2}' | awk -F',' '{print $1}' \
    | sort -n | uniq -c | tail -15
  echo "--- $m : top acceptance lengths by freq (active)"
  zcat b_lce1_${m}.log.gz 2>/dev/null \
    | grep -a 'SpecDecoding metrics' | grep -a -v 'Accepted throughput: 0.0' \
    | awk -F'Mean acceptance length: ' '{print $2}' | awk -F',' '{printf "%s\n", $1}' \
    | sort -n | uniq -c | sort -rn | head -8
done
echo
echo "=== f8e4m4 : JIT / jit_monitor lines (with timestamps, first 12)"
zcat b_lce1_f8e4m4.log.gz 2>/dev/null | grep -a -E 'JIT|jit_monitor' | head -12
echo
echo "=== f8e4ns : JIT / jit_monitor lines (first 10)"
zcat b_lce1_f8e4ns.log.gz 2>/dev/null | grep -a -E 'JIT|jit_monitor' | head -10
echo
echo "=== mtp4 : route / SPLITS / flash_attn lines (first 8, uniq by content)"
zcat b_lce1_mtp4.log.gz 2>/dev/null | grep -a -E 'SPLITS|flash_attn route|turboquant_attn' | sed 's/^[^ ]* [^ ]* //' | sort | uniq -c | sort -rn | head -8
echo
echo "=== f8e4m4 : any 'fp8' lines (all, uniq)"
zcat b_lce1_f8e4m4.log.gz 2>/dev/null | grep -a -i 'fp8' | sed 's/^[^ ]* [^ ]* //' | sort | uniq -c | sort -rn | head -12
echo
echo "=== f8e4m4 : generation throughput distribution during decode (nonzero)"
zcat b_lce1_f8e4m4.log.gz 2>/dev/null | grep -a 'Avg generation throughput' \
  | grep -a -v 'generation throughput: 0.0' \
  | grep -a -o -E 'generation throughput: [0-9.]+' | awk '{print $3}' \
  | sort -n | awk '{a[NR]=$1} END {print "n="NR, "min="a[1], "p50="a[int(NR/2)], "max="a[NR]}'
echo "=== mtp4 : same"
zcat b_lce1_mtp4.log.gz 2>/dev/null | grep -a 'Avg generation throughput' \
  | grep -a -v 'generation throughput: 0.0' \
  | grep -a -o -E 'generation throughput: [0-9.]+' | awk '{print $3}' \
  | sort -n | awk '{a[NR]=$1} END {print "n="NR, "min="a[1], "p50="a[int(NR/2)], "max="a[NR]}'
echo "=== f8e4ns : same"
zcat b_lce1_f8e4ns.log.gz 2>/dev/null | grep -a 'Avg generation throughput' \
  | grep -a -v 'generation throughput: 0.0' \
  | grep -a -o -E 'generation throughput: [0-9.]+' | awk '{print $3}' \
  | sort -n | awk '{a[NR]=$1} END {print "n="NR, "min="a[1], "p50="a[int(NR/2)], "max="a[NR]}'
