#!/bin/bash
# h_live_capture.sh — emergency live-stack capture for wedged worker 609.
set -u
D=/root/build/lce1/bootH
date -u
echo '=== fr headers ==='
grep -n '^== ' "$D/fr_tail400_t0.txt"
echo '=== TP0 (609) fr last 15 ==='
sed -n '/fr_609/,/fr_638/p' "$D/fr_tail400_t0.txt" | tail -15
echo '=== py-spy / gdb availability ==='
docker exec lsv-test sh -c 'which py-spy; ls /opt/venv/bin/ 2>/dev/null | grep -i spy; which gdb; ls /usr/bin | grep -x gdb'
echo '=== hot threads 609 (utime+stime jiffies) ==='
docker exec lsv-test sh -c 'for t in /proc/609/task/*; do st=$(awk "{print \$3}" $t/stat 2>/dev/null); u=$(awk "{print \$14+\$15}" $t/stat 2>/dev/null); w=$(cat $t/wchan 2>/dev/null); echo "$u $(basename $t) $st $w"; done | sort -rn | head -6'
echo '=== hot threads 638 ==='
docker exec lsv-test sh -c 'for t in /proc/638/task/*; do st=$(awk "{print \$3}" $t/stat 2>/dev/null); u=$(awk "{print \$14+\$15}" $t/stat 2>/dev/null); w=$(cat $t/wchan 2>/dev/null); echo "$u $(basename $t) $st $w"; done | sort -rn | head -6'
echo '=== 609/638 state + serve tail ==='
docker exec lsv-test sh -c 'awk "{print \$3}" /proc/609/stat /proc/638/stat 2>/dev/null'
docker exec lsv-test sh -c 'tail -5 /root/serve_full.log | grep -v "GET /"'
echo '=== fds/mappings hint: 609 maps count ==='
docker exec lsv-test sh -c 'wc -l /proc/609/maps 2>/dev/null'
echo LIVE_CAPTURE_DONE
