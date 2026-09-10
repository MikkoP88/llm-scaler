#!/bin/bash
# chain_diag5.sh — take over when diag4 completes (MASTER_DIAG4_DONE),
# settle, and launch master_diag5.sh (bake validation + q1/k2 residual
# discriminators). Single-shot guard via marker file.
set -u
M=/root/build/lce1/master_diag.log
GUARD=/root/build/lce1/.chain5_launched

[ -e "$GUARD" ] && exit 0

for i in $(seq 1 480); do
  grep -q 'MASTER_DIAG4_DONE$' "$M" 2>/dev/null && break
  sleep 30
done

# grace: let diag4's prod restore settle
sleep 45

touch "$GUARD"
pkill -f "master_diag[4]" 2>/dev/null
sleep 3
echo "CHAIN5_GO" >> "$M"
cd /root/build && nohup bash master_diag5.sh >/dev/null 2>&1 &
echo "diag5 launched $(date +%m-%d' '%H:%M:%S)" >> "$M"
