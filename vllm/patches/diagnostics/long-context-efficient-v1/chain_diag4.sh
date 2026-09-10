#!/bin/bash
# chain_diag4.sh — take over from diag3 the moment dF-t8's block ends,
# skipping diag3's tail prod-restore bounce (a mid-chain prod boot+teardown
# wastes ~15 min and has died 1/1 today; the REAL lane restore is owed only
# at campaign end — diag4 ends with prod_restore127.sh itself).
# Single-shot guard via marker file.
set -u
M=/root/build/lce1/master_diag.log
GUARD=/root/build/lce1/.chain4_launched

[ -e "$GUARD" ] && exit 0

for i in $(seq 1 120); do
  grep -q 'BLOCK dF-t8 end' "$M" 2>/dev/null && break
  sleep 30
done

# grace: let diag3 finish cancel probe + docker cp + gzip of the block log
sleep 45

touch "$GUARD"
# kill diag3 before/early-in its prod restore, plus any orphans of it
pkill -f "master_diag[3]" 2>/dev/null
pkill -f "prod_restore12[7]" 2>/dev/null
pkill -f "serve_benc[h]" 2>/dev/null
sleep 3
docker rm -f lsv-test >/dev/null 2>&1
sleep 20   # settle (xe driver protection)

echo "CHAIN4_GO" >> "$M"
cd /root/build && nohup bash master_diag4.sh >/dev/null 2>&1 &
echo "diag4 launched $(date +%m-%d' '%H:%M:%S)" >> "$M"
