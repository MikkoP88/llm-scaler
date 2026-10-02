#!/bin/bash
# onset_ctx.sh — for each stall capture, print the lines immediately before
# the first ASYNC-EVENT-STALL fence line (the onset context).
set -u
for d in stall_p14_e5m2 stall2_p13_e5m2 stall3_p12_e4m3 stall4_p14_e4m3_freshnode; do
  F=/root/build/lce1/$d/serve_tail.log
  echo "===== $d ====="
  N=$(grep -n 'ASYNC-EVENT-STALL: num_accepted' "$F" | head -1 | cut -d: -f1)
  echo "first-fence-line=$N of $(wc -l < "$F")"
  if [ -n "${N:-}" ]; then
    A=$((N-18)); [ "$A" -lt 1 ] && A=1
    sed -n "${A},$((N-1))p" "$F" | grep -v 'GET /health' | tail -15
  fi
done
