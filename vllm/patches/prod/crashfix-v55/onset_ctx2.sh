#!/bin/bash
# onset_ctx2.sh — last NORMAL (non-ERROR) lines before the fence block.
set -u
for d in stall_p14_e5m2 stall2_p13_e5m2 stall3_p12_e4m3 stall4_p14_e4m3_freshnode; do
  F=/root/build/lce1/$d/serve_tail.log
  echo "===== $d ====="
  N=$(grep -n 'ASYNC-EVENT-STALL: num_accepted' "$F" | head -1 | cut -d: -f1)
  # walk back to the first line of the contiguous ERROR block
  B=$N
  while [ "$B" -gt 1 ] && grep -q 'ERROR' <(sed -n "$((B-1))p" "$F"); do B=$((B-1)); done
  echo "error-block-starts=$B"
  A=$((B-22)); [ "$A" -lt 1 ] && A=1
  sed -n "${A},$((B-1))p" "$F" | grep -v 'GET /health' | tail -20
done
