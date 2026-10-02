#!/bin/bash
# repro_sustain.sh — sustained wedge-trigger attempts: N back-to-back
# dt_warmup_v53 runs (each = p1-p14 incl. p11 stagger, p12 solo long
# decode, p13 concurrent, p14 full 5-stream cycle) on the STANDING
# instrumented boot. Auto-captures at first fence and stops.
set -u
N="${1:-4}"
for i in $(seq 1 "$N"); do
  echo "=== SUSTAIN ROUND $i/$(date +%H:%M:%S) ==="
  python3 /root/build/dt_warmup_v53.py >> /root/build/lce1/sustain_warmup.out 2>&1
  echo "round $i exit=$? $(date +%H:%M:%S)"
  F=$(docker exec lsv-test sh -c 'grep -c ASYNC-EVENT-STALL /root/serve_full.log' 2>/dev/null)
  echo "fence-hits=$F"
  if [ "${F:-0}" != "0" ]; then
    echo "WEDGE IN ROUND $i — capturing"
    bash /root/build/wedge_capture.sh "repro_wedge_r${i}"
    exit 2
  fi
done
echo "SUSTAIN_COMPLETE_NO_WEDGE ($N rounds)"
