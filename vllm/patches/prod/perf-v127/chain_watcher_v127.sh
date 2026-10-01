#!/bin/bash
# chain_watcher_v127.sh — append a compact progress line every 120s for the
# validate_v1226.sh chain + lane queue + watchdog drift, so the W1 record
# survives without continuous manual polls. Runs up to 6h then exits.
W=/root/build/v127_stage/chain_watch.log
P=/root/build/_v1226_sustain.txt
V=/root/build/_v1226_validate.txt
L=/root/build/lane_watchdog.log
end=$(( $(date +%s) + 21600 ))
echo "=== chain_watcher start $(date '+%F %T') ===" >> $W
while [ $(date +%s) -lt $end ]; do
  S=$(tail -1 $P 2>/dev/null)
  D=$(tail -1 $V 2>/dev/null)
  Q=$(curl -s -m 5 http://localhost:8000/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | tr -s ' ' | cut -d ' ' -f2 | paste -sd/ -)
  X=$(grep -cE 'DRIFT:' $L 2>/dev/null); X=${X:-0}
  echo "$(date '+%T') sustain=[$S] validate=[$D] queue=$Q drift_lines=$X" >> $W
  # stop when the chain is done
  grep -q 'VALIDATE_V1226_DONE\|ABORT' $V 2>/dev/null && { echo "=== chain ended $(date '+%T') ===" >> $W; exit 0; }
  sleep 120
done
echo "=== chain_watcher expiry $(date '+%T') ===" >> $W
