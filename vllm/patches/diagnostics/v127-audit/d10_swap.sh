#!/bin/bash
# d10_swap.sh — v127-audit D10 discriminator: df7_tq4 deep cells REVERSED
# (221000 first, then 115000; prompts bit-identical to lane df7_tq4 via the
# same dt_ctxscan2.py + nwords). If the acceptance collapse follows the
# PROMPT -> second cell (115000) collapses again (4.2-class); if it follows
# POSITION/first-decode state -> first cell (221000) collapses instead.
set -u
OUT=/root/build/bench127/d10swap
mkdir -p "$OUT"
R="$OUT/report.txt"
M=/root/build/bench127/master.log

echo "== D10SWAP teardown+boot $(date +%H:%M:%S)" | tee -a "$R" "$M"
docker rm -f lsv-test >/dev/null 2>&1
sleep 5
if bash /root/build/serve_bench.sh turboquant_4bit_nc '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' \
    0.9 262144 b_d10swap '' '' > "$OUT/boot.out" 2>&1; then
  echo "BOOT_OK" | tee -a "$R"
else
  echo "BOOT_FAIL" | tee -a "$R"; tail -15 "$OUT/boot.out" | tee -a "$R"; exit 1
fi

# health wait
HOK=0
for i in $(seq 1 60); do
  sleep 10
  if curl -s -o /dev/null -m 4 http://localhost:8000/health; then HOK=1; break; fi
done
[ "$HOK" = "1" ] && echo "HEALTH_OK $(date +%H:%M:%S)" | tee -a "$R" || { echo "HEALTH_TIMEOUT" | tee -a "$R"; exit 1; }

# warmup p1-p9 (lane parity: lane 2 warmed before deep)
ND0=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
nohup python3 /root/build/dt_warmup_v51.py >/dev/null 2>&1 &
WOK=0
for i in $(seq 1 90); do
  sleep 10
  ND=$(grep -c WARMUP_DONE /root/dt_warmup.log 2>/dev/null || echo 0)
  if [ "$ND" -gt "$ND0" ]; then WOK=1; break; fi
done
[ "$WOK" = "1" ] && echo "WARMUP_OK $(date +%H:%M:%S)" | tee -a "$R" || echo "WARMUP_TIMEOUT (continuing)" | tee -a "$R"

# REVERSED deep cells, timestamps around each for acceptance correlation
echo "CELL_A_221000_START $(date +%H:%M:%S)" | tee -a "$R"
python3 /root/build/dt_ctxscan2.py d10swap "221000" 256 > "$OUT/cellA.out" 2>&1
grep CTXSCAN2 "$OUT/cellA.out" | tee -a "$R"
echo "CELL_A_END_B_START $(date +%H:%M:%S)" | tee -a "$R"
python3 /root/build/dt_ctxscan2.py d10swap "115000" 256 > "$OUT/cellB.out" 2>&1
grep CTXSCAN2 "$OUT/cellB.out" | tee -a "$R"
echo "CELL_B_END $(date +%H:%M:%S)" | tee -a "$R"

# preserve engine log (bind-mounted) + snapshot SpecDecoding windows
docker cp lsv-test:/root/b_d10swap.log "$OUT/b_d10swap.log" >/dev/null 2>&1 || cp /root/b_d10swap.log "$OUT/b_d10swap.log" 2>/dev/null || true
grep "Mean acceptance" "$OUT/b_d10swap.log" | tail -80 > "$OUT/accept_windows.txt" || true
docker rm -f lsv-test >/dev/null 2>&1
echo "D10SWAP_DONE $(date +%H:%M:%S)" | tee -a "$R" "$M"
