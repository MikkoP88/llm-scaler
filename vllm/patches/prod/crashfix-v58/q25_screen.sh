#!/bin/bash
# q25_screen.sh — F-phase gate screen on fw-aligned host (GuC 70.72.1).
# Gates: f8ref MUST-equal, p1_soak, bench3 cold/warm, q17_perf cold/warm.
cd /root/build
L=lce1

echo "=== GATE 1: f8ref (numerics MUST-equal cb8c3851b897/68332ec7c31b/05c88ff03b0c)"
python3 f8ref.py q25e5m2 > $L/f8ref_q25e5m2.txt 2>&1
cat $L/f8ref_q25e5m2.txt

echo "=== GATE 2: p1_soak (expect 16/16 bad=0)"
python3 p1_soak.py > $L/p1_soak_q25.out 2>&1
tail -6 $L/p1_soak_q25.out

echo "=== GATE 3: bench3 COLD"
python3 dt_bench3.py > $L/bench3_Q25_cold.out 2>&1
tail -12 $L/bench3_Q25_cold.out

echo "=== GATE 4: q17_perf COLD (solo + conc4)"
bash q17_perf.sh > $L/q25_perf_cold.out 2>&1
tail -8 $L/q25_perf_cold.out

echo "=== GATE 5: bench3 WARM"
python3 dt_bench3.py > $L/bench3_Q25_warm.out 2>&1
tail -12 $L/bench3_Q25_warm.out

echo "=== GATE 6: q17_perf WARM"
bash q17_perf.sh > $L/q25_perf_warm.out 2>&1
tail -8 $L/q25_perf_warm.out

echo "=== guard fires + census"
docker exec lsv-test sh -c 'grep -c "v58 P1 uniform-decode prefill guard" /root/serve_full.log' || true
dmesg | grep -c 'Engine reset' || true
echo Q25_SCREEN_DONE $(date +%H:%M:%S)
