#!/bin/bash
# nv_screen.sh — NEO-version screen gates (crashfix-v58 §24 F bisect).
#   usage: nv_screen.sh <LABEL>
# Gates (in order):
#   1. f8ref numerics MUST-equal cb8c3851b897/68332ec7c31b/05c88ff03b0c
#      (mixed IGC stacks that change codegen -> hash drift = FAIL)
#   2. q17 solo+conc4 with GPU-clock mode-guard (2800=boost / 950=parked;
#      §24 mode-flip lesson: parked readings are invalid, re-boost via
#      prefill then re-measure)
#   3. bench3 cold (TTFT + decode cells + conc8 + acceptance)
#   4. q17 warm re-run
# Verdict bands (stock cert): solo 49-51.5, conc4 123-148, decode
# 2k>=390 16k>=420(w) 65k>=340, acc 0.74±0.01. 26.18-class = solo ~38.
cd /root/build
L=lce1
LBL="${1:?label}"
CERT=cb8c3851b897

echo "=== [$LBL] GATE 1: f8ref MUST-equal"
timeout 360 python3 -u f8ref.py nv_$LBL 2>&1 | tee $L/f8ref_nv_$LBL.out | grep -o 'hashes=.*' || true
if grep -q "$CERT" $L/f8ref_nv_$LBL.out; then echo "F8REF_PASS"; else echo "F8REF_FAIL"; fi

echo "=== [$LBL] GATE 2: q17 solo+conc4 (clock-guarded)"
( for i in 1 2 3 4 5 6; do
    xpu-smi dump --device 1 --metrics all --number 1 > $L/nv_${LBL}_clk_$i.txt 2>/dev/null
    sleep 4
  done ) &
CLKPID=$!
bash q17_perf.sh > $L/nv_${LBL}_q17_cold.out 2>&1
kill $CLKPID 2>/dev/null
tail -4 $L/nv_${LBL}_q17_cold.out
grep -hoE '[0-9]{3,4}(\.[0-9]+)?' $L/nv_${LBL}_clk_*.txt 2>/dev/null | sort -rn | head -3 > $L/nv_${LBL}_clkmax.txt || true
echo "clock samples (gpu freq field, top-3): $(tr '\n' ' ' < $L/nv_${LBL}_clkmax.txt 2>/dev/null)"

echo "=== [$LBL] GATE 3: bench3 COLD"
python3 dt_bench3.py > $L/bench3_NV${LBL}_cold.out 2>&1
tail -7 $L/bench3_NV${LBL}_cold.out

echo "=== [$LBL] GATE 4: q17 warm"
bash q17_perf.sh > $L/nv_${LBL}_q17_warm.out 2>&1
tail -4 $L/nv_${LBL}_q17_warm.out

echo "=== [$LBL] census + resets"
docker exec lsv-test sh -c "dpkg -l | grep -E 'intel-opencl-icd|intel-igc|libigdgmm|libze-intel-gpu1' | awk '{print \$2, \$3}'"
echo "engine_resets=$(dmesg | grep -ac 'Engine reset')"
echo "NV_SCREEN_DONE [$LBL] $(date +%H:%M:%S)"
