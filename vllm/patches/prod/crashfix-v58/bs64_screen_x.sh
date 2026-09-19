#!/bin/bash
# bs64_screen_x.sh — bs64 lane screen on EXTENDED bench3 (§24 Q: +ctx128k).
# Identical protocol to bs64_screen.sh; GATE 3/3b run dt_bench3x.py
# (2k/16k/65k/128k + conc8 + acceptance) instead of dt_bench3.py.
# usage: bs64_screen_x.sh <LABEL> <CERT_PROBE1>
#   e5m2 cert  cb8c3851b897   (68332ec7c31b / 05c88ff03b0c)
#   e4m3 cert  52f598e7d38a   (6b1c26403bfc / 95e24129958b)
cd /root/build
L=lce1
LBL="${1:?label}"
CERT="${2:?cert probe1}"

echo "=== [$LBL] GATE 1: f8ref (cert=$CERT)"
timeout 360 python3 -u f8ref.py "bs64_$LBL" 2>&1 | tee "$L/f8ref_bs64_$LBL.out" | grep -o 'hashes=.*' || true
grep -q 'distinct=OK' "$L/f8ref_bs64_$LBL.out" && echo F8REF_DETERMINISTIC || echo F8REF_NONDET
grep -q "$CERT" "$L/f8ref_bs64_$LBL.out" && echo F8REF_EXACT || echo F8REF_NOT_EXACT

echo "=== [$LBL] GATE 2: q17 solo+conc4 (clock-guarded)"
( for i in 1 2 3 4 5 6; do
    xpu-smi dump --device 1 --metrics all --number 1 > "$L/bs64_${LBL}_clk_$i.txt" 2>/dev/null
    sleep 4
  done ) &
CLKPID=$!
bash q17_perf.sh > "$L/bs64_${LBL}_q17_cold.out" 2>&1
kill $CLKPID 2>/dev/null
tail -4 "$L/bs64_${LBL}_q17_cold.out"

echo "=== [$LBL] GATE 3: bench3x COLD"
python3 dt_bench3x.py > "$L/bench3X_BS64${LBL}_cold.out" 2>&1
tail -8 "$L/bench3X_BS64${LBL}_cold.out"

echo "=== [$LBL] GATE 3b: bench3x WARM"
python3 dt_bench3x.py > "$L/bench3X_BS64${LBL}_warm.out" 2>&1
tail -8 "$L/bench3X_BS64${LBL}_warm.out"

echo "=== [$LBL] GATE 4: q17 warm"
bash q17_perf.sh > "$L/bs64_${LBL}_q17_warm.out" 2>&1
tail -4 "$L/bs64_${LBL}_q17_warm.out"

echo "=== [$LBL] census"
docker exec lsv-test sh -c "grep -o -- '--block-size [0-9]*' /root/serve_user.sh; grep -m3 'block_size' /root/serve_full.log" 2>/dev/null || true
echo "engine_resets=$(dmesg | grep -ac 'Engine reset')"
echo "BS64_SCREEN_X_DONE [$LBL] $(date +%H:%M:%S)"
