#!/bin/sh
# v64 validation chain on final config (K=2 interleave, budget 1024, mnbt 8192):
# 1) solo-cold baseline at mnbt 8192 (vs 16384's 112.7 s)
# 2) full battery (S1/L2/C4 + xgrammar + thinking + preemption/async checks)
# 3) 3x 14-phase wedge drill
OUT=/root/build/_v64_validate.txt
echo "== SOLO COLD mnbt8192 seed114 ==" > $OUT
python3 /root/build/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 114 >> $OUT 2>&1
echo "== BATTERY (K2@1024) ==" >> $OUT
sh /root/build/battery_v64.sh >> $OUT 2>&1
grep -q BATTERY_V64_DONE /root/build/_v64_battery.txt || echo BATTERY_INCOMPLETE >> $OUT
echo "== WEDGE DRILL 3x ==" >> $OUT
bash /root/build/repro_sustain.sh 3 > /root/build/_v64_sustain.txt 2>&1
DRILL_RC=$?
tail -6 /root/build/_v64_sustain.txt >> $OUT
echo "drill_rc=$DRILL_RC" >> $OUT
echo VALIDATE_V64_DONE >> $OUT
