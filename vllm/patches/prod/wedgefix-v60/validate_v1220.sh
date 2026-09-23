#!/bin/sh
# v1220 validation chain on the fresh V1220 lane: 1) solo-cold baseline
# 2) full battery (S1/L2/C4 + xgrammar + thinking + preemption/async
# checks) 3) 3x 14-phase wedge drill 4) post JIT recheck. v65
# instrumentation must stay absent throughout.
OUT=/root/build/_v1220_validate.txt
echo "== SOLO COLD seed114 ==" > $OUT
python3 /root/build/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 114 >> $OUT 2>&1
echo "== BATTERY ==" >> $OUT
sh /root/build/battery_v64.sh >> $OUT 2>&1
grep -q BATTERY_V64_DONE /root/build/_v64_battery.txt || echo BATTERY_INCOMPLETE >> $OUT
echo "== WEDGE DRILL 3x ==" >> $OUT
bash /root/build/repro_sustain.sh 3 > /root/build/_v1220_sustain.txt 2>&1
DRILL_RC=$?
tail -6 /root/build/_v1220_sustain.txt >> $OUT
echo "drill_rc=$DRILL_RC" >> $OUT
echo "== POST-VALIDATE JIT RECHECK (must be 0) ==" >> $OUT
docker exec lsv-test sh -c "grep -cE 'expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel' /root/serve_full.log || true" >> $OUT 2>&1
echo "== V66 REGRESSION: fairness probe (decode during 106k prefill) ==" >> $OUT
cd /root/build && python3 probe_fair_v63.py >> $OUT 2>&1
echo VALIDATE_V1220_DONE >> $OUT
