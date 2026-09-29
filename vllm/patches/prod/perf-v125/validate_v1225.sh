#!/bin/sh
# v1225 validation chain on the fresh V1225 lane: 1) solo-cold baseline
# 2) full battery (S1/L2/C4 + xgrammar + thinking + preemption/async
# checks) 3) 3x 14-phase wedge drill 4) post JIT recheck. v65
# instrumentation must stay absent throughout.
OUT=/root/build/_v1225_validate.txt
echo "== SOLO COLD seed114 ==" > $OUT
python3 /root/build/probe_solo_cold.py http://127.0.0.1:8000 qwen3.8-27b-fp8 114 >> $OUT 2>&1
echo "== BATTERY ==" >> $OUT
sh /root/build/battery_v64.sh >> $OUT 2>&1
grep -q BATTERY_V64_DONE /root/build/_v64_battery.txt || echo BATTERY_INCOMPLETE >> $OUT
echo "== WEDGE DRILL 3x ==" >> $OUT
bash /root/build/repro_sustain.sh 3 > /root/build/_v1225_sustain.txt 2>&1
DRILL_RC=$?
tail -6 /root/build/_v1225_sustain.txt >> $OUT
echo "drill_rc=$DRILL_RC" >> $OUT
echo "== POST-VALIDATE JIT RECHECK (must be 0) ==" >> $OUT
docker exec lsv-test sh -c "grep -cE 'expand_kernel|rejection_greedy_sample_kernel|eagle_prepare_inputs_padded_kernel|eagle_prepare_next_token_padded_kernel|eagle_step_slot_mapping_metadata_kernel|_zero_kv_blocks_kernel|_compute_slot_mapping_kernel|batch_memcpy_kernel' /root/serve_full.log || true" >> $OUT 2>&1
echo "== V66 REGRESSION: fairness probe (decode during 106k prefill) ==" >> $OUT
cd /root/build && python3 probe_fair_v63.py >> $OUT 2>&1
echo "== V88 WEDGE ACCEPTANCE: serialized 24 (historically DEAD at 17) ==" >> $OUT
OK=0; FAIL=0; H="200"
for i in $(seq 1 24); do
  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")
  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
  sleep 6
  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)
  [ "$H" != "200" ] && { echo "SERIALIZED DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }
done
[ "$H" = "200" ] && echo "SERIALIZED SURVIVED ok=$OK fail=$FAIL" >> $OUT
echo "== V88 WEDGE ACCEPTANCE: burst_harsh x2 ==" >> $OUT
bash /root/build/burst_harsh.sh v1225_a 2>&1 | tail -2 >> $OUT
bash /root/build/burst_harsh.sh v1225_b 2>&1 | tail -2 >> $OUT
grep -q "SURVIVED" /root/build/lce1/burst_v1225_a.log || echo BURST_A_NOT_SURVIVED >> $OUT
grep -q "SURVIVED" /root/build/lce1/burst_v1225_b.log || echo BURST_B_NOT_SURVIVED >> $OUT
echo "== WEDGES: engine resets in dmesg (must be 0 new) ==" >> $OUT
dmesg | grep -c "Engine reset" >> $OUT 2>&1
echo "== V1225 SERIALIZED 24 LEG 2 ==" >> $OUT
OK=0; FAIL=0; H="200"
for i in $(seq 1 24); do
  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")
  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
  sleep 6
  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)
  [ "$H" != "200" ] && { echo "SERIALIZED2 DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }
done
[ "$H" = "200" ] && echo "SERIALIZED2 SURVIVED ok=$OK fail=$FAIL" >> $OUT
echo "== V1225 SERIALIZED 24 LEG 3 ==" >> $OUT
OK=0; FAIL=0; H="200"
for i in $(seq 1 24); do
  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")
  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))
  sleep 6
  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)
  [ "$H" != "200" ] && { echo "SERIALIZED3 DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }
done
[ "$H" = "200" ] && echo "SERIALIZED3 SURVIVED ok=$OK fail=$FAIL" >> $OUT
echo "== V1225 burst_harsh leg C ==" >> $OUT
bash /root/build/burst_harsh.sh v1225_c 2>&1 | tail -2 >> $OUT
grep -q "SURVIVED" /root/build/lce1/burst_v1225_c.log || echo BURST_C_NOT_SURVIVED >> $OUT
echo "== V1225 PARSER BATTERY (engine-direct, qwen3_coder) ==" >> $OUT
cd /root/build && python3 cc_parser_battery_v89.py >> $OUT 2>&1 || echo PARSER_BATTERY_FAIL >> $OUT
echo "== V1225 SOLO GENSPEED (expect 120+ async (floor 70 = sync parity)) ==" >> $OUT
cd /root/build && python3 bench_genspeed.py 1 1024 3 >> $OUT 2>&1
echo "== V123 ASYNC GENSPEED 4x1024 (acceptance: aggregate >= 100; sync was 72-74) ==" >> $OUT
cd /root/build && python3 bench_genspeed.py 4 1024 1 >> $OUT 2>&1
echo "== V123 SERVE LOG ERRORS (acceptance: 0 tracebacks after boot) ==" >> $OUT
docker exec lsv-test sh -c "grep -ciE \"traceback|EngineDeadError\" /root/serve_full.log" >> $OUT 2>&1
echo "== V124 RUNTIME FP8 RESOLUTION (acceptance: float8_e4m3fn float8_e5m2) ==" >> $OUT
docker exec lsv-test /opt/venv/bin/python3 -c "import torch; from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C; a=C.gated_delta_net_state_dtype(torch.float16, \"auto\", \"fp8_e4m3\"); b=C.gated_delta_net_state_dtype(torch.float16, \"auto\", \"fp8_e5m2\"); print(a[1], b[1])" >> $OUT 2>&1
echo "== V125 C7 FUNCTIONAL ON LANE (informational dup of the pre-chain gate) ==" >> $OUT
docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-test /opt/venv/bin/python3 -c "import vllm._xpu_ops" > /root/build/_v1225_c7ref.txt 2>&1; echo "c7_refuse_rc=$?" >> $OUT
tail -1 /root/build/_v1225_c7ref.txt >> $OUT 2>&1
docker exec lsv-test /opt/venv/bin/python3 -c "import vllm._xpu_ops; print(\"C7_UNSET_IMPORT_OK\")" >> $OUT 2>&1
echo VALIDATE_V1225_DONE >> $OUT
