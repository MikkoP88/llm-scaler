#!/bin/sh
# validate_v1222_run.sh — v1.2.22 promotion validation (fresh-boot lane).
# Bake + fresh boot + solo-cold(seed110, done manually) are already complete;
# this generates sanity_v1222/validate_v1222 from the certified v1221 pair
# (image-generation discipline: sed v1221->v1222 only, then APPEND v1222-
# specific legs) and runs the full chain: sanity (incl. v66 admission PASS
# gate + v88 wheel check + v1221/v1222 marker pedigree), validate (solo cold
# seed114 + battery + 3x 14-phase drill + JIT recheck + fairness + serialized
# 24 x3 legs + burst_harsh x3 + parser battery engine-direct + solo genspeed
# + resets). Acceptance is grepped by the caller.
LOG=/root/build/lce1/v1222_validate_run.log
{
echo "=== VALIDATE_V1222_RUN start $(date +%T) ==="

# 1. sanity_v1222.sh from sanity_v1221.sh (sed only) + EXTRA: wheel + pedigree
S=/root/build/sanity_v1222.sh
sed -e 's/v1221/v1222/g' -e 's/V1221/V1222/g' /root/build/sanity_v1221.sh > "$S"
printf '%s\n' \
  'echo "== V88 WHEEL CHECK ==" >> $R' \
  'docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1' \
  'docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1221_baked /root/.llm_scaler_exp_v1222_baked 2>&1" >> $R 2>&1' \
  'echo "== V1222 SERVE CONFIG (parser + capture sizes) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -o \"--tool-call-parser [a-z0-9_]*\" /root/serve_user.sh" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c cudagraph_capture_sizes /root/serve_user.sh" >> $R 2>&1' \
  'echo "SANITY_V1222_EXTRA_DONE" >> $R' >> "$S"
grep -q SANITY_V1222_DONE "$S" || { echo ABORT_SANITY_GEN; exit 1; }

# 2. validate_v1222.sh from validate_v1221.sh (sed, drop DONE) + v1222 legs
V=/root/build/validate_v1222.sh
sed -e 's/v1221/v1222/g' -e 's/V1221/V1222/g' /root/build/validate_v1221.sh | grep -v '^echo VALIDATE_V1222_DONE' > "$V"
printf '%s\n' \
  'echo "== V1222 SERIALIZED 24 LEG 2 ==" >> $OUT' \
  'OK=0; FAIL=0; H="200"' \
  'for i in $(seq 1 24); do' \
  '  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")' \
  '  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))' \
  '  sleep 6' \
  '  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)' \
  '  [ "$H" != "200" ] && { echo "SERIALIZED2 DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }' \
  'done' \
  '[ "$H" = "200" ] && echo "SERIALIZED2 SURVIVED ok=$OK fail=$FAIL" >> $OUT' \
  'echo "== V1222 SERIALIZED 24 LEG 3 ==" >> $OUT' \
  'OK=0; FAIL=0; H="200"' \
  'for i in $(seq 1 24); do' \
  '  out=$(curl -s --max-time 20 -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"Count from 1 to 5.\",\"max_tokens\":24,\"temperature\":0}" -o /dev/null -w "%{http_code}")' \
  '  [ "$out" = "200" ] && OK=$((OK+1)) || FAIL=$((FAIL+1))' \
  '  sleep 6' \
  '  H=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://localhost:8000/health)' \
  '  [ "$H" != "200" ] && { echo "SERIALIZED3 DEAD cycle $i ok=$OK fail=$FAIL" >> $OUT; break; }' \
  'done' \
  '[ "$H" = "200" ] && echo "SERIALIZED3 SURVIVED ok=$OK fail=$FAIL" >> $OUT' \
  'echo "== V1222 burst_harsh leg C ==" >> $OUT' \
  'bash /root/build/burst_harsh.sh v1222_c 2>&1 | tail -2 >> $OUT' \
  'grep -q "SURVIVED" /root/build/lce1/burst_v1222_c.log || echo BURST_C_NOT_SURVIVED >> $OUT' \
  'echo "== V1222 PARSER BATTERY (engine-direct, qwen3_coder) ==" >> $OUT' \
  'cd /root/build && python3 cc_parser_battery_v89.py >> $OUT 2>&1 || echo PARSER_BATTERY_FAIL >> $OUT' \
  'echo "== V1222 SOLO GENSPEED (expect ~73-74 with 85 sizes) ==" >> $OUT' \
  'cd /root/build && python3 bench_genspeed.py 1 1024 3 >> $OUT 2>&1' \
  'echo VALIDATE_V1222_DONE >> $OUT' >> "$V"
grep -q VALIDATE_V1222_DONE "$V" || { echo ABORT_VALIDATE_GEN; exit 1; }
sh -n "$S" || { echo ABORT_SANITY_SYNTAX; exit 1; }
sh -n "$V" || { echo ABORT_VALIDATE_SYNTAX; exit 1; }
echo "v1222 sanity/validate generated $(date +%T)"

# 3. sanity (acceptance: DONE + admission verdict=PASS)
sh /root/build/sanity_v1222.sh > /dev/null 2>&1
grep -q SANITY_V1222_EXTRA_DONE /root/build/_v1222_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1222_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1222_sanity.txt || true
echo "sanity+admission PASS $(date +%T)"

# 4. full validate chain
sh /root/build/validate_v1222.sh > /dev/null 2>&1
grep -q VALIDATE_V1222_DONE /root/build/_v1222_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1222_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED SURVIVED ok=24' /root/build/_v1222_validate.txt || { echo ABORT_SERIALIZED1_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED2 SURVIVED ok=24' /root/build/_v1222_validate.txt || { echo ABORT_SERIALIZED2_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED3 SURVIVED ok=24' /root/build/_v1222_validate.txt || { echo ABORT_SERIALIZED3_NOT_CLEAN; exit 1; }
grep -q 'BURST_A_NOT_SURVIVED\|BURST_B_NOT_SURVIVED\|BURST_C_NOT_SURVIVED' /root/build/_v1222_validate.txt && { echo ABORT_BURST_NOT_CLEAN; exit 1; } || true
grep -q 'BATTERY_INCOMPLETE' /root/build/_v1222_validate.txt && { echo ABORT_BATTERY; exit 1; } || true
grep -q 'PARSER_BATTERY_FAIL' /root/build/_v1222_validate.txt && { echo ABORT_PARSER_BATTERY; exit 1; } || true
echo "validate CLEAN $(date +%T)"
echo "=== VALIDATE_V1222_RUN COMPLETE $(date +%T) ==="
echo VALIDATE_V1222_RUN_COMPLETE
} > "$LOG" 2>&1
