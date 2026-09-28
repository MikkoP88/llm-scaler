#!/bin/sh
# validate_v1223_run.sh — v1.2.23-raw promotion validation (fresh-boot lane,
# v123 posture: --async-scheduling + VLLM_RPC_TIMEOUT=60000 + barrier 0).
# Bake + host reboot + fresh boot via repro_bootV1223.sh are done before this.
# Generates sanity_v1223/validate_v1223 from the certified v1222 pair
# (sed v1222->v1223 only), fixes the genspeed expectation label, then APPENDS
# v123-specific legs and runs the full chain: sanity (v66 admission PASS +
# v123 posture gates: async ENGAGED, barrier default "False False 0", serve
# env BARRIER=0 + RPC_TIMEOUT=60000, v123 marker + patcher --check),
# validate (solo cold seed114 + battery + 3x 14-phase drill + JIT recheck +
# fairness + serialized 24 x3 + burst_harsh x3 + parser battery + solo
# genspeed + 4-stream async genspeed + resets). Caller greps acceptance.
LOG=/root/build/lce1/v1223_validate_run.log
{
echo "=== VALIDATE_V1223_RUN start $(date +%T) ==="

# 1. sanity_v1223.sh from sanity_v1222.sh (sed only) + v123 posture extras
S=/root/build/sanity_v1223.sh
sed -e 's/v1222/v1223/g' -e 's/V1222/V1223/g' /root/build/sanity_v1222.sh > "$S"
printf '%s\n' \
  'echo "== V88 WHEEL CHECK ==" >> $R' \
  'docker exec lsv-test /opt/venv/bin/pip freeze 2>/dev/null | grep -i vllm-xpu-kernels >> $R 2>&1' \
  'echo "== V123 PEDIGREE ==" >> $R' \
  'docker exec lsv-test sh -c "ls /root/.llm_scaler_exp_v1221_baked /root/.llm_scaler_exp_v1222_baked /root/.llm_scaler_exp_v1223_baked /root/.v1223_jit_warmed 2>&1" >> $R 2>&1' \
  'echo "== V123 SERVE CONFIG (parser + sizes + ASYNC REQUIRED) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -o \"--tool-call-parser [a-z0-9_]*\" /root/serve_user.sh" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c cudagraph_capture_sizes /root/serve_user.sh" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c -- \"--async-scheduling --port 8000\" /root/serve_user.sh" >> $R 2>&1' \
  'echo "== V123 BARRIER DEFAULT (acceptance: False False 0) ==" >> $R' \
  'docker cp /root/build/patch_barrier_v123_bake.py lsv-test:/root/patch_barrier_v123_bake.py' \
  'docker exec lsv-test /opt/venv/bin/python3 /root/patch_barrier_v123_bake.py --check >> $R 2>&1' \
  'docker exec lsv-test /opt/venv/bin/python3 -c "from vllm.v1.worker import gpu_model_runner as g; print(g._SPEC_DRAFT_BARRIER, g._SPEC_DRAFT_BARRIER_HOST, g._SPEC_DRAFT_BARRIER_MIN_CTX)" >> $R 2>&1' \
  'echo "== V123 ASYNC ENGAGED (acceptance: >=1 both) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -c \"Asynchronous scheduling is enabled\" /root/serve_full.log" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"async_scheduling.: True\" /root/serve_full.log" >> $R 2>&1' \
  'echo "== V123 SERVE ENV (acceptance: BARRIER=0 + RPC_TIMEOUT=60000) ==" >> $R' \
  'docker exec lsv-test sh -c "P=\$(pgrep -f \"vllm serve\" | head -1); tr \"\\0\" \"\\n\" < /proc/\$P/environ | grep -E \"^VLLM_XPU_SPEC_DRAFT_BARRIER=|^VLLM_RPC_TIMEOUT=\"" >> $R 2>&1' \
  'echo "== V123 NOTE: BARRIER HB /dev/shm section above EXPECTED EMPTY (mode 0) ==" >> $R' \
  'echo "SANITY_V1223_EXTRA_DONE" >> $R' >> "$S"
grep -q SANITY_V1223_EXTRA_DONE "$S" || { echo ABORT_SANITY_GEN; exit 1; }

# 2. validate_v1223.sh from validate_v1222.sh (sed, fix genspeed label, drop DONE) + v123 legs
V=/root/build/validate_v1223.sh
sed -e 's/v1222/v1223/g' -e 's/V1222/V1223/g' /root/build/validate_v1222.sh \
  | sed -e 's/expect ~73-74 with 85 sizes/expect 120+ async (floor 70 = sync parity)/' \
  | grep -v '^echo VALIDATE_V1223_DONE' > "$V"
printf '%s\n' \
  'echo "== V123 ASYNC GENSPEED 4x1024 (acceptance: aggregate >= 100; sync was 72-74) ==" >> $OUT' \
  'cd /root/build && python3 bench_genspeed.py 4 1024 1 >> $OUT 2>&1' \
  'echo "== V123 SERVE LOG ERRORS (acceptance: 0 tracebacks after boot) ==" >> $OUT' \
  'docker exec lsv-test sh -c "grep -ciE \"traceback|EngineDeadError\" /root/serve_full.log" >> $OUT 2>&1' \
  'echo VALIDATE_V1223_DONE >> $OUT' >> "$V"
grep -q VALIDATE_V1223_DONE "$V" || { echo ABORT_VALIDATE_GEN; exit 1; }
sh -n "$S" || { echo ABORT_SANITY_SYNTAX; exit 1; }
sh -n "$V" || { echo ABORT_VALIDATE_SYNTAX; exit 1; }
echo "v1223 sanity/validate generated $(date +%T)"

# 3. sanity (acceptance: EXTRA_DONE + admission PASS + async engaged + barrier posture)
sh /root/build/sanity_v1223.sh > /dev/null 2>&1
grep -q SANITY_V1223_EXTRA_DONE /root/build/_v1223_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1223_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1223_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "ABORT_ASYNC_NOT_ENGAGED got=$ASY"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1223_sanity.txt)
[ "$BARR" = "False False 0" ] || { echo "ABORT_BARRIER_DEFAULT got=$BARR"; exit 1; }
grep -q 'V123_BARRIER APPLIED' /root/build/_v1223_sanity.txt || { echo ABORT_V123_MARKER; exit 1; }
grep -q 'VLLM_XPU_SPEC_DRAFT_BARRIER=0' /root/build/_v1223_sanity.txt || { echo ABORT_BARRIER_ENV; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1223_sanity.txt || { echo ABORT_RPC_TIMEOUT_ENV; exit 1; }
echo "sanity+admission+v123-posture PASS $(date +%T)"

# 4. full validate chain
sh /root/build/validate_v1223.sh > /dev/null 2>&1
grep -q VALIDATE_V1223_DONE /root/build/_v1223_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1223_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED SURVIVED ok=24' /root/build/_v1223_validate.txt || { echo ABORT_SERIALIZED1_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED2 SURVIVED ok=24' /root/build/_v1223_validate.txt || { echo ABORT_SERIALIZED2_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED3 SURVIVED ok=24' /root/build/_v1223_validate.txt || { echo ABORT_SERIALIZED3_NOT_CLEAN; exit 1; }
grep -q 'BATTERY_INCOMPLETE' /root/build/_v1223_validate.txt && { echo ABORT_BATTERY; exit 1; } || true
grep -q 'PARSER_BATTERY_FAIL' /root/build/_v1223_validate.txt && { echo ABORT_PARSER_BATTERY; exit 1; } || true
grep -qE 'BURST_[ABC]_NOT_SURVIVED' /root/build/_v1223_validate.txt && { echo ABORT_BURST_NOT_CLEAN; exit 1; } || true
JIT=$(awk '/POST-VALIDATE JIT RECHECK/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1223_validate.txt)
# v123 recalibration (KNOWN_ISSUES #28 + v1222-raw precedent +2): monitor lines are
# first-use loads/compiles; the -raw-round gate is a bounded triton cache delta.
# Strict zero-writes remains the PRODUCTION-image gate (ship warm-verify + gates floor).
CACHE_NOW=$(docker exec lsv-test sh -c 'ls /root/.triton/cache | wc -l' | tr -d '[:space:]')
CACHE_RAW=$(docker run --rm --entrypoint sh llm-scaler-exp:v1.2.23-raw -c 'ls /root/.triton/cache | wc -l' 2>/dev/null | tr -d '[:space:]')
JIT_DELTA=$((CACHE_NOW - CACHE_RAW))
echo "jit_recheck_lines=$JIT cache_delta=$JIT_DELTA (bound 2; v1222-raw precedent +2)"
if [ "$JIT_DELTA" -lt 0 ] || [ "$JIT_DELTA" -gt 2 ]; then { echo "ABORT_JIT_CACHE_DELTA got=$JIT_DELTA"; exit 1; }; fi
TB=$(awk '/V123 SERVE LOG ERRORS/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1223_validate.txt)
[ "$TB" = "0" ] || { echo "ABORT_SERVE_TRACEBACKS got=$TB"; exit 1; }
RS=$(awk '/engine resets in dmesg/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1223_validate.txt)
[ "$RS" = "0" ] || { echo "ABORT_ENGINE_RESETS got=$RS"; exit 1; }
grep -m1 'SANITY finish' /root/build/_v1223_sanity.txt || true
echo "== V123 GENSPEED EVIDENCE (verify manually: solo 120+ class / 4x1024 >= 100 agg) =="
grep -A6 'SOLO GENSPEED' /root/build/_v1223_validate.txt | head -10
grep -A6 'ASYNC GENSPEED' /root/build/_v1223_validate.txt | head -10
echo "=== VALIDATE_V1223_RUN ALL GATES PASS $(date +%T) ==="
} > "$LOG" 2>&1
tail -40 "$LOG"
grep -q 'VALIDATE_V1223_RUN ALL GATES PASS' "$LOG" || exit 1
echo V1223_VALIDATION_COMPLETE
