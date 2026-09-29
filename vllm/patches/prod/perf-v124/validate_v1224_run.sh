#!/bin/sh
# validate_v1224_run.sh — v1.2.24-raw promotion validation (fresh-boot lane).
# v124 posture = v123 posture (--async-scheduling + VLLM_RPC_TIMEOUT=60000 +
# barrier 0, parser coder, 85 capture sizes) + P16 non-spec gate fix (the
# v31.1 inductor-disable gate widened to TP>1) + P19.5a fp8 mamba SSM cache
# plumbing (inert on the certified fp16 serve). Bake + host reboot + fresh
# boot via repro_bootV1224.sh are done before this runs.
# Chain: generates sanity_v1224/validate_v1224 from the certified v1223 pair
# (sed v1223->v1224; the v123 barrier patcher name is PROTECTED — the script
# is version-agnostic and still prints its literal V123_BARRIER marker),
# appends v124 code-delta legs, then runs the full battery: sanity (v66
# admission PASS + posture gates + v124 deltas) + validate (solo cold seed114
# + battery + 3x 14-phase drill + JIT recheck + fairness + serialized 24 x3
# + burst_harsh x3 + parser battery + solo genspeed + 4-stream async genspeed
# + resets) + v124 asserts (P16/P19.5a markers, runtime fp8 resolution, mamba
# fp16 serve default, triton cache delta bound vs v1.2.24-raw).
LOG=/root/build/lce1/v1224_validate_run.log
{
echo "=== VALIDATE_V1224_RUN start $(date +%T) ==="

# 1. sanity_v1224.sh from sanity_v1223.sh (sed; PROTECT the v123 patcher name)
S=/root/build/sanity_v1224.sh
sed -e 's/v1223/v1224/g' -e 's/V1223/V1224/g' \
    -e 's/patch_barrier_v1224_bake/patch_barrier_v123_bake/g' \
    /root/build/sanity_v1223.sh > "$S"
printf '%s\n' \
  'echo "== V124 CODE DELTAS (acceptance: P16=2; P19.5a cache=1 mamba=1 ops=1) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -c \"v124 P16\" /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/config/cache.py" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/mamba_utils.py" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"v124 P19.5a\" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py" >> $R 2>&1' \
  'echo "== V124 MAMBA SERVE DEFAULT (acceptance: 1 / 0 — plumbing inert) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -c -- \"mamba-ssm-cache-dtype float16\" /root/serve_user.sh" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c -- \"mamba-ssm-cache-dtype fp8\" /root/serve_user.sh" >> $R 2>&1' \
  'echo "SANITY_V1224_EXTRA2_DONE" >> $R' >> "$S"
grep -q SANITY_V1224_EXTRA2_DONE "$S" || { echo ABORT_SANITY_GEN; exit 1; }

# 2. validate_v1224.sh from validate_v1223.sh (sed; drop DONE; + v124 legs)
V=/root/build/validate_v1224.sh
sed -e 's/v1223/v1224/g' -e 's/V1223/V1224/g' /root/build/validate_v1223.sh \
  | grep -v '^echo VALIDATE_V1224_DONE' > "$V"
printf '%s\n' \
  'echo "== V124 RUNTIME FP8 RESOLUTION (acceptance: float8_e4m3fn float8_e5m2) ==" >> $OUT' \
  'docker exec lsv-test /opt/venv/bin/python3 -c "import torch; from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C; a=C.gated_delta_net_state_dtype(torch.float16, \"auto\", \"fp8_e4m3\"); b=C.gated_delta_net_state_dtype(torch.float16, \"auto\", \"fp8_e5m2\"); print(a[1], b[1])" >> $OUT 2>&1' \
  'echo VALIDATE_V1224_DONE >> $OUT' >> "$V"
grep -q VALIDATE_V1224_DONE "$V" || { echo ABORT_VALIDATE_GEN; exit 1; }
sh -n "$S" || { echo ABORT_SANITY_SYNTAX; exit 1; }
sh -n "$V" || { echo ABORT_VALIDATE_SYNTAX; exit 1; }
echo "v1224 sanity/validate generated $(date +%T)"

# 3. sanity (acceptance: EXTRA_DONE + admission PASS + async + barrier posture)
sh /root/build/sanity_v1224.sh > /dev/null 2>&1
grep -q SANITY_V1224_EXTRA_DONE /root/build/_v1224_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -q SANITY_V1224_EXTRA2_DONE /root/build/_v1224_sanity.txt || { echo ABORT_SANITY2; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1224_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
# NB: the inherited posture legs keep their literal 3-char "V123" headers
# (the generation sed only renames the 4-char V1223) — key on the ACTUAL
# label text, not the version.
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1224_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "ABORT_ASYNC_NOT_ENGAGED got=$ASY"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1224_sanity.txt)
echo "$BARR" | grep -q 'False False 0' || { echo "ABORT_BARRIER_DEFAULT got=$BARR"; exit 1; }
# The patcher's literal output is version-agnostic ("V123_BARRIER APPLIED").
grep -q 'V123_BARRIER APPLIED' /root/build/_v1224_sanity.txt || { echo ABORT_BARRIER_MARKER; exit 1; }
grep -q 'VLLM_XPU_SPEC_DRAFT_BARRIER=0' /root/build/_v1224_sanity.txt || { echo ABORT_BARRIER_ENV; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1224_sanity.txt || { echo ABORT_RPC_TIMEOUT_ENV; exit 1; }
echo "sanity+admission+v124-posture PASS $(date +%T)"

# 4. full validate chain
sh /root/build/validate_v1224.sh > /dev/null 2>&1
grep -q VALIDATE_V1224_DONE /root/build/_v1224_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1224_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED SURVIVED ok=24' /root/build/_v1224_validate.txt || { echo ABORT_SERIALIZED1_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED2 SURVIVED ok=24' /root/build/_v1224_validate.txt || { echo ABORT_SERIALIZED2_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED3 SURVIVED ok=24' /root/build/_v1224_validate.txt || { echo ABORT_SERIALIZED3_NOT_CLEAN; exit 1; }
grep -q 'BATTERY_INCOMPLETE' /root/build/_v1224_validate.txt && { echo ABORT_BATTERY; exit 1; } || true
grep -q 'PARSER_BATTERY_FAIL' /root/build/_v1224_validate.txt && { echo ABORT_PARSER_BATTERY; exit 1; } || true
grep -qE 'BURST_[ABC]_NOT_SURVIVED' /root/build/_v1224_validate.txt && { echo ABORT_BURST_NOT_CLEAN; exit 1; } || true
JIT=$(awk '/POST-VALIDATE JIT RECHECK/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1224_validate.txt)
# v123 recalibration stands (KNOWN_ISSUES #28): monitor lines are first-use
# loads; the -raw-round gate is the bounded triton cache delta (<=2,
# v1222-raw precedent +2). Strict zero-writes remains the PRODUCTION gate.
CACHE_NOW=$(docker exec lsv-test sh -c 'ls /root/.triton/cache | wc -l' | tr -d '[:space:]')
CACHE_RAW=$(docker run --rm --entrypoint sh llm-scaler-exp:v1.2.24-raw -c 'ls /root/.triton/cache | wc -l' 2>/dev/null | tr -d '[:space:]')
JIT_DELTA=$((CACHE_NOW - CACHE_RAW))
echo "jit_recheck_lines=$JIT cache_now=$CACHE_NOW cache_raw=$CACHE_RAW delta=$JIT_DELTA (bound 2)"
if [ "$JIT_DELTA" -lt 0 ] || [ "$JIT_DELTA" -gt 2 ]; then { echo "ABORT_JIT_CACHE_DELTA got=$JIT_DELTA"; exit 1; }; fi
TB=$(awk '/V123 SERVE LOG ERRORS/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1224_validate.txt)
[ "$TB" = "0" ] || { echo "ABORT_SERVE_TRACEBACKS got=$TB"; exit 1; }
RS=$(awk '/engine resets in dmesg/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1224_validate.txt)
[ "$RS" = "0" ] || { echo "ABORT_ENGINE_RESETS got=$RS"; exit 1; }
FP8RT=$(grep -A1 'V124 RUNTIME FP8 RESOLUTION' /root/build/_v1224_validate.txt | tail -1)
echo "$FP8RT" | grep -q 'float8_e4m3fn' || { echo "ABORT_FP8_RESOLUTION got=$FP8RT"; exit 1; }
echo "$FP8RT" | grep -q 'float8_e5m2' || { echo "ABORT_FP8_RESOLUTION got=$FP8RT"; exit 1; }

# 5. v124 code-delta asserts (direct against the live lane)
P16M=$(docker exec lsv-test sh -c 'grep -c "v124 P16" /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py' | tr -d '[:space:]')
P195C=$(docker exec lsv-test sh -c 'grep -c "v124 P19.5a" /opt/venv/lib/python3.12/site-packages/vllm/config/cache.py' | tr -d '[:space:]')
P195M=$(docker exec lsv-test sh -c 'grep -c "v124 P19.5a" /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/mamba_utils.py' | tr -d '[:space:]')
P195O=$(docker exec lsv-test sh -c 'grep -c "v124 P19.5a" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py' | tr -d '[:space:]')
MFP16=$(docker exec lsv-test sh -c 'grep -c -- "mamba-ssm-cache-dtype float16" /root/serve_user.sh' | tr -d '[:space:]')
MFP8=$(docker exec lsv-test sh -c 'grep -c -- "mamba-ssm-cache-dtype fp8" /root/serve_user.sh' | tr -d '[:space:]')
echo "v124 deltas: p16=$P16M p195_cache=$P195C p195_mamba=$P195M p195_ops=$P195O mamba_fp16=$MFP16 mamba_fp8=$MFP8"
[ "$P16M" -eq 2 ] 2>/dev/null || { echo ABORT_P16_MARKERS; exit 1; }
[ "$P195C" -ge 1 ] 2>/dev/null || { echo ABORT_P195_CACHE; exit 1; }
[ "$P195M" -ge 1 ] 2>/dev/null || { echo ABORT_P195_MAMBA; exit 1; }
[ "$P195O" -ge 1 ] 2>/dev/null || { echo ABORT_P195_OPS; exit 1; }
[ "$MFP16" -eq 1 ] 2>/dev/null || { echo ABORT_MAMBA_DEFAULT_NOT_FP16; exit 1; }
[ "$MFP8" -eq 0 ] 2>/dev/null || { echo ABORT_MAMBA_FP8_IN_SERVE; exit 1; }

grep -m1 'SANITY finish' /root/build/_v1224_sanity.txt || true
echo "== V124 GENSPEED EVIDENCE (verify manually: solo parity ~70+ / 4x1024 >= 100 agg) =="
grep -A6 'SOLO GENSPEED' /root/build/_v1224_validate.txt | head -10
grep -A6 'ASYNC GENSPEED' /root/build/_v1224_validate.txt | head -10
echo "=== VALIDATE_V1224_RUN ALL GATES PASS $(date +%T) ==="
} > "$LOG" 2>&1
tail -40 "$LOG"
grep -q 'VALIDATE_V1224_RUN ALL GATES PASS' "$LOG" || exit 1
echo V1224_VALIDATION_COMPLETE
