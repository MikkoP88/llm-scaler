#!/bin/sh
# validate_v1225_run.sh — v1.2.25-raw promotion validation (fresh-boot lane).
# v125 posture = v124 posture (async + RPC 60000 + barrier 0, parser coder, 85
# capture sizes, P16 non-spec gate, P19.5a fp8 SSM plumbing inert on fp16) +
# opsall ROOT FIX (custom_ops='all' restored after the TP>1 compile gate) +
# C7 fp8-state import-time refusal (P23D/P23F conviction baked in-code).
# Bake + host reboot + fresh boot via repro_bootV1225.sh are done before this
# runs. Chain: generates sanity_v1225/validate_v1225 from the v1224 pair (sed
# v1224->v1225; literal V123 barrier-patcher names and V123 posture headers
# are version-agnostic and survive untouched), appends v125 legs, then runs
# the full battery: sanity (v66 admission PASS + posture gates + v124+v125
# deltas) + validate (solo cold seed114 + battery + 3x 14-phase drill + JIT
# recheck + fairness + serialized 24 x3 + burst_harsh x3 + parser battery +
# solo genspeed + 4-stream async genspeed + resets) + v125 asserts (opsall/C7
# markers, C7 functional refusal ON THE LANE, 'v125 root fix' boot line) +
# P23F QUALITY GATE (fresh probe A: 0/80 concurrent wrong + 0 flips + 30/30
# tools; post-battery probe B: 0/24 wrong + 30/30 tools) + triton cache delta
# bound vs v1.2.25-raw.
LOG=/root/build/lce1/v1225_validate_run.log
{
echo "=== VALIDATE_V1225_RUN start $(date +%T) ==="

# 1. sanity_v1225.sh from sanity_v1224.sh (sed; v123 literals untouched)
S=/root/build/sanity_v1225.sh
sed -e 's/v1224/v1225/g' -e 's/V1224/V1225/g' \
    /root/build/sanity_v1224.sh > "$S"
printf '%s\n' \
  'echo "== V125 CODE DELTAS (acceptance: opsall=1; c7>=1; fixline>=1) ==" >> $R' \
  'docker exec lsv-test sh -c "grep -c \"perf-v125 root fix\" /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"llm-scaler v125 C7\" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py" >> $R 2>&1' \
  'docker exec lsv-test sh -c "grep -c \"v125 root fix\" /root/serve_full.log" >> $R 2>&1' \
  'echo "SANITY_V1225_EXTRA3_DONE" >> $R' >> "$S"
grep -q SANITY_V1225_EXTRA3_DONE "$S" || { echo ABORT_SANITY_GEN; exit 1; }

# 2. validate_v1225.sh from validate_v1224.sh (sed; drop DONE; + v125 legs)
V=/root/build/validate_v1225.sh
sed -e 's/v1224/v1225/g' -e 's/V1224/V1225/g' /root/build/validate_v1224.sh \
    | grep -v '^echo VALIDATE_V1225_DONE' > "$V"
printf '%s\n' \
  'echo "== V125 C7 FUNCTIONAL ON LANE (informational dup of the pre-chain gate) ==" >> $OUT' \
  'docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-test /opt/venv/bin/python3 -c "import vllm._xpu_ops" > /root/build/_v1225_c7ref.txt 2>&1; echo "c7_refuse_rc=$?" >> $OUT' \
  'tail -1 /root/build/_v1225_c7ref.txt >> $OUT 2>&1' \
  'docker exec lsv-test /opt/venv/bin/python3 -c "import vllm._xpu_ops; print(\"C7_UNSET_IMPORT_OK\")" >> $OUT 2>&1' \
  'echo VALIDATE_V1225_DONE >> $OUT' >> "$V"
grep -q VALIDATE_V1225_DONE "$V" || { echo ABORT_VALIDATE_GEN; exit 1; }
sh -n "$S" || { echo ABORT_SANITY_SYNTAX; exit 1; }
sh -n "$V" || { echo ABORT_VALIDATE_SYNTAX; exit 1; }
echo "v1225 sanity/validate generated $(date +%T)"

# 3. sanity (acceptance: EXTRA_DONE + admission PASS + async + barrier posture)
sh /root/build/sanity_v1225.sh > /dev/null 2>&1
grep -q SANITY_V1225_EXTRA_DONE /root/build/_v1225_sanity.txt || { echo ABORT_SANITY; exit 1; }
grep -q SANITY_V1225_EXTRA2_DONE /root/build/_v1225_sanity.txt || { echo ABORT_SANITY2; exit 1; }
grep -q SANITY_V1225_EXTRA3_DONE /root/build/_v1225_sanity.txt || { echo ABORT_SANITY3; exit 1; }
grep -q 'ADMISSION_DONE verdict=PASS' /root/build/_v1225_sanity.txt || { echo ABORT_ADMISSION_NOT_PASS; exit 1; }
# NB: inherited posture legs keep their literal 3-char "V123" headers —
# key on the ACTUAL label text, not the version.
ASY=$(awk '/V123 ASYNC ENGAGED/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1225_sanity.txt)
[ "$ASY" -ge 1 ] 2>/dev/null || { echo "ABORT_ASYNC_NOT_ENGAGED got=$ASY"; exit 1; }
BARR=$(awk '/V123 BARRIER DEFAULT/{f=1;next} f&&/False/{print;exit}' /root/build/_v1225_sanity.txt)
echo "$BARR" | grep -q 'False False 0' || { echo "ABORT_BARRIER_DEFAULT got=$BARR"; exit 1; }
grep -q 'V123_BARRIER APPLIED' /root/build/_v1225_sanity.txt || { echo ABORT_BARRIER_MARKER; exit 1; }
grep -q 'VLLM_XPU_SPEC_DRAFT_BARRIER=0' /root/build/_v1225_sanity.txt || { echo ABORT_BARRIER_ENV; exit 1; }
grep -q 'VLLM_RPC_TIMEOUT=60000' /root/build/_v1225_sanity.txt || { echo ABORT_RPC_TIMEOUT_ENV; exit 1; }
echo "sanity+admission+v125-posture PASS $(date +%T)"

# 4. v125 code-delta asserts (before the long chain, fail fast)
OPS=$(docker exec lsv-test sh -c 'grep -c "perf-v125 root fix" /opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py' | tr -d '[:space:]')
C7M=$(docker exec lsv-test sh -c 'grep -c "llm-scaler v125 C7" /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py' | tr -d '[:space:]')
FX=$(docker exec lsv-test sh -c 'grep -c "v125 root fix" /root/serve_full.log' | tr -d '[:space:]')
echo "v125 deltas: opsall=$OPS c7=$C7M fixline=$FX"
[ "$OPS" -ge 1 ] 2>/dev/null || { echo ABORT_OPSALL_MARKER; exit 1; }
[ "$C7M" -ge 1 ] 2>/dev/null || { echo ABORT_C7_MARKER; exit 1; }
[ "$FX" -ge 1 ] 2>/dev/null || { echo ABORT_OPSALL_BOOTLINE; exit 1; }
C7REF=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-test /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "rc=$?")
echo "$C7REF" | grep -q 'v125 C7' || { echo ABORT_C7_NO_MESSAGE; exit 1; }
echo "$C7REF" | tail -1 | grep -q 'rc=1' || { echo ABORT_C7_NO_RAISE; exit 1; }
C7UNSET=$(docker exec lsv-test /opt/venv/bin/python3 -c 'import vllm._xpu_ops; print("OK")' 2>&1 | tail -1)
[ "$C7UNSET" = "OK" ] || { echo "ABORT_C7_UNSET_IMPORT got=$C7UNSET"; exit 1; }
echo "v125 C7 functional PASS (refuse+raise on =2, clean import unset)"

# 4.5 P23F QUALITY PROBE A — must run on the GENUINELY FRESH boot, i.e.
# BEFORE any battery traffic (the generated validate legs only carry the
# informational C7 dup; probe A runs here, gated in section 7).
python3 /root/build/p23f_probe.py A > /root/build/_v1225_p23fA.txt 2>&1
grep -E 'SUMMARY|FLIP' /root/build/_v1225_p23fA.txt
grep -q 'P23F_PROBE_DONE' /root/build/_v1225_p23fA.txt || { echo ABORT_P23F_PROBE_INCOMPLETE; exit 1; }

# 5. full validate chain
sh /root/build/validate_v1225.sh > /dev/null 2>&1
grep -q VALIDATE_V1225_DONE /root/build/_v1225_validate.txt || { echo ABORT_VALIDATE; exit 1; }
grep -q 'SUSTAIN_COMPLETE_NO_WEDGE' /root/build/_v1225_sustain.txt || { echo ABORT_DRILL_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED SURVIVED ok=24' /root/build/_v1225_validate.txt || { echo ABORT_SERIALIZED1_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED2 SURVIVED ok=24' /root/build/_v1225_validate.txt || { echo ABORT_SERIALIZED2_NOT_CLEAN; exit 1; }
grep -q 'SERIALIZED3 SURVIVED ok=24' /root/build/_v1225_validate.txt || { echo ABORT_SERIALIZED3_NOT_CLEAN; exit 1; }
grep -q 'BATTERY_INCOMPLETE' /root/build/_v1225_validate.txt && { echo ABORT_BATTERY; exit 1; } || true
grep -q 'PARSER_BATTERY_FAIL' /root/build/_v1225_validate.txt && { echo ABORT_PARSER_BATTERY; exit 1; } || true
grep -qE 'BURST_[ABC]_NOT_SURVIVED' /root/build/_v1225_validate.txt && { echo ABORT_BURST_NOT_CLEAN; exit 1; } || true
JIT=$(awk '/POST-VALIDATE JIT RECHECK/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1225_validate.txt)
# v123 recalibration stands (KNOWN_ISSUES #28): monitor lines are first-use
# loads; the -raw-round gate is the bounded triton cache delta (<=2,
# v1222-raw precedent). Strict zero-writes remains the PRODUCTION gate.
CACHE_NOW=$(docker exec lsv-test sh -c 'ls /root/.triton/cache | wc -l' | tr -d '[:space:]')
CACHE_RAW=$(docker run --rm --entrypoint sh llm-scaler-exp:v1.2.25-raw -c 'ls /root/.triton/cache | wc -l' 2>/dev/null | tr -d '[:space:]')
JIT_DELTA=$((CACHE_NOW - CACHE_RAW))
echo "jit_recheck_lines=$JIT cache_now=$CACHE_NOW cache_raw=$CACHE_RAW delta=$JIT_DELTA (bound 2)"
if [ "$JIT_DELTA" -lt 0 ] || [ "$JIT_DELTA" -gt 2 ]; then { echo "ABORT_JIT_CACHE_DELTA got=$JIT_DELTA"; exit 1; }; fi
TB=$(awk '/V123 SERVE LOG ERRORS/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1225_validate.txt)
[ "$TB" = "0" ] || { echo "ABORT_SERVE_TRACEBACKS got=$TB"; exit 1; }
RS=$(awk '/engine resets in dmesg/{f=1;next} f&&/^[0-9]+$/{print;exit}' /root/build/_v1225_validate.txt)
[ "$RS" = "0" ] || { echo "ABORT_ENGINE_RESETS got=$RS"; exit 1; }

# 6. v124 lineage asserts (inherited markers, still exact)
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

# 7. P23F QUALITY GATE (the P24-recorded ship gate, now on the certified
# posture: fresh probe A must be PERFECT — 0/80 concurrent wrong, 0 flips,
# 0 serial wrong, 30/30 tools clean)
grep -q 'MATH\[fresh\] SUMMARY wrong=0/80' /root/build/_v1225_p23fA.txt || { echo ABORT_P23F_FRESH_MATH; grep 'MATH\[fresh\] SUMMARY' /root/build/_v1225_p23fA.txt; exit 1; }
grep -q 'MATH\[fresh\] SERIAL SUMMARY wrong=0/80 concurrent-vs-serial flips=0' /root/build/_v1225_p23fA.txt || { echo ABORT_P23F_FRESH_SERIAL; grep 'SERIAL SUMMARY' /root/build/_v1225_p23fA.txt; exit 1; }
grep -q 'TOOL\[fresh\] SUMMARY clean=30 salad=0 other=0 error=0' /root/build/_v1225_p23fA.txt || { echo ABORT_P23F_FRESH_TOOLS; grep 'TOOL\[fresh\] SUMMARY' /root/build/_v1225_p23fA.txt; exit 1; }

# 8. post-battery probe B (the battery = drills+bursts is the P23D prefix;
# certified posture must stay perfect under it: 0/24 wrong + 30/30 tools)
python3 /root/build/p23f_probe.py B > /root/build/_v1225_p23fB.txt 2>&1
grep -q 'MATH\[prefix\] SUMMARY wrong=0/24' /root/build/_v1225_p23fB.txt || { echo ABORT_P23F_POST_MATH; grep 'MATH\[prefix\] SUMMARY' /root/build/_v1225_p23fB.txt; exit 1; }
grep -q 'TOOL\[prefix\] SUMMARY clean=30 salad=0 other=0 error=0' /root/build/_v1225_p23fB.txt || { echo ABORT_P23F_POST_TOOLS; grep 'TOOL\[prefix\] SUMMARY' /root/build/_v1225_p23fB.txt; exit 1; }
echo "P23F quality gate PASS: fresh 0/80+0 flips+30/30, post-battery 0/24+30/30"

grep -m1 'SANITY finish' /root/build/_v1225_sanity.txt || true
echo "== V125 GENSPEED EVIDENCE (verify manually: solo parity ~70+ / 4x1024 >= 100 agg) =="
grep -A6 'SOLO GENSPEED' /root/build/_v1225_validate.txt | head -10
grep -A6 'ASYNC GENSPEED' /root/build/_v1225_validate.txt | head -10
echo "=== VALIDATE_V1225_RUN ALL GATES PASS $(date +%T) ==="
} > "$LOG" 2>&1
tail -40 "$LOG"
grep -q 'VALIDATE_V1225_RUN ALL GATES PASS' "$LOG" || exit 1
echo V1225_VALIDATION_COMPLETE; echo V1225_VALIDATION_COMPLETE >> "$LOG"
