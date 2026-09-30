#!/usr/bin/env python3
"""mk_stage5_bake_v1226.py — build stage5_bake_v1226.sh from the certified
stage5_bake_v1225.sh (host-side transform; run on 10.20.3.65).

v1.2.26 = v1.2.25-raw + ONE v126 change:
  DUALBRIDGE (patch_v126_dualbridge.py, P28-certified): the prefill/mixed
  GDN bridge replaces the C5-era static/env-gated surface; the legacy
  VLLM_XPU_GDN_FP8_NATIVE=2 route is refused in-code ("llm-scaler v126"
  marker supersedes "llm-scaler v125 C7"). Ship-posture evidence
  (SHIPMAT2 lane): speed matrix bench_run7n agg4x256 236.80/253.74,
  solo200 75.58/75.69, agg8x256 423.11/422.93 — ABOVE the cert16s4 fp16
  control on every axis (v125 fp8 deficit inverted).

NOT baked (diagnostic/convicted, hard absence gates):
  - poolpath (fp8 SSM pools — FORMAT-IMPOSSIBLE: e5m2 pool NaN from the
    first traffic tick (P29M live evidence); e4m3 pool cannot seed at
    absmax>=1280 vs 448 cast ceiling (P29I)). fp16 GDN pool is the
    certified endgame; fp8_e4m3 KV stays.
  - P29H/P29M instrumentation (P29H WEDGES the engine on BOTH pool
    dtypes; never ships).
  - v131 kernels .so (P29B-FIX ring-row snapshot = 3x decode step cost:
    131ms vs 50ms cadence, measured 09-30). Production lgrf .so sha
    1d9dcf4e... asserted INHERITED.

Transform strategy: anchored literal replaces with exact-count asserts;
v125 opsall/C7 apply-blocks become assert-blocks (already baked into the
v1.2.25-raw base); dualbridge apply + gates inserted; absence gates
inserted; pedigree/jit stamps extended; commit target v1.2.26-raw; boot
builder V1225 -> V1226.
"""
import sys

SRC = "/root/build/stage5_bake_v1225.sh"
DST = "/root/build/stage5_bake_v1226.sh"

HDR_NEW = '''#!/bin/bash
# stage5_bake_v1226.sh — gate and bake llm-scaler-exp:v1.2.26-raw. v1.2.26 =
# v1.2.25-raw + ONE v126 change (NO kernel wheel change — v88 wheel inherited
# and ASSERTED string-clean; no scheduler change — v63/v64/v66 stay; no
# serve-config change — async+coder+85-sizes+barrier-default-0+mamba-fp16+
# e4m3-KV inherited from v1.2.25-raw and only GATED here, never re-applied):
#   1. DUALBRIDGE (patch_v126_dualbridge.py, P28-certified on this lane):
#      prefill/mixed GDN bridge in _xpu_ops.py; the legacy
#      VLLM_XPU_GDN_FP8_NATIVE=2 route is REFUSED in-code ("llm-scaler v126"
#      marker; supersedes the v125 C7 marker text, refusal semantics kept).
#      Ship-posture evidence (SHIPMAT2, prod .so): bench_run7n agg4x256
#      236.80/253.74 (cert16s4 181.81/244.88), solo200 75.58/75.69 (75.06/
#      74.89), agg8x256 423.11/422.93 — fp8-KV posture ABOVE the fp16
#      control on every axis. opsall ROOT FIX + v125 C7 semantics inherited
#      from v1.2.25-raw and ASSERTED here (never re-applied).
# HARD ABSENCE (v126 round convictions, never baked):
#   - poolpath fp8 SSM pools: FORMAT-IMPOSSIBLE (P29M: e5m2 pool NaN from
#     the first traffic tick; P29I: e4m3 cannot seed >=1280 vs 448 ceiling).
#     Certified endgame = fp16 GDN pool + fp8_e4m3 KV.
#   - P29H/P29M instrumentation (P29H wedges on BOTH pool dtypes).
#   - v131 kernels .so (ring-row snapshot = 3x step cost, 131 vs 50 ms).
# Spec MTP x4 + XGrammar-2 0.2.7 support ASSERTED by gates (hard requirement).
# Bake source: FRESH container from llm-scaler-exp:v1.2.25-raw (NOT the
# running lane). Lane MUST BE DOWN first (bake serves on :8000 for warm).
'''

with open(SRC) as f:
    t = f.read()


def rep(s, old, new, expect):
    n = s.count(old)
    if n != expect:
        raise SystemExit("anchor %r count %d != %d" % (old[:60], n, expect))
    return s.replace(old, new)


# 1) header
hdr_end = t.index("set -u")
t = HDR_NEW + t[hdr_end:]

# 2) log name
t = rep(t, "L=/root/build/lce1/v1225_bake.log",
        "L=/root/build/lce1/v1226_bake.log", 1)
t = rep(t, "=== V1225 BAKE v1.2.25 ",
        "=== V1226 BAKE v1.2.26 ", 1)

# 3) required-files list: drop the two v125 patch files (asserted, not
#    applied), add dualbridge; boot script source is now V1225
t = rep(t,
        "for f in patch_v125_opsall_fix.py patch_v125_c7_fp8state_raise.py \\\n"
        "         patch_sched_v66_bake.py probe_jitwarm_v63.py warm_ext_v1219.py \\\n"
        "         t2_xgrammar_probe.py repro_bootV1224.sh; do",
        "for f in patch_v126_dualbridge.py patch_sched_v66_bake.py \\\n"
        "         probe_jitwarm_v63.py warm_ext_v1219.py t2_xgrammar_probe.py \\\n"
        "         repro_bootV1225.sh; do", 1)

# 4) base image (docker run + comment line survived header rewrite; only
#    the docker run line matters)
t = rep(t, "llm-scaler-exp:v1.2.24-raw || { echo \"BAKE ABORT: container run failed\"; exit 1; }",
        "llm-scaler-exp:v1.2.25-raw || { echo \"BAKE ABORT: container run failed\"; exit 1; }", 1)

# 5) opsall apply block -> assert block (inherited from v1.2.25-raw)
OPS_OLD = '''echo "--- v125 patch 1: opsall root fix (custom_ops='all' after TP>1 compile gate) ---"
docker cp /root/build/patch_v125_opsall_fix.py lsv-bake:/root/patch_v125_opsall_fix.py
docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v125_opsall_fix.py
OPSEXIT=$(docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v125_opsall_fix.py | tail -1 | grep -c -e V125_OPSFIX_OK -e V125_OPSFIX_ALREADY)
ck opsall_patch_exit 1 "$OPSEXIT"
ck opsall_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'perf-v125 root fix' $XPUF | awk '{print (\\$1>=1)?1:0}'")"
ck opsall_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py").read()); print(1)' | tail -1)"'''
OPS_NEW = '''echo "--- v125 opsall root fix INHERITED from v1.2.25-raw (asserted, never re-applied) ---"
ck opsall_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'perf-v125 root fix' $XPUF | awk '{print (\\$1>=1)?1:0}'")"
ck opsall_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/platforms/xpu.py").read()); print(1)' | tail -1)"'''
t = rep(t, OPS_OLD, OPS_NEW, 1)

# 6) C7 apply block -> assert + DUALBRIDGE apply block
C7_OLD = '''echo "--- v125 patch 2: C7 fp8-state import-time refusal ---"
ck c7_base_clean 0 "$(docker exec lsv-bake grep -c 'VLLM_XPU_GDN_FP8_NATIVE' $XO)"
docker cp /root/build/patch_v125_c7_fp8state_raise.py lsv-bake:/root/patch_v125_c7_fp8state_raise.py
docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v125_c7_fp8state_raise.py
C7EXIT=$(docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v125_c7_fp8state_raise.py | tail -1 | grep -c -e V125_C7_OK -e V125_C7_ALREADY)
ck c7_patch_exit 1 "$C7EXIT"
ck c7_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v125 C7' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck c7_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py").read()); print(1)' | tail -1)"
# functional refusal: =2 and =1 must raise with the C7 message; unset must import clean
C7_2=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "RC=$?")
echo "$C7_2" | tail -3
ck c7_refuse_2_raise 1 "$(echo "$C7_2" | grep -c 'v125 C7')"
ck c7_refuse_2_rc RC=1 "$(echo "$C7_2" | tail -1)"
C7_1=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=1 lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "RC=$?")
ck c7_refuse_1_raise 1 "$(echo "$C7_1" | grep -c 'v125 C7')"
ck c7_refuse_1_rc RC=1 "$(echo "$C7_1" | tail -1)"
C7_0=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops; print("IMPORT_OK")' 2>&1; echo "RC=$?")
ck c7_unset_import_clean 1 "$(echo "$C7_0" | grep -c IMPORT_OK)"
ck c7_unset_rc RC=0 "$(echo "$C7_0" | tail -1)"'''
C7_NEW = '''echo "--- v125 C7 refusal INHERITED (asserted via v126 markers after dualbridge) ---"

echo "--- v126 patch: DUALBRIDGE (prefill/mixed GDN bridge + =2 legacy-route refusal) ---"
ck dual_base_clean 0 "$(docker exec lsv-bake grep -c 'llm-scaler v126' $XO)"
docker cp /root/build/patch_v126_dualbridge.py lsv-bake:/root/patch_v126_dualbridge.py
DBEXIT=$(docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v126_dualbridge.py | tail -1 | grep -c -e V126_DUALBRIDGE_OK -e V126_DUALBRIDGE_ALREADY)
ck dualbridge_patch_exit 1 "$DBEXIT"
ck dualbridge_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 DUAL BRIDGE' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck dualbridge_c7_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 C7' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck dualbridge_v125_c7_gone 0 "$(docker exec lsv-bake grep -c 'llm-scaler v125 C7' $XO)"
ck dualbridge_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py").read()); print(1)' | tail -1)"
# functional refusal: =2 must raise with the v126 message; unset must import clean
C7_2=$(docker exec -e VLLM_XPU_GDN_FP8_NATIVE=2 lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops' 2>&1; echo "RC=$?")
echo "$C7_2" | tail -3
ck c7_refuse_2_raise 1 "$(echo "$C7_2" | grep -c 'llm-scaler v126')"
ck c7_refuse_2_rc RC=1 "$(echo "$C7_2" | tail -1)"
C7_0=$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops; print("IMPORT_OK")' 2>&1; echo "RC=$?")
ck c7_unset_import_clean 1 "$(echo "$C7_0" | grep -c IMPORT_OK)"
ck c7_unset_rc RC=0 "$(echo "$C7_0" | tail -1)"

echo "--- v126 round-artifact ABSENCE (poolpath/P29H/P29M/v131 .so never baked) ---"
GDN=$SP/model_executor/layers/mamba/gdn_linear_attn.py
ck no_poolpath_markers 0 "$(docker exec lsv-bake grep -c 'v126 P29C' $GDN)"
ck no_p29h_markers 0 "$(docker exec lsv-bake sh -c "grep -rc -e 'P29H' -e 'P29M' -e 'v126_p29h' $SP/model_executor/layers/mamba/ | awk -F: '{s+=\\$2} END{print s+0}'")"
ck no_p29_marker_file 1 "$(docker exec lsv-bake sh -c "test ! -e /root/.v126_p29h && echo 1 || echo 0")"
ck no_p29_root_artifacts 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE 'p29[bhm]_|p29m_mag|p29h_diag' || true")"
PKGSO=/opt/venv/lib/python3.12/site-packages/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
SOSHA=$(docker exec lsv-bake sha256sum "$PKGSO" | awk '{print $1}')
ck prod_lgrf_so_sha 1d9dcf4e7a1c8db6e94b0673c51f58935d03de359dc109c41bbac6df00f85dff "$SOSHA"
echo "(v131 diagnostic .so sha e050551e... asserted absent by the exact-sha gate above)"'''
t = rep(t, C7_OLD, C7_NEW, 1)

# 7) C5/C6 absence gate comment mentions stay; add v125 dev-serve variants
#    gate line kept. Next: jit warm stamp touch -> v1226 (+ inherited v1225)
t = rep(t, "docker exec lsv-bake touch /root/.v1225_jit_warmed",
        "docker exec lsv-bake test -f /root/.v1225_jit_warmed && "
        "echo 'GATE-OK   v1225_jit_warm_stamp' || "
        "{ echo 'GATE-FAIL v1225_jit_warm_stamp'; FAIL=1; }\n"
        "docker exec lsv-bake touch /root/.v1226_jit_warmed", 1)

# 8) pedigree: add v1225 marker assert + v1226 touch
t = rep(t,
        'docker exec lsv-bake test -f /root/.llm_scaler_exp_v1224_baked && echo "GATE-OK   v1224_pedigree_marker" || { echo "GATE-FAIL v1224_pedigree_marker"; FAIL=1; }',
        'docker exec lsv-bake test -f /root/.llm_scaler_exp_v1224_baked && echo "GATE-OK   v1224_pedigree_marker" || { echo "GATE-FAIL v1224_pedigree_marker"; FAIL=1; }\n'
        'docker exec lsv-bake test -f /root/.llm_scaler_exp_v1225_baked && echo "GATE-OK   v1225_pedigree_marker" || { echo "GATE-FAIL v1225_pedigree_marker"; FAIL=1; }', 1)

# 9) jit-warm stamp list: original ends at v1224 (v1225 stamp is touched by
#    THIS bake in v1225's case, but IS present in the v1.2.25-raw base) —
#    append v1225 (inherited) + v1226 (touched this bake) after the v1224 line
t = rep(t,
        'docker exec lsv-bake test -f /root/.v1224_jit_warmed && echo "GATE-OK   v1224_jit_warm_stamp" || { echo "GATE-FAIL v1224_jit_warm_stamp"; FAIL=1; }',
        'docker exec lsv-bake test -f /root/.v1224_jit_warmed && echo "GATE-OK   v1224_jit_warm_stamp" || { echo "GATE-FAIL v1224_jit_warm_stamp"; FAIL=1; }\n'
        'docker exec lsv-bake test -f /root/.v1225_jit_warmed && echo "GATE-OK   v1225_jit_warm_stamp" || { echo "GATE-FAIL v1225_jit_warm_stamp"; FAIL=1; }\n'
        'docker exec lsv-bake test -f /root/.v1226_jit_warmed && echo "GATE-OK   v1226_jit_warm_stamp" || { echo "GATE-FAIL v1226_jit_warm_stamp"; FAIL=1; }', 1)

# 10) commit + boot builder: image tag, script cleanup, boot script names
t = rep(t, "docker commit lsv-bake llm-scaler-exp:v1.2.25-raw",
        "docker commit lsv-bake llm-scaler-exp:v1.2.26-raw", 1)
t = rep(t, "docker images llm-scaler-exp:v1.2.25-raw --format",
        "docker images llm-scaler-exp:v1.2.26-raw --format", 1)
t = rep(t,
        "docker exec lsv-bake rm -f /root/patch_v125_opsall_fix.py /root/patch_v125_c7_fp8state_raise.py",
        "docker exec lsv-bake rm -f /root/patch_v126_dualbridge.py", 1)
t = rep(t, 'ck no_bake_scripts_left 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE \'patch_v125\' || true")"',
        'ck no_bake_scripts_left 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE \'patch_v126|patch_v125\' || true")"', 1)
t = rep(t, "docker exec lsv-bake touch /root/.llm_scaler_exp_v1225_baked",
        "docker exec lsv-bake touch /root/.llm_scaler_exp_v1226_baked", 1)

# 11) boot-script builder: V1224 source -> V1225 source, V1225 target -> V1226
t = rep(t,
        "echo \"--- build repro_bootV1225.sh from V1224 (image + marker; async gate already inverted in lineage) ---\"\n"
        "B=/root/build/repro_bootV1225.sh\n"
        "cp /root/build/repro_bootV1224.sh \"$B\"\n"
        "sed -i 's/llm-scaler-exp:v1\\.2\\.24-raw/llm-scaler-exp:v1.2.25-raw/g' \"$B\"\n"
        "sed -i 's|test -f /root/.llm_scaler_exp_v1224_baked|test -f /root/.llm_scaler_exp_v1225_baked|' \"$B\"",
        "echo \"--- build repro_bootV1226.sh from V1225 (image + marker; async gate already inverted in lineage) ---\"\n"
        "B=/root/build/repro_bootV1226.sh\n"
        "cp /root/build/repro_bootV1225.sh \"$B\"\n"
        "sed -i 's/llm-scaler-exp:v1\\.2\\.25-raw/llm-scaler-exp:v1.2.26-raw/g' \"$B\"\n"
        "sed -i 's|test -f /root/.llm_scaler_exp_v1225_baked|test -f /root/.llm_scaler_exp_v1226_baked|' \"$B\"\n"
        "# v126: dualbridge marker gate on boot (image carries it baked) —\n"
        "# inserted via python after the seds (sed multiline quoting is fragile)\n"
        "python3 - \"$B\" <<'PYBB'\n"
        "import sys\n"
        "p = sys.argv[1]\n"
        "s = open(p).read()\n"
        "a = 'test -f /root/.llm_scaler_exp_v1226_baked || { echo \"BOOT_$MODE MARKER MISSING\"; exit 9; }'\n"
        "g = (a + '\\n'\n"
        "     'DBN=$(docker exec lsv-test sh -c '\n"
        "     '\\'grep -c \\\"llm-scaler v126 DUAL BRIDGE\\\" '\n"
        "     '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\\' | tr -d \\\"[:space:]\\\")' '\\n'\n"
        "     '[ \"$DBN\" -ge 1 ] 2>/dev/null || { echo \"BOOT V1226 DUALBRIDGE MISSING got=$DBN\"; exit 9; }')\n"
        "assert s.count(a) == 1, 'marker line count %d' % s.count(a)\n"
        "open(p, 'w').write(s.replace(a, g))\n"
        "print('V1226_BOOT_DUALBRIDGE_GATE_ADDED')\n"
        "PYBB", 1)
t = rep(t,
        'echo "--- diff V1224 -> V1225 boot (expect image + marker lines only) ---"\n'
        'diff /root/build/repro_bootV1224.sh "$B" || true\n'
        "grep -nE 'v1\\.2\\.25|v1225_baked' \"$B\" | head -8",
        'echo "--- diff V1225 -> V1226 boot (expect image + marker + dualbridge-gate lines only) ---"\n'
        'diff /root/build/repro_bootV1225.sh "$B" || true\n'
        "grep -nE 'v1\\.2\\.26|v1226_baked|DUALBRIDGE' \"$B\" | head -10", 1)
t = rep(t, 'echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="',
        'echo "=== BAKE DONE (image committed, boot script ready) $(date +%T) ==="', 1)
t = rep(t, "echo STAGE5_BAKE_V1225_DONE", "echo STAGE5_BAKE_V1226_DONE", 1)
t = rep(t, 'tail -100 "$L"', 'tail -100 "$L"', 1)

with open(DST, "w") as f:
    f.write(t)
print("wrote", DST, len(t), "bytes")
print("sanity: v126 mentions:", t.count("v126") if "v126" in t else 0)
print("commit target count:", t.count("v1.2.26-raw"))
print("base image count:", t.count("llm-scaler-exp:v1.2.25-raw"))
print("leftover v1225 applies:", t.count("patch_v125_c7_fp8state_raise.py"))
