#!/usr/bin/env python3
"""mk_stage5_bake_v1227.py — generate stage5_bake_v1227.sh from the v1226
bake (repo lineage discipline: every transformation asserts its anchor).

v1.2.27 = v1.2.26-raw + ONE v127 change, a CAPABILITY delta only:

  1. M0-LIVE COLLECTOR baked DORMANT (m0_live_collect_v127.py, P42):
     per-feature runmax telemetry thread; inert unless /root/.v127_m0live
     exists (marker-file law). Enables future live scale harvests without
     a re-patch boot round.

NO knob deltas (D-knob program CLOSED P39-P41: mnbt 16384 / v63 4096 /
v66 5.0 all measured+rejected — the certified values are the optimum).
The scaled-space bridge patch is EXCLUDED BY CONVICTION (P43:
recurrence-compounding law — fp8 storage of the recurrent GDN state
collapses structured decode at ~step 6; fp16 is the minimum viable
recurrent-state storage). Gates assert its absence from the image.

Inherited from v1.2.26-raw and ASSERTED (never re-applied): dualbridge
(+C7 refusal), opsall root fix, v88 wheel string-clean, prod lgrf .so
exact sha, full v1218..v1226 lineage, serve-config posture
(async+coder+85-sizes+barrier-0+mamba-fp16+e4m3-KV+spec-MTPx4), xgrammar
0.2.7, JIT warm chain.
"""
import sys

SRC = "/root/build/stage5_bake_v1226.sh"
DST = "/root/build/stage5_bake_v1227.sh"

s = open(SRC, encoding="utf-8").read()


def rep(old: str, new: str, n: int = 1) -> None:
    global s
    assert s.count(old) == n, "anchor %r count %d != %d" % (old[:60], s.count(old), n)
    s = s.replace(old, new)


# --- header + log -----------------------------------------------------------
rep("# stage5_bake_v1226.sh — gate and bake llm-scaler-exp:v1.2.26-raw. v1.2.26 =",
    "# stage5_bake_v1227.sh — gate and bake llm-scaler-exp:v1.2.27-raw (mk_stage5_bake_v1227.py\n"
    "# from the v1226 bake). v1.2.27 = v1.2.26-raw + ONE v127 change (M0-LIVE COLLECTOR")
rep("L=/root/build/lce1/v1226_bake.log", "L=/root/build/lce1/v1227_bake.log")
rep('echo "=== V1226 BAKE v1.2.26 $(date +%F\' \'%T) ==="',
    'echo "=== V1227 BAKE v1.2.27 $(date +%F\' \'%T) ==="')

# --- required-files precheck: m0live replaces dualbridge as the applied patch
rep("""for f in patch_v126_dualbridge.py patch_sched_v66_bake.py \\
         probe_jitwarm_v63.py warm_ext_v1219.py t2_xgrammar_probe.py \\
         repro_bootV1225.sh; do""",
    """for f in v127_stage/m0_live_collect_v127.py patch_sched_v66_bake.py \\
         probe_jitwarm_v63.py warm_ext_v1219.py t2_xgrammar_probe.py \\
         repro_bootV1226.sh; do""")

# --- base image: v1.2.26-raw (dualbridge already baked + asserted below) -----
rep("echo \"--- fresh bake container from v1.2.24-raw ---\"",
    'echo "--- fresh bake container from v1.2.26-raw (v126 stack INHERITED) ---"')
rep("    llm-scaler-exp:v1.2.25-raw || { echo \"BAKE ABORT: container run failed\"; exit 1; }",
    "    llm-scaler-exp:v1.2.26-raw || { echo \"BAKE ABORT: container run failed\"; exit 1; }")

# --- the v126 apply section becomes: inherited-assert + v127 collector apply -
rep("""echo "--- v125 C7 refusal INHERITED (asserted via v126 markers after dualbridge) ---"

echo "--- v126 patch: DUALBRIDGE (prefill/mixed GDN bridge + =2 legacy-route refusal) ---"
ck dual_base_clean 0 "$(docker exec lsv-bake grep -c 'llm-scaler v126' $XO)"
docker cp /root/build/patch_v126_dualbridge.py lsv-bake:/root/patch_v126_dualbridge.py
DBEXIT=$(docker exec lsv-bake /opt/venv/bin/python3 /root/patch_v126_dualbridge.py | tail -1 | grep -c -e V126_DUALBRIDGE_OK -e V126_DUALBRIDGE_ALREADY)
ck dualbridge_patch_exit 1 "$DBEXIT"
ck dualbridge_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 DUAL BRIDGE' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck dualbridge_c7_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 C7' $XO | awk '{print (\\$1>=1)?1:0}'")"
""",
    """echo "--- v125 C7 refusal INHERITED (asserted via v126 markers) ---"

echo "--- v126 DUALBRIDGE INHERITED from v1.2.26-raw (asserted, never re-applied) ---"
ck dualbridge_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 DUAL BRIDGE' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck dualbridge_c7_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v126 C7' $XO | awk '{print (\\$1>=1)?1:0}'")"

echo "--- v127 patch: M0-LIVE COLLECTOR baked DORMANT (P42 telemetry; scaled-space EXCLUDED by P43) ---"
ck m0_base_clean 0 "$(docker exec lsv-bake grep -c 'llm-scaler v127 M0LIVE-COLLECTOR' $XO)"
ck scaledspace_code_absent 0 "$(docker exec lsv-bake grep -c 'llm-scaler v127 SCALEDSPACE' $XO)"
docker cp /root/build/v127_stage/m0_live_collect_v127.py lsv-bake:/root/m0_live_collect_v127.py
M0EXIT=$(docker exec lsv-bake /opt/venv/bin/python3 /root/m0_live_collect_v127.py | tail -1 | grep -c -e V127_M0LIVE_OK -e V127_M0LIVE_ALREADY)
ck m0live_patch_exit 1 "$M0EXIT"
ck m0live_marker 1 "$(docker exec lsv-bake sh -c "grep -c 'llm-scaler v127 M0LIVE-COLLECTOR' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck m0live_syntax 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import ast; ast.parse(open("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py").read()); print(1)' | tail -1)"
ck m0live_import_clean 1 "$(docker exec lsv-bake /opt/venv/bin/python3 -c 'import vllm._xpu_ops; print("IMPORT_OK")' 2>&1 | grep -c IMPORT_OK)"
# DORMANT by construction: no armed marker, no scales, no dumps anywhere
ck m0live_armed_marker_absent 1 "$(docker exec lsv-bake sh -c 'test ! -e /root/.v127_m0live && echo 1 || echo 0')"
ck m0live_dumps_absent 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -c m0live_runmax || true")"
ck scaledspace_scales_absent 1 "$(docker exec lsv-bake sh -c 'test ! -e /root/v127_scales_e4m3.pt && echo 1 || echo 0')"
ck scaledspace_arm_marker_absent 1 "$(docker exec lsv-bake sh -c 'test ! -e /root/.v127_scaledspace && echo 1 || echo 0')"
""")

# --- warm-serve evidence: add the dormancy proof -----------------------------
rep("""ck bake_opsall_fired 1 "$(docker exec lsv-bake sh -c "grep -c 'v125 root fix' /root/serve_full.log | awk '{print (\\$1>=1)?1:0}'")\"""",
    """ck bake_opsall_fired 1 "$(docker exec lsv-bake sh -c "grep -c 'v125 root fix' /root/serve_full.log | awk '{print (\\$1>=1)?1:0}'")"
ck bake_m0live_dormant 0 "$(docker exec lsv-bake sh -c "grep -c 'M0LIVE collector engaged' /root/serve_full.log || true")\"""")

# --- stamps: keep the inherited asserts, add v1227 ---------------------------
rep("""docker exec lsv-bake test -f /root/.v1225_jit_warmed && echo "GATE-OK   v1225_jit_warm_stamp" || { echo "GATE-FAIL v1225_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1226_jit_warmed && echo "GATE-OK   v1226_jit_warm_stamp" || { echo "GATE-FAIL v1226_jit_warm_stamp"; FAIL=1; }""",
    """docker exec lsv-bake test -f /root/.v1225_jit_warmed && echo "GATE-OK   v1225_jit_warm_stamp" || { echo "GATE-FAIL v1225_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.v1226_jit_warmed && echo "GATE-OK   v1226_jit_warm_stamp" || { echo "GATE-FAIL v1226_jit_warm_stamp"; FAIL=1; }
docker exec lsv-bake test -f /root/.llm_scaler_exp_v1226_baked && echo "GATE-OK   v1226_pedigree_marker" || { echo "GATE-FAIL v1226_pedigree_marker"; FAIL=1; }""")
rep("docker exec lsv-bake touch /root/.v1226_jit_warmed",
    "docker exec lsv-bake touch /root/.v1227_jit_warmed")

# --- commit section ----------------------------------------------------------
rep('echo "--- gates PASS: committing v1.2.25-raw ---"',
    'echo "--- gates PASS: committing v1.2.27-raw ---"')
rep("""docker exec lsv-bake rm -f /root/patch_v126_dualbridge.py
ck no_bake_scripts_left 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE 'patch_v126|patch_v125' || true")"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1226_baked
docker commit lsv-bake llm-scaler-exp:v1.2.26-raw
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.26-raw --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'""",
    """docker exec lsv-bake rm -f /root/m0_live_collect_v127.py \\
    /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py.bak_v127m0
ck no_bake_scripts_left 0 "$(docker exec lsv-bake sh -c "ls /root/ | grep -cE 'patch_v126|patch_v125|m0_live_collect' || true")"
ck no_v127_bak_files 0 "$(docker exec lsv-bake sh -c "ls /opt/venv/lib/python3.12/site-packages/vllm/ | grep -c bak_v127 || true")"
docker exec lsv-bake touch /root/.llm_scaler_exp_v1227_baked
docker commit lsv-bake llm-scaler-exp:v1.2.27-raw
echo "commit_rc=$?"
docker images llm-scaler-exp:v1.2.27-raw --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'""")

# --- repro boot build: V1227 from V1226 (+ collector boot gate) ---------------
rep("""echo "--- build repro_bootV1226.sh from V1225 (image + marker; async gate already inverted in lineage) ---"
B=/root/build/repro_bootV1226.sh
cp /root/build/repro_bootV1225.sh "$B"
sed -i 's/llm-scaler-exp:v1\\.2\\.25-raw/llm-scaler-exp:v1.2.26-raw/g' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1225_baked|test -f /root/.llm_scaler_exp_v1226_baked|' "$B"
# v126: dualbridge marker gate on boot (image carries it baked) —
# inserted via python after the seds (sed multiline quoting is fragile)
python3 - "$B" <<'PYBB'
import sys
p = sys.argv[1]
s = open(p).read()
a = 'test -f /root/.llm_scaler_exp_v1226_baked || { echo "BOOT_$MODE MARKER MISSING"; exit 9; }'
g = (a + '\\n'
     'DBN=$(docker exec lsv-test sh -c '
     '\\'grep -c \\"llm-scaler v126 DUAL BRIDGE\\" '
     '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\\' | tr -d \\"[:space:]\\")' '\\n'
     '[ "$DBN" -ge 1 ] 2>/dev/null || { echo "BOOT V1226 DUALBRIDGE MISSING got=$DBN"; exit 9; }')
assert s.count(a) == 1, 'marker line count %d' % s.count(a)
open(p, 'w').write(s.replace(a, g))
print('V1226_BOOT_DUALBRIDGE_GATE_ADDED')
PYBB
bash -n "$B" || { echo "BOOT SCRIPT SYNTAX FAIL"; exit 1; }
echo "--- diff V1225 -> V1226 boot (expect image + marker + dualbridge-gate lines only) ---"
diff /root/build/repro_bootV1225.sh "$B" || true
grep -nE 'v1\\.2\\.26|v1226_baked|DUALBRIDGE' "$B" | head -10""",
    """echo "--- build repro_bootV1227.sh from V1226 (image + marker + collector boot gate) ---"
B=/root/build/repro_bootV1227.sh
cp /root/build/repro_bootV1226.sh "$B"
sed -i 's/llm-scaler-exp:v1\\.2\\.26-raw/llm-scaler-exp:v1.2.27-raw/g' "$B"
sed -i 's|test -f /root/.llm_scaler_exp_v1226_baked|test -f /root/.llm_scaler_exp_v1227_baked|' "$B"
# v1227: m0live-collector marker gate on boot (image carries it baked) —
# inserted via python after the seds (sed multiline quoting is fragile)
python3 - "$B" <<'PYBB'
import sys
p = sys.argv[1]
s = open(p).read()
a = 'test -f /root/.llm_scaler_exp_v1227_baked || { echo "BOOT_$MODE MARKER MISSING"; exit 9; }'
g = (a + '\\n'
     'M0N=$(docker exec lsv-test sh -c '
     '\\'grep -c \\"llm-scaler v127 M0LIVE-COLLECTOR\\" '
     '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\\' | tr -d \\"[:space:]\\")' '\\n'
     '[ "$M0N" -ge 1 ] 2>/dev/null || { echo "BOOT V1227 M0LIVE-COLLECTOR MISSING got=$M0N"; exit 9; }')
assert s.count(a) == 1, 'marker line count %d' % s.count(a)
open(p, 'w').write(s.replace(a, g))
print('V1227_BOOT_M0LIVE_GATE_ADDED')
PYBB
bash -n "$B" || { echo "BOOT SCRIPT SYNTAX FAIL"; exit 1; }
echo "--- diff V1226 -> V1227 boot (expect image + marker + m0live-gate lines only) ---"
diff /root/build/repro_bootV1226.sh "$B" || true
grep -nE 'v1\\.2\\.27|v1227_baked|M0LIVE' "$B" | head -10""")

rep('echo STAGE5_BAKE_V1226_DONE', 'echo STAGE5_BAKE_V1227_DONE')

open(DST, "w", encoding="utf-8").write(s)
print("WROTE %s (%d lines)" % (DST, s.count("\n") + 1))
