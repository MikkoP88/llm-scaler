#!/usr/bin/env python3
"""mk_gates_v1227_lane.py — build gates_v1227_lane.sh from the certified
gates_v1226_lane.sh (host-side transform).

v127 ship delta vs v1226: the image adds ONLY the DORMANT m0-live collector
(_xpu_ops.py telemetry thread, inert unless /root/.v127_m0live — marker-file
law) plus the v1227 pedigree/jit stamps. The C7 surface STAYS v126 (the
dualbridge is untouched this round; 'v126' does not contain 'v1226', so the
blanket rename cannot reach it). The scaled-space patch is EXCLUDED by P43
(recurrence-compounding law) — the ship gates must assert its ABSENCE from
code, markers, scales, and dumps, so no later arming of the known
wrong-answer mode can pass a ship review.

Transforms (ordered):
  1. blanket v1226 -> v1227 (names, logs, markers)
  2. blanket v1.2.26 -> v1.2.27 (image tags only; independent tokens)
  3. pedigree + jit lists: re-insert v1226 before v1227 (blanket collapsed)
  4. NEW v127 gates after the v126 block: m0live marker present,
     scaled-space code/marker/scales/dumps absent, no bak_v127 files
  5. header note
"""
SRC = "/root/build/gates_v1226_lane.sh"
DST = "/root/build/gates_v1227_lane.sh"

with open(SRC) as f:
    t = f.read()

# 1) + 2) blanket renames
n_v = t.count("v1226")
t = t.replace("v1226", "v1227")
n_img = t.count("v1.2.26")
t = t.replace("v1.2.26", "v1.2.27")

# 2b) banner literal (uppercase survives the lowercase blanket);
#     V123_ID/V123RAW_ID are intentionally-kept generic variable names
n_gate_ban = t.count("V1226 SHIP GATES")
assert n_gate_ban == 3, "gates banners=%d" % n_gate_ban
t = t.replace("V1226 SHIP GATES", "V1227 SHIP GATES")

# 3) lists: blanket collapsed 'v1226' -> 'v1227'; re-insert v1226
a_ped = "for M in v1218 v1220 v1221 v1222 v1223 v1224 v1225 v1227; do"
a_jit = "for M in v64 v1220 v1221 v1222 v1223 v1224 v1225 v1227; do"
assert t.count(a_ped) == 1, "pedigree list anchor=%d" % t.count(a_ped)
assert t.count(a_jit) == 1, "jit list anchor=%d" % t.count(a_jit)
t = t.replace(a_ped, "for M in v1218 v1220 v1221 v1222 v1223 v1224 v1225 v1226 v1227; do")
t = t.replace(a_jit, "for M in v64 v1220 v1221 v1222 v1223 v1224 v1225 v1226 v1227; do")

# 4) NEW v127 gates after the v126 block (last gate of the v126 insertion)
# 2026-10-01 HOTFIX: the anchor originally ended at ...print s+0}'" WITHOUT
# the line's closing `)"` — a line PREFIX, so the replace split the v126
# command mid-line and stranded the orphaned `)"` on the v127_no_bak_files
# line; bash parsed a mega command substitution swallowing the v127 block and
# ck() saw invisible trailing bytes -> GATE-FAIL expected=0 got=0. The anchor
# must match through the line's true end: ...print s+0}'")" (fix_gates_v1227.py
# repaired the generated script in place).
anchor = ('ck v126_no_p29h_markers 0 "$(docker exec lsv-gate sh -c '
          '"grep -rc -e \'P29H\' -e \'P29M\' -e \'v126_p29h\' '
          '$SP/model_executor/layers/mamba/ | awk -F: \'{s+=\\$2} END{print s+0}\'")"')
assert t.count(anchor) == 1, "v126 block tail anchor=%d" % t.count(anchor)
newblock = anchor + '''

echo "--- v127 deltas: m0live collector DORMANT + scaled-space absence (P43 law) ---"
ck v127_m0live_marker 1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler v127 M0LIVE-COLLECTOR' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck v127_scaledspace_code_absent 0 "$(docker exec lsv-gate grep -c 'llm-scaler v127 SCALEDSPACE' $XO)"
ck v127_m0live_unarmed 1 "$(docker exec lsv-gate sh -c 'test ! -e /root/.v127_m0live && echo 1 || echo 0')"
ck v127_ss_unarmed 1 "$(docker exec lsv-gate sh -c 'test ! -e /root/.v127_scaledspace && echo 1 || echo 0')"
ck v127_scales_absent 1 "$(docker exec lsv-gate sh -c 'test ! -e /root/v127_scales_e4m3.pt && echo 1 || echo 0')"
ck v127_m0live_dumps_absent 0 "$(docker exec lsv-gate sh -c "ls /root/ | grep -c m0live_runmax || true")"
ck v127_no_bak_files 0 "$(docker exec lsv-gate sh -c "ls $SP/ | grep -c bak_v127 || true")"'''
t = t.replace(anchor, newblock)

# 5) header note
hdr = ("# mk_gates_v1227_lane.py note: v127 = v1226 gates + dormant m0live\n"
       "# collector present + scaled-space ABSENCE asserted (P43 recurrence-\n"
       "# compounding law: no arming of the wrong-answer mode can pass ship) —\n"
       "# everything else (v126 dualbridge/C7, prod .so sha, serve config,\n"
       "# v123 layer, v60-v66 lineage, xgrammar/spec, pedigree v1218..v1227,\n"
       "# triton cache floor) unchanged from the certified v1226 gates.\n")
t = t.replace("#!/bin/bash\n", "#!/bin/bash\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("v1227 count:", t.count("v1227"), " v1226 count:", t.count("v1226"))
print("v1.2.27 count:", t.count("v1.2.27"), " v1.2.26 count:", t.count("v1.2.26"))
print("m0live gates:", t.count("v127_m0live"))
print("pedigree list ok:", "v1224 v1225 v1226 v1227; do" in t)
print("jit list ok:", "v64 v1220 v1221 v1222 v1223 v1224 v1225 v1226 v1227; do" in t)
print("C7 still v126:", t.count("llm-scaler v126 C7"))
