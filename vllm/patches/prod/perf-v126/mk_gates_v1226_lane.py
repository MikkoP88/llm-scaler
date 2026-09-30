#!/usr/bin/env python3
"""mk_gates_v1226_lane.py — build gates_v1226_lane.sh from the certified
gates_v1225_lane.sh (host-side transform).

v126 ship delta vs v1225: the image adds ONLY the dualbridge patch
(_xpu_ops.py: prefill/mixed GDN bridge + in-code refusal of the legacy
VLLM_XPU_GDN_FP8_NATIVE route). The v125 C7 marker text is superseded
("llm-scaler v125 C7" -> "llm-scaler v126 C7"), the env-name reference
count inside _xpu_ops.py moves 2 -> 3 (comment + environ.get + RuntimeError
message; measured on the SHIPMAT2 lane), the production lgrf .so must be
the EXACT certified sha (never the v131 diagnostic build), and the P29H/
P29M/poolpath round artifacts must be absent. Pedigree and jit-stamp lists
gain v1226 (and must KEEP v1225 — the blanket rename collapses it, so both
lists are re-expanded with anchored replaces).

Transforms (ordered):
  1. blanket v1225 -> v1226 (names, logs, markers)
  2. blanket v1.2.25 -> v1.2.26 (image tags only; 'v1.2.25' does not
     contain 'v1225', so both blankets are independent)
  3. pedigree + jit lists: re-insert v1225 before v1226 (blanket collapsed)
  4. C7 surface -> v126: marker grep, env-refs expectation 2 -> 3,
     functional-refusal message greps, section header
  5. NEW v126 gates after the C7 functional block: DUAL BRIDGE marker,
     v125-C7-gone, prod .so exact sha, P29H/P29M/poolpath absence
  6. header note
"""
SRC = "/root/build/gates_v1225_lane.sh"
DST = "/root/build/gates_v1226_lane.sh"

with open(SRC) as f:
    t = f.read()

# 1) + 2) blanket renames
n_v = t.count("v1225")
t = t.replace("v1225", "v1226")
n_img = t.count("v1.2.25")
t = t.replace("v1.2.25", "v1.2.26")

# 2b) banner literal (uppercase V1225 survives the lowercase blanket);
#     V123_ID/V123RAW_ID are intentionally-kept generic variable names
n_gate_ban = t.count("V1225 SHIP GATES")
assert n_gate_ban == 3, "gates banners=%d" % n_gate_ban
t = t.replace("V1225 SHIP GATES", "V1226 SHIP GATES")

# 3) lists: blanket collapsed 'v1225' -> 'v1226'; re-insert v1225
a_ped = "for M in v1218 v1220 v1221 v1222 v1223 v1224 v1226; do"
a_jit = "for M in v64 v1220 v1221 v1222 v1223 v1224 v1226; do"
assert t.count(a_ped) == 1, "pedigree list anchor=%d" % t.count(a_ped)
assert t.count(a_jit) == 1, "jit list anchor=%d" % t.count(a_jit)
t = t.replace(a_ped, "for M in v1218 v1220 v1221 v1222 v1223 v1224 v1225 v1226; do")
t = t.replace(a_jit, "for M in v64 v1220 v1221 v1222 v1223 v1224 v1225 v1226; do")

# 3b) v124_p195_sycl_bridge: ==1 -> >=1. In v1225 the site held exactly one
#     marker; the v126 dualbridge replaces that block and carries the tag
#     forward in its lineage comment, where the phrase "v124 P19.5a" appears
#     twice (P29T wording). The battery's P195OPS assert already uses >=1
#     (validate_v1226_run.sh: "[ "$P195O" -ge 1 ]"); the ship gate must match
#     or the P29T lineage fix fails its own ship. Uniqueness of the v126
#     surface is enforced by v126_dualbridge_marker / v126_c7_marker (==1).
#     NOTE: this awk sits OUTSIDE the sh -c "..." quotes (piped from docker
#     exec), so the program must use a BARE $1 — the \$1 form is the inner-sh
#     escaping and only applies to the awks inside sh -c (run-2 lesson:
#     \$1 here made awk error out and the gate read got= empty).
OLD_BRIDGE = ('ck v124_p195_sycl_bridge 1 "$(docker exec lsv-gate sh -c '
              '"grep -c \'v124 P19.5a\' $SPR/vllm/_xpu_ops.py" | '
              'tr -d \'[:space:]\')"')
NEW_BRIDGE = ('ck v124_p195_sycl_bridge 1 "$(docker exec lsv-gate sh -c '
              '"grep -c \'v124 P19.5a\' $SPR/vllm/_xpu_ops.py" | '
              'awk \'{print ($1>=1)?1:0}\')"')
assert t.count(OLD_BRIDGE) == 1, "p195 bridge gate anchor=%d" % t.count(OLD_BRIDGE)
t = t.replace(OLD_BRIDGE, NEW_BRIDGE)

# 4) C7 surface -> v126
OLD_MARK = 'ck v125_c7_marker 1 "$(docker exec lsv-gate sh -c "grep -c \'llm-scaler v125 C7\' $XO | awk \'{print (\\$1>=1)?1:0}\'")"'
NEW_MARK = 'ck v126_c7_marker 1 "$(docker exec lsv-gate sh -c "grep -c \'llm-scaler v126 C7\' $XO | awk \'{print (\\$1>=1)?1:0}\'")"'
assert t.count(OLD_MARK) == 1, "c7 marker gate anchor=%d" % t.count(OLD_MARK)
t = t.replace(OLD_MARK, NEW_MARK)

OLD_REFS = 'ck v125_fp8_native_refs_guard_only 2 "$(docker exec lsv-gate grep -c \'VLLM_XPU_GDN_FP8_NATIVE\' $XO)"'
NEW_REFS = 'ck v126_c7_native_refs_guard_only 3 "$(docker exec lsv-gate grep -c \'VLLM_XPU_GDN_FP8_NATIVE\' $XO)"'
assert t.count(OLD_REFS) == 1, "c7 refs gate anchor=%d" % t.count(OLD_REFS)
t = t.replace(OLD_REFS, NEW_REFS)

n_msg = t.count("grep -c 'v125 C7'")
assert n_msg == 2, "c7 functional msg anchors=%d" % n_msg
t = t.replace("grep -c 'v125 C7'", "grep -c 'llm-scaler v126'")

OLD_HDR = 'echo "--- v125 deltas: opsall root fix + C7 fp8-state refusal ---"'
NEW_HDR = ('echo "--- v125 opsall root fix + v126 dualbridge C7 refusal '
           '(C7 markers are v126 now) ---"')
assert t.count(OLD_HDR) == 1, "c7 section header anchor=%d" % t.count(OLD_HDR)
t = t.replace(OLD_HDR, NEW_HDR)

# 5) NEW v126 gates after the C7 functional block
anchor = 'ck c7g_unset_rc RC=0 "$(echo "$C7G0" | tail -1)"'
assert t.count(anchor) == 1, "c7g anchor=%d" % t.count(anchor)
newblock = anchor + '''

echo "--- v126 deltas: dualbridge + production .so + P29 round-artifact absence ---"
ck v126_dualbridge_marker 1 "$(docker exec lsv-gate sh -c "grep -c 'llm-scaler v126 DUAL BRIDGE' $XO | awk '{print (\\$1>=1)?1:0}'")"
ck v126_v125c7_gone 0 "$(docker exec lsv-gate grep -c 'llm-scaler v125 C7' $XO)"
PKGSO=$SPR/custom_esimd_kernels_vllm/custom_esimd_kernels_lgrf.cpython-312-x86_64-linux-gnu.so
SOSHA=$(docker exec lsv-gate sha256sum "$PKGSO" | awk '{print $1}')
ck prod_lgrf_so_sha 1d9dcf4e7a1c8db6e94b0673c51f58935d03de359dc109c41bbac6df00f85dff "$SOSHA"
GDNL=$SP/model_executor/layers/mamba/gdn_linear_attn.py
ck v126_no_poolpath_markers 0 "$(docker exec lsv-gate grep -c 'v126 P29C' $GDNL)"
ck v126_no_p29h_markers 0 "$(docker exec lsv-gate sh -c "grep -rc -e 'P29H' -e 'P29M' -e 'v126_p29h' $SP/model_executor/layers/mamba/ | awk -F: '{s+=\\$2} END{print s+0}'")"'''
t = t.replace(anchor, newblock)

# 6) header note
hdr = ("# mk_gates_v1226_lane.py note: v126 = v1225 gates + dualbridge surface\n"
       "# (v126 C7 marker replaces v125 C7 text, env refs 3, DUAL BRIDGE marker\n"
       "# gate, prod .so exact-sha gate, P29H/P29M/poolpath absence) — everything\n"
       "# else (v124/v125 deltas, serve config, v123 layer, v60-v66 lineage,\n"
       "# xgrammar/spec, pedigree v1218..v1226, triton cache floor) unchanged.\n")
t = t.replace("#!/bin/bash\n", "#!/bin/bash\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("v1226 count:", t.count("v1226"), " v1225 count:", t.count("v1225"))
print("v1.2.26 count:", t.count("v1.2.26"), " v1.2.25 count:", t.count("v1.2.25"))
print("DUAL BRIDGE gate:", t.count("v126_dualbridge_marker"))
print("p195 bridge >=1 gate:", t.count(NEW_BRIDGE))
print("pedigree list ok:", "v1224 v1225 v1226; do" in t)
print("jit list ok:", "v64 v1220 v1221 v1222 v1223 v1224 v1225 v1226; do" in t)
