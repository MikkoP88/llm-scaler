#!/usr/bin/env python3
"""mk_validate_v1226.py — build validate_v1226_run.sh from the certified
validate_v1225_run.sh (host-side transform).

The v126 ship posture (v1.2.25-raw + standing stack + DUALBRIDGE, prod
.so, fp16 SSM + e4m3 KV) changes exactly one assertion surface: the C7
import-time refusal moved from the v125 marker ("llm-scaler v125 C7",
env-gated legacy route) to the v126 marker ("llm-scaler v126", the =2
legacy route is refused and superseded by the in-code dual bridge). The
dualbridge marker itself is a NEW required gate.

Transforms (ordered):
  1. v1225 -> v1226 (file names, logs, DONE markers; v1.2.25-raw and
     "v125 root fix"/"perf-v125 root fix" literals contain no 'v1225'
     and survive untouched — they are still the accurate code tags).
  2. "llm-scaler v125 C7" -> "llm-scaler v126 C7" (marker grep).
  3. C7REF message check 'v125 C7' -> 'llm-scaler v126'.
  4. Insert the dualbridge marker gate before the C7-functional-PASS
     echo in section 4.
Everything else (sanity/admission, drills, serial24 x3, burst_harsh x3,
parser battery, JIT/cache-delta bound vs v1.2.25-raw, lineage markers,
P23F quality gate A/B) runs unchanged.
"""
SRC = "/root/build/validate_v1225_run.sh"
DST = "/root/build/validate_v1226_run.sh"

with open(SRC) as f:
    t = f.read()

# 1) version names
t = t.replace("v1225", "v1226")

# 2) C7 marker rename (the only marker the dualbridge patch superseded)
n_c7 = t.count('llm-scaler v125 C7')
t = t.replace('llm-scaler v125 C7', 'llm-scaler v126 C7')

# 3) C7 functional-refusal message match (message now reads
#    "llm-scaler v126: VLLM_XPU_GDN_FP8_NATIVE=2 is a legacy route ...")
n_msg = t.count("grep -q 'v125 C7'")
t = t.replace("grep -q 'v125 C7'", "grep -q 'llm-scaler v126'")

# 4) dualbridge marker gate before the C7 functional PASS echo
anchor = 'echo "v125 C7 functional PASS (refuse+raise on =2, clean import unset)"'
if t.count(anchor) != 1:
    raise SystemExit("C7-PASS anchor count %d != 1" % t.count(anchor))
dblock = (
    'DBM=$(docker exec lsv-test sh -c \'grep -c "llm-scaler v126 DUAL BRIDGE" '
    '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\' | '
    "tr -d '[:space:]')\n"
    '[ "$DBM" -ge 1 ] 2>/dev/null || { echo "ABORT_DUALBRIDGE_MARKER '
    'got=$DBM"; exit 1; }\n'
    'echo "v126 dualbridge marker=$DBM"\n')
t = t.replace(anchor, dblock + anchor)

# header note
hdr = ("# mk_validate_v1226.py note: v126 ship-posture variant — C7 marker "
       "renamed v125->v126\n# (dualbridge supersedes the legacy =2 route), "
       "dualbridge marker gate added; all else identical to v1225.\n")
t = t.replace("#!/bin/sh\n", "#!/bin/sh\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("c7 marker replaces:", n_c7, "(expect 2: EXTRA3 printf + section-4 grep)")
print("c7 msg replaces:", n_msg, "(expect 1)")
print("v1226 count:", t.count("v1226"))
print("v1.2.25-raw literals kept:", t.count("v1.2.25-raw"))
print("v125 root fix kept:", t.count("v125 root fix"))
