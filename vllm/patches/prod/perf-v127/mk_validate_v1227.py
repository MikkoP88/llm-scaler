#!/usr/bin/env python3
"""mk_validate_v1227.py — build validate_v1227_run.sh from the certified
validate_v1226_run.sh (host-side transform).

The v127 ship posture (v1.2.26-raw stack + DORMANT m0-live collector baked
into _xpu_ops.py) adds exactly one assertion surface: the collector marker
must be present on the booted lane AND the collector must be dormant (no
/root/.v127_m0live) with the scaled-space patch absent (P43 recurrence-
compounding law — it must never be armable on a shipped posture).

Transforms (ordered):
  1. v1226 -> v1227 (file names, logs, internal seds generate
     sanity_v1227.sh / validate_v1227.sh writing _v1227_*.txt)
  2. -raw literals shift one version: lane/tag v1.2.26-raw -> v1.2.27-raw;
     cache-floor reference v1.2.25-raw -> v1.2.26-raw
  3. Insert the v127 collector gates after the v126 dualbridge marker echo
Everything else (sanity/admission, drills, serial24 x3, burst_harsh x3,
parser battery, JIT/cache-delta bound, lineage markers, P23F quality gate
A/B) runs unchanged. Caps literals (V1225_*) are rename carryovers — the
ship precondition greps them by design.
"""
SRC = "/root/build/validate_v1226_run.sh"
DST = "/root/build/validate_v1227_run.sh"

with open(SRC) as f:
    t = f.read()

# 1) version names
n_v = t.count("v1226")
t = t.replace("v1226", "v1227")

# 2) -raw literals (order matters: 26->27 first so 25->26 cannot collide)
n_r27 = t.count("v1.2.26-raw")
t = t.replace("v1.2.26-raw", "v1.2.27-raw")
n_r26 = t.count("v1.2.25-raw")
t = t.replace("v1.2.25-raw", "v1.2.26-raw")

# 3) v127 collector gates after the dualbridge marker echo
anchor = 'echo "v126 dualbridge marker=$DBM"'
if t.count(anchor) != 1:
    raise SystemExit("dualbridge echo anchor count %d != 1" % t.count(anchor))
m0block = (
    'M0M=$(docker exec lsv-test sh -c \'grep -c "llm-scaler v127 M0LIVE-COLLECTOR" '
    '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\' | '
    "tr -d '[:space:]')\n"
    '[ "$M0M" -ge 1 ] 2>/dev/null || { echo "ABORT_M0LIVE_MARKER got=$M0M"; exit 1; }\n'
    'M0D=$(docker exec lsv-test sh -c \'test ! -e /root/.v127_m0live && echo 1 || echo 0\')\n'
    '[ "$M0D" = "1" ] || { echo "ABORT_M0LIVE_ARMED"; exit 1; }\n'
    'SSD=$(docker exec lsv-test sh -c \'grep -c "llm-scaler v127 SCALEDSPACE" '
    '/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py\' | '
    "tr -d '[:space:]')\n"
    '[ "$SSD" = "0" ] 2>/dev/null || { echo "ABORT_SCALEDSPACE_PRESENT got=$SSD"; exit 1; }\n'
    'echo "v127 m0live collector marker=$M0M dormant=$M0D scaledspace_absent=$SSD"\n')
t = t.replace(anchor, anchor + "\n" + m0block)

# header note
hdr = ("# mk_validate_v1227.py note: v127 ship-posture variant — m0live collector\n"
       "# present+dormant gate and scaled-space absence gate (P43) after the\n"
       "# dualbridge gate; -raw refs shifted 26->27 (lane) and 25->26 (floor);\n"
       "# all else identical to v1226.\n")
t = t.replace("#!/bin/sh\n", "#!/bin/sh\n" + hdr, 1)

with open(DST, "w") as f:
    f.write(t)

print("wrote", DST)
print("v1226->v1227 replaces:", n_v)
print("v1.2.26-raw -> v1.2.27-raw:", n_r27)
print("v1.2.25-raw -> v1.2.26-raw:", n_r26)
print("m0live gate:", t.count("ABORT_M0LIVE"))
print("dualbridge gate kept:", t.count("llm-scaler v126 DUAL BRIDGE"))
