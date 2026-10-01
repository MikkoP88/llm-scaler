#!/usr/bin/env python3
"""fix_restore_v128b.py — add the V1227_XGCOMPACT leg knob to
boot_v1227_restore.sh (both copies: /root/build/v127_stage/ = the copy the
watchdog invokes, /root/build/ = synced sibling).

EDIT A (knob block, after the V1227_SCALEDSPACE block):
  V1227_XGCOMPACT=1 appends
    --structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}
  to the generated serve cmdline (sed-insert before the unique
  --async-scheduling anchor, line-verified unique in the baked
  serve_user.sh) and hard-gates its presence (exit 18).

  This is the v128 WS-D engine-side fix candidate: compile_json_schema(
  any_whitespace=False) makes inter-token whitespace ILLEGAL in guided-JSON
  grammars, so the whitespace-attractor degeneration class (P38 quiet
  sample 4/11; xg_sweep_v128: whitespace runaway in every arm incl. the
  pattern-bounded pb1 control) becomes impossible by construction. The
  digit-attractor class on unbounded numeric fields is NOT affected —
  measured separately by xg_flag_probe_v128. Installed surface verified:
  xgrammar 0.2.7 compile_json_schema(any_whitespace=...); guidance NOT
  installed in the image (the auto-mode fallback branch is dead code, so
  pinning backend=xgrammar is zero-regression). Default 0 = certified
  v1.2.27 posture byte-identical.

EDIT B: BOOT echo line gains xgc=/perreq= leg stamps.
EDIT C: header doc line after the V1227_PERREQ doc.

Idempotent: no-op if V1227_XGCOMPACT already present.
Line/anchor-through-EOL law throughout; bash -n verified after edit.
"""
import subprocess
import sys

FILES = ["/root/build/v127_stage/boot_v1227_restore.sh",
         "/root/build/boot_v1227_restore.sh"]

ANCHOR_SCALEDSPACE = (
    'if [ "${V1227_SCALEDSPACE:-0}" = "1" ]; then\n'
    '  docker exec lsv-test sed -i "s/--mamba-ssm-cache-dtype float16/'
    '--mamba-ssm-cache-dtype fp8_e4m3/" /root/serve_user_v1227.sh\n'
    "fi\n"
)

XGCOMPACT_BLOCK = ANCHOR_SCALEDSPACE + '''
# --- optional v128 XGCOMPACT leg (WS-D engine-side fix candidate;
#     see fix_restore_v128b.py / xg_flag_probe_v128.py) ---
if [ "${V1227_XGCOMPACT:-0}" = "1" ]; then
  docker exec lsv-test sed -i 's|--async-scheduling|--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true} --async-scheduling|' /root/serve_user_v1227.sh
  docker exec lsv-test grep -q -F -- '--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}' /root/serve_user_v1227.sh \\
    || { echo "BOOT_$MODE XGCOMPACT_FLAG_MISSING"; exit 18; }
fi
'''

OLD_BOOTLINE_MARK = "m0live=${V1227_M0LIVE:-0} "
NEW_BOOTLINE_MARK = "m0live=${V1227_M0LIVE:-0} xgc=${V1227_XGCOMPACT:-0} perreq=${V1227_PERREQ:-0} "

# the knobs doc header is INLINE (comma-separated), so both doc blocks
# append after the last knob-doc line (the D2 passthrough line)
KNOBDOC_ANCHOR = "# passthrough for the D2 sweep leg), V1227_IMAGE (v1.2.27 since ship 10:35),\n"
DOC_BLOCK = KNOBDOC_ANCHOR + (
    "# V1227_PERREQ (0; 1 applies+arms the v128 D4 per-request split\n"
    "# telemetry patch — see patch_perreq_v128.py),\n"
    "# V1227_XGCOMPACT (0; 1 appends --structured-outputs-config\n"
    '# {"backend":"xgrammar","disable_any_whitespace":true} to the serve\n'
    "# cmdline — v128 WS-D fix candidate: guided-JSON grammars compiled\n"
    "# whitespace-free, killing the whitespace-attractor degeneration class\n"
    "# by construction; JSON output becomes compact),\n"
)

for path in FILES:
    with open(path) as f:
        t = f.read()

    if "V1227_XGCOMPACT" in t:
        print(path, "-> already patched, no-op")
        continue

    assert t.count(ANCHOR_SCALEDSPACE) == 1, \
        "%s scaledspace anchor=%d" % (path, t.count(ANCHOR_SCALEDSPACE))
    t = t.replace(ANCHOR_SCALEDSPACE, XGCOMPACT_BLOCK)

    lines = t.splitlines(keepends=True)
    n_boot = 0
    for i, ln in enumerate(lines):
        if OLD_BOOTLINE_MARK in ln and "echo " in ln:
            lines[i] = ln.replace(OLD_BOOTLINE_MARK, NEW_BOOTLINE_MARK)
            n_boot += 1
    assert n_boot == 1, "%s bootline edits=%d" % (path, n_boot)
    t = "".join(lines)

    assert t.count(KNOBDOC_ANCHOR) == 1, \
        "%s knobdoc anchor=%d" % (path, t.count(KNOBDOC_ANCHOR))
    t = t.replace(KNOBDOC_ANCHOR, DOC_BLOCK)

    with open(path, "w") as f:
        f.write(t)

    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    with open(path) as f:
        t2 = f.read()
    print(path)
    print("  xgcompact knob blocks:", t2.count("V1227_XGCOMPACT"))
    print("  flag gate (exit 18):", "XGCOMPACT_FLAG_MISSING" in t2)
    print("  bootline stamps xgc/perreq:", NEW_BOOTLINE_MARK in t2)
    print("  bash -n:", "SYNTAX_OK" if r.returncode == 0 else "FAIL " + r.stderr)

print("fix_restore_v128b DONE")
