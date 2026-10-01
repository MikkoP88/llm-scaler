#!/usr/bin/env python3
"""fix_restore_v128.py — two edits to boot_v1227_restore.sh (host, both
copies: /root/build/ and /root/build/v127_stage/ — the latter is what the
watchdog actually invokes; the former is a synced sibling).

EDIT 1 (P46 hardening — restore-path posture currency):
  V1227_POSTURE default flips swift -> ckpt. The swift branch reproduces
  the PRE-certification designated posture (/models/swift bind + runtime
  --quantization fp8 + --chat-template flags), which since the v1.2.27
  ship is no longer the certified surface — the checkpoint-folded
  posture (pre-quantized fp8 export at /models/target, template baked
  into tokenizer_config.json, NO runtime flags; the drift guard asserts
  the flags' absence and labels it posture=certified-swift-fp8) is the
  script's else branch. A default-env watchdog restore must land on the
  CERTIFIED surface, so the default becomes ckpt (= else branch). The
  swift branch stays reachable for explicit historical legs only.

EDIT 2 (P47 prep — D4 per-request split leg hook):
  V1227_PERREQ=1 boot leg, mirroring the V1227_M0LIVE hook: apply
  patch_perreq_v128.py inside the fresh container BEFORE serve start,
  verify the marker, arm /root/.v128_perreq, clear stale JSONL. Default
  0 = dormant (watchdog restores never touch the patch).

Line-based transforms (anchor-through-EOL law; no mid-line prefixes).
"""
import sys

FILES = ["/root/build/v127_stage/boot_v1227_restore.sh",
         "/root/build/boot_v1227_restore.sh"]

HOOK_ANCHOR = ("  docker exec lsv-test touch /root/.v127_m0live\n"
               "  docker exec lsv-test rm -f /root/m0live_runmax_*.pt\n"
               "fi\n")
HOOK_BLOCK = HOOK_ANCHOR + '''
# --- optional v128 PERREQ leg (WS-D D4 per-request split telemetry;
#     dormant patch unless /root/.v128_perreq armed; the frontend appends
#     one JSON line per finished request to /root/v128_perreq.jsonl;
#     harvest = docker cp out after the leg, analyze with d4_split_v128) ---
if [ "${V1227_PERREQ:-0}" = "1" ]; then
  docker cp /root/build/v128_stage/patch_perreq_v128.py lsv-test:/root/patch_perreq_v128.py
  docker exec lsv-test /opt/venv/bin/python3 /root/patch_perreq_v128.py | tee /root/build/lce1/v128_perreq_apply.log
  docker exec lsv-test grep -c "llm-scaler v128 PERREQ" /opt/venv/lib/python3.12/site-packages/vllm/v1/engine/output_processor.py
  docker exec lsv-test touch /root/.v128_perreq
  docker exec lsv-test rm -f /root/v128_perreq.jsonl
fi
'''

NEW_SWIFT_LINE = ("#   ckpt   (default) CERTIFIED since the v1.2.27 ship: pre-quantized fp8\n"
                  "#                     swift export mounted at /models/target, chat template\n"
                  "#                     baked into tokenizer_config.json, NO runtime\n"
                  "#                     --quantization/--chat-template flags (drift guard\n"
                  "#                     asserts their absence; = the script's else branch)\n"
                  "#   swift  explicit-legs-only PRE-certification designated posture:\n"
                  "#                     /models/swift bind + runtime quant/template flags\n")

for path in FILES:
    with open(path) as f:
        lines = f.readlines()

    out = []
    n_posture = n_hdr = n_hook = 0
    for ln in lines:
        if ln == 'V1227_POSTURE="${V1227_POSTURE:-swift}"\n':
            out.append('V1227_POSTURE="${V1227_POSTURE:-ckpt}"\n')
            n_posture += 1
            continue
        if ln.startswith("#   swift  (default)"):
            out.append(NEW_SWIFT_LINE)
            n_hdr += 1
            continue
        if ln.startswith("# V1227_IMAGE "):
            out.append(ln)
            out.append("# V1227_PERREQ (0; 1 applies+arms the v128 D4 per-request split\n"
                       "# telemetry patch — see patch_perreq_v128.py),\n")
            continue
        out.append(ln)

    t = "".join(out)
    assert n_posture == 1, "%s posture flips=%d" % (path, n_posture)
    assert n_hdr == 1, "%s header rewrites=%d" % (path, n_hdr)
    if "V1227_PERREQ" not in t.split(HOOK_ANCHOR)[0]:
        pass  # hook inserted below
    assert t.count(HOOK_ANCHOR) == 1, "%s hook anchor=%d" % (
        path, t.count(HOOK_ANCHOR))
    t = t.replace(HOOK_ANCHOR, HOOK_BLOCK)

    with open(path, "w") as f:
        f.write(t)

    with open(path) as f:
        t2 = f.read()
    print(path)
    print("  posture default ckpt:", 'V1227_POSTURE:-ckpt}' in t2)
    print("  perreq hook:", t2.count("V1227_PERREQ"))
    print("  m0live block intact:", t2.count("V1227_M0LIVE"))

print("fix_restore_v128 DONE")
