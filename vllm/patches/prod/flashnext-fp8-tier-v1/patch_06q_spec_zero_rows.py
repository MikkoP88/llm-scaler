#!/usr/bin/env python3
"""FN8_06q (d11 boundary-death fix): PLE spec S=0 skip instead of kernel death.

d11 serve died 16:45:59 2026-10-12 under real traffic (request from
10.100.8.6, 2085 output tokens, num_computed_tokens=8187 of
max_model_len 8192):
  - At the context boundary the scheduler scheduled a PARTIAL spec
    window (4 of the 5 MTP tokens; SchedulerOutput showed
    num_scheduled_tokens=4, scheduled_spec_decode_tokens=[-1,-1,-1]).
  - _ple_forward_spec took the 06n recompute path
    (s_indx.shape[0] != S*L) -> S = min(map, T) // L = 4 // 5 = 0,
    n_spec_toks = 0, hs_pad = hidden_states[arange(0)].view(0, L, D).
  - ple(hs_pad, ...) -> norm_key -> _norm's
    x.reshape(*x.shape[:-1], -1, self.group_size) on a 0-element
    [0, 5, 12800] tensor -> RuntimeError "cannot reshape tensor of 0
    elements into shape [0, 5, -1, 2560] ... ambiguous" -> worker dead
    -> EngineDeadError -> every subsequent request 500.

06n/06o (DEC40) fixed S too LARGE (per-token start_loc read as
per-seq); this is the complementary case T < L -> S = 0, which still
reached the kernel.

Fix: after the 06n recompute, if S == 0 skip the spec kernel block
entirely — zero PLE contribution for the degenerate step (identity on
the layer's residual-add path); n_spec_toks is 0 so n_nonspec_toks = T
and every row of outs is still written by the non-spec metadata group
or the 06p stateless fallback.  Loud FN8_06q ERROR log, first 10 hits
+ every 100th.  Capture never takes this branch with S=0 baked
meaningfully (FULL_DECODE_ONLY dummies are uniform-width), but the
skip itself is shape-static and bake-safe either way.

Edit (qwen4_exp.py, _ple_forward_spec spec kernel block):
  Q1  guard the kernel body: `if S == 0: <log+skip> else: <body>`.

Idempotent; py_compile gate; backup .fn8_06q.bak; EOL aware;
anchor==1; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06q"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06q"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

GUARD_LF = '''            if S == 0:
                # 06q (d11 boundary-death fix): a context-boundary spec
                # step can schedule FEWER than L tokens (16:45:59
                # 2026-10-12: request at 8187/8192 computed, 4 of the
                # 5-token MTP window scheduled) -> the 06n recompute
                # S = T // L = 0 -> hs_pad [0, L, D] -> the PLE kernel's
                # group-norm reshape(x.shape[:-1], -1, group_size) dies
                # on 0 elements ("unspecified dimension size -1 ...
                # ambiguous") -> EngineDead.  Skip the kernel for the
                # degenerate step: zero PLE contribution on the
                # residual-add path; n_spec_toks=0 so the non-spec
                # group below (or the 06p stateless fallback) still
                # writes every row of outs.
                _FN8_PLE_PATH_REC["spec_zero"] = (
                    _FN8_PLE_PATH_REC.get("spec_zero", 0) + 1)
                _n6q = _FN8_PLE_PATH_REC["spec_zero"]
                if _n6q <= 10 or _n6q % 100 == 0:
                    logger.error(
                        "FN8_06q PLE spec S=0 boundary skip: T=%d L=%d "
                        "n_spec=%d hit=%d", T, L, n_spec_toks, _n6q)
            else:
'''

ANCHOR_SPEC_FIX = '"spec_fix", "spec"'
KERNEL_START = "            D = hidden_states.shape[-1]"
KERNEL_END = "        # ---- non-spec group"


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06q.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if MARKER in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


# anchor: the 06n sentinel call must occur exactly once
if src.count(ed(ANCHOR_SPEC_FIX)) != 1:
    print(f"{TAG} FAIL: anchor spec_fix occurs "
          f"{src.count(ed(ANCHOR_SPEC_FIX))}x (want 1)")
    sys.exit(2)

a = src.index(ed(ANCHOR_SPEC_FIX))
s = src.index(ed(KERNEL_START), a)
e = src.index(ed(KERNEL_END), s)
body = src[s:e]

# re-indent the whole kernel block (+4) under the else:
lines = body.split(eol)
out_lines = []
for ln in lines:
    out_lines.append(("    " + ln) if ln.strip() else ln)
new_body = eol.join(out_lines)

out = src[:s] + ed(GUARD_LF) + new_body + src[e:]

bak = path + ".fn8_06q.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8", newline=""
) as tf:
    tf.write(out)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e2:
    print(f"{TAG} FAIL: compile error -> {e2}")
    sys.exit(3)
finally:
    os.unlink(tmp)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(out)
print(
    f"{TAG} [{REL}] edited "
    f"(eol={'CRLF' if eol == chr(13) + chr(10) else 'LF'}), md5={md5(path)}"
)
print(f"{TAG} ALL_EDITS_OK")
