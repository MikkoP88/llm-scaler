#!/usr/bin/env python3
"""FN8_06d3 (DEC40): widen the FN8_PLE_STUB module gate — "=2" was dead code.

Root cause (d8s3 tries 1-2, 2026-10-12, banked /stage/bank/d8s3_try2_fatal.log):
FN8_PLE_STUB=2 posture reached graph capture for the first time (d8s/d8s2
never got past the load wedge; FN8_07 fixed that).  The module gate

    _FN8_PLE_STUB = os.environ.get("FN8_PLE_STUB", "") == "1"

is False for "2", so the 06d2 SAFE-stub branch (constant REAL row 0,
nested INSIDE `if _FN8_PLE_STUB:`) never ran.  Forward fell through to
torch.ops.vllm.ple_ngram_gather, whose backend does a blocking D2H
(flat_ids.to("cpu", non_blocking=False)) — inside torch.xpu.graph capture
that raises Level-Zero's

    RuntimeError: wait method cannot be used for an event associated
    with a command graph.

(D8 crash-2 law, capture-illegal host op — same landmine, new error
string under FULL_DECODE_ONLY capture.)  Stack: _dummy_run -> layer
forward 1141 -> _ple_forward_flat 925 -> _ple_forward_spec 1034 -> PLE
forward 691 -> embedding forward 600 -> _ple_ngram_gather 351.

Fix: gate activates for ANY non-empty FN8_PLE_STUB; the inner 06d2 branch
still distinguishes "2" (real-row-0 constant) from "1"/other (zeros).
No other reader of _FN8_PLE_STUB depends on the == "1" distinction.

Idempotent; py_compile gate; backup .fn8_06d3.bak; EOL aware; anchor==1;
--restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06d3"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06d3"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''_FN8_PLE_STUB = os.environ.get("FN8_PLE_STUB", "") == "1"
'''
NEW_LF = '''# 06d3: gate was == "1" — the 06d2 SAFE stub (=2, nested inside the
# `if _FN8_PLE_STUB:` block) was DEAD CODE, so STUB=2 fell through to the
# 06c host-gather op whose blocking D2H is capture-ILLEGAL (d8s3 capture
# fatal: "wait method cannot be used for an event associated with a
# command graph").  Activate for any non-empty value; the inner branch
# still distinguishes "2" (real row 0) from zeros.
_FN8_PLE_STUB = os.environ.get("FN8_PLE_STUB", "") != ""
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06d3.bak"
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


n = src.count(ed(OLD_LF))
if n != 1:
    print(f"{TAG} FAIL: anchor 'stub_gate' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06d3.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ed(OLD_LF), ed(NEW_LF))

with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8", newline=""
) as tf:
    tf.write(out)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print(f"{TAG} FAIL: compile error -> {e}")
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
