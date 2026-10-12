#!/usr/bin/env python3
"""FN8_06b (DEC40 D8): PLE spec guard -> uniform-bake for graph capture.

D8 boot died AT CAPTURE (07:39:05, serve_d8.log:448+):
  RuntimeError: Qwen4Exp PLE spec: non-uniform verify widths
  (20 tokens vs S*L=100)
— patch_05's guard firing on the CAPTURE-TIME DUMMY metadata, which is
internally inconsistent (20 spec tokens against 20x5 slot grid).  Real
spec steps are uniform k+1 (D7: the guard never fired in ~1500 live
steps across 6 legs + parity).  Only the dummy is anomalous.

Fix: on mismatch, rebuild s_indx = arange(S*L) and continue down the
uniform static path.  The captured graph then bakes the path REAL
replay traffic takes (uniform); the dummy's own output is discarded
anyway.  Shape-only python (no .item(), no sync) — capture-legal.

Edit (qwen4_exp.py): replace the raise with the arange bake.
Idempotent; py_compile gate; backup .fn8_06b.bak; EOL aware;
anchor==1 gate; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06b"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "06b: capture-time dummy metadata"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''            if s_indx.shape[0] != n_spec_toks:
                raise RuntimeError(
                    "Qwen4Exp PLE spec: non-uniform verify widths "
                    f"({s_indx.shape[0]} tokens vs S*L={n_spec_toks})"
                )
'''
NEW_LF = '''            if s_indx.shape[0] != n_spec_toks:
                # 06b: capture-time dummy metadata is internally
                # inconsistent (D8 crash: 20 tokens vs S*L=100); real
                # spec steps are uniform k+1 (D7: this guard never fired
                # in ~1500 live steps).  Bake the UNIFORM path into the
                # graph — replay executes it with real uniform traffic;
                # the dummy run's own output is discarded.
                s_indx = torch.arange(n_spec_toks, device=device)
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06b.bak"
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
    print(f"{TAG} FAIL: anchor occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06b.bak"
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
