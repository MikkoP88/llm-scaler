#!/usr/bin/env python3
"""FN8_06d4 (DEC40): fix the 06d2 SAFE-stub row shape — table rows are
PER-HEAD, not [embedding_dim].

Root cause (d8s3 fix-gate boot try 1, 2026-10-12 11:10:37, banked
/stage/bank/d8s3_fix2_fatal.log): with the 06d3 gate fix, capture finally
reached the 06d2 stub branch — which then died on

    RuntimeError: The expanded size of the tensor (2560) must match the
    existing size (160) at non-singleton dimension 2.
    Target sizes: [1, 4096, 2560].  Tensor sizes: [160]

The PLE host table layout is [total_rows, head_dim_per_ngram=160]; the
REAL op path is ngram_ids [R,L,heads=16] -> gather -> [R,L,16,160] ->
view -> [R,L,2560] (heads*head_dim == ple_embed_dim).  The 06d2 stub
dequantized ONE row ([160]) and expand()-ed it to embedding_dim 2560 —
size mismatch.

Fix: tile the dequantized row 0 across heads (repeat) so the constant is
[heads*head_dim] = [2560].  Values stay real-row-0 magnitudes (d8s
crash-5 NaN-drift law intact); quality remains garbage BY DESIGN (perf
probe only).

Idempotent; py_compile gate; backup .fn8_06d4.bak; EOL aware; anchor==1;
--restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06d4"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06d4"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''                    self._fn8_stub_row = _o.to(torch.float16).reshape(-1)
                    row = self._fn8_stub_row
'''
NEW_LF = '''                    # 06d4: table rows are PER-HEAD (last dim
                    # head_dim_per_ngram=160), not [embedding_dim]; the
                    # real op gathers [R,L,heads,160] then flattens to
                    # [R,L,heads*160=2560].  Tile row 0 across the heads
                    # so the expand below hits the right shape.
                    self._fn8_stub_row = (
                        _o.to(torch.float16).reshape(-1).repeat(
                            self.embedding_dim // _o.shape[-1]))
                    row = self._fn8_stub_row
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06d4.bak"
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
    print(f"{TAG} FAIL: anchor 'stub_row' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06d4.bak"
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
