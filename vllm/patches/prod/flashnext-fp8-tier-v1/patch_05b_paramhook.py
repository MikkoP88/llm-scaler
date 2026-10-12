#!/usr/bin/env python3
"""FN8_05b (DEC40 D7): opt vLLM Parameters out of __torch_function__.

D4 py-spy (pyspy_d4_tp0/tp1.raw): 9.5% of the 250 ms eager step is
vllm/model_executor/parameter.py:126 __torch_function__ — a pure
passthrough (kwargs default + super() call) that nonetheless forces
python-mode dispatch on EVERY op touching a vLLM Parameter (linear
forwards hit it per weight, per layer, per step).  Weight loading never
relies on op interception (tf_count audit: the hook body only forwards).
torch core's own opt-out is torch._C._disabled_torch_function_impl —
assign it at the Base class and the per-op tax disappears for every
Parameter subclass.

Edit (parameter.py): inject the assignment between BasevLLMParameter and
its first subclass.  Idempotent; py_compile gate; backup .fn8_05b.bak;
EOL aware; anchor==1 gate.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_05b"
REL = "model_executor/parameter.py"
MARKER = "_disabled_torch_function_impl"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''class _ColumnvLLMParameter(BasevLLMParameter):
'''
NEW_LF = '''# FN8_05b (DEC40 D7): the per-op __torch_function__ passthrough above
# forces python-mode dispatch on every op touching a vLLM Parameter
# (py-spy D4: 9.5% of the eager step = parameter.py:126).  The hook is a
# pure passthrough; weight loading never relies on op interception.
# Route to torch core's C impl — the same opt-out torch itself uses.
BasevLLMParameter.__torch_function__ = torch._C._disabled_torch_function_impl


class _ColumnvLLMParameter(BasevLLMParameter):
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_05b.bak"
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

if "\nimport torch\n" not in src and not src.startswith("import torch\n"):
    print(f"{TAG} FAIL: parameter.py has no bare 'import torch'")
    sys.exit(5)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


n = src.count(ed(OLD_LF))
if n != 1:
    print(f"{TAG} FAIL: anchor 'column_class' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_05b.bak"
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
