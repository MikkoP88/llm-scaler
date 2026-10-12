#!/usr/bin/env python3
"""FN8_06g (DEC40): runner hook — call model.fn8_ple_prestage in the eager
window between _preprocess and dispatch (set_forward_context / _model_forward).

Site: execute_model, right after the input-prep `with` block closes —
input_ids (the exact tensor the model receives) and attn_metadata (built at
_build_attention_metadata) are both in scope, and NO capture context is
active yet.  The hook's host work (per-step PLE row staging for the 06f
device banks) is therefore always EAGER; the only sync is the ~KB ids D2H
inside Qwen4ExpNGramEmbedding.fn8_prestage.  During graph-capture driver
steps the same execute_model path runs eagerly BEFORE capture begins, so
banks are staged (and were allocated at load via 06f) before any record.

Defensive by mandate: missing attribute -> silent no-op; any exception ->
warning + step continues (the graph reads last-step rows for one step —
finite, in-distribution, quality-degraded only).

Idempotent; py_compile gate; backup .fn8_06g.bak; EOL aware; anchor==1;
--restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06g"
REL = "v1/worker/gpu_model_runner.py"
MARKER = "FN8_06g"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''        # Run the model.
        # Use persistent buffers for CUDA graphs.
'''
NEW_LF = '''        # FN8_06g: PLE row prestage (eager window before dispatch).  The
        # host-side n-gram table gather for THIS step lands in fixed device
        # banks that captured decode graphs slice+view (06f).  Runs outside
        # set_forward_context and any capture; the only sync is a ~KB ids
        # D2H.  Defensive: never break the step — on failure the graph
        # replays on last-step rows (finite, in-distribution) and we log.
        _fn8_ple_ps = getattr(self.model, "fn8_ple_prestage", None)
        if _fn8_ple_ps is not None:
            try:
                _fn8_ple_ps(input_ids, attn_metadata)
            except Exception as _fn8_e:  # noqa: BLE001 (telemetry mandate)
                logger.warning(
                    "FN8_06g: PLE prestage failed (step continues on stale "
                    "rows): %s", _fn8_e)

        # Run the model.
        # Use persistent buffers for CUDA graphs.
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06g.bak"
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
    print(f"{TAG} FAIL: anchor 'run_model' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06g.bak"
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
