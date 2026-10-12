#!/usr/bin/env python3
"""FN8_06d (DEC40 D8): env-gated PLE zero-stub — graph-perf probe.

Root finding (D8 boot 3, 08:23:59): under spec+TP2 the fork takes the
xpu.py `if _unsafe_spec_tp_graph:` guard branch (TORCH_COMPILE_DISABLE=1,
"whole-step XPU graph capture alone is clean and faster") which SKIPS the
entire else-chain where every splitting_ops append lives — with dynamo
off there is NO piecewise partitioner, splitting_ops is consulted by
nothing, and the whole-step capture records the RAW eager model.  The
registered custom op (06c) therefore still executed inside capture
(same "wait method cannot be used for an event associated with a command
graph" at the op's D2H) and VLLM_XPU_SPEC_EXTRA_SPLITS is inert in this
posture.

The correct-quality fix (host-side prestage of PLE rows between the
eager drafter head and the verify replay, in-graph pinned->device copy)
is a large build.  Before building it, PROVE the lever: FN8_PLE_STUB=1
returns zeros for the n-gram embedding (capture-legal; output quality
garbage — parity NOT expected; wall-clock step time indicative of the
graphed ceiling).  Gate decision:
  * graphed step >= ~3.2 tok/step at <= 80 ms  -> build the prestage
  * graphed step still ~250 ms                 -> graphs lever dead,
    pivot (Q4-of-125B / other)

Edit (qwen4_exp.py): stub flag + stub counter in the existing 06c
telemetry rec + zero-return branch at the head of
Qwen4ExpNGramEmbedding.forward.  Default OFF (behavior = 06c).

Idempotent; py_compile gate; backup .fn8_06d.bak; EOL aware;
anchor==1 gates; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06d"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_PLE_STUB"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1: stub flag on the 06c telemetry rec line -----------------------------
OLD1_LF = '''_FN8_PLE_OP_REC = {"calls": 0, "rows": 0, "empty": 0, "ms": 0.0}
'''
NEW1_LF = '''_FN8_PLE_OP_REC = {"calls": 0, "rows": 0, "empty": 0, "ms": 0.0,
                   "stub": 0}
_FN8_PLE_STUB = os.environ.get("FN8_PLE_STUB", "") == "1"
'''

# --- E2: zero-return branch at head of forward --------------------------------
OLD2_LF = '''        FN8_06c: single registered-op call (vllm::ple_ngram_gather) —
        the split-list boundary; everything D2H/host/H2D/dequant runs
        eager between captured pieces.  _dequant_rows kept above for
        reference/parity but no longer on this path.
        """
        emb = torch.ops.vllm.ple_ngram_gather(
'''
NEW2_LF = '''        FN8_06c: single registered-op call (vllm::ple_ngram_gather) —
        the split-list boundary; everything D2H/host/H2D/dequant runs
        eager between captured pieces.  _dequant_rows kept above for
        reference/parity but no longer on this path.

        FN8_06d: FN8_PLE_STUB=1 -> zero embedding (graph-perf probe;
        D8 boot-3 law: no partitioner under spec+TP2, the op cannot be
        split, so the only capture-legal stand-in is zeros).  Quality
        is garbage by design — perf probe ONLY.
        """
        if _FN8_PLE_STUB:
            _FN8_PLE_OP_REC["stub"] += 1
            return torch.zeros(
                *ngram_ids.shape[:-1], self.embedding_dim,
                dtype=torch.float16, device=ngram_ids.device)
        emb = torch.ops.vllm.ple_ngram_gather(
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06d.bak"
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


for name, old_lf in (("rec_line", OLD1_LF), ("forward_head", OLD2_LF)):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_06d.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ed(OLD1_LF), ed(NEW1_LF)).replace(ed(OLD2_LF), ed(NEW2_LF))

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
