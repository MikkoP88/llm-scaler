#!/usr/bin/env python3
"""FN8_06d2 (DEC40 D8s2): SAFE constant-row PLE stub — graph gate probe.

d8s crash-5 (09:13:07): the 06d ZERO stub captured clean (3/3) but the
first leg request died at decode token ~18 with
    IndexKernelUtils.h:63 "vectorized gather kernel index out of bounds"
-> EngineDead.  Zero-magnitude PLE output lets activations drift until
an fp8 quant overflows -> NaN logits -> sampler garbage ids -> embedding
gather OOB (18 tokens of finite-but-wrong decode first = drift, not a
deterministic 06e shape bug, which would die on step 1).

FN8_PLE_STUB=2 returns a constant REAL table row (row 0, dequantized
once with the layer's real scale, cached on the module): identical cost
shape to the planned 06f prestage, in-distribution magnitudes, quality
still wrong (probe ONLY) but numerics stable -> the leg completes ->
graphed step_ms gate number.

Gate (unchanged from 06d):
  * >= 3.2 tok/step at <= ~80 ms -> build 06f real prestage
  * still ~250 ms                 -> graphs lever dead, pivot (Q4)
FN8_PLE_STUB=1 keeps the original zero stub for reference.

Idempotent; py_compile gate; backup .fn8_06d2.bak; EOL aware;
anchor==1 gates; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06d2"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "06d2:"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- safe stub branch inside the 06d stub block --------------------------------
OLD_LF = '''        if _FN8_PLE_STUB:
            _FN8_PLE_OP_REC["stub"] += 1
            return torch.zeros(
                *ngram_ids.shape[:-1], self.embedding_dim,
                dtype=torch.float16, device=ngram_ids.device)
'''
NEW_LF = '''        if _FN8_PLE_STUB:
            _FN8_PLE_OP_REC["stub"] += 1
            if os.environ.get("FN8_PLE_STUB", "") == "2":
                # 06d2: SAFE stub — constant REAL table row (row 0,
                # dequantized once with the layer scale, cached).  The
                # 06d zero stub NaN-drifted to a gather OOB at decode
                # token ~18 (d8s crash-5); real-row magnitudes keep the
                # activations in-distribution so the probe leg survives.
                # Cost shape == planned 06f prestage (no host gather on
                # the step path).  Lazy init relies on the eager profile
                # run preceding capture (upstream PinnedHostEmbedding
                # invariant, same law).
                row = getattr(self, "_fn8_stub_row", None)
                if row is None or row.device != ngram_ids.device:
                    _t = self._host_table[:1].to(ngram_ids.device)
                    _o = _t.to(torch.float32)
                    if self._col_block_size is None:
                        if self._host_scale_inv.numel() == 1:
                            _o = _o * self._host_scale_inv.to(
                                _o.device).to(torch.float32)
                        elif self._host_scale.numel() == 1:
                            _o = _o * self._host_scale.to(
                                _o.device).to(torch.float32)
                    else:
                        _sc = self._host_scale_inv.to(
                            _o.device).to(torch.float32)
                        _o = _o * _sc[0].view(1, -1)
                    self._fn8_stub_row = _o.to(torch.float16).reshape(-1)
                    row = self._fn8_stub_row
                return row.expand(
                    *ngram_ids.shape[:-1], self.embedding_dim)
            return torch.zeros(
                *ngram_ids.shape[:-1], self.embedding_dim,
                dtype=torch.float16, device=ngram_ids.device)
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06d2.bak"
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
    print(f"{TAG} FAIL: anchor 'stub_block' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06d2.bak"
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
