#!/usr/bin/env python3
"""FN8_06c (DEC40 D8): PLE n-gram gather -> registered custom op split.

D8 boot-2 died AT CAPTURE (07:57:00, serve_d8.log):
  RuntimeError: wait method cannot be used for an event associated
  with a command graph.   (qwen4_exp.py:489 flat_ids.to("cpu"))
— the PLE n-gram embedding's blocking D2H, inside a captured piece.
The table is HOST-RESIDENT BY DESIGN (~51 GB padded_vocab*head_dim fp8
— no VRAM on dual B70 can hold it), so the D2H->host gather->H2D can
never be capture-legal.

Fix (option G, moe/GDN split precedent on this exact stack): wrap the
WHOLE gather+dequant as torch.library.custom_op("vllm::ple_ngram_gather")
+ register_fake (dynamo-traceable, per esimd_fake_ops law) and add the
name to the piecewise split list via
VLLM_XPU_SPEC_EXTRA_SPLITS=...vllm::ple_ngram_gather — split ops
execute EAGER between captured pieces (xpu.py splitting_ops law;
"unmatched names are harmless").  partition_rules.should_split matches
the FX call_function node's OpOverload name — a registered torch op
qualifies.

Edits (qwen4_exp.py):
  E1  module scope before class Qwen4ExpNGramEmbedding: counter rec +
      custom_op registration (gather body == current forward/_dequant_rows
      math, verbatim) + fake impl + atexit telemetry dump (standing
      telemetry directive).
  E2  Qwen4ExpNGramEmbedding.forward -> single torch.ops call + view.

Idempotent; py_compile gate; backup .fn8_06c.bak; EOL aware;
anchor==1 gates; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06c"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "ple_ngram_gather"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1: module-level op registration before the embedding class -----------
OLD1_LF = '''class Qwen4ExpNGramEmbedding(nn.Module):
'''
NEW1_LF = '''# FN8_06c (D8 graphs): whole n-gram gather as a registered custom op so
# the piecewise splitter runs it EAGER between captured pieces (split-list
# entry vllm::ple_ngram_gather via VLLM_XPU_SPEC_EXTRA_SPLITS).  The table
# is host-resident BY DESIGN (~51 GB — no VRAM can hold it), so the
# blocking D2H/H2D here can never be capture-legal; D8 crash-2 died
# exactly at this site.  Split precedent: moe_ops + v31 GDN arm.
_FN8_PLE_OP_REC = {"calls": 0, "rows": 0, "empty": 0, "ms": 0.0}


@torch.library.custom_op("vllm::ple_ngram_gather", mutates_args=())
def _ple_ngram_gather(ngram_ids: torch.Tensor, table: torch.Tensor,
                      scale_inv: torch.Tensor, scale: torch.Tensor,
                      col_block: int, embedding_dim: int) -> torch.Tensor:
    """Gather host table rows for flat ids; dequant to fp16 (T, K).

    Body is the pre-06c forward + _dequant_rows math, verbatim: blocking
    D2H of ids -> host index_select (uint8 view for fp8) -> blocking H2D
    of rows -> device dequant (scalar or [128,128]-block scales).
    """
    import time as _t
    _t0 = _t.time()
    id_device = ngram_ids.device
    flat_ids = ngram_ids.reshape(-1)
    if flat_ids.numel() == 0:
        _FN8_PLE_OP_REC["calls"] += 1
        _FN8_PLE_OP_REC["empty"] += 1
        _FN8_PLE_OP_REC["ms"] += _t.time() - _t0
        return torch.zeros(0, embedding_dim, dtype=torch.float16,
                           device=id_device)
    ids_cpu = flat_ids.to("cpu", non_blocking=False).long()
    if table.dtype == torch.float8_e4m3fn:
        rows = table.view(torch.uint8).index_select(0, ids_cpu).view(
            torch.float8_e4m3fn)
    else:
        rows = table.index_select(0, ids_cpu)
    rows_dev = rows.to(id_device, non_blocking=False)
    out = rows_dev.to(torch.float32)
    if col_block == 0:
        if scale_inv.numel() == 1:
            out = out * scale_inv.to(out.device).to(torch.float32)
        elif scale.numel() == 1:
            out = out * scale.to(out.device).to(torch.float32)
        # else: no scales at all — pure values cast up (load-time logged)
        res = out.to(torch.float16)
    else:
        sc = scale_inv.to(out.device).to(torch.float32)  # (V/128, K)
        scale_rows = sc[ids_cpu.to(id_device) // col_block]  # (T, K)
        width = rows_dev.shape[-1]
        for k, start in enumerate(range(0, width, col_block)):
            end = min(start + col_block, width)
            out[..., start:end] = out[..., start:end] * scale_rows[
                ..., k, None]
        res = out.to(torch.float16)
    _FN8_PLE_OP_REC["calls"] += 1
    _FN8_PLE_OP_REC["rows"] += int(flat_ids.numel())
    _FN8_PLE_OP_REC["ms"] += _t.time() - _t0
    return res


@_ple_ngram_gather.register_fake
def _ple_ngram_gather_fake(ngram_ids, table, scale_inv, scale, col_block,
                           embedding_dim):
    return torch.zeros(ngram_ids.reshape(-1).shape[0], embedding_dim,
                       dtype=torch.float16, device=ngram_ids.device)


def _fn8_ple_op_dump():
    import sys as _s
    print(f"FN8_PLE_OP {_FN8_PLE_OP_REC}", file=_s.stderr, flush=True)


import atexit as _fn8_atexit  # noqa: E402 (FN8_06c telemetry)
_fn8_atexit.register(_fn8_ple_op_dump)


class Qwen4ExpNGramEmbedding(nn.Module):
'''

# --- E2: forward -> op call ---------------------------------------------------
OLD2_LF = '''    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather host table rows for (R, L, heads) ids -> (R, L, embedding_dim)
        fp16 on the ids' device."""
        id_device = ngram_ids.device
        flat_ids = ngram_ids.reshape(-1)
        num_rows = flat_ids.numel()
        if num_rows == 0:
            return torch.zeros(
                *ngram_ids.shape[:-1], self.embedding_dim,
                dtype=torch.float16, device=id_device,
            )
        # P0: synchronous D2H of the ids, host gather, H2D of the rows.
        # (2.56 KB/token; optimization — pinned staging + async copies — is a
        # later-phase item, see PHASES.md.)
        ids_cpu = flat_ids.to("cpu", non_blocking=False).long()
        if self._host_table.dtype == torch.float8_e4m3fn:
            rows = self._host_table.view(torch.uint8).index_select(0, ids_cpu).view(torch.float8_e4m3fn)
        else:
            rows = self._host_table.index_select(0, ids_cpu)
        rows_dev = rows.to(id_device, non_blocking=False)
        emb = self._dequant_rows(rows_dev, ids_cpu.to(id_device))
        # (R, L, heads) ids -> (R, L, heads*head_dim) == ple_embed_dim,
        # already the reference flatten(-2) result.
        return emb.view(*ngram_ids.shape[:-1], self.embedding_dim)
'''
NEW2_LF = '''    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather host table rows for (R, L, heads) ids -> (R, L, embedding_dim)
        fp16 on the ids' device.

        FN8_06c: single registered-op call (vllm::ple_ngram_gather) —
        the split-list boundary; everything D2H/host/H2D/dequant runs
        eager between captured pieces.  _dequant_rows kept above for
        reference/parity but no longer on this path.
        """
        emb = torch.ops.vllm.ple_ngram_gather(
            ngram_ids, self._host_table, self._host_scale_inv,
            self._host_scale, self._col_block_size or 0, self.embedding_dim,
        )
        # (R, L, heads) ids -> (R, L, heads*head_dim) == ple_embed_dim,
        # already the reference flatten(-2) result.
        return emb.view(*ngram_ids.shape[:-1], self.embedding_dim)
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06c.bak"
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


for name, old_lf in (("class_anchor", OLD1_LF), ("forward_body", OLD2_LF)):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_06c.bak"
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
