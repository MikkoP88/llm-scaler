#!/usr/bin/env python3
"""llm-scaler v39b (TQ continuation prefill): rotate q+new-K, not cached-K.

_continuation_prefill dequants the whole cached prefix per layer per
chunk and then INVERSE-ROTATES every cached K row back to the original
space (k_flat @ Pi_half over cached_len*Hk rows) so it can run FA2 with
the raw query and raw new-chunk keys. That is an O(context) GEMM pass
(+ full-prefix fp16 read/write traffic + temp) repeated per layer per
continuation chunk — pure overhead at deep context.

PATCH: leave the cached K in its Hadamard-rotated domain and rotate the
NEW rows instead — query (q_len*Hq rows) and key_chunk (q_len*Hk rows)
— before the FA2 concat. Scores are orthogonally equivalent: H is
symmetric orthonormal, so (q@H)·(k@H) == q·k exactly (V is never
rotated; output space unchanged).

Cost: O(q_len) GEMM replaces O(cached_len) GEMM (last chunk of a 65k
prefill at 8k chunks: ~8x fewer rows, ~2x average). NOT bit-exact vs
the v38 path (fp16 rounding order differs — same math, different
association); numerically equivalent. Validate with its own battery
AFTER v39a (which IS bit-exact and must gate on probe hash
0ce080630035 first).

Idempotent; FAILS on tree drift; compile-before-write; backup at
<file>.pre39b.
"""
import os
import py_compile
import shutil

P = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/"
     "turboquant_attn.py")
MARKER = "# llm-scaler v39b"

OLD = """\
        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
            Pi_half = layer._tq_Pi_half
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
            k_flat = k_flat @ Pi_half
            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                0, 1
            )  # (cached_len, Hk, D) — already fp16
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)
"""

NEW = """\
        # llm-scaler v39b: keep cached K in its rotated domain; rotate the
        # NEW rows (query + key_chunk) instead. Orthogonally equivalent
        # scores (H symmetric orthonormal: (q@H)·(k@H) == q·k); drops the
        # O(cached_len*Hk) inverse-rotation GEMM per layer per chunk to
        # O(q_len*(Hq+Hk)) and the full-prefix k_flat temp. NOT bit-exact
        # vs v38 (fp16 rounding order); numerically equivalent. V and the
        # output space are unchanged (V is never rotated).
        if not self.tq_config.key_fp8:
            Pi_half = layer._tq_Pi_half
            query = (query.reshape(-1, D) @ Pi_half).reshape(q_len, Hq, D)
            key_chunk = (key_chunk.reshape(-1, D) @ Pi_half).reshape(
                q_len, Hk, D
            )
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D) — stays rotated (fp16)
        else:
            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                0, 1
            )  # (cached_len, Hk, D)
"""

src = open(P).read()
if MARKER in src:
    print("V39B_QROT: already patched")
    raise SystemExit(0)
n = src.count(OLD)
if n != 1:
    print("V39B_QROT FAIL: anchor count=%d (want 1) — tree drift" % n)
    raise SystemExit(1)
src = src.replace(OLD, NEW)
tmp = P + ".v39btmp"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
bak = P + ".pre39b"
if not os.path.exists(bak):
    shutil.copy2(P, bak)
os.replace(tmp, P)
py_compile.compile(P, doraise=True)
d = os.path.dirname(P)
for f in os.listdir(d):
    if f.startswith("__pycache__") or f.endswith(".pyc"):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)
print("V39B_QROT OK: continuation rotates q+newK (O(q_len)) instead of "
      "cached K (O(cached_len)); backup at %s.pre39b" % P)
