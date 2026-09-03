#!/usr/bin/env python3
"""llm-scaler v38 (#18 FIX): keep fp8-KV decode OFF the ESIMD fast path.

ROOT CAUSE (v38, 2026-09-03, fully isolated):
  The PAGED_ATTN_ESIMD_INSERTED_v1 gate routed fp16-Q/XPU-graph/head-256/
  GQA>=2 decoder decode to eagle_ops.page_attn_decode (compiled ESIMD
  kernel). That kernel is RUN-TO-RUN NONDETERMINISTIC on fp8 KV:
    v38quant.py (fixed inputs, 999 calls, bit-compare):
      fp8@511  eager: 969/998 calls differ from call-0 (same 235-elem
                      signature — MULTISTABLE discrete outputs)
      fp8@512  eager: 132/998; graph: 998/998 (toggles 2 modes/replay)
      fp8@1024 eager: 988/998; fp8@2048: 150/998; fp8@117: ~1/50
      fp16@512 eager:   5/998 single-element (latent, much rarer)
    magnitudes 1-2 fp16 ULP — invisible on fat-margin tokens, flips
    knife-edge argmax tokens -> the #18 "bimodality" (coherent alternate
    generations). Engine proof: same request x6 batch=1 cache-hit gave 4
    distinct outputs WITH the gate active; 10/10 identical + f8ref x5 +
    700-token gens x4 through the 512/1024 boundary hot zone all
    bit-identical with the gate disabled.

  The reroute target (vxk FA2 varlen kernel, 3cab97a + #357 fp8-KV
  optimize) is deterministic (50/50 all split configs), tail-garbage
  invariant, fp32-reference correct to 1.4e-5 — AND FASTER on this
  stack: 65k warm 28.26 vs 23.99 tok/s (+18%; 4bit prod 22.28), 2k warm
  33.5-33.6 vs 23.99 (+40%, 4bit parity 33.49).

PATCH: extend the ESIMD gate to exclude fp8* kv_cache_dtype lanes.
VLLM_XPU_ALLOW_ESIMD_F8=1 restores the old route for A/B. fp16/auto and
4bit TQ lanes are untouched (4bit uses the TQ backend regardless).

Idempotent; FAILS on tree drift; compile-before-write.
"""
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"
MARKER = "# llm-scaler v38: fp8 decode must not use the ESIMD kernel"

OLD = """                    and query.dtype == torch.float16
                    and os.environ.get(
                        "VLLM_XPU_ENABLE_XPU_GRAPH", "0") not in ("", "0")
                ):"""

NEW = """                    # llm-scaler v38: fp8 decode must not use the ESIMD kernel
                    # (#18 root cause: eagle_ops.page_attn_decode is
                    # run-to-run nondeterministic on fp8 KV — see
                    # v38_esimd_reroute.py header. vxk FA2 is deterministic,
                    # correct, and faster. A/B force-back:
                    # VLLM_XPU_ALLOW_ESIMD_F8=1.)
                    and not (
                        self.kv_cache_dtype.startswith("fp8")
                        and os.environ.get(
                            "VLLM_XPU_ALLOW_ESIMD_F8", "0") != "1")
                    and query.dtype == torch.float16
                    and os.environ.get(
                        "VLLM_XPU_ENABLE_XPU_GRAPH", "0") not in ("", "0")
                ):"""

src = open(P).read()
if MARKER in src:
    print("V38_ESIMD: already patched")
    raise SystemExit(0)
n = src.count(OLD)
if n != 1:
    print("V38_ESIMD FAIL: anchor count=%d (tree drift)" % n)
    raise SystemExit(1)
src = src.replace(OLD, NEW)
tmp = P + ".v38tmp"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
import os
os.replace(tmp, P)
py_compile.compile(P, doraise=True)
print("V38_ESIMD OK: fp8 decode rerouted to vxk FA2 (env override "
      "VLLM_XPU_ALLOW_ESIMD_F8=1)")
