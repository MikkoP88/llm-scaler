#!/usr/bin/env python3
"""patch_v125_c6_staticraise.py — v125 P22C6: retire the static fp8 bridge
with a LOUD raise.

Conviction (Runs 7k/7l, PHASES.md): the P22C5 static fp8 spec bridge
(gather -> fp16 copy -> arange remap -> kernel -> scatter-home) corrupts
multi-request spec batches — slot-stable 3/8 streams bad in eager AND
captured modes, dup=0, correct first token then word-salad explosion —
behaving like cross-request state crosstalk that identical states mask.
The exact mechanism stayed design-analysis-opaque after full kernel
(gated_delta_rule.hpp) and call-site (_xpu_ops.py) verification, so the
bridge is RETIRED, not fixed.

Reachability at the moment it would engage (=1, decode-only fp8, ESIMD
declined): n>1 spec verify — the convicted path — or a pure-decode batch
of >128 tokens, impossible under max-num-seqs 64. n=1 spec never arrives
(ESIMD takes num_spec_decodes==1). So the bridge has NO validated regime
at ANY n: every would-be engagement raises.

The validated fp8 postures (Run 7m, P22C5_N2VAL ALL_CLEAN both formats,
every n>1 probe family):
  spec fp8  -> VLLM_XPU_GDN_FP8_NATIVE=2 + P22B wheel (pool-straight)
  ns fp8    -> =1 (ESIMD decode + v124 unique-bridge prefill)

Requires the 'v125 P22C5' marker (patches the C5-patched file). No new
backup: restore convention stays `cp _xpu_ops.py.pre_v125_c5 _xpu_ops.py`
(the certified pre-C5 state). The downstream static gather/remap/scatter
code inserted by C5 becomes unreachable by construction; the P25 bake
variant may strip it for cleanliness — this patch keeps the edit surface
minimal.

Run inside the container AFTER patch_v125_s4fp8_bridge.py:
  docker cp patch_v125_c6_staticraise.py lsv-test:/root/
  docker exec lsv-test python3 /root/patch_v125_c6_staticraise.py
"""
import py_compile
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "v125 P22C6"
C5_MARKER = "v125 P22C5"

src = open(P).read()

if MARKER in src:
    print("V125_C6_ALREADY")
    sys.exit(0)

if C5_MARKER not in src:
    print("V125_C6_ABORT_NO_C5")
    sys.exit(1)

old = """    # v125 P22C5: decode-only fp8 batches without the P22B wheel take the
    # STATIC bridge (capture-safe). Pre-empts the unique bridge — torch.unique
    # has a value-dependent output shape and must never be captured.
    _fp8_static = (
        _fp8_ssm
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )
    if _fp8_static:
        _fp8_ssm = False
        if not globals().get("_GDN_FP8_STATIC_LOGGED"):
            globals()["_GDN_FP8_STATIC_LOGGED"] = True
            print(
                "v125 P22C5 STATIC_FP8_BRIDGE ssm_dtype=%s"
                " (decode-only fp8 via static fp16 copy; pool-straight needs"
                " the P22B wheel + =2)" % ssm_pool.dtype,
                flush=True,
            )
"""
new = """    # v125 P22C5/P22C6: decode-only fp8 batches without the P22B wheel.
    # P22C6 RETIRES the static bridge: it corrupts multi-request spec
    # batches (Run 7k/7l — slot-stable 3/8 streams bad, eager AND
    # captured, dup=0; cross-request state crosstalk that identical
    # states mask; the n=1 "clean" evidence never touched it because
    # ESIMD takes num_spec_decodes==1). Validated postures: spec fp8 =
    # =2 + P22B wheel (pool-straight, ALL_CLEAN every n, both formats —
    # Run 7m); ns fp8 = =1 (ESIMD decode + unique-bridge prefill). Any
    # would-be engagement raises LOUDLY — never silent corruption. The
    # downstream static gather/remap/scatter code is unreachable by
    # construction.
    _fp8_static = (
        _fp8_ssm
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )
    if _fp8_static:
        raise RuntimeError(
            "v125 P22C6: static fp8 decode bridge RETIRED — it corrupts "
            "multi-request spec batches (v125 Run 7k/7l). fp8 spec "
            "requires VLLM_XPU_GDN_FP8_NATIVE=2 with the P22B wheel "
            "(fp8 DISPATCH_STATE_DTYPE branches in _xpu_C.abi3.so); "
            "ns fp8 keeps =1 (ESIMD decode + unique-bridge prefill)."
        )
"""

n = src.count(old)
assert n == 1, f"anchor not unique (count={n})"
src = src.replace(old, new, 1)

with open(P, "w") as f:
    f.write(src)

py_compile.compile(P, doraise=True)
print("V125_C6_OK")
