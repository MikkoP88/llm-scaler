#!/usr/bin/env python3
"""p22b2_python_patch.py — v125 P22B2: decode-only fp8-native routing.

WHY: the first P22B switch (VLLM_XPU_GDN_FP8_NATIVE=1 -> _fp8_ssm=False
globally) passed the fp8 pool to the SYCL gdn_attention op for EVERY call.
The op's host splits internally: decode-length sequences hit the v125
fp8-native decode kernel (works), but prefill-length sequences route to
xe_2/chunk_gated_delta_rule_kernels_xe2.hpp whose DISPATCH_STATE_DTYPE
still rejects fp8 -> leg e4m3n/e5m2n died at EngineCore-init profile
prefill ("ssm_state dtype must be float32/float16/bfloat16").

The xe2 C++ fix was attempted and BLOCKED: cute's reorder machinery
(cute/algorithm/reorder.hpp upcast_subbyte_t) eagerly instantiates
cutlass::platform::numeric_limits<c10::Float8_e4m3fn/e5m2>, which has no
specialization -> 12 hard build errors. Decision: split on the python
side instead — the impl's single op call already receives
attn_metadata.num_prefills, which is the exact discriminator the SYCL
host uses to route prefill vs decode.

THE FIX: native fp8 ONLY when the batch contains zero prefill sequences
(num_prefills == 0 — pure decode/spec batches, the CC-fleet steady state
under spec MTP x4). Any batch with prefill tokens takes the certified
fp16 bridge (gather -> fp16 ssm_run -> scatter), whose decode sequences
run the original fp16 decode kernel — correct, just not native for that
step. Mixed prefill+decode coalesced batches therefore always bridge.

Idempotent (marker 'v125 P22B2'). Applies ON TOP of p22b_python_patch.py
(the original P22B gate block is replaced wholesale).
"""
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
src = open(P, encoding="utf-8").read()

if "v125 P22B2" in src:
    print("ALREADY_APPLIED")
    sys.exit(0)

OLD = '''    # v125 P22B: fp8-native mode bypasses the bridge below — the fp8 pool
    # goes straight to the fp8-native SYCL kernel with the original index
    # tensors. One-time loud evidence line for the boot gate.
    _pool_fp8 = ssm_pool.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if _pool_fp8 and _gdn_fp8_native_enabled():
        if not globals().get("_GDN_FP8_NATIVE_LOGGED"):
            globals()["_GDN_FP8_NATIVE_LOGGED"] = True
            print(
                "v125 P22B FP8_NATIVE ENGAGED ssm_dtype=%s" % ssm_pool.dtype,
                flush=True,
            )
        _fp8_ssm = False
    else:
        _fp8_ssm = _pool_fp8
'''

NEW = '''    # v125 P22B2: native fp8 for DECODE-ONLY batches. Inside the single
    # gdn_attention op, prefill-length sequences route to the xe2 chunk
    # kernel (xe_2/chunk_gated_delta_rule_kernels_xe2.hpp) whose cute
    # reorder machinery cannot host c10 fp8 StateT (cutlass
    # numeric_limits<c10::Float8_*> undefined -> 12 build errors), so any
    # batch containing prefill tokens must take the certified fp16 bridge
    # below. Decode/spec-only batches — the steady state under spec MTP
    # x4 — pass the pool straight to the fp8-native decode kernel.
    _pool_fp8 = ssm_pool.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    _native_ok = (
        _pool_fp8
        and _gdn_fp8_native_enabled()
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )
    if _native_ok:
        if not globals().get("_GDN_FP8_NATIVE_LOGGED"):
            globals()["_GDN_FP8_NATIVE_LOGGED"] = True
            print(
                "v125 P22B FP8_NATIVE ENGAGED ssm_dtype=%s decode_only=1"
                " (P22B2: prefill batches take the fp16 bridge)" % ssm_pool.dtype,
                flush=True,
            )
        _fp8_ssm = False
    else:
        _fp8_ssm = _pool_fp8
'''

assert src.count(OLD) == 1, "P22B gate block not found verbatim: %d" % src.count(OLD)
src = src.replace(OLD, NEW, 1)

# post-conditions
assert "v125 P22B2" in src
assert src.count("attn_metadata.num_prefills == 0") == 1
assert 'P22B FP8_NATIVE ENGAGED' in src  # boot-gate grep string survives
# the bridge + op call below the gate must be untouched
assert "_fp8_ssm = _pool_fp8" in src
assert "ssm_state=ssm_run if _fp8_ssm else self.kv_cache[1]" in src

compile(src, P, "exec")  # hard syntax gate before any write
open(P, "w", encoding="utf-8").write(src)
print("P22B2_PYTHON_PATCHED")
