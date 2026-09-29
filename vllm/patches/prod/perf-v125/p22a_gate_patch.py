#!/usr/bin/env python3
"""p22a_gate_patch.py — v125 P22A: admit fp8 SSM pools into the ESIMD
fused conv+GDN decode path (gdn_linear_attn.py).

ORDERING CONSTRAINT (hard): this gate extension is ONLY valid together
with the P22A fp8-StateT ESIMD .so (custom_esimd_kernels_lgrf). With the
OLD .so, an fp8 pool reaching esimd_gdn_conv_fused* is reinterpreted as
fp16 bytes -> NaN state cascade. Apply this patch ONLY as part of the
swap step that also installs the rebuilt .so (p22a swap script order:
.so first, op-level fp8 smoke test, then this patch, then serve legs).

What changes: the lazy one-time `_gdn_conv_state_fp16_ok` check admits
kv_cache[1] (SSM pool) in fp8_e4m3fn/fp8_e5m2 when
VLLM_XPU_GDN_FP8_NATIVE=1 (same switch as the SYCL surface, frozen
helper _gdn_fp8_native_enabled in vllm/_xpu_ops.py). conv_state
(kv_cache[0]) must stay fp16 — the conv kernel reads it as fp16 in
every variant. Everything else in the path passes the pool tensor
straight through to the op, whose .sycl host now dispatches on dtype.

Idempotent (marker 'v125 P22A'). Hard syntax gate before write.
"""
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py"
src = open(P, encoding="utf-8").read()

if "v125 P22A" in src:
    print("ALREADY_APPLIED")
    sys.exit(0)

OLD = '''        if not self._gdn_conv_state_fp16_ok:
            if (
                self.kv_cache[0].dtype == torch.float16
                and self.kv_cache[1].dtype == torch.float16
            ):
                self._gdn_conv_state_fp16_ok = True
            else:
                return False
'''

NEW = '''        if not self._gdn_conv_state_fp16_ok:
            # v125 P22A: with the fp8-StateT ESIMD kernels installed, an
            # fp8 SSM pool is native-capable under VLLM_XPU_GDN_FP8_NATIVE=1
            # (same switch as the SYCL surface); conv_state stays fp16 in
            # every variant. NEVER admit fp8 here without the matching .so.
            from vllm._xpu_ops import _gdn_fp8_native_enabled

            _ssm_pool_ok = self.kv_cache[1].dtype == torch.float16 or (
                self.kv_cache[1].dtype
                in (torch.float8_e4m3fn, torch.float8_e5m2)
                and _gdn_fp8_native_enabled()
            )
            if (
                self.kv_cache[0].dtype == torch.float16
                and _ssm_pool_ok
            ):
                self._gdn_conv_state_fp16_ok = True
                if self.kv_cache[1].dtype != torch.float16:
                    print(
                        "v125 P22A ESIMD_FP8_ELIGIBLE ssm_dtype=%s"
                        % self.kv_cache[1].dtype,
                        flush=True,
                    )
            else:
                return False
'''

assert src.count(OLD) == 1, "gate block not found verbatim: %d" % src.count(OLD)
src = src.replace(OLD, NEW, 1)

assert "v125 P22A" in src
# NEW references the helper twice: the import line + the call site.
assert src.count("_gdn_fp8_native_enabled") == 2
assert "torch.float8_e4m3fn" in src and "torch.float8_e5m2" in src

compile(src, P, "exec")
open(P, "w", encoding="utf-8").write(src)
print("P22A_GATE_PATCHED")
