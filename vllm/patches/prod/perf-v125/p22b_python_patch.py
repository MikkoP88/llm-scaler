#!/usr/bin/env python3
"""p22b_python_patch.py — v125 P22B: VLLM_XPU_GDN_FP8_NATIVE switch in
vllm/_xpu_ops.py (run INSIDE lsv-test via docker exec; docker cp first).

When VLLM_XPU_GDN_FP8_NATIVE=1 and the SSM pool is fp8, the v124 P19.5a
gather/remap/scatter bridge is bypassed: self.kv_cache[1] goes STRAIGHT to
the (now fp8-native) SYCL op with the original index tensors. Unset/0 keeps
the bridge byte-identical — back-compat with the shipped v1.2.24 wheel.

Implementation detail: the single semantic change is extending the
`_fp8_ssm` gate with `and not _gdn_fp8_native_enabled()`. Every existing
`if _fp8_ssm` branch then naturally takes the native pass-through path
(ssm_state=self.kv_cache[1], original indices, no scatter) — no other call
site edits. The mode is frozen at first call (pre-capture; capture replays
must never observe a mode change) and its engagement is logged once, loud,
for the boot gate to grep. An old wheel under =1 fails loudly inside the
kernel's own dtype TORCH_CHECK — never a silent fallback. Idempotent.
"""
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
src = open(P).read()

if "v125 P22B" in src:
    print("ALREADY_APPLIED")
    sys.exit(0)

# --- edit 1: module-level resolver, inserted just above the impl def ---
DEF = "def _gdn_attention_core_xpu_impl("
assert DEF in src, "impl def not found"
resolver = '''
# llm-scaler v125 P22B: fp8-native SYCL gdn_attention switch. When the
# installed vllm-xpu-kernels wheel accepts fp8 ssm_state (DISPATCH_STATE_DTYPE
# fp8 branches), VLLM_XPU_GDN_FP8_NATIVE=1 passes the fp8 pool STRAIGHT to
# the op — the v124 P19.5a gather/remap/scatter bridge is skipped entirely.
# Frozen at first call (pre-capture; capture replays must never see a mode
# change). An old wheel under =1 aborts loudly inside the kernel's dtype
# TORCH_CHECK — never a silent fallback.
_GDN_FP8_NATIVE_MODE = None  # None = unresolved, then 0/1 (v125 P22B)


def _gdn_fp8_native_enabled():
    global _GDN_FP8_NATIVE_MODE
    if _GDN_FP8_NATIVE_MODE is None:
        _GDN_FP8_NATIVE_MODE = (
            1 if os.environ.get("VLLM_XPU_GDN_FP8_NATIVE", "") == "1" else 0
        )
    return _GDN_FP8_NATIVE_MODE == 1


'''
src = src.replace(DEF, resolver + DEF, 1)

# `import os` must be present at module level
head = src.split(DEF, 1)[0]
if "\nimport os\n" not in head and "import os," not in head:
    first_import = head.index("\nimport ")
    src = src.replace("\nimport ", "\nimport os\nimport ", 1)

# --- edit 2: the gate (single semantic change) ---
OLD = """    ssm_pool = self.kv_cache[1]
    _fp8_ssm = ssm_pool.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)"""
NEW = """    ssm_pool = self.kv_cache[1]
    # v125 P22B: fp8-native mode bypasses the bridge below — the fp8 pool
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
        _fp8_ssm = _pool_fp8"""
assert OLD in src, "fp8 gate anchor not found"
src = src.replace(OLD, NEW, 1)

# --- post-conditions ---
assert src.count("v125 P22B") >= 2
assert "_gdn_fp8_native_enabled()" in src
assert 'VLLM_XPU_GDN_FP8_NATIVE' in src
compile(src, P, "exec")

open(P, "w").write(src)
print("P22B_PYTHON_PATCHED")
