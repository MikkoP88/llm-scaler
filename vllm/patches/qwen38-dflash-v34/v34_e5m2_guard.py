# llm-scaler v34: allow fp8_e5m2 KV cache with fp8 checkpoints on XPU
# (KNOWN_ISSUES #19). Upstream attention.py raises unconditionally; e5m2
# KV is exercised elsewhere in the stack (kernel dtypes, triton kv8 view)
# so the guard is the only blocker. Env-gated opt-in:
#   VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1
# CAVEAT under test: the checkpoint's k_scale/v_scale were calibrated for
# e4m3 KV; reusing them for e5m2 storage shifts quantization error. This
# patch is a probe lane, not a validated numerics lane.
import py_compile

AT = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py"
at = open(AT).read()
if "llm-scaler v34: e5m2-with-fp8-ckpt opt-in" not in at:
    old = """        if layer.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
"""
    new = """        # llm-scaler v34: e5m2-with-fp8-ckpt opt-in (KNOWN_ISSUES #19).
        # Upstream rejects unconditionally; XPU stack handles e5m2 KV
        # dtypes end-to-end, so allow it behind an explicit env for the
        # probe lane. Scales remain the checkpoint's e4m3-calibrated ones.
        import os as _os
        if layer.kv_cache_dtype == "fp8_e5m2" and _os.environ.get(
            "VLLM_XPU_ALLOW_E5M2_FP8_CKPT", "0"
        ) != "1":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
"""
    assert at.count(old) == 1, f"anchor={at.count(old)}"
    at = at.replace(old, new)
    open(AT, "w").write(at)
    py_compile.compile(AT, doraise=True)
    print("V34E5M2 OK: guard env-gated (VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1 to allow)")
else:
    print("V34E5M2: already applied")
