# llm-scaler v34 (#18): pin the XPU FA2 KV split count via
# VLLM_XPU_FA2_PIN_SPLITS=N so the C++ reduction tree is batch-state
# invariant (fp8-nospec greedy bimodality fix candidate).
# Why not upstream VLLM_BATCH_INVARIANT=1: it also reroutes the attention
# selector (selector.py:153 mamba/GDN batch-invariance raise / backend
# switch) which crash-loops this stack during model init; and its
# num_splits=1 pin would recreate the v33 #15 serial-scan pathology at
# long ctx (2 CTAs scanning 65k KV). Fixed N=64 gives a deterministic
# split tree AND parallel scan (v33 NSEG sweep: 64 = optimal).
# num_splits semantics in the C++ decode branch: 0 = dynamic heuristic
# (batch-dependent = the bimodality source), N>0 = exact N splits.
import py_compile

FA = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"
fa = open(FA).read()
if "llm-scaler v34: pin FA2 KV split count" not in fa:

    old = """        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1
"""
    new = """        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        # llm-scaler v34: pin FA2 KV split count (#18 bimodality fix).
        # VLLM_XPU_FA2_PIN_SPLITS=N (N>=1) forces an exact, batch-state
        # invariant split count without touching backend selection.
        import os as _os34
        try:
            _v34_pin_splits = int(_os34.environ.get("VLLM_XPU_FA2_PIN_SPLITS", "0"))
        except ValueError:
            _v34_pin_splits = 0
        if _v34_pin_splits >= 1:
            max_num_splits = _v34_pin_splits
"""
    assert fa.count(old) == 1, f"anchor={fa.count(old)}"
    fa = fa.replace(old, new)

    open(FA, "w").write(fa)
    py_compile.compile(FA, doraise=True)
    print("V34F8BI OK: FA2 KV split pin (VLLM_XPU_FA2_PIN_SPLITS)")
else:
    print("V34F8BI: already applied")
