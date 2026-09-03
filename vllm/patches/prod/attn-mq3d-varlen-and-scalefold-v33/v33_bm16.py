# llm-scaler v33: small multi-query (MTP verify, q<=8) keeps BLOCK_M=16.
# The wrapper's BLOCK_M=32 widening was tuned for 256-token canvas
# attention; for verify-sized q it pads tl.dot's M dim 32/12 ~2.7x wasted
# XMX work. 16 is the XMX minimum tile and doubles CTA count.
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
s = open(P).read()
old = """    if max_seqlen_q > 1 and num_queries_per_kv <= 16:
        BLOCK_M = 32
"""
new = """    # llm-scaler v33: small multi-query (MTP verify) keeps BLOCK_M=16 —
    # q_len<=8 rows do not need the 32-wide M tile (padded XMX waste).
    if max_seqlen_q > 8 and num_queries_per_kv <= 16:
        BLOCK_M = 32
"""
if "llm-scaler v33: small multi-query (MTP verify) keeps BLOCK_M=16" not in s:
    assert s.count(old) == 1, f"anchor={s.count(old)}"
    s = s.replace(old, new)
    open(P, "w").write(s)
    py_compile.compile(P, doraise=True)
print("V33BM16 OK")
