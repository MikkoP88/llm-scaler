# llm-scaler v33: fold per-tensor fp8 KV scales into S/P tiles in the
# unified attention kernel. Mode-1 (_cast_kv_tile) multiplied EVERY KV
# element by the scale via an f32 detour (to(f32)*scale->to(f16)) — 2 x
# 8192 f32 ops per tile. s*(Q.K) == Q.(s*K) and (P*s)@V == P@(s*V), so
# applying the scalar scales on the S score tile and P probability tile
# ([16,32] = 512 elems, 16x fewer) is mathematically identical up to f32
# rounding. Only the v33 fp8-verify path uses mode 1 on this deployment
# (verified: sole unified_attention callers = flash_attn v33 helper +
# triton_attn backend, unused on XPU).
# Idempotent per site (site markers), single write at the end.
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
s = open(P).read()

# --- site 1: cast becomes a plain element cast ---
old1 = """    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)
"""
new1 = """    if KV_QUANT_MODE == 1:
        # llm-scaler v33: plain element cast — the per-tensor scale is
        # applied on the S/P tiles in the main loop (16x fewer elements;
        # identical math: s*(Q.K) == Q.(s*K), (P*s)@V == P@(s*V)).
        return data.to(Q.dtype)
    return data.to(Q.dtype)
"""
if "llm-scaler v33: plain element cast" not in s:
    assert s.count(old1) == 1, f"anchor1={s.count(old1)}"
    s = s.replace(old1, new1)

# --- site 2: hoist scalar loads before the tile loop (kernel-body depth) ---
old2 = """    context_len = seq_len - cur_batch_query_len
"""
new2 = """    context_len = seq_len - cur_batch_query_len
    # llm-scaler v33: hoist per-tensor fp8 scales (mode 1) for the
    # S/P-tile application below.
    if KV_QUANT_MODE == 1:
        k_sc_v33 = tl.load(k_scale)
        v_sc_v33 = tl.load(v_scale)
"""
if "hoist per-tensor fp8 scales" not in s:
    assert s.count(old2) == 1, f"anchor2={s.count(old2)}"
    s = s.replace(old2, new2)

# --- site 3: S tile carries k_scale ---
old3 = """        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)
"""
new3 = """        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        elif KV_QUANT_MODE == 1:
            # llm-scaler v33: per-tensor fp8 — k_scale on the score tile.
            S += (scale * k_sc_v33) * tl.dot(Q, K)
        else:
            S += scale * tl.dot(Q, K)
"""
if "k_scale on the score tile" not in s:
    assert s.count(old3) == 1, f"anchor3={s.count(old3)}"
    s = s.replace(old3, new3)

# --- site 4: P tile carries v_scale ---
old4 = """        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)
"""
new4 = """        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        elif KV_QUANT_MODE == 1:
            # llm-scaler v33: per-tensor fp8 — v_scale on the P tile.
            P_v33 = (P * v_sc_v33).to(V.dtype)
            acc += tl.dot(P_v33, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)
"""
if "v_scale on the P tile" not in s:
    assert s.count(old4) == 1, f"anchor4={s.count(old4)}"
    s = s.replace(old4, new4)

open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("V33SCALEFOLD OK: fp8 per-tensor scales on S/P tiles")
