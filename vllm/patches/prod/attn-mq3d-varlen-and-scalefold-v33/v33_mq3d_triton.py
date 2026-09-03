# llm-scaler v33: fix fp8-KV speculative-verify pathology (#15).
# Mechanism: fp8 KV -> FlashAttention backend -> C++ _vllm_fa2_C. Verify
# (q_len=k+1, max_seqlen_q>1) routes to branch1 (chunk_prefill) with NO KV
# splits: BLOCK_M=32 / nqpk=6 -> BLOCK_Q=5, q_len=2 packs into ONE q-block,
# grid (1, Hk=2/GPU after TP2) = 2 CTAs serially scan the whole context
# (249.5 ms/step @65k measured; 7.09 tok/s incl. prefill). num_splits_kv=32
# plumbed through the shim was IGNORED by branch1 (bit-identical output,
# same wall -> probe dead).
# Fix: route small-q paged fp8 batches to the in-tree Triton
# `unified_attention` 3D split path. That kernel already supports paged KV,
# per-row causal ramps (context_len + query_pos — the exact MTP verify
# ramp), per-tensor fp8 caches with scalar descales (KVQuantMode
# .FP8_PER_TENSOR -> _cast_kv_tile multiplies by tl.load(scale)), tl.dot
# XMX math, and a per-token segment reduction (reduce_segments). Its only
# limitation was the wrapper refusing 3D for max_seqlen_q>1.
# Patches: (A) wrapper 3D gate q<=8, (B) module helper + routing in
# flash_attn.py. Rollback: VLLM_XPU_TRITON_MQ3D=0.
import py_compile

TA = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
FA = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"

# --- site A: unified_attention wrapper allows 3D for small multi-query ---
ta = open(TA).read()
if "llm-scaler v33: allow the 3D split path" not in ta:
    oldA = """        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
"""
    newA = """        # llm-scaler v33: allow the 3D split path for SMALL multi-query
        # batches (MTP verify, q_len=k+1<=8). The per-row causal ramp and
        # per-token segment reduction make it mathematically identical to
        # the 2D path; 2D serializes verify-sized q through ~Hk CTAs.
        or (max_seqlen_q > 1 and max_seqlen_q > 8)
        or num_seqs > seq_threshold_3D
"""
    assert ta.count(oldA) == 1, f"anchorA={ta.count(oldA)}"
    ta = ta.replace(oldA, newA)
    open(TA, "w").write(ta)
    py_compile.compile(TA, doraise=True)

# --- site B1: routing at the verify (paged multi-query) call site ---
fa = open(FA).read()
oldB1 = """                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
"""
newB1 = """                # llm-scaler v33: paged multi-query (MTP verify) on fp8 KV
                # -> Triton 3D split path (C++ branch1 serializes: 2 CTAs
                # per GPU scan the whole context, 249.5 ms/step @65k).
                if (
                    _V33_TRITON_MQ3D
                    and isinstance(self.kv_cache_dtype, str)
                    and self.kv_cache_dtype.startswith("fp8")
                    and block_table is not None
                    and seqused_k is not None
                    and 1 < max_seqlen_q <= 8
                    and attn_metadata.causal is True
                    and sliding_window_size[0] < 0
                    and not self.logits_soft_cap
                    and q_descale is None
                ):
                    _v33_mq3d_varlen(
                        self,
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        output[:num_actual_tokens],
                        cu_seqlens_q,
                        seqused_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        block_table,
                        k_descale,
                        v_descale,
                    )
                    return output
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
"""
if "llm-scaler v33: paged multi-query (MTP verify) on fp8 KV" not in fa:
    assert fa.count(oldB1) == 1, f"anchorB1={fa.count(oldB1)}"
    fa = fa.replace(oldB1, newB1)

# --- site B2: module-level helper appended at EOF ---
helper = '''

# ---------------------------------------------------------------------------
# llm-scaler v33: Triton 3D split path for paged multi-query (MTP verify)
# on fp8 KV caches. See the routing block in FlashAttentionImpl.forward.
# Rollback: VLLM_XPU_TRITON_MQ3D=0. Segment count tunable via
# VLLM_XPU_MQ3D_SEGS (default 64).
_V33_TRITON_MQ3D = os.environ.get("VLLM_XPU_TRITON_MQ3D", "1") != "0"
_V33_MQ3D_SEGS = max(1, int(os.environ.get("VLLM_XPU_MQ3D_SEGS", "64") or 64))


def _v33_mq3d_varlen(
    impl,
    q,
    key_cache,
    value_cache,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    block_table,
    k_descale,
    v_descale,
):
    """Small-q paged fp8 verify via the Triton unified 3D split kernel."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    ntok, Hq, D = q.shape
    D_pad = 1 << (D - 1).bit_length()
    nseg = _V33_MQ3D_SEGS
    bufs = getattr(impl, "_v33_mq3d_bufs", None)
    if (
        bufs is None
        or bufs[0].shape[0] < ntok
        or bufs[0].shape[1] < Hq
        or bufs[0].shape[2] < nseg
    ):
        bufs = (
            torch.empty(
                ntok, Hq, nseg, D_pad, dtype=torch.float32, device=q.device
            ),
            torch.empty(ntok, Hq, nseg, dtype=torch.float32, device=q.device),
            torch.empty(ntok, Hq, nseg, dtype=torch.float32, device=q.device),
        )
        impl._v33_mq3d_bufs = bufs
    seg_o, seg_m, seg_e = bufs
    kv8 = (
        torch.float8_e5m2
        if impl.kv_cache_dtype == "fp8_e5m2"
        else torch.float8_e4m3fn
    )
    if k_descale is None:
        k_descale = torch.ones(1, dtype=torch.float32, device=q.device)
    if v_descale is None:
        v_descale = torch.ones(1, dtype=torch.float32, device=q.device)
    unified_attention(
        q=q,
        k=key_cache.view(kv8),
        v=value_cache.view(kv8),
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=impl.scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=4096,
        num_par_softmax_segments=nseg,
        softmax_segm_output=seg_o[:ntok],
        softmax_segm_max=seg_m[:ntok],
        softmax_segm_expsum=seg_e[:ntok],
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
'''
assert "def _v33_mq3d_varlen" not in fa
fa = fa + helper
open(FA, "w").write(fa)
py_compile.compile(FA, doraise=True)
print("V33MQ3D OK: fp8 verify -> triton unified 3D split path")
