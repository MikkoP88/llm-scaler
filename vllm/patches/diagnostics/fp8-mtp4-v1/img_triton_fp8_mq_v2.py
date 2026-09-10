# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""llm-scaler v51: Triton multi-query flash-decode kernel for fp8 paged KV.

Root fix for the fp8-KV + speculative-decoding slowness (v50 F6 /
B5/B5b/B5d/B5e conviction: multi-row verify on fp8 KV ran tforward
d=4976 ms / step). The conviction was made with the v33 route ACTIVE
(``_V33_TRITON_MQ3D`` routes paged multi-query fp8 to the general-purpose
``unified_attention`` 3D-segment kernel) — i.e. the general kernel IS the
slow path, not a missing route. This module is a dedicated flash-decoding
kernel for the exact shape spec verify produces:

    q      [N, Hq, D] with N == q_len * B (UNIFORM decode: every request
           contributes exactly q_len = k+1 rows, 1 < q_len <= 8)
    kv     paged fp8 pool halves [num_blocks, block_size, Hk, D]
           (fp8_e4m3fn or fp8_e5m2 bytes, per-tensor k/v scales)
    seqlens [B], block_table [B, max_blocks]

Design (lineage: harmonized patch 08 TQ MQ kernel, prod since v19):
  * stage 1 grid (B, Hq, SPLITS): each program owns one (request, q-head,
    kv-split), scans its split span in BLOCK_KV tiles, loads each K/V tile
    ONCE and scores ALL q rows against it (per-row online softmax with
    per-row causal limits limit[j] = q0 + j — v32 register-lean per-row
    static_range accumulation, temps stay [BLOCK_KV, BLOCK_D]).
  * stage 2 grid (B, Hq): blind combine across splits — empty partials
    carry lse = -inf and contribute zero weight, so no per-row split-range
    recomputation is needed on the host or in-kernel.
  * uniform split ranges derive from each program's OWN seq_len (loaded
    from the seq_lens device tensor in-kernel — no host sync, CUDA/XPU
    graph capture safe).
  * fp8 loads are uint8 -> bitcast(float8e4nv | float8e5) -> fp32, scaled
    by the per-tensor k_descale/v_descale fp32 scalars loaded from device
    memory (matches vxk FA2 descale semantics; no D2H sync).
  * Determinism: fixed split layout + fixed-order fp32 stage-2 combine —
    every output element is produced by exactly one program with a fixed
    reduction order, so run-to-run output is bit-stable for identical
    inputs (unlike the convicted nondeterministic ESIMD fp8 path, v38).

Knobs (env):
  VLLM_XPU_FP8_MQ          =0 disables the route (falls back to v33) —
                             read in flash_attn.py, not here.
  VLLM_FP8MQ_BLOCK_KV      KV tile width, default 16.
  VLLM_FP8MQ_SPLITS        kv split count, default 32.
  VLLM_FP8MQ_STAGE1_WARPS  default 1 (patch-08 lineage: 1 warp measured
                             fastest at depth on Xe2; 2/4 were slower).
  VLLM_FP8MQ_STAGE1_STAGES default 2 (fp8 tiles are 1 B/elem; prefetch
                             two tiles to hide latency).

Routing gate lives in FlashAttentionImpl._inner_forward (v51): fp8 KV +
paged + UNIFORM q (N == max_query_len * B) + 1 < q_len <= 8 + causal +
no sliding window + no softcap + head_size <= 256. Non-uniform shapes
(chunked prefill mixed batches) fall through to the v33 route / vxk FA2.
"""

import os

import torch
import triton
import triton.language as tl

# llm-scaler fp8-mtp4-v1 (2026-09-09): default retuned after live A/B on
# llm-scaler-exp:v1.2.5 (fp8-mtp4-v1/REPORT.md). Tiles 32/4/2 are the
# measured optimum (BLOCK_KV=8/1/1 was 1.6-10x WORSE at long ctx). The
# collapse driver is SPLITS-scaled (mid scratch + per-step cost):
# 64 -> ~5.8 s/step flat; 32 -> 330-460 ms/step floor; validated ladder
# value baked below. Env knobs stay live for override.
_V51_FP8MQ_BLOCK_KV = max(1, int(os.environ.get("VLLM_FP8MQ_BLOCK_KV", "32") or 32))
_V51_FP8MQ_SPLITS = max(1, int(os.environ.get("VLLM_FP8MQ_SPLITS", "16") or 16))
_V51_FP8MQ_STAGE1_WARPS = max(
    1, int(os.environ.get("VLLM_FP8MQ_STAGE1_WARPS", "4") or 4)
)
_V51_FP8MQ_STAGE1_STAGES = max(
    1, int(os.environ.get("VLLM_FP8MQ_STAGE1_STAGES", "2") or 2)
)


_V51_FP8MQ_DOT = os.environ.get("VLLM_FP8MQ_DOT", "0") == "1"


@triton.jit
def _fp8_mq_stage1(
    Q_ptr,  # [B * Q_LEN, Hq, D] query dtype (uniform rows: seq b at b*Q_LEN)
    K8_ptr,  # [num_blocks, block_size, Hk, D] uint8 (fp8 bytes)
    V8_ptr,  # [num_blocks, block_size, Hk, D] uint8 (fp8 bytes)
    BT_ptr,  # [B, max_num_blocks] int32/int64
    Seq_lens_ptr,  # [B] int32 — full seq len incl. the Q_LEN verify rows
    Kscale_ptr,  # fp32 scalar tensor (per-tensor k descale)
    Vscale_ptr,  # fp32 scalar tensor (per-tensor v descale)
    Mid_ptr,  # [B, Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] fp32 partials
    # strides
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_bt_b,
    stride_mb,
    stride_mh,
    stride_ms,
    # constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,  # actual query rows per seq (1 < Q_LEN <= Q_BLOCK)
    Q_BLOCK: tl.constexpr,  # padded power-of-2 row tile
    IS_E5M2: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    seq_len = tl.load(Seq_lens_ptr + bid)
    # Tokens visible to query row 0 (row j sits at position q0 - 1 + j and
    # attends kv < q0 + j; seq_len = q0 + Q_LEN - 1).
    q0 = seq_len - (Q_LEN - 1)
    row_limit = q0 + q_offs  # [Q_BLOCK]
    seq_len_max = seq_len

    k_scale = tl.load(Kscale_ptr)
    v_scale = tl.load(Vscale_ptr)

    split_len = tl.cdiv(seq_len_max, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len_max)

    # NOTE: no early return — every program must write its (possibly
    # all-empty) partial so stage 2 can combine blindly. Empty partials
    # are lse = -inf / acc = 0 and get zero weight.

    # Query tile: [Q_BLOCK, BLOCK_D] fp32 (padded rows -> 0).
    q_base = (bid * Q_LEN + q_offs)[:, None] * stride_qb + hid * stride_qh
    q_tile = tl.load(
        Q_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators — one lane per query row.
    m_prev = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            BT_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        k_bases = (
            block_nums * stride_kb
            + page_off.to(tl.int64) * stride_kp
            + tl.cast(kv_head, tl.int64) * stride_kh
        )
        v_bases = (
            block_nums * stride_vb
            + page_off.to(tl.int64) * stride_vp
            + tl.cast(kv_head, tl.int64) * stride_vh
        )

        # Per-row causal mask: [Q_BLOCK, BLOCK_KV]
        att_mask = (kv_offs[None, :] < row_limit[:, None]) & kv_mask[None, :]

        # K tile [BLOCK_KV, BLOCK_D] fp32 — dequant fp8 -> fp32 * k_scale.
        k_raw = tl.load(
            K8_ptr + k_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            k_f = k_raw.to(tl.float8e5, bitcast=True).to(tl.float32) * k_scale
        else:
            k_f = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * k_scale

        # Per-row scores (v32 register-lean static loop; temps stay
        # [BLOCK_KV, BLOCK_D]).
        scores = tl.zeros([Q_BLOCK, BLOCK_KV], dtype=tl.float32)
        for _r in tl.static_range(Q_LEN):
            _q_r = tl.sum(
                tl.where((q_offs == _r)[:, None], q_tile, 0.0), axis=0
            )
            _s_r = (
                tl.sum(
                    tl.where(d_mask[None, :], _q_r[None, :] * k_f, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where((q_offs == _r)[:, None], _s_r[None, :], scores)
        scores = tl.where(att_mask, scores, -float("inf"))

        # V tile [BLOCK_KV, BLOCK_D] fp32 — dequant fp8 -> fp32 * v_scale.
        v_raw = tl.load(
            V8_ptr + v_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            v_f = v_raw.to(tl.float8e5, bitcast=True).to(tl.float32) * v_scale
        else:
            v_f = v_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * v_scale

        # Online softmax update (a row with no valid token in this tile
        # keeps m = -inf -> guarded exponent below).
        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [Q_BLOCK]
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp(m_prev - n_safe)  # exp(-inf - 0) = 0 when empty
        p = tl.exp(scores - n_safe[:, None])  # exp(-inf) = 0 for masked

        # Per-row weighted value accumulation (v32 pattern).
        acc = acc * re_scale[:, None]
        for _r in tl.static_range(Q_LEN):
            _p_r = tl.sum(tl.where((q_offs == _r)[:, None], p, 0.0), axis=0)
            _a_r = tl.sum(_p_r[:, None] * v_f, axis=0)
            acc = acc + tl.where((q_offs == _r)[:, None], _a_r[None, :], 0.0)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Store partial: [Q_BLOCK, D+1]; rows >= Q_LEN hold benign finite
    # values (stage 2 only stores rows < Q_LEN to the output).
    out_base = bid * stride_mb + hid * stride_mh + sid * stride_ms
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_ptr + out_base + q_offs[:, None] * (HEAD_DIM + 1) + d_offs[None, :],
        acc / safe_l[:, None],
        mask=d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)  # -inf + log(1) = -inf for empty rows
    tl.store(Mid_ptr + out_base + q_offs * (HEAD_DIM + 1) + HEAD_DIM, lse)


@triton.jit
def _fp8_mq_stage2(
    Mid_ptr,  # [B, Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] fp32
    Out_ptr,  # [B * Q_LEN, Hq, D] query dtype
    stride_mb,
    stride_mh,
    stride_ms,
    stride_obs,
    stride_oh,
    Q_LEN: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    row_stride = HEAD_DIM + 1
    base = bid * stride_mb + hid * stride_mh

    m_run = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_D], dtype=tl.float32)

    # Blind combine: empty partials carry lse = -inf -> zero weight.
    for sid in range(NUM_KV_SPLITS):
        sbase = base + sid * stride_ms
        tlogic = tl.load(Mid_ptr + sbase + q_offs * row_stride + HEAD_DIM)
        tv = tl.load(
            Mid_ptr + sbase + q_offs[:, None] * row_stride + d_offs[None, :],
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        n_e_max = tl.maximum(m_run, tlogic)
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        old_scale = tl.exp(m_run - n_safe)
        exp_logic = tl.where(tlogic == -float("inf"), 0.0, tl.exp(tlogic - n_safe))
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        m_run = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    result = acc / safe_sum[:, None]
    out_rows = bid * Q_LEN + q_offs
    tl.store(
        Out_ptr + out_rows[:, None] * stride_obs + hid * stride_oh + d_offs[None, :],
        result,
        mask=q_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _fp8_mq_stage1_dot(
    Q_ptr,  # [B * Q_LEN, Hq, D] query dtype (uniform rows: seq b at b*Q_LEN)
    K8_ptr,  # [num_blocks, block_size, Hk, D] uint8 (fp8 bytes)
    V8_ptr,  # [num_blocks, block_size, Hk, D] uint8 (fp8 bytes)
    BT_ptr,  # [B, max_num_blocks] int32/int64
    Seq_lens_ptr,  # [B] int32 — full seq len incl. the Q_LEN verify rows
    Kscale_ptr,  # fp32 scalar tensor (per-tensor k descale)
    Vscale_ptr,  # fp32 scalar tensor (per-tensor v descale)
    Mid_ptr,  # [B, Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] fp32 partials
    # strides
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_bt_b,
    stride_mb,
    stride_mh,
    stride_ms,
    # constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,  # actual query rows per seq (1 < Q_LEN <= Q_BLOCK)
    Q_BLOCK: tl.constexpr,  # padded power-of-2 row tile (>= 16 here)
    IS_E5M2: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # fp8-mtp4-v1 DOT variant: identical split layout / per-row causal
    # limits / stage-2 contract as _fp8_mq_stage1; the per-tile QK^T and
    # PV contractions run on the matrix units via tl.dot (fp16 operands,
    # fp32 accumulate) instead of 2*Q_LEN masked-reduction vector passes.
    # Per-tensor descale scalars fold AFTER the dots (exact algebra).
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    seq_len = tl.load(Seq_lens_ptr + bid)
    q0 = seq_len - (Q_LEN - 1)
    row_limit = q0 + q_offs  # [Q_BLOCK]
    seq_len_max = seq_len

    k_scale = tl.load(Kscale_ptr)
    v_scale = tl.load(Vscale_ptr)

    split_len = tl.cdiv(seq_len_max, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len_max)

    # NOTE: no early return — every program must write its (possibly
    # all-empty) partial so stage 2 can combine blindly.

    # Query tile: [Q_BLOCK, BLOCK_D] fp16 (padded rows -> 0).
    q_base = (bid * Q_LEN + q_offs)[:, None] * stride_qb + hid * stride_qh
    q_tile = tl.load(
        Q_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    # Online softmax accumulators — one lane per query row (fp32).
    m_prev = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            BT_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        k_bases = (
            block_nums * stride_kb
            + page_off.to(tl.int64) * stride_kp
            + tl.cast(kv_head, tl.int64) * stride_kh
        )
        v_bases = (
            block_nums * stride_vb
            + page_off.to(tl.int64) * stride_vp
            + tl.cast(kv_head, tl.int64) * stride_vh
        )

        # Per-row causal mask: [Q_BLOCK, BLOCK_KV]
        att_mask = (kv_offs[None, :] < row_limit[:, None]) & kv_mask[None, :]

        # K tile [BLOCK_KV, BLOCK_D] fp16 — dequant fp8 -> fp16 (no scale;
        # the per-tensor k_scale folds into the score scalar below).
        k_raw = tl.load(
            K8_ptr + k_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            k16 = k_raw.to(tl.float8e5, bitcast=True).to(tl.float16)
        else:
            k16 = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)

        # scores [Q_BLOCK, BLOCK_KV] fp32 — matrix-unit dot + folded scales.
        scores = tl.dot(q_tile, tl.trans(k16)) * (ATTN_SCALE * k_scale)
        scores = tl.where(att_mask, scores, -float("inf"))

        # V tile [BLOCK_KV, BLOCK_D] fp16 — dequant fp8 -> fp16 (v_scale
        # folds into the final store: sum p*(v*s) == (sum p*v)*s).
        v_raw = tl.load(
            V8_ptr + v_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            v16 = v_raw.to(tl.float8e5, bitcast=True).to(tl.float16)
        else:
            v16 = v_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)

        # Online softmax update (a row with no valid token in this tile
        # keeps m = -inf -> guarded exponent below).
        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [Q_BLOCK]
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp(m_prev - n_safe)  # exp(-inf - 0) = 0 when empty
        p = tl.exp(scores - n_safe[:, None])  # exp(-inf) = 0 for masked

        # PV on the matrix units; accumulate in fp32.
        acc = acc * re_scale[:, None]
        acc = tl.dot(p.to(tl.float16), v16, acc)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Store partial: [Q_BLOCK, D+1]; rows >= Q_LEN hold benign finite
    # values (stage 2 only stores rows < Q_LEN to the output).
    out_base = bid * stride_mb + hid * stride_mh + sid * stride_ms
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_ptr + out_base + q_offs[:, None] * (HEAD_DIM + 1) + d_offs[None, :],
        (acc * v_scale) / safe_l[:, None],
        mask=d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)  # -inf + log(1) = -inf for empty rows
    tl.store(Mid_ptr + out_base + q_offs * (HEAD_DIM + 1) + HEAD_DIM, lse)


def triton_fp8_mq_attention(
    impl,
    query: torch.Tensor,  # [N, Hq, D], N == q_len * B (uniform rows)
    key_cache: torch.Tensor,  # [num_blocks, block_size, Hk, D] fp8 view
    value_cache: torch.Tensor,  # same
    output: torch.Tensor,  # [N, Hq, D] query dtype
    seq_lens: torch.Tensor,  # [B] int32
    block_table: torch.Tensor,  # [B, max_num_blocks]
    q_len: int,
    k_descale: torch.Tensor,  # fp32 scalar tensor (expanded ok)
    v_descale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Batched multi-query fp8 flash-decode attention (spec verify).

    Caller (FlashAttentionImpl._inner_forward v51 route) guarantees the
    uniform-q shape; everything below is host-int math on metadata only —
    no device->host syncs, XPU-graph-capture safe. Scratch and uint8 KV
    views are cached on the impl (pool tensors are immutable per boot);
    the grow-only scratch policy matches the v33/v34 pattern and the
    descending capture order of the cudagraph pool (largest B captured
    first, later captures reuse the buffer).
    """
    N, Hq, D = query.shape
    B = N // q_len
    Hk = key_cache.shape[2]
    block_size = key_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    Q_BLOCK = triton.next_power_of_2(max(q_len, 2))
    if _V51_FP8MQ_DOT and Q_BLOCK < 16:
        # tl.dot needs M >= 16; pad the row tile (stage 2 only stores
        # rows < Q_LEN, so the padded rows are inert).
        Q_BLOCK = 16
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = _V51_FP8MQ_SPLITS
    is_e5m2 = key_cache.dtype == torch.float8_e5m2

    # Cached uint8 views of the fp8 pool halves + ones fallback for a
    # missing descale (v34 pattern: keyed by pool tensor identity).
    st = getattr(impl, "_v51_fp8mq_state", None)
    if st is None or st[0] is not key_cache or st[1] is not value_cache:
        st = (
            key_cache,
            value_cache,
            key_cache.view(torch.uint8),
            value_cache.view(torch.uint8),
            torch.ones(1, dtype=torch.float32, device=device),
        )
        impl._v51_fp8mq_state = st
    _, _, k8, v8, ones = st
    if k_descale is None:
        k_descale = ones
    if v_descale is None:
        v_descale = ones

    mid = getattr(impl, "_v51_fp8mq_mid", None)
    if (
        mid is None
        or mid.shape[0] < B
        or mid.shape[1] < Hq
        or mid.shape[2] < NUM_KV_SPLITS
        or mid.shape[3] < Q_BLOCK
        or mid.shape[4] < D + 1
    ):
        mid = torch.empty(
            B, Hq, NUM_KV_SPLITS, Q_BLOCK, D + 1,
            dtype=torch.float32, device=device,
        )
        impl._v51_fp8mq_mid = mid
    mid = mid[:B, :Hq, :NUM_KV_SPLITS, :Q_BLOCK]

    grid = (B, Hq, NUM_KV_SPLITS)
    _stage1_kernel = _fp8_mq_stage1_dot if _V51_FP8MQ_DOT else _fp8_mq_stage1
    _stage1_kernel[grid](
        query,
        k8,
        v8,
        block_table,
        seq_lens,
        k_descale,
        v_descale,
        mid,
        query.stride(0),
        query.stride(1),
        k8.stride(0),
        k8.stride(1),
        k8.stride(2),
        v8.stride(0),
        v8.stride(1),
        v8.stride(2),
        block_table.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        Q_LEN=q_len,
        Q_BLOCK=Q_BLOCK,
        IS_E5M2=1 if is_e5m2 else 0,
        ATTN_SCALE=softmax_scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=_V51_FP8MQ_BLOCK_KV,
        num_warps=_V51_FP8MQ_STAGE1_WARPS,
        num_stages=_V51_FP8MQ_STAGE1_STAGES,
    )

    grid2 = (B, Hq)
    _fp8_mq_stage2[grid2](
        mid,
        output,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        output.stride(0),
        output.stride(1),
        Q_LEN=q_len,
        Q_BLOCK=Q_BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        HEAD_DIM=D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return output

