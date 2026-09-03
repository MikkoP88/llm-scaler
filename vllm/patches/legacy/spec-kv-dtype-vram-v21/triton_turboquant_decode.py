# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
import os
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

# Stage-1 launch tuning knobs (env-overridable for A/B testing without
# rebuilding). BLOCK_KV = KV tokens per inner-loop tile: the kernel was
# designed for 16 (see the _tq_decode_stage1 constexpr doc) but shipped
# with a conservative 4; larger tiles amortize the dependent
# block-table -> slot address -> key -> score -> value load chain that
# dominates at deep context with many KV splits.
#
# Measured on 2x Arc Pro B70 (TP=2, qwen3.8-27b fp8, 117k-token deep
# decode, steady tok/s vs defaults 4/1): BLOCK_KV=16 -> -72% deep,
# BLOCK_KV=32 -> -83% deep, 16+2 warps -> -20% deep, and 32+2 warps is
# FATAL (EngineDeadError on first request, GPU engine resets). The wide
# tiles spill registers / starve latency hiding on Xe2 at num_warps=1,
# and adaptive KV splits multiply the effect at deep context. The
# shipped defaults (4 / 1 warp) are optimal on this hardware; do not
# raise BLOCK_KV past 8 without re-validating deep-context stability.
_TQ_BLOCK_KV = max(1, int(os.environ.get("VLLM_TQ_BLOCK_KV", "4") or 4))
_TQ_STAGE1_WARPS = max(1, int(os.environ.get("VLLM_TQ_STAGE1_WARPS", "1") or 1))
_TQ_STAGE1_STAGES = max(1, int(os.environ.get("VLLM_TQ_STAGE1_STAGES", "1") or 1))

_FP8_E4B15: dict[int, int] = {}


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0.
    On non-CUDA platforms (e.g. XPU), always returns 0 (use e4nv format).
    """
    if device not in _FP8_E4B15:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, :], q_rot[None, :] * k_float, 0.0),
                    axis=1,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, :], q_rot[None, :] * c_vals, 0.0),
                axis=1,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms * term1 * ATTN_SCALE
            scores = tl.where(kv_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    # Store partial result
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _stream_capturing() -> bool:
    try:
        return torch.accelerator.is_current_stream_capturing()
    except Exception:
        return False


# Shared grow-only scratch for callers that pass neither pre-allocated
# buffers nor a buf_holder (e.g. the TQ continuation fast path). Layers
# execute sequentially, so one buffer per (device, name) suffices; this
# avoids pumping multi-MB mid_o tensors through the XPU caching allocator
# on every call. Never used while a graph is being captured.
_SCRATCH: dict = {}


def _get_scratch(
    device: torch.device, name: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    if _stream_capturing():
        return torch.empty(shape, dtype=dtype, device=device)
    dev = device.index if device.index is not None else 0
    store = _SCRATCH.setdefault(dev, {})
    if name == "output":
        store = store.setdefault("output_by_dtype", {})
        name = str(dtype)
    buf = store.get(name)
    if (
        buf is None
        or buf.dtype != dtype
        or buf.shape[0] < shape[0]
        or buf.shape[1:] != tuple(shape[1:])
    ):
        buf = torch.empty(shape, dtype=dtype, device=device)
        store[name] = buf
    return buf[: shape[0]]


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # PIECEWISE-graph fallback: the workspace manager is locked at capture
    # time, so the caller passes mid_o/output/lse bufs as None. Reuse the
    # scratch cached on the layer from the first call instead of
    # re-allocating it every step — the buffers were stored on buf_holder
    # but never read back, pumping per-layer scratch (~MBs/step at B=1)
    # through the XPU caching allocator, which wedges the xe engines under
    # XPU graphs when free memory is low.
    if mid_o_buf is None and buf_holder is not None:
        mid_o_buf = getattr(buf_holder, "_tq_mid_o_buf", None)
    if output_buf is None and buf_holder is not None:
        output_buf = getattr(buf_holder, "_tq_output_buf", None)
    if lse_buf is None and buf_holder is not None:
        lse_buf = getattr(buf_holder, "_tq_lse_buf", None)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    if key_fp8:
        q_rot = query.contiguous()
    else:
        if PiT is None:
            PiT = Pi.T.contiguous()
        # Cache the cast + GEMM temporaries on the layer (grow-only) so the
        # MSE path is allocation-free after the first call.
        n_flat = B * Hq
        q_float_buf = (
            getattr(buf_holder, "_tq_q_float_buf", None)
            if buf_holder is not None
            else _get_scratch(device, "q_float", (n_flat, D), torch.float32)
        )
        if q_float_buf is None or q_float_buf.shape[0] < n_flat:
            q_float_buf = torch.empty(n_flat, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_q_float_buf = q_float_buf
        q_rot_buf = (
            getattr(buf_holder, "_tq_q_rot_buf", None)
            if buf_holder is not None
            else _get_scratch(device, "q_rot", (n_flat, D), torch.float32)
        )
        if q_rot_buf is None or q_rot_buf.shape[0] < n_flat:
            q_rot_buf = torch.empty(n_flat, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_q_rot_buf = q_rot_buf
        q_float = q_float_buf[:n_flat]
        q_float.copy_(query.reshape(n_flat, D))
        q_rot = torch.mm(q_float, PiT, out=q_rot_buf[:n_flat]).view(B, Hq, D)

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    elif buf_holder is not None:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        buf_holder._tq_mid_o_buf = mid_o
    else:
        mid_o = _get_scratch(
            device, "mid_o", (B, Hq, NUM_KV_SPLITS, D + 1), torch.float32
        )

    # Stage 1: split-KV tiled attention scoring + value accumulation
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_KV = _TQ_BLOCK_KV
    grid = (B, Hq, NUM_KV_SPLITS)
    _tq_decode_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        num_warps=_TQ_STAGE1_WARPS,
        num_stages=_TQ_STAGE1_STAGES,
    )

    # Stage 2: Reduce across KV splits
    # Output in query dtype — eliminates float16_copy kernel after stage2
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    elif buf_holder is not None:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        buf_holder._tq_output_buf = output
    else:
        output = _get_scratch(device, "output", (B, Hq, D), out_dtype)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    elif buf_holder is not None:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        buf_holder._tq_lse_buf = lse
    else:
        lse = _get_scratch(device, "lse", (B, Hq), torch.float32)

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    return output  # already in query dtype


# ---------------------------------------------------------------------------
# llm-scaler v19: multi-query (speculative-verify) decode variant.
#
# A dflash/dspark verify step runs q_len = k+1 (<= 8) query tokens per
# request against the compressed KV cache. The single-query kernel above
# forced the backend into the "synthetic decode" trick: q_len separate
# kernel calls, each re-scanning the ENTIRE compressed context -> ~q_len x
# TQ KV bandwidth per verify step, which is why TQ spec decoding was
# ~5x slower than nospec at depth. The kernels below load each KV tile
# ONCE and score all Q_BLOCK query rows against it (flash-decoding style,
# per-row online softmax with per-row causal limits limit[j] = q0 + j).
#
# Split convention: uniform split ranges derived from the row-0 limit + Q_LEN
# (== the request's true seq_len). Stage 2 combines blindly across splits:
# empty per-row partials are written as lse = -inf / acc = 0 and contribute
# exp(-inf - M) = 0 weight, so no per-row split-range recomputation is
# needed (the single-query _fwd_kernel_stage2 recomputes ranges from each
# row's seq_len, which would NOT match this layout when
# cdiv(seq_len_j, SPLITS) != cdiv(seq_len_max, SPLITS)).
# ---------------------------------------------------------------------------

# num_warps=1 measured fastest at depth (cached=16384, q_len=5): MSE keys
# 1.39x / FP8 keys 2.66x vs the v18 synthetic-decode path; warps=2/4 are
# slower (0.88x/0.49x MSE) - matches the single-query kernel's lineage
# (wide tiles + more warps spill registers on Xe2).
_TQ_MQ_STAGE1_WARPS = max(1, int(os.environ.get("VLLM_TQ_MQ_STAGE1_WARPS", "1") or 1))
_TQ_MQ_STAGE1_STAGES = max(1, int(os.environ.get("VLLM_TQ_MQ_STAGE1_STAGES", "1") or 1))


@triton.jit
def _tq_mq_decode_stage1(
    Q_rot_ptr,  # [Q_BLOCK(padded rows), Hq, D] float32 (rows >= Q_LEN masked)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    Block_table_ptr,  # [B, max_num_blocks] int32
    Q0_lens_ptr,  # [B] int32: tokens visible to query row 0 (cached_len + 1)
    Centroids_ptr,  # [n_centroids] float32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] float32
    # Strides
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,  # actual query rows (1 < Q_LEN <= Q_BLOCK)
    Q_BLOCK: tl.constexpr,  # padded power-of-2 row tile
    # TQ layout constants
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    # Score constants
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    NON_CAUSAL: tl.constexpr = 0,
):
    bid = tl.program_id(0)  # batch index
    hid = tl.program_id(1)  # q_head index
    sid = tl.program_id(2)  # kv_split index

    kv_head = hid // KV_GROUP_SIZE

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN

    q0 = tl.load(Q0_lens_ptr + bid)
    if NON_CAUSAL:
        # llm-scaler v21: DFlash draft step — the caller passes the FULL
        # seq_len in Q0_lens; every query row attends the whole stored
        # context (the chunk's K/V were pre-stored by the drafter's
        # precompute, so there is no causal ramp to apply).
        seq_len_max = q0
    else:
        # Per-row causal limits: row j sees q0 + j tokens.
        row_limit = q0 + q_offs  # [Q_BLOCK]
        # Uniform split ranges over the full seq_len (row-0 limit + Q_LEN - 1).
        seq_len_max = q0 + (Q_LEN - 1)

    split_len = tl.cdiv(seq_len_max, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len_max)

    # NOTE: no early return — every program must write its (possibly
    # all-empty) partial so stage 2 can combine blindly. An empty partial
    # is lse = -inf / acc = 0 and gets zero weight.

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query tile: [Q_BLOCK, BLOCK_D] float32 (padded rows -> 0)
    q_base = (bid * Q_BLOCK + q_offs)[:, None] * stride_qb + hid * stride_qh
    q_rot = tl.load(
        Q_rot_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators — one lane per query row
    m_prev = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration, all query
    # rows share the same KV tile loads.
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # Per-row causal mask: [Q_BLOCK, BLOCK_KV]
        if NON_CAUSAL:
            # Full-context: every valid query row attends every valid token
            # (padded rows get -inf scores and are dropped by stage 2).
            att_mask = q_mask[:, None] & kv_mask[None, :]
        else:
            att_mask = (kv_offs[None, :] < row_limit[:, None]) & kv_mask[None, :]

        # ============================================================
        # COMPUTE ATTENTION SCORES: [Q_BLOCK, BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(d_mask[None, None, :], q_rot[:, None, :] * k_float[None, :, :], 0.0),
                    axis=2,
                )
                * ATTN_SCALE
            )
            scores = tl.where(att_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(d_mask[None, None, :], q_rot[:, None, :] * c_vals[None, :, :], 0.0),
                axis=2,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms[None, :] * term1 * ATTN_SCALE
            scores = tl.where(att_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (per-row; a row with no valid token in
        # this tile keeps m = -inf -> guarded exponent below)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [Q_BLOCK]
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp(m_prev - n_safe)  # exp(-inf - 0) = 0 when empty
        p = tl.exp(scores - n_safe[:, None])  # exp(-inf) = 0 for masked

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D] (shared)
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION (per-row)
        # ============================================================
        acc = acc * re_scale[:, None] + tl.sum(p[:, :, None] * values[None, :, :], axis=1)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Store partial result: [Q_BLOCK, D+1] block (rows padded -> benign
    # finite values; stage 2 only stores rows < Q_LEN to the output).
    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_o_ptr + out_base + q_offs[:, None] * (HEAD_DIM + 1) + d_offs[None, :],
        acc / safe_l[:, None],
        mask=d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)  # -inf + log(1) = -inf for empty rows
    tl.store(Mid_o_ptr + out_base + q_offs * (HEAD_DIM + 1) + HEAD_DIM, lse)


@triton.jit
def _tq_mq_fwd_stage2(
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] float32
    Out_ptr,  # [B * Q_BLOCK, Hq, D] query dtype
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_obs,
    stride_oh,
    Q_LEN: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < Lv

    row_stride = Lv + 1
    base = bid * stride_mid_b + hid * stride_mid_h

    m_run = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_DV], dtype=tl.float32)

    # Blind combine: empty partials carry lse = -inf -> zero weight.
    for sid in range(NUM_KV_SPLITS):
        sbase = base + sid * stride_mid_s
        tlogic = tl.load(Mid_o_ptr + sbase + q_offs * row_stride + Lv)
        tv = tl.load(
            Mid_o_ptr + sbase + q_offs[:, None] * row_stride + d_offs[None, :],
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
    out_rows = bid * Q_BLOCK + q_offs
    tl.store(
        Out_ptr + out_rows[:, None] * stride_obs + hid * stride_oh + d_offs[None, :],
        result,
        mask=q_mask[:, None] & d_mask[None, :],
    )


def triton_turboquant_mq_decode_attention(
    query: torch.Tensor,  # [Q_LEN, Hq, D] — original query (B == 1)
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [1, max_num_blocks] int32
    q0_seq_lens: torch.Tensor,  # [1] int32 — cached_len + 1 (row-0 limit)
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    max_num_kv_splits: int = 32,
    non_causal: bool = False,
) -> torch.Tensor:
    """Multi-query TQ decode attention for speculative verify steps.

    One pass over the compressed KV per (head, split): all Q_LEN query
    rows are scored against each shared KV tile (flash-decoding style),
    replacing Q_LEN full-context rescans of the synthetic-decode path.
    Returns: output tensor [Q_LEN, Hq, D] in query's dtype.

    llm-scaler v21: non_causal=True serves DFlash draft steps — q0_seq_lens
    then carries the FULL seq_len and every query row attends the whole
    stored context (no causal ramp), matching the synthetic-decode
    non-causal branch semantics in turboquant_attn.py.
    """
    Q_LEN, Hq, D = query.shape
    assert Q_LEN <= 8, "MQ verify kernel is sized for q_len <= 8"
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)
    Q_BLOCK = triton.next_power_of_2(max(Q_LEN, 2))
    NUM_KV_SPLITS = max_num_kv_splits

    # q_rot = q @ Pi.T — same GEMM/buffer pattern as the single-query
    # launcher, grown to Q_LEN * Hq rows (padded rows are never loaded).
    if key_fp8:
        q_rot = query.contiguous()
    else:
        if PiT is None:
            PiT = Pi.T.contiguous()
        n_flat = Q_LEN * Hq
        q_float_buf = _get_scratch(device, "mq_q_float", (n_flat, D), torch.float32)
        q_rot_buf = _get_scratch(device, "mq_q_rot", (n_flat, D), torch.float32)
        q_float = q_float_buf[:n_flat]
        q_float.copy_(query.reshape(n_flat, D))
        q_rot = torch.mm(q_float, PiT, out=q_rot_buf[:n_flat]).view(Q_LEN, Hq, D)

    mid_o = _get_scratch(
        device, "mq_mid_o", (1, Hq, NUM_KV_SPLITS, Q_BLOCK, D + 1), torch.float32
    )

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    grid = (1, Hq, NUM_KV_SPLITS)
    _tq_mq_decode_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        q0_seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        Q_LEN=Q_LEN,
        Q_BLOCK=Q_BLOCK,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_KV=_TQ_BLOCK_KV,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        NON_CAUSAL=1 if non_causal else 0,
        num_warps=_TQ_MQ_STAGE1_WARPS,
        num_stages=_TQ_MQ_STAGE1_STAGES,
    )

    output = _get_scratch(
        device, "output", (Q_BLOCK, Hq, D), query.dtype
    )  # dtype-keyed scratch shared with the single-query path

    grid2 = (1, Hq)
    _tq_mq_fwd_stage2[grid2](
        mid_o,
        output,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        Q_LEN=Q_LEN,
        Q_BLOCK=Q_BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output[:Q_LEN]  # already in query dtype
