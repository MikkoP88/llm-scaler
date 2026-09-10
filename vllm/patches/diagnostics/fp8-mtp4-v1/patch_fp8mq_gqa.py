#!/usr/bin/env python3
"""fp8-mtp4-v3 patcher: GQA-batched stage-1 flash-decode kernel.

Diagnosis chain (fp8-mtp4-v1/REPORT.md §3.6-§3.9): after SPLITS=16 the
mtp4 step slope is still ~2 µs/KVtok; the dV/dM step-time series forces
a k-INDEPENDENT ctx-scaled floor (~0.5-0.8 µs/KVtok). The v51 stage-1
grid is (B, Hq, SPLITS) with kv_head = hid // KV_GROUP_SIZE: on this
model (24q/4kv, TP2 -> Hq=12, Hk=2, KV_GROUP_SIZE=6) every KV slice is
loaded and dequantized 6x redundantly, once per q-head program.

This patch adds `_fp8_mq_stage1_gqa` (knob VLLM_FP8MQ_GQA, default
configurable via --gqa-default): grid (B, Hk * NPROG, SPLITS) where each
program owns one kv-head and a GQA_ROWS=16 block of the
(q-head-in-group x q-position) rows — KVG*Q_LEN rows padded up to a
multiple of 16 across NPROG programs. KV tiles are loaded/dequantized
ONCE per row-block instead of once per q-head (6x -> 2x at k4: 30 rows
-> 2 blocks). Scoring and PV run on the matrix units via tl.dot (fp16
operands, fp32 accumulate, per-tensor descales folded after — same
exact algebra as the v2 dot variant). The mid-buffer layout and stage-2
combine are UNCHANGED: each mid element (b, hid, sid, qp) is still
written by exactly one program with a fixed reduction order, so the
bit-determinism contract is preserved.

Patcher protocol: anchors must match exactly once (drift protection);
py_compile gates the write; --gqa-default sets the knob default.
"""

import argparse
import pathlib
import py_compile
import sys

TARGET = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/"
    "triton_fp8_mq.py"
)

KNOB_ANCHOR = '_V51_FP8MQ_DOT = os.environ.get("VLLM_FP8MQ_DOT", "0") == "1"\n'
KNOB_NEW = (
    '_V51_FP8MQ_DOT = os.environ.get("VLLM_FP8MQ_DOT", "0") == "1"\n'
    "\n"
    "_V51_FP8MQ_GQA = os.environ.get(\"VLLM_FP8MQ_GQA\", \"{GQAD}\") == \"1\"\n"
)

GQA_KERNEL = '''

@triton.jit
def _fp8_mq_stage1_gqa(
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
    Q_LEN: tl.constexpr,  # actual query rows per seq (1 < Q_LEN <= 8)
    Q_BLOCK: tl.constexpr,  # mid row tile (layout contract with stage 2)
    IS_E5M2: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    NPROG: tl.constexpr,  # row-block programs per kv-head
    GQA_ROWS: tl.constexpr,  # rows per program (16: tl.dot minimum M)
):
    # fp8-mtp4-v3 GQA variant: each program owns one (request, kv-head,
    # kv-split) and a 16-row block of the (q-head-in-group x q-position)
    # rows. K/V tiles are loaded and dequantized once per row block —
    # not once per q-head — removing the KV_GROUP_SIZE-fold redundant
    # reads of the v51 grid (B, Hq, SPLITS). Mid layout / stage-2
    # contract unchanged; every mid element is still written by exactly
    # one program (fixed reduction order, bit-deterministic).
    bid = tl.program_id(0)  # batch index
    pid1 = tl.program_id(1)  # kv_head * NPROG + row_block
    sid = tl.program_id(2)  # kv_split index

    kv_head = pid1 // NPROG
    rowblk = pid1 % NPROG

    r_loc = tl.arange(0, GQA_ROWS)
    r_glob = rowblk * GQA_ROWS + r_loc  # row = i * Q_LEN + qp in group
    row_valid = r_glob < KV_GROUP_SIZE * Q_LEN
    i_head = r_glob // Q_LEN  # q-head index within the kv group [GQA_ROWS]
    qp = r_glob % Q_LEN  # query position [GQA_ROWS]
    hid = kv_head * KV_GROUP_SIZE + i_head  # global q-head [GQA_ROWS]

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    seq_len = tl.load(Seq_lens_ptr + bid)
    q0 = seq_len - (Q_LEN - 1)
    row_limit = q0 + qp  # [GQA_ROWS] per-row causal limits
    seq_len_max = seq_len

    k_scale = tl.load(Kscale_ptr)
    v_scale = tl.load(Vscale_ptr)

    split_len = tl.cdiv(seq_len_max, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len_max)

    # NOTE: no early return — every program must write its (possibly
    # all-empty) partial so stage 2 can combine blindly.

    # Query tile [GQA_ROWS, BLOCK_D] fp16 (padded rows -> 0). Row
    # (hid, qp) lives at query row bid * Q_LEN + qp.
    q_base = (
        (bid * Q_LEN + qp).to(tl.int64)[:, None] * stride_qb
        + hid.to(tl.int64)[:, None] * stride_qh
    )
    q_tile = tl.load(
        Q_ptr + q_base + d_offs[None, :],
        mask=row_valid[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    # Online softmax accumulators — one lane per (head, q-pos) row.
    m_prev = tl.full([GQA_ROWS], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([GQA_ROWS], dtype=tl.float32)
    acc = tl.zeros([GQA_ROWS, BLOCK_D], dtype=tl.float32)

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

        # Per-row causal mask: [GQA_ROWS, BLOCK_KV]
        att_mask = (kv_offs[None, :] < row_limit[:, None]) & kv_mask[None, :]

        # K tile [BLOCK_KV, BLOCK_D] fp16 — dequant fp8 -> fp16 (no
        # scale; the per-tensor k_scale folds into the score scalar).
        k_raw = tl.load(
            K8_ptr + k_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            k16 = k_raw.to(tl.float8e5, bitcast=True).to(tl.float16)
        else:
            k16 = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)

        # scores [GQA_ROWS, BLOCK_KV] fp32 — matrix-unit dot + fold.
        scores = tl.dot(q_tile, tl.trans(k16)) * (ATTN_SCALE * k_scale)
        scores = tl.where(att_mask, scores, -float("inf"))

        # V tile [BLOCK_KV, BLOCK_D] fp16 (v_scale folds at the store).
        v_raw = tl.load(
            V8_ptr + v_bases[:, None] + d_offs[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if IS_E5M2:
            v16 = v_raw.to(tl.float8e5, bitcast=True).to(tl.float16)
        else:
            v16 = v_raw.to(tl.float8e4nv, bitcast=True).to(tl.float16)

        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [GQA_ROWS]
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp(m_prev - n_safe)
        p = tl.exp(scores - n_safe[:, None])

        # PV on the matrix units; accumulate in fp32.
        acc = acc * re_scale[:, None]
        acc = tl.dot(p.to(tl.float16), v16, acc)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Store partials: row (hid, qp) -> mid[bid, hid, sid, qp, :].
    out_base = (
        bid * stride_mb
        + hid.to(tl.int64) * stride_mh
        + sid.to(tl.int64) * stride_ms
    )  # [GQA_ROWS]
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_ptr
        + out_base[:, None]
        + qp[:, None] * (HEAD_DIM + 1)
        + d_offs[None, :],
        (acc * v_scale) / safe_l[:, None],
        mask=row_valid[:, None] & d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)  # -inf + log(1) = -inf for empty rows
    tl.store(
        Mid_ptr + out_base + qp * (HEAD_DIM + 1) + HEAD_DIM,
        lse,
        mask=row_valid,
    )

'''

LAUNCH_ANCHOR = """    grid = (B, Hq, NUM_KV_SPLITS)
    _stage1_kernel = _fp8_mq_stage1_dot if _V51_FP8MQ_DOT else _fp8_mq_stage1
    _stage1_kernel[grid](
"""
LAUNCH_NEW = """    if _V51_FP8MQ_GQA:
        NPROG = triton.cdiv(kv_group_size * q_len, 16)
        grid = (B, Hk * NPROG, NUM_KV_SPLITS)
        _stage1_kernel = _fp8_mq_stage1_gqa
        _extra = {"NPROG": NPROG, "GQA_ROWS": 16}
    else:
        grid = (B, Hq, NUM_KV_SPLITS)
        _stage1_kernel = _fp8_mq_stage1_dot if _V51_FP8MQ_DOT else _fp8_mq_stage1
        _extra = {}
    _stage1_kernel[grid](
"""

CALL_ANCHOR = """        BLOCK_KV=_V51_FP8MQ_BLOCK_KV,
        num_warps=_V51_FP8MQ_STAGE1_WARPS,
        num_stages=_V51_FP8MQ_STAGE1_STAGES,
    )
"""
CALL_NEW = """        BLOCK_KV=_V51_FP8MQ_BLOCK_KV,
        **_extra,
        num_warps=_V51_FP8MQ_STAGE1_WARPS,
        num_stages=_V51_FP8MQ_STAGE1_STAGES,
    )
"""


def patch(path: str, gqa_default: int) -> None:
    p = pathlib.Path(path)
    src = p.read_text()
    gqad = "1" if gqa_default else "0"

    def sub_once(anchor: str, new: str, what: str) -> None:
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"PATCH-ANCHOR-FAIL {what}: found {n} occurrences")
        return src.replace(anchor, new)

    if "_fp8_mq_stage1_gqa" in src:
        print("PATCH-SKIP: gqa kernel already present")
        return

    src = sub_once(KNOB_ANCHOR, KNOB_NEW.format(GQAD=gqad), "knob")
    src = sub_once(
        "\n\ndef triton_fp8_mq_attention(",
        GQA_KERNEL + "\ndef triton_fp8_mq_attention(",
        "kernel-insert",
    )
    src = sub_once(LAUNCH_ANCHOR, LAUNCH_NEW, "launcher-select")
    src = sub_once(CALL_ANCHOR, CALL_NEW, "launcher-call")
    p.write_text(src)
    py_compile.compile(str(p), doraise=True)
    print("PATCHED triton_fp8_mq.py: +_fp8_mq_stage1_gqa "
          f"(VLLM_FP8MQ_GQA default {gqad}), NPROG launcher")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--gqa-default", type=int, choices=[0, 1], default=0)
    a = ap.parse_args()
    patch(a.target, a.gqa_default)
