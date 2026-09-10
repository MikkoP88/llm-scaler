#!/usr/bin/env python3
"""patch_fp8mq_dot.py — add a tl.dot (matrix-unit) stage-1 variant to the
v51 fp8mq kernel, behind env knob VLLM_FP8MQ_DOT (default off).

Rationale (fp8-mtp4-v1 diag4/diag5, 2026-09-09, host 2x Arc Pro B70):
the shipped stage-1 scores and accumulates Q_LEN rows via
``tl.static_range`` masked-reduction loops — 2*Q_LEN full
[BLOCK_KV, BLOCK_D] fp32 vector passes per tile (10 at MTP k=4), plus a
per-row extraction reduction — on the VECTOR ALUs, with no tensor
contraction. That formulation is the convicted long-ctx residual
(~2.2 us/KV-token/step, ctx-linear, present at every SPLITS value and
in the TQ-kernel lane too). The dot variant keeps the split layout,
per-row causal limits, blind stage-2 combine and fixed-order
determinism contract IDENTICAL, but runs QK^T and PV as fp16 tl.dot
with fp32 accumulate (2 dots per tile), folding the per-tensor descale
scalars AFTER the dots ((q·(k*s)) == (q·k)*s for scalar s).

Requires Q_BLOCK >= 16 (tl.dot M minimum); the launcher pads the row
tile when the knob is on (stage 2 only stores rows < Q_LEN, so padding
is inert). fp16 numerics: q(bf16)->fp16 lossless for normals; fp8->fp16
exact; p <= 1; dots accumulate in fp32. Softmax state stays fp32.

This patcher fails loudly if any anchor is not found exactly once
(protects against a drifted base image).
"""
import argparse
import py_compile
import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_fp8_mq.py"
)

KNOB_SRC = '''_V51_FP8MQ_DOT = os.environ.get("VLLM_FP8MQ_DOT", "{DOTD}") == "1"


'''

DOT_KERNEL = '''@triton.jit
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


'''

LAUNCHER_BUMP = '''    Q_BLOCK = triton.next_power_of_2(max(q_len, 2))
    if _V51_FP8MQ_DOT and Q_BLOCK < 16:
        # tl.dot needs M >= 16; pad the row tile (stage 2 only stores
        # rows < Q_LEN, so the padded rows are inert).
        Q_BLOCK = 16
'''

LAUNCHER_CALL_OLD = """    grid = (B, Hq, NUM_KV_SPLITS)
    _fp8_mq_stage1[grid]("""
LAUNCHER_CALL_NEW = """    grid = (B, Hq, NUM_KV_SPLITS)
    _stage1_kernel = _fp8_mq_stage1_dot if _V51_FP8MQ_DOT else _fp8_mq_stage1
    _stage1_kernel[grid]("""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(DEFAULT_TARGET), type=Path)
    ap.add_argument("--dot-default", choices=["0", "1"], default="0")
    args = ap.parse_args()

    path = args.target
    src = path.read_text()
    orig = src

    def replace_once(text, old, new, what):
        n = text.count(old)
        if n != 1:
            print(f"FAIL: expected exactly 1 occurrence of {what!r}, "
                  f"found {n} — base image drifted?")
            return None
        return text.replace(old, new, 1)

    # 1) module knob, before the first @triton.jit kernel
    src = replace_once(
        src,
        "@triton.jit\ndef _fp8_mq_stage1(",
        KNOB_SRC.format(DOTD=args.dot_default) + "@triton.jit\ndef _fp8_mq_stage1(",
        "stage1 kernel def anchor",
    )
    if src is None:
        return 1

    # 2) dot kernel body, before the launcher
    src = replace_once(
        src,
        "def triton_fp8_mq_attention(",
        DOT_KERNEL + "def triton_fp8_mq_attention(",
        "launcher def anchor",
    )
    if src is None:
        return 1

    # 3) launcher Q_BLOCK pad
    src = replace_once(
        src,
        "    Q_BLOCK = triton.next_power_of_2(max(q_len, 2))\n",
        LAUNCHER_BUMP,
        "Q_BLOCK assignment",
    )
    if src is None:
        return 1

    # 4) launcher call-site dispatch
    src = replace_once(src, LAUNCHER_CALL_OLD, LAUNCHER_CALL_NEW,
                       "stage1 launch site")
    if src is None:
        return 1

    if src == orig:
        print("NOTE: no changes needed (dot patch already applied?)")
        return 0

    path.write_text(src)
    py_compile.compile(str(path), doraise=True)

    print("PATCHED: _V51_FP8MQ_DOT knob (default "
          f"{args.dot_default}), _fp8_mq_stage1_dot kernel, launcher dispatch")
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
