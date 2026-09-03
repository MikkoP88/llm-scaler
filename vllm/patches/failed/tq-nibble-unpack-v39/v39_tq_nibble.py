#!/usr/bin/env python3
"""llm-scaler v39a (TQ decode perf): nibble-split 4-bit unpack.

ROOT CAUSE (#15 follow-up): the TurboQuant kernels' 4-bit unpack does
2x the necessary byte loads plus a wasted 16-bit assembly step:

  MSE K (MSE_BITS==4) in _tq_decode_stage1 + _tq_mq_decode_stage1:
      every dim loads byte d//2 AND byte d//2+1, then assembles
      raw16 = b0 | (b1<<8) and extracts (raw16 >> shift) & 0xF.
      4-bit indices never straddle byte boundaries, so the +1 load is
      ALWAYS discarded -> half the K byte traffic is pure overhead,
      plus the or/shift ALU on a [BLOCK_KV, BLOCK_D] tile every loop
      iteration.
  V (VQB==4), same two kernels: each dim loads byte d//2; every byte
      is fetched twice (once for even d, once for odd d).
  _tq_full_dequant_kv (continuation-prefill dequant): both patterns,
      1D per (pos, head).

#15 telemetry attributed ~26ms of the ~40ms 65k-context scan to MSE
unpack + centroid-gather ALU; halving the loads and deleting the
16-bit assembly attacks that directly.

BIT-EXACTNESS (holds for all 256 byte values):
  MSE_BITS==4: even d: b & 0xF == (b0 | (b1<<8)) & 0xF
               odd  d: (b >> 4) == ((b0 | (b1<<8)) >> 4) & 0xF
               because (b1<<8)>>4 = b1<<4 contributes only bits >= 4.
  VQB==4: even d = low nibble, odd d = high nibble in both paths.
  tl.interleave(lo, hi) lays out [lo0, hi0, lo1, hi1, ...] == d =
  0,1,2,3,...  Downstream reduction trees untouched -> outputs are
  bit-identical. Empirically verified by v39_check.py (old vs new
  module from the pristine backup, all presets, on XPU).

3-bit K (k3v4_nc / 3bit_nc) keeps the straddling 16-bit path (3-bit
indices DO cross byte boundaries). KEY_FP8 (k8v4 K) never enters the
MSE branch. The VQB==4 V-side fix applies to every 4-bit-V preset.

Idempotent; FAILS on tree drift; compile-before-write; pristine
backup written next to the target as <file>.pre39.
"""
import os
import py_compile
import shutil

P = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/"
     "triton_turboquant_decode.py")
MARKER = "# llm-scaler v39:"

# ---------------------------------------------------------------------------
# Sites 1+2 (count=2): 2D MSE unpack in _tq_decode_stage1 and
# _tq_mq_decode_stage1 (byte-identical text).
# ---------------------------------------------------------------------------
OLD_2D_MSE = """\
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
"""

NEW_2D_MSE = """\
            # llm-scaler v39: MSE_BITS==4 nibble-split — one byte load per
            # two dims. The 16-bit assembly's +1 byte load is always
            # discarded when 4-bit indices never straddle byte boundaries,
            # so half the loads and the or/shift ALU were pure overhead.
            # Bit-exact: even d = low nibble, odd d = high nibble.
            # 3-bit keeps the straddling 16-bit path below.
            if MSE_BITS == 4:
                nb_offs = tl.arange(0, BLOCK_D // 2)
                nb_mask = nb_offs < MSE_BYTES
                nb_raw = tl.load(
                    KV_cache_ptr + slot_bases[:, None] + nb_offs[None, :],
                    mask=kv_mask[:, None] & nb_mask[None, :],
                    other=0,
                ).to(tl.int32)
                mse_idx = tl.interleave(nb_raw & 0xF, (nb_raw >> 4) & 0xF)
            else:
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
"""

# ---------------------------------------------------------------------------
# Sites 3+4 (count=2): 2D VQB==4 unpack in the same two kernels.
# ---------------------------------------------------------------------------
OLD_2D_V4 = """\
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)
"""

NEW_2D_V4 = """\
            # llm-scaler v39: nibble-split — one byte load per two dims
            # (each byte was fetched twice: once for even d, once for odd
            # d). Bit-exact: even d = low nibble, odd d = high nibble.
            nb_offs = tl.arange(0, BLOCK_D // 2)
            nb_vmask = nb_offs < VAL_DATA_BYTES
            nb_raw = tl.load(
                KV_cache_ptr + val_bases[:, None] + nb_offs[None, :],
                mask=kv_mask[:, None] & nb_vmask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = tl.interleave(nb_raw & 0xF, (nb_raw >> 4) & 0xF).to(
                tl.float32
            )
"""

# ---------------------------------------------------------------------------
# Site 5 (count=1): 1D MSE unpack in _tq_full_dequant_kv.
# ---------------------------------------------------------------------------
OLD_1D_MSE = """\
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
"""

NEW_1D_MSE = """\
        # llm-scaler v39: MSE_BITS==4 nibble-split (see decode-kernel
        # note): one byte load per two dims; 3-bit path unchanged.
        if MSE_BITS == 4:
            nb_offs = tl.arange(0, BLOCK_D // 2)
            nb_mask = nb_offs < MSE_BYTES
            nb_raw = tl.load(
                KV_cache_ptr + slot_base + nb_offs, mask=nb_mask, other=0
            ).to(tl.int32)
            mse_idx = tl.interleave(nb_raw & 0xF, (nb_raw >> 4) & 0xF)
        else:
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
"""

# ---------------------------------------------------------------------------
# Site 6 (count=1): 1D VQB==4 unpack in _tq_full_dequant_kv.
# ---------------------------------------------------------------------------
OLD_1D_V4 = """\
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)
"""

NEW_1D_V4 = """\
        # llm-scaler v39: nibble-split (see decode-kernel note): one byte
        # load per two dims; bit-exact (even d = low nibble).
        nb_offs = tl.arange(0, BLOCK_D // 2)
        nb_vmask = nb_offs < VAL_DATA_BYTES
        nb_raw = tl.load(
            KV_cache_ptr + val_base + nb_offs, mask=nb_vmask, other=0
        ).to(tl.int32)
        v_idx = tl.interleave(nb_raw & 0xF, (nb_raw >> 4) & 0xF).to(tl.float32)
"""

SITES = [
    ("2D MSE (stage1 + mq)", OLD_2D_MSE, NEW_2D_MSE, 2),
    ("2D V4  (stage1 + mq)", OLD_2D_V4, NEW_2D_V4, 2),
    ("1D MSE (full_dequant)", OLD_1D_MSE, NEW_1D_MSE, 1),
    ("1D V4  (full_dequant)", OLD_1D_V4, NEW_1D_V4, 1),
]

src = open(P).read()
if MARKER in src:
    print("V39_TQ_NIBBLE: already patched")
    raise SystemExit(0)

for name, old, new, want in SITES:
    n = src.count(old)
    if n != want:
        print("V39_TQ_NIBBLE FAIL: %s anchor count=%d (want %d) — tree drift"
              % (name, n, want))
        raise SystemExit(1)

for _, old, new, _ in SITES:
    src = src.replace(old, new)

tmp = P + ".v39tmp"
open(tmp, "w").write(src)
py_compile.compile(tmp, doraise=True)
bak = P + ".pre39"
if not os.path.exists(bak):
    shutil.copy2(P, bak)
os.replace(tmp, P)
py_compile.compile(P, doraise=True)
# purge stale bytecode so the next import recompiles
d = os.path.dirname(P)
for f in os.listdir(d):
    if f.startswith("__pycache__") or f.endswith(".pyc"):
        p = os.path.join(d, f)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)
print("V39_TQ_NIBBLE OK: 6 sites patched (2x 2D MSE, 2x 2D V4, 1D MSE, "
      "1D V4); pristine backup at %s.pre39" % P)
