#!/usr/bin/env python3
"""p29d_traj.py — v126 P29D diagnosis: e4m3 lane quality collapse (68/80
wrong, 0/30 tools, degeneration after token 1, serial/solo included).

P28-green ran ALL fp8 decode through the dual bridge (kernel always saw
fp16 pools; fp8 was only storage). P29c is the first serve exposure of the
NATIVE fp8 ride (poolpath E2b + P29A register chain + v131 snapshot). The
op-level gates passed but they are single-round, synthetic-magnitude and —
critically — solo-vs-batch bitwise (both sides run the SAME code, so a
checkpoint-store defect is invisible to them). Real serve compounds state
round over round at real value ranges.

This probe reproduces the SERVE dataflow: many consecutive verify rounds,
each feeding its checkpointed pool state into the next round, with REAL
model GDN parameters (A_log / dt_bias / conv1d from the fp8 checkpoint —
real decay rates and magnitudes) and realistic inputs.

Configs (identical inputs each round):
  REF      fp16 pool, native op                (v1.2.25 production class)
  NAT8     fp8_e4m3 pool, native op            (P29c failing class)
  BRIDGE8  fp8_e4m3 storage, but each round the kernel sees a dequantized
           fp16 clone and stores are re-quantized after (P28-green class)

Readout per round: out cosine vs REF (+ pool-row cos NAT8 vs BRIDGE8).
If BRIDGE8 tracks REF while NAT8 diverges -> the native fp8 execution
(chain or fp8 store/load primitives at real ranges) is convicted; if both
track -> single-layer kernel numerics are exonerated at real dynamics and
the hunt moves to the serve integration.
"""
import json
import glob
import sys

import torch
from safetensors import safe_open

DEV = "xpu"
K = V = 128
H = 8
HV = 24
NC = 512          # pool slots
NST = 5           # tokens per verify round (MTP x4)
ROUNDS = 30
SCALE = float(K ** -0.5)
DIM = 2 * H * K + 2 * HV * V     # 8192: q|k|v|z


def load_real_params():
    """Real layer-0 GDN params from the fp8 checkpoint (value-range
    realism; exact TP slicing is irrelevant for trajectory realism)."""
    pref = "model.language_model.layers.0.linear_attn."
    shard = "/models/target/layers-0.safetensors"
    try:
        out = {}
        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            for label, key in (("A_log", pref + "A_log"),
                               ("dt_bias", pref + "dt_bias"),
                               ("conv_w", pref + "conv1d.weight")):
                if key not in keys:
                    print("missing:", key)
                    return None
                out[label] = f.get_tensor(key)
        out["conv_b"] = torch.zeros(out["conv_w"].shape[0])
        return out
    except Exception as e:  # noqa: BLE001
        print("load_real_params failed:", e)
        return None


def build_real(rp, seed):
    """Real-magnitude kernel params, matching p29fix_verify shapes."""
    torch.manual_seed(seed)
    A_log = rp["A_log"].to(torch.float32).flatten()
    dt_bias = rp["dt_bias"].to(torch.float32).flatten()
    hv_ofs = (A_log.numel() - HV) // 2     # middle slice: real range either way
    A_log16 = A_log[hv_ofs:hv_ofs + HV].half().to(DEV)
    dtb16 = dt_bias[hv_ofs:hv_ofs + HV].half().to(DEV)
    cw = rp["conv_w"].to(torch.float32).flatten(1)
    if cw.shape[0] < DIM:
        raise RuntimeError("conv weight rows %d < DIM %d" % (cw.shape[0], DIM))
    row_ofs = (cw.shape[0] - DIM) // 2
    cw16 = cw[row_ofs:row_ofs + DIM].half().to(DEV)
    cb = rp["conv_b"].to(torch.float32).flatten()
    cb16 = cb[row_ofs:row_ofs + DIM].half().to(DEV)
    print("real params: A_log[min,max]=%.4f,%.4f dt_bias[min,max]=%.4f,%.4f"
          % (A_log.min(), A_log.max(), dt_bias.min(), dt_bias.max()))
    print("             conv_w[std]=%.4f qkvz_scale=0.5 ba_scale=0.5"
          % cw16.float().std().item())
    return cw16, cb16, A_log16, dtb16


def run_spec(op, cw, cb, A_log, dt_bias, dt, pool, conv, qkvz, ba,
             rows_idx, accepted):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    op(qkvz, conv, cw, cb, idx, A_log, dt_bias, ba, pool,
       out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out


def cos(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    return (torch.dot(af, bf) /
            (af.norm() * bf.norm() + 1e-30)).item()


def trajectory(op, cw, cb, A_log, dtb, scale, seed, verbose):
    """One multi-round trajectory at the given input/pool scale.
    Returns (final_nat8_cos, final_bridge8_cos, worst_poolcos, out_absmax)."""
    torch.manual_seed(seed)
    pool16_0 = (torch.randn(NC, HV, V, K, dtype=torch.float16,
                            device=DEV) * 0.05 * scale)
    conv0 = (torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01)
    rounds_inputs = []
    for _r in range(ROUNDS):
        qkvz = (torch.randn(NST, DIM, dtype=torch.float16, device=DEV)
                * 0.5 * scale)
        ba = (torch.randn(NST, 2 * HV, dtype=torch.float16, device=DEV)
              * 0.5 * scale)
        rounds_inputs.append((qkvz, ba))
    acc_cycle = [4, 3, 2, 1, 2, 3, 4, 5]     # every init_col over time
    slots = list(range(40, 40 + NST))        # consecutive per-seq ring rows

    pools = {
        "REF": (pool16_0.clone(), "native_fp16"),
        "NAT8": (pool16_0.to(torch.float8_e4m3fn).contiguous(), "native_fp8"),
        "BRIDGE8": (pool16_0.to(torch.float8_e4m3fn).contiguous(), "bridge"),
    }
    convs = {k: conv0.clone() for k in pools}
    out_absmax = 0.0
    final = {}
    worst_poolcos = 1.0

    for r in range(ROUNDS):
        acc = [acc_cycle[r % len(acc_cycle)]]
        qkvz, ba = rounds_inputs[r]
        outs = {}
        for name, (pool, mode) in pools.items():
            if mode == "bridge":
                rows = torch.tensor(slots, device=DEV)
                pool_h = pool.to(torch.float16).contiguous()
                out = run_spec(op, cw, cb, A_log, dtb, torch.float16,
                               pool_h, convs[name], qkvz, ba, slots, acc)
                pool.view(torch.uint8).index_copy_(
                    0, rows,
                    pool_h[rows].to(pool.dtype).contiguous()
                    .view(torch.uint8))
            else:
                out = run_spec(op, cw, cb, A_log, dtb, pool.dtype, pool,
                               convs[name], qkvz, ba, slots, acc)
            outs[name] = out
            if name == "REF":
                out_absmax = max(out_absmax, out.abs().max().item())
        poolcos = cos(pools["NAT8"][0].float().flatten(),
                      pools["BRIDGE8"][0].float().flatten())
        worst_poolcos = min(worst_poolcos, poolcos)
        for name in ("NAT8", "BRIDGE8"):
            final[name] = cos(outs[name], outs["REF"])
        if verbose and (r % 10 == 0 or r >= ROUNDS - 2):
            print("  r=%02d NAT8=%.5f BRIDGE8=%.5f poolcos=%.6f outabs=%.1f"
                  % (r, final["NAT8"], final["BRIDGE8"], poolcos,
                     out.abs().max().item()))
    return final["NAT8"], final["BRIDGE8"], worst_poolcos, out_absmax


def main():
    print("=== P29D trajectory diagnosis (real params, %d rounds) ===" % ROUNDS)
    from custom_esimd_kernels_vllm import (
        esimd_gdn_conv_fused_seq_spec as op,
    )
    rp = load_real_params()
    if rp is None:
        print("REAL PARAMS NOT FOUND — falling back to synthetic ranges")
        torch.manual_seed(4)
        rp = {"A_log": torch.randn(48) * 0.4,
              "dt_bias": torch.randn(48) * 0.4,
              "conv_w": torch.randn(8192, 4) * 0.05,
              "conv_b": torch.zeros(8192)}
    cw, cb, A_log, dtb = build_real(rp, 1234)

    verdicts = []
    for scale in (0.5, 1.0, 2.0, 6.0):
        print("\n--- scale=%.1f (pool/qkvz/ba scaled; saturation probe) ---"
              % scale)
        n8, b8, wpc, oam = trajectory(op, cw, cb, A_log, dtb, scale,
                                      20260929, verbose=True)
        print("  FINAL scale=%.1f NAT8=%.6f BRIDGE8=%.6f worst_poolcos=%.6f"
              " out_absmax=%.1f" % (scale, n8, b8, wpc, oam))
        verdicts.append((scale, n8, b8, wpc))

    print("\n=== P29D summary ===")
    for scale, n8, b8, wpc in verdicts:
        print("scale=%.1f NAT8=%.6f BRIDGE8=%.6f poolcos=%.6f"
              % (scale, n8, b8, wpc))
    n8_bad = any(n8 < 0.99 for _, n8, _, _ in verdicts)
    b8_ok = all(b8 > 0.99 for _, _, b8, _ in verdicts)
    if b8_ok and n8_bad:
        print("P29D_VERDICT: NATIVE_FP8_CONVICTED (bridge tracks, native diverges)")
    elif all(n8 > 0.99 for _, n8, _, _ in verdicts):
        print("P29D_VERDICT: KERNEL_EXONERATED (both track across ranges)")
    else:
        print("P29D_VERDICT: MIXED — inspect per-scale values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
