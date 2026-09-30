#!/usr/bin/env python3
"""p29i_probe.py — v126 P29I: REAL-SCALE overflow repro at op level.

P29H v8 ring conviction: under serve traffic (qkvz_absmax ~60) the
native fp8 spec call writes NaN into the e4m3 SSM pool from the very
first wide-graph replay; op-level probes were green only because their
synthetic inputs (randn*0.5 -> absmax ~2-3) sat 20-100x below real
scale. e4m3fn max is 448; overflow maps to NaN.

This probe sweeps the input scale with an otherwise P29F-shaped call
(nsd=7 nst=5, K=V=128, ring rows per seq) over ROUNDS of fresh inputs
(compounding walk), each round comparing:
  - live e4m3 pool  (native posture: writes round-trip through fp8)
  - fp16 twin pool  (same starting values, writes stay fp16)
Per-round: out diff, NaN counts, and the per-(seq,slot) NaN map of the
ring rows — pinning WHICH write overflows first (intermediate ring-slot
spill vs final accepted checkpoint).
"""
import sys

import torch

DEV = "xpu"
K = V = 128
H = 8
HV = 24
NC = 512
NST = 5
NSD = 7
ROUNDS = 5
SCALE = float(K ** -0.5)
DIM = 2 * H * K + 2 * HV * V

SCALES = [0.5, 5.0, 15.0, 30.0, 60.0]


def main():
    print("=== P29I real-scale overflow sweep (nsd=%d nst=%d rounds=%d) ==="
          % (NSD, NST, ROUNDS))
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec as op
    torch.manual_seed(77)
    cw = (torch.randn(DIM, 4, device=DEV) * 0.1).half()
    cb = torch.zeros(DIM, device=DEV, dtype=torch.float16)
    A_log = (torch.randn(HV, device=DEV) * 0.1).half()
    dtb = (torch.randn(HV, device=DEV) * 0.1).half()

    for s in SCALES:
        torch.manual_seed(4242)
        pool16_0 = (torch.randn(NC, HV, V, K, device=DEV) * 5.0).half()
        conv_0 = (torch.randn(NC, 3, DIM, device=DEV) * 1.0).half()
        pool = pool16_0.to(torch.float8_e4m3fn).contiguous()
        conv = conv_0.clone()
        twin = pool.to(torch.float16).contiguous()   # same starting values
        conv_t = conv_0.clone()

        idx = torch.zeros(NSD, NST, dtype=torch.int32, device=DEV)
        for q in range(NSD):
            idx[q] = torch.arange(q * NST, q * NST + NST,
                                  dtype=torch.int32, device=DEV)
        tok = torch.arange(NSD * NST, dtype=torch.int32, device=DEV)
        acc = torch.full((NSD,), 4, dtype=torch.int32, device=DEV)
        out = torch.zeros(NSD * NST, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros_like(out)
        out_t = torch.zeros_like(out)
        z_t = torch.zeros_like(out)

        pmax0 = pool.to(torch.float16).abs().max().item()
        print("-- scale %.1f (qkvz absmax ~%.0f) pool start absmax=%.1f"
              % (s, s * 3.3, pmax0))
        for r in range(ROUNDS):
            torch.manual_seed(9000 + r)
            qkvz = (torch.randn(NSD * NST, DIM, device=DEV) * (s / 2)).half()
            ba = (torch.randn(NSD * NST, 2 * HV, device=DEV) * 0.5).half()
            op(qkvz, conv, cw, cb, idx, A_log, dtb, ba, pool, out, z,
               tok, acc, NSD, NST, H, HV, K, V, SCALE)
            op(qkvz, conv_t, cw, cb, idx, A_log, dtb, ba, twin, out_t, z_t,
               tok, acc, NSD, NST, H, HV, K, V, SCALE)
            torch.xpu.synchronize()
            pf = pool.to(torch.float16)
            diff = (out.float() - out_t.float()).abs()
            md = diff.max().item()
            nan_o = int(torch.isnan(out.float()).sum())
            nan_map = torch.isnan(pf[idx.to(torch.long)]).any(
                dim=-1).any(dim=-1)  # (nsd, nst) per slot
            nan_slots = nan_map.sum(dim=0).tolist()
            pmax = pf.abs().max().item()
            tmax = twin.abs().max().item()
            print("  r%d md_out=%9.4f nan_out=%6d | pool nan slots/seq=%s "
                  "| pool absmax=%8.1f twin absmax=%8.1f"
                  % (r, md, nan_o, nan_slots, pmax, tmax))
            if not torch.isfinite(torch.tensor(md)):
                break
            acc_n = [4, 2, 3, 1][r % 4]
            idx.copy_(torch.roll(idx, shifts=acc_n, dims=1))
            acc.fill_(acc_n)
        del pool, conv, twin, conv_t
    print("P29I_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
