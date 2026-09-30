#!/usr/bin/env python3
"""p29i_probe2.py — v126 P29I run-2: WHICH real-scale input kills the pool?

Run-1 verdict: pure-randn qkvz up to absmax ~198 with ba fixed at
randn*0.5 stays clean — magnitude alone does not convict. The un-swept
suspect is ba: the SSM state increment is dt*beta*B with
dt = softplus(ba_dt + dt_bias), so increment ~ ba_absmax^2 * |B|.
Run-1's fixed ba (absmax ~1.7 -> dt ~1.9) gave increments ~90, under
e4m3's 448. Serve-real ba (same order as qkvz_absmax 60-67) implies
increments ~1e4: far over 448 (e4m3 -> NaN), inside 65504 (fp16 OK),
inside 57344 (e5m2 OK).

Sweeps (each vs an fp16 twin on identical starting values):
  A) ba scale [0.5, 2, 5, 15, 30, 60]  — qkvz at serve scale, pool
     start randn*5: pins the NaN threshold per pool dtype.
  B) pool start scale [5, 50, 200, 400] — ba at modest 2: is the
     prefill-accumulated state magnitude alone fatal?
  C) heavy-tail outliers: ba = randn*0.5 with 0.05% of entries at
     +-U(200, 2000), qkvz serve scale: real-activation-outlier class.
Reported per round: increment-driving stats (ba/dt absmax), md_out vs
twin, ring-row NaN count, live pool absmax, twin absmax.
"""
import sys

import torch
import torch.nn.functional as F

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

BA_SCALES = [0.5, 2.0, 5.0, 15.0, 30.0, 60.0]
POOL_SCALES = [5.0, 50.0, 200.0, 400.0]
FDTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def run_leg(op, cw, cb, A_log, dtb, fd, qkvz_gen, ba_gen, pool_gen, tag):
    torch.manual_seed(4242)
    pool16_0 = pool_gen().half()
    conv_0 = (torch.randn(NC, 3, DIM, device=DEV) * 1.0).half()
    pool = pool16_0.to(fd).contiguous()
    conv = conv_0.clone()
    twin = pool.to(torch.float16).contiguous()
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

    for r in range(ROUNDS):
        torch.manual_seed(9000 + r)
        qkvz = qkvz_gen().half()
        ba = ba_gen().half()
        dt_est = F.softplus(ba[:, HV:].float() + dtb.float())
        op(qkvz, conv, cw, cb, idx, A_log, dtb, ba, pool, out, z,
           tok, acc, NSD, NST, H, HV, K, V, SCALE)
        op(qkvz, conv_t, cw, cb, idx, A_log, dtb, ba, twin, out_t, z_t,
           tok, acc, NSD, NST, H, HV, K, V, SCALE)
        torch.xpu.synchronize()
        pf = pool.to(torch.float16)
        md = (out.float() - out_t.float()).abs().max().item()
        nan_rows = int(torch.isnan(pf[idx.to(torch.long)]).any(
            dim=-1).any(dim=-1).any(dim=-1).sum())
        nan_o = int(torch.isnan(out.float()).sum())
        pmax = pf.abs().max().item()
        tmax = twin.abs().max().item()
        print("  %s r%d ba_max=%7.1f dt_max=%7.2f md_out=%10.4f "
              "nan_out=%6d ringnan=%3d/%d poolmax=%9.1f twinmax=%9.1f"
              % (tag, r, ba.abs().max().item(), dt_est.max().item(),
                 md, nan_o, nan_rows, NSD * NST, pmax, tmax))
        if md != md:  # NaN diff — pool poisoned, further rounds moot
            break
        acc_n = [4, 2, 3, 1][r % 4]
        idx.copy_(torch.roll(idx, shifts=acc_n, dims=1))
        acc.fill_(acc_n)
    del pool, conv, twin, conv_t


def main():
    print("=== P29I run-2: ba/pool/outlier sweeps, e4m3 + e5m2 vs fp16 "
          "twin (nsd=%d nst=%d) ===" % (NSD, NST))
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec as op
    torch.manual_seed(77)
    cw = (torch.randn(DIM, 4, device=DEV) * 0.1).half()
    cb = torch.zeros(DIM, device=DEV, dtype=torch.float16)
    A_log = (torch.randn(HV, device=DEV) * 0.1).half()
    dtb = (torch.randn(HV, device=DEV) * 0.1).half()

    qkvz_serve = (lambda: torch.randn(NSD * NST, DIM, device=DEV) * 15.0)

    print("-- sweep A: ba scale (qkvz serve-scale, pool start randn*5)")
    for s in BA_SCALES:
        for fd in FDTYPES:
            run_leg(op, cw, cb, A_log, dtb, fd,
                    qkvz_serve,
                    lambda s=s: torch.randn(NSD * NST, 2 * HV, device=DEV) * s,
                    lambda: torch.randn(NC, HV, V, K, device=DEV) * 5.0,
                    "A ba=%4.1f %s" % (s, str(fd).split(".")[-1]))

    print("-- sweep B: pool start scale (ba modest 2.0, qkvz serve-scale)")
    for s in POOL_SCALES:
        for fd in FDTYPES:
            run_leg(op, cw, cb, A_log, dtb, fd,
                    qkvz_serve,
                    lambda: torch.randn(NSD * NST, 2 * HV, device=DEV) * 2.0,
                    lambda s=s: torch.randn(NC, HV, V, K, device=DEV) * s,
                    "B pool=%5.1f %s" % (s, str(fd).split(".")[-1]))

    print("-- sweep C: heavy-tail ba outliers (0.05% at +-U(200,2000))")
    def ba_outlier():
        ba = torch.randn(NSD * NST, 2 * HV, device=DEV) * 0.5
        m = torch.rand_like(ba) < 0.0005
        ba[m] = (torch.rand_like(ba[m]) * 1800 + 200) * \
            torch.where(torch.rand_like(ba[m]) < 0.5, -1.0, 1.0)
        return ba
    for fd in FDTYPES:
        run_leg(op, cw, cb, A_log, dtb, fd, qkvz_serve, ba_outlier,
                lambda: torch.randn(NC, HV, V, K, device=DEV) * 5.0,
                "C outl %s" % str(fd).split(".")[-1])
    print("P29I2_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
