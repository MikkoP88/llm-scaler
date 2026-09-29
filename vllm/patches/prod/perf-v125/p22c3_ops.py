#!/usr/bin/env python3
"""p22c3_ops.py (v125 P22-C C3) — op-level fp16 vs fp8 ESIMD kernel wall
benchmark. WHY: the fp8 superiority gate needs per-op speed evidence, not
only end-to-end serve numbers: if the fp8 pool path is SLOWER at op
level (dequant/requant overhead not amortized), the serve legs' parity
is luck and the vectorization question (kernel-side fp8 math vs
convert-on-load) stays open. Serve must be DOWN for clean timing.

Ops and signatures verified against production call sites
(gdn_linear_attn.py ~1063 spec / ~1100 non-spec, esimd_utils.py:221):
  A. esimd_gdn_conv_fused_seq(qkvz, conv_state, cw, cb, state_idx,
     A_log16, dt_bias, ba, ssm_state, state_idx, out, z, n_dec,
     H, HV, K, V, scale)
  B. esimd_gdn_conv_fused_seq_spec(qkvz, conv_state, cw, cb,
     spec_state_indices(S,nst), A_log16, dt_bias, ba, ssm_state, out,
     z, token_indx, num_accepted(S), nsd, nst, H, HV, K, V, scale)
State-slot fidelity: EVERY slot index is DISTINCT (spec rows carry nst
candidate slots each = production traffic, not nst identical reads);
pool dtype is the single variable (fp16 / e4m3 / e5m2); A_log/dt_bias/
conv weights stay fp16 exactly as the call site feeds them.

Sweep: rows/seqs in 1,2,4,8,16,32,64. Timing: sync + perf_counter,
100 warmup / 300 timed; report us/call, us/row, ~GB/s (pool r+w +
qkvz/ba in + out/z out + conv read).

PASS (exit 0): every fp8 cell <= 1.10x its fp16 counterpart (noise
margin; the gate needs no REGRESSION — wins are reported, not required).
"""
import sys
import time

import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512           # pool slots (>= 64*5 so the biggest spec cell fits)
WARM, ITERS = 100, 300
DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]


def bench(fn, warm=WARM, iters=ITERS):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us/call


def params(seed):
    """Shared fp16 params exactly as the call site feeds them."""
    torch.manual_seed(seed)
    cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
    cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
    A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
    dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
    conv = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
    pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05
    return cw, cb, A_log16, dt_bias, conv, pool16


def make_inputs(rows, seed):
    torch.manual_seed(seed)
    qkvz = torch.randn(rows, DIM, dtype=torch.float16, device=DEV) * 0.1
    ba = torch.randn(rows, 2 * HV, dtype=torch.float16, device=DEV) * 0.5
    return qkvz, ba


def cell_seq(dt):
    """Non-spec decode op cell."""
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq
    cw, cb, A_log16, dt_bias, conv, pool16 = params(1234)
    pool = pool16.clone() if dt == torch.float16 else pool16.to(dt)
    scale = float(K ** -0.5)

    def run(rows):
        qkvz, ba = make_inputs(rows, rows)
        out = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
        idx = torch.arange(rows, dtype=torch.int32, device=DEV)
        p = pool[:rows]
        c = conv[:rows]
        fn = (lambda: esimd_gdn_conv_fused_seq(
            qkvz, c, cw, cb, idx, A_log16, dt_bias, ba, p, idx,
            out, z, rows, H, HV, K, V, scale))
        us = bench(fn)
        # bytes: qkvz+ba in, out+z out, conv read, pool state read+write
        b = (qkvz.numel() * 2 + ba.numel() * 2 + out.numel() * 2
             + z.numel() * 2 + c.numel() * 2
             + p.numel() * 2 * p.element_size())
        return us, b
    return run


def cell_spec(dt):
    """Spec verify op cell (nsd=S seqs, nst=5 tokens each)."""
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec
    cw, cb, A_log16, dt_bias, conv, pool16 = params(4321)
    pool = pool16.clone() if dt == torch.float16 else pool16.to(dt)
    nsd, nst = 1, 5
    rows = nsd * nst
    scale = float(K ** -0.5)

    def run(seqs):
        n = seqs * rows
        qkvz, ba = make_inputs(n, seqs)
        # every slot distinct: row i carries candidate slots i*5..i*5+4
        ssi = torch.arange(n, dtype=torch.int32, device=DEV).view(seqs, rows)
        tix = torch.arange(n, dtype=torch.int32, device=DEV)
        num_acc = torch.full((seqs,), 4, dtype=torch.int32, device=DEV)
        out = torch.zeros(n, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros(n, HV, V, dtype=torch.float16, device=DEV)
        p = pool[:n]
        c = conv[:n]
        fn = (lambda: esimd_gdn_conv_fused_seq_spec(
            qkvz, c, cw, cb, ssi, A_log16, dt_bias, ba, p,
            out, z, tix, num_acc, seqs, nst, H, HV, K, V, scale))
        us = bench(fn)
        b = (qkvz.numel() * 2 + ba.numel() * 2 + out.numel() * 2
             + z.numel() * 2 + c.numel() * 2
             + p.numel() * 2 * p.element_size())
        return us, b
    return run


def main():
    print("loading custom_esimd_kernels_vllm ...")
    import custom_esimd_kernels_vllm  # noqa: F401
    print("module import OK; pool slots=%d DIM=%d" % (NC, DIM))

    fails = 0
    for label, builder, sweep in (("seq", cell_seq, (1, 2, 4, 8, 16, 32, 64)),
                                  ("spec", cell_spec, (1, 2, 4, 8, 16, 32, 64))):
        base = {}
        for name, dt in DTYPES:
            run = builder(dt)
            for n in sweep:
                us, b = run(n)
                gbs = b / (us * 1e-6) / 1e9
                print(f"[{label} {name}] n={n:3d} {us:9.2f} us "
                      f"{us / n:8.2f} us/row ~{gbs:7.1f} GB/s")
                if name == "fp16":
                    base[n] = us
                else:
                    r = us / base[n]
                    verdict = "OK" if r <= 1.10 else "REGRESS"
                    if r > 1.10:
                        fails += 1
                    print(f"[{label} {name}] n={n:3d} vs fp16 ratio={r:.3f} "
                          f"[{verdict}]")
    print("P22C3_FAILS=%d" % fails)
    if fails:
        sys.exit(1)
    print("P22C3_OPS_PASS")


if __name__ == "__main__":
    main()
