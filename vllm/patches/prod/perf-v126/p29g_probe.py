#!/usr/bin/env python3
"""p29g_probe.py — v126 P29G: graphed multi-round trajectory (the last
op-level serve condition P29D/P29E/P29F did not model).

Serve reality: ONE decode graph captured at boot is REPLAYED round over
round; each verify round's checkpoint writes rotate the ring (next round
reads a different ring row as its t==0 init) and the state compounds
through the SAME frozen capture. P29D tested this eagerly (green);
P29E/P29F tested single-round captures (green).

P29G: capture the spec wrapper ONCE per dtype, replay ROUNDS times with
per-round fresh inputs + rotated ring indices + accepted counts; compare
each round's outputs and the final pool bytes against an identical EAGER
trajectory. Divergence at e4m3/e5m2 but not fp16 re-convicts the op level
with a minimal repro; all-green escalates the hunt to serve integration.
"""
import sys

import torch

DEV = "xpu"
K = V = 128
H = 8
HV = 24
NC = 512
NST = 5
NSD = 8
ROUNDS = 30
SCALE = float(K ** -0.5)
DIM = 2 * H * K + 2 * HV * V

DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]


def bv(t):
    return t.view(torch.uint8) if t.dtype != torch.float16 else t.view(torch.int16)


def cos(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    return (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-30)).item()


def main():
    print("=== P29G graphed multi-round trajectory (nsd=%d rounds=%d) ==="
          % (NSD, ROUNDS))
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec as op
    torch.manual_seed(77)
    cw = (torch.randn(DIM, 4, device=DEV) * 0.1).half()
    cb = torch.zeros(DIM, device=DEV, dtype=torch.float16)
    A_log = (torch.randn(HV, device=DEV) * 0.1).half()
    dtb = (torch.randn(HV, device=DEV) * 0.1).half()

    acc_cycle = [4, 2, 3, 1, 5, 2, 4, 3]

    for name, dt in DTYPES:
        torch.manual_seed(4242)
        pool = ((torch.randn(NC, HV, V, K, device=DEV) * 0.05).half()
                if dt == torch.float16 else
                (torch.randn(NC, HV, V, K, device=DEV) * 0.05).half().to(dt)
                .contiguous())
        conv = (torch.randn(NC, 3, DIM, device=DEV) * 0.01).half()

        # eager twin: identical starting state, driven eagerly
        pool_e = pool.clone()
        conv_e = conv.clone()

        idx_s = torch.zeros(NSD, NST, dtype=torch.int32, device=DEV)
        for s in range(NSD):
            idx_s[s] = torch.arange(s * NST, s * NST + NST,
                                    dtype=torch.int32, device=DEV)
        tok_s = torch.arange(NSD * NST, dtype=torch.int32, device=DEV)
        acc_s = torch.full((NSD,), 4, dtype=torch.int32, device=DEV)
        qkvz_s = (torch.randn(NSD * NST, DIM, device=DEV) * 0.5).half()
        ba_s = (torch.randn(NSD * NST, 2 * HV, device=DEV) * 0.5).half()
        out_s = torch.zeros(NSD * NST, HV, V, dtype=torch.float16, device=DEV)
        z_s = torch.zeros_like(out_s)
        idx_e, tok_e, acc_e = idx_s.clone(), tok_s.clone(), acc_s.clone()
        qkvz_e, ba_e = qkvz_s.clone(), ba_s.clone()
        out_e = torch.zeros_like(out_s)
        z_e = torch.zeros_like(out_s)

        def call_s():
            op(qkvz_s, conv, cw, cb, idx_s, A_log, dtb, ba_s, pool,
               out_s, z_s, tok_s, acc_s, NSD, NST, H, HV, K, V, SCALE)

        def call_e():
            op(qkvz_e, conv_e, cw, cb, idx_e, A_log, dtb, ba_e, pool_e,
               out_e, z_e, tok_e, acc_e, NSD, NST, H, HV, K, V, SCALE)

        call_s()
        torch.xpu.synchronize()
        # reset state so graphed and eager twins start identically
        pool.copy_(pool_e.clone() if dt == torch.float16 else
                   pool_e.to(dt))          # both now share fresh content
        conv.copy_(conv_e)
        pool_e = pool.clone()
        conv_e = conv.clone()
        try:
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                call_s()
        except Exception as e:  # noqa: BLE001
            print("  %s: CAPTURE-ERROR %r" % (name, e))
            continue
        torch.xpu.synchronize()
        # capture-exec mutated conv/pool; resync eager twin to match exactly
        pool_e.copy_(pool if dt == torch.float16 else pool)
        conv_e.copy_(conv)
        # eager twin must re-run its own call to stay content-identical?
        # No: capture executed call_s on (pool, conv). We copy that state
        # to the twin so both trajectories resume from identical bytes.

        worst = {}
        worst_round_cos = 1.0
        graph_out_ok = True
        for r in range(ROUNDS):
            torch.manual_seed(9000 + r)
            q = (torch.randn(NSD * NST, DIM, device=DEV) * 0.5).half()
            b = (torch.randn(NSD * NST, 2 * HV, device=DEV) * 0.5).half()
            a = acc_cycle[r % len(acc_cycle)]
            # ring rotation: roll each seq's ring by the accepted count so
            # the next t==0 init row is the last checkpointed row
            idx_s.copy_(torch.roll(idx_s, shifts=a, dims=1))
            acc_s.fill_(a)
            qkvz_s.copy_(q)
            ba_s.copy_(b)
            idx_e.copy_(idx_s)
            acc_e.copy_(acc_s)
            qkvz_e.copy_(q)
            ba_e.copy_(b)
            call_e()
            torch.xpu.synchronize()
            g.replay()
            torch.xpu.synchronize()
            if not torch.equal(out_s, out_e):
                graph_out_ok = False
            worst_round_cos = min(worst_round_cos, cos(out_s, out_e))
        # final pool comparison on the rotated ring rows (all touched rows)
        rows = idx_s.reshape(-1).to(torch.long)
        pool_ok = bool(torch.equal(bv(pool[rows]), bv(pool_e[rows])))
        conv_ok = bool(torch.equal(conv[rows], conv_e[rows]))
        full_pool_ok = bool(torch.equal(bv(pool), bv(pool_e)))
        print("  %-5s per-round out bitwise=%s worst_cos=%.6f "
              "ring pool=%s conv=%s FULL pool=%s"
              % (name, "OK" if graph_out_ok else "BAD", worst_round_cos,
                 "OK" if pool_ok else "BAD", "OK" if conv_ok else "BAD",
                 "OK" if full_pool_ok else "BAD"))
        print("P29G %-5s out_bitwise=%d worst_cos=%.6f ring_pool=%d conv=%d "
              "full_pool=%d"
              % (name, graph_out_ok, worst_round_cos, pool_ok, conv_ok,
                 full_pool_ok))
        del pool, conv, pool_e, conv_e
    print("P29G_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
