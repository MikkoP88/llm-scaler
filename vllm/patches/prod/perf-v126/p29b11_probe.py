#!/usr/bin/env python3
"""p29b11_probe.py — v126 P29B11: on-device first-divergence capture via the
instrumented esimd_gdn_conv_fused_seq_spec_dbg op (v130 .so).

Runs the exculpated T3 config (zero pool + zero qkvz, nsd=2, distinct
slots, accs [2,4], HV=24, e4m3) in batch and solo form with dbg records,
then diffs seq1's records against the solo's per (t, hv, tid, field).
Causality order: global_t > prev/save idx > s2_0 (conv chain in) >
conv0/conv_hi (own compute) > SLM q/k/v > h0_pre (pool load) > A/dt/b/a
> q_inv/k_inv > kv0 > res0. The FIRST field class that diverges names
the corrupted stage. Also a random-input leg.
"""
import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DT = torch.float8_e4m3fn
NF = 20

F = ["glb_t", "prev", "save", "conv0", "s2_0", "q_lo", "k_lo", "v0",
     "v1", "h0pre", "A", "dtb", "b", "a", "q_inv", "k_inv", "kv0",
     "res0", "convhi", "epoch"]
CAUS = [0, 1, 2, 4, 3, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]

import custom_esimd_kernels_vllm  # loads the lgrf .so + op registrations
op_dbg = torch.ops.custom_esimd_kernels_vllm.esimd_gdn_conv_fused_seq_spec_dbg

torch.manual_seed(2929)
cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
conv0 = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05

torch.manual_seed(31337)
qkvzB = torch.randn(4 * NST, DIM, dtype=torch.float16, device=DEV) * 0.1
baB = torch.randn(4 * NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5


def spec_dbg(pool, conv, qkvz, ba, rows_idx, accepted):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    dbg = torch.full((n * NST, HV, 64, NF), float("nan"),
                     dtype=torch.float32, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    op_dbg(
        qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
        out, z, tok, acc, n, NST, H, HV, K, V, SCALE, dbg)
    torch.xpu.synchronize()
    return out, dbg


def fp8(t):
    return t.to(DT).contiguous()


S1 = list(range(45, 50))


def leg(zero, tag):
    pool_src = torch.zeros_like(pool16) if zero else pool16
    q_src = torch.zeros_like(qkvzB) if zero else qkvzB
    oB, dB = spec_dbg(fp8(pool_src), conv0.clone(),
                      q_src[0:2 * NST].contiguous(),
                      baB[0:2 * NST].contiguous(),
                      list(range(40, 50)), [2, 4])
    oS, dS = spec_dbg(fp8(pool_src), conv0.clone(),
                      q_src[NST:2 * NST].contiguous(),
                      baB[NST:2 * NST].contiguous(), S1, [4])
    print("=== %s ===" % tag)
    print("out t0 band:", "".join(
        "o" if torch.equal(oB[NST, hv], oS[0, hv]) else "X"
        for hv in range(HV)))
    for t in range(NST):
        # batch row for seq1 token t vs solo row t
        A = dB[NST + t]   # (HV, 64, NF)
        B = dS[t]
        ne = A != B      # NaN != NaN is True — but records are written
        per_field = ne.any(dim=1)          # (HV, NF) any tid
        cnt = ne.sum(dim=(0, 1))           # (NF,) total mismatches
        first = next((f for f in CAUS if cnt[f] > 0), None)
        print("t=%d first-divergent field: %s   (mismatch counts: %s)"
              % (t, F[first] if first is not None else "NONE",
                 " ".join("%s=%d" % (F[i], cnt[i]) for i in CAUS
                          if cnt[i] > 0)))
        if t == 0 and first is not None:
            # sample one failing (hv, tid) for the first field
            hv_hit, tid_hit = None, None
            for f in CAUS:
                m = (A[:, :, f] != B[:, :, f])
                if m.any():
                    hv_hit, tid_hit = m.nonzero()[0].tolist()
                    print("  field %s first hit hv=%d tid=%d batch=%.6f"
                          " solo=%.6f" % (
                              F[f], hv_hit, tid_hit,
                              A[hv_hit, tid_hit, f].item(),
                              B[hv_hit, tid_hit, f].item()))
                    break
        if first is None:
            break


leg(True, "T3 zero-pool+zero-qkvz")
leg(False, "random inputs")
print("PROBE11_DONE")
