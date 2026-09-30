#!/usr/bin/env python3
"""p29b9_probe.py — v126 P29B9: identical-twin execution determinism.

p29b8 killed every input-channel theory: pool=0 + qkvz=0 still diverges for
(seq>=1, hv>=8). Source audited correct 3x (geometry, staging, barriers,
addressing all verified). Remaining hypothesis: the EXECUTION itself is
nondeterministic across concurrently-scheduled work-groups.

  M1  nsd=2 twins, IDENTICAL slots (both 45..49), identical inputs+acc.
      Position 0 vs position 1 share every address, byte and math step.
      Divergence  -> execution nondeterminism convicted outright.
      Match       -> divergence requires distinct slots (cross-wg address
                     interaction); go to M2.
  M2  nsd=2 twins, DISTINCT slots (40..44 vs 45..49), identical inputs.
      Out rows are position-private; divergence here with M1 matching =
                     the distinct-slot configuration itself is the trigger.
  M3  M1 x4 repeats: is the twin divergence stable in-process?
  M4  M1 with pool=0 AND qkvz=0: divergence with fully constant inputs.
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

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa

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


def spec(pool, conv, qkvz, ba, rows_idx, accepted):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    esimd_gdn_conv_fused_seq_spec(
        qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
        out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out


def fp8(t):
    return t.to(DT).contiguous()


# twin inputs: 2 copies of req0's rows
Q2 = qkvzB[0:NST].repeat(2, 1).contiguous()
B2 = baB[0:NST].repeat(2, 1).contiguous()


def twins(slots0, slots1, zero):
    pool_src = torch.zeros_like(pool16) if zero else pool16
    qkv_src = torch.zeros_like(Q2) if zero else Q2
    ba_src = B2
    return spec(fp8(pool_src), conv0.clone(), qkv_src, ba_src,
                slots0 + slots1, [2, 2])


def band(o, tag):
    per_hv = "".join(
        "o" if torch.equal(o[0, hv], o[NST, hv]) else "X"
        for hv in range(HV))
    full = torch.equal(o[0:NST], o[NST:2 * NST])
    print("%-34s t0-per-hv %s  all-tokens-eq=%s" % (tag, per_hv, full))


S0, S1 = list(range(40, 45)), list(range(45, 50))
band(twins(S1, S1, False), "M1 same-slots twins")
band(twins(S0, S1, False), "M2 distinct-slots twins")
for r in range(4):
    band(twins(S1, S1, False), "M3 same-slots rep%d" % r)
band(twins(S1, S1, True), "M4 same-slots zero-inputs")
print("PROBE9_DONE")
