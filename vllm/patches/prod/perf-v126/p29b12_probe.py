#!/usr/bin/env python3
"""p29b12_probe.py — v126 P29B12: pre-existence test on the STOCK
production .so (sha 1d9dcf4e..., extracted from image v1.2.25-raw).

Runs the p29b10 discriminator geometry with the fp16 pool (the stock
.so predates fp8 SSM state). If nsd=2 HV=24 batch-vs-solo diverges for
seq1 hv>=8 here too, the nsd>1 double_v defect PRE-EXISTS the entire
v126/v125 line — an upstream bug never before exercised (production
gates nsd==1 for the ESIMD spec path), not a v129 regression.
"""
import torch

K = V = 128
H = 8
HV = 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5
SCALE = float(K ** -0.5)

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa

torch.manual_seed(2929)
cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
conv0 = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
pool = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05

torch.manual_seed(31337)
qkvzB = torch.randn(4 * NST, DIM, dtype=torch.float16, device=DEV) * 0.1
baB = torch.randn(4 * NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5


def spec(conv, qkvz, ba, rows_idx, accepted):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    esimd_gdn_conv_fused_seq_spec(
        qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool.clone(),
        out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out


S1 = list(range(45, 50))
for rep in range(3):
    oB = spec(conv0.clone(), qkvzB[0:2 * NST].contiguous(),
              baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
    oS = spec(conv0.clone(), qkvzB[NST:2 * NST].contiguous(),
              baB[NST:2 * NST].contiguous(), S1, [4])
    band = "".join(
        "o" if torch.equal(oB[NST, hv], oS[0, hv]) else "X"
        for hv in range(HV))
    print("STOCK .so fp16 nsd=2 rep%d seq1-t0 band: %s" % (rep, band))
print("PROBE12_DONE")
