#!/usr/bin/env python3
"""p29b6_probe.py — v126 P29B6: cross-slot / cross-t equality scan.

Facts: hv>=8 of seq>=1 execute fully (nonzero out, non-pristine pool, conv
solo-equal) but hold DIFFERENT values than the solo, deterministically
(e4m3). If their stores/loads land at a SHIFTED address (fixed bug), the
batch's bytes must appear SOMEWHERE ELSE in the solo universe:
  - pool: pB[slot=45][hv] vs pS[X][hv] for ALL X in 0..511
  - out : oB[seq1,t][hv]  vs oS[t'][hv]   for all t' (cross-t shift)
  - also against the pristine pool bytes.
If NO match anywhere -> truly novel values -> the math inputs differ.
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


def fresh():
    return pool16.to(DT).contiguous()


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
    return out, z


S1 = list(range(45, 50))
R1 = qkvzB[NST:2 * NST].contiguous()
B1 = baB[NST:2 * NST].contiguous()

# batch nsd=2
oB, _ = spec(fresh(), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
             baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
pB = fresh()
spec(pB, conv0.clone(), qkvzB[0:2 * NST].contiguous(),
     baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
# solo req1
oS, _ = spec(fresh(), conv0.clone(), R1, B1, S1, [4])
pS = fresh()
spec(pS, conv0.clone(), R1, B1, S1, [4])
pristine = fresh()

# --- pool scan: pB[45][hv] vs pS[X][hv] and pristine[X][hv], all X --------
BAD = list(range(8, 24))
for hv in (10, 15, 23):
    row = pB[45][hv].view(torch.uint8)
    hits = [X for X in range(NC)
            if torch.equal(row, pS[X][hv].view(torch.uint8))]
    phits = [X for X in range(NC)
             if torch.equal(row, pristine[X][hv].view(torch.uint8))]
    print("pool pB[45][hv=%d]: solo-slot hits=%s pristine hits=%s"
          % (hv, hits, phits))

# --- out scan: oB[5+t][hv] vs oS[t'][hv] all t' ---------------------------
for hv in (10, 15, 23):
    for t in range(NST):
        row = oB[5 + t][hv]
        hits = [tp for tp in range(NST) if torch.equal(row, oS[tp][hv])]
        if t == 0 or hits:
            print("out oB[seq1,t=%d][hv=%d]: solo-t hits=%s" % (t, hv, hits))

# --- numeric relationship: is oB seq1 t0 a SCALED/SHIFTED version? -------
for hv in (10, 15, 23):
    a = oB[5][hv].float()
    b = oS[0][hv].float()
    ratio = (a / b).abs()
    print("hv=%d: batch/solo ratio min=%.4f max=%.4f mean=%.4f  "
          "corr=%.6f  b_zero_lanes=%d"
          % (hv, ratio.min().item(), ratio.max().item(), ratio.mean().item(),
             torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
             (b == 0).sum().item()))
print("PROBE6_DONE")
