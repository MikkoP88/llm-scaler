#!/usr/bin/env python3
"""p29b4_probe.py — v126 P29B4: forensic isolation of the seq>=1 t=0 SSM
divergence on the DETERMINISTIC e4m3 lane (run_det=True in p29b3).

Facts so far: conv checkpoints BITWISE MATCH solos for every req (conv chain
+ q/k/v SLM staging identical); outputs + ssm pool differ from t=0 for
seq1/seq2 in every dtype. Inputs are byte-identical between batch and solo
=> the seed load itself must resolve differently.

Discriminators:
  Z1  zero-state reference: solo with idx[init_col] = -1 forces
      state_base == nullptr -> h0 = 0. If batch seq1 t0 == that, the
      batch's seed load saw NOTHING (zeros / null).
  Z2  pattern seed: pool[seed slot] filled with 0.25 everywhere; solo must
      show it (differs from random-seed solo); if batch seq1 t0 IGNORES the
      pattern -> the load never touched that slot.
  Z3  cross-slot address forensics: seed seq0's slot with pattern P0 and
      seq1's slot with P1 (different); does batch seq1's t0 equal the
      P1-solo or the P0-solo (reads neighbor via state_row shift)?
  Z4  z rows: z passthrough is independent of the SSM math; z equality
      between batch and solo pins the divergence to the SSM segment.
  Z5  which V-rows (vi0 = 2*tid) and hv differ in pool[45] (t0 save slot)
      batch vs solo: confined pattern => addressing; scrambled => timing.
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


R1 = qkvzB[NST:2 * NST].contiguous()
B1 = baB[NST:2 * NST].contiguous()
S1 = list(range(45, 50))            # req1 slots, seed col 3 -> slot 48

# nsd=2 batch: req0 (slots 40..44) + req1 (slots 45..49), acc [2,4]
oB, zB = spec(fresh(), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
              baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])

# solo req1, random seed (the reference that batch seq1 must equal)
oR, _ = spec(fresh(), conv0.clone(), R1, B1, S1, [4])

# Z1: seed slot disabled (idx[3] = -1 -> state_base null -> h0 = 0)
S1n = [45, 46, 47, -1, 49]
oN, _ = spec(fresh(), conv0.clone(), R1, B1, S1n, [4])

# Z2: pattern seed 0.25 in slot 48
pP = fresh()
pP[48] = 0.25
oP, _ = spec(pP, conv0.clone(), R1, B1, S1, [4])

# Z3: distinct patterns — seq0 seed slot 41 <- 0.125, seq1 seed slot 48 <- 0.25
pX = fresh()
pX[41] = 0.125
pX[48] = 0.25
oX, _ = spec(pX, conv0.clone(), qkvzB[0:2 * NST].contiguous(),
             baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
# solo controls for Z3
oP0, _ = None, None
p0 = fresh()
p0[41] = 0.125
oQ0, _ = spec(p0, conv0.clone(), qkvzB[0:NST].contiguous(),
              baB[0:NST].contiguous(), list(range(40, 45)), [2])

b_t0 = oB[5:6]
print("batch seq1 t0 == random-seed solo t0 :", torch.equal(b_t0, oR[0:1]))
print("Z1 batch seq1 t0 == NULL-seed solo t0:", torch.equal(b_t0, oN[0:1]))
print("Z2 pattern solo t0 != random solo t0 :",
      not torch.equal(oP[0:1], oR[0:1]), "(pattern visible in solo)")
print("Z2 batch(seq1 pat 0.25) t0 == pattern-solo t0:",
      torch.equal(oX[5:6], oP[0:1]))
print("Z3 batch seq1 t0 == seq0-seed-pattern solo t0:",
      torch.equal(oX[5:6], oQ0[0:1]), "(reads neighbor's slot?)")

# Z4: z rows for seq1 — independent of SSM math
zR = spec(fresh(), conv0.clone(), R1, B1, S1, [4])[1]
print("Z4 z rows seq1 batch == solo         :",
      torch.equal(zB[5:10], zR[0:5]))

# Z5: which hv / vi0 rows differ in pool[45] (t0 save slot) batch vs solo
pB = fresh()
spec(pB, conv0.clone(), qkvzB[0:2 * NST].contiguous(),
     baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
pS = fresh()
spec(pS, conv0.clone(), R1, B1, S1, [4])
d = (pB[45].float() != pS[45].float())
nz = d.any(dim=2).nonzero()               # (hv, v) pairs differing
hvs = sorted(set(nz[:, 0].tolist()))
vs = sorted(set(nz[:, 1].tolist()))
tid_rows = sorted({v // 2 for v in vs})    # vi0 = 2*tid -> tid = v//2
print("Z5 pool[45] diff: hv=%s" % hvs)
print("    v rows=%s" % vs)
print("    tids=%s" % tid_rows)
print("PROBE4_DONE")
