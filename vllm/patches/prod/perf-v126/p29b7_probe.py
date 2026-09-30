#!/usr/bin/env python3
"""p29b7_probe.py — v126 P29B7: contamination topology.

New facts: hv=23 out bitwise CORRECT while hv 8..22 novel (corr ~0.53);
no cross-slot byte matches => not an address shift. Now:
  M1  full (seq x hv) out-correctness matrix, nsd=2 and nsd=4 (e4m3)
  M2  seq0 INPUT perturbation: scale seq0's qkvz/ba rows; seq1 identical
      -> does seq1's output change? (cross-seq data leak)
  M3  seq0 SLOT perturbation: seq0 on 70..74 instead of 40..44
      -> does seq1's output change? (cross-seq address-space leak)
  M4  nsd perturbation: seq1 same inputs/slots/position, nsd = 2,3,4
      -> does seq1's output change? (grid-size dependence)
  M5  run-to-run: same nsd=2 batch 3x, per-hv stability of the wrong bytes
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
    return out


S1 = list(range(45, 50))
R1 = qkvzB[NST:2 * NST].contiguous()
B1 = baB[NST:2 * NST].contiguous()


def solo_out(i, acc_i, slots_i):
    return spec(fresh(), conv0.clone(),
                qkvzB[i * NST:(i + 1) * NST].contiguous(),
                baB[i * NST:(i + 1) * NST].contiguous(), slots_i, [acc_i])


# ---- M1: (seq x hv) correctness matrices ----------------------------------
for nsd in (2, 4):
    slots = list(range(40, 40 + nsd * NST))
    accs = [2, 4, 1, 3][:nsd]
    oB = spec(fresh(), conv0.clone(), qkvzB[:nsd * NST].contiguous(),
              baB[:nsd * NST].contiguous(), slots, accs)
    print("--- M1 nsd=%d out==solo matrix (o=ok X=diff) ---" % nsd)
    for i in range(nsd):
        oS = solo_out(i, accs[i], slots[i * NST:(i + 1) * NST])
        marks = "".join(
            "o" if torch.equal(oB[i * NST + t], oS[t]) else "X"
            for t in range(0, NST, NST))  # t=0 only per seq for compactness
        hvmarks = "".join(
            "o" if torch.equal(oB[i * NST, hv], oS[0, hv]) else "X"
            for hv in range(HV))
        print("  seq%d t0 %s | hv %s" % (i, marks, hvmarks))

# ---- M2: seq0 input perturbation ------------------------------------------
oA = spec(fresh(), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
          baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
q2 = qkvzB[0:2 * NST].clone()
q2[0:NST] *= 2.0                     # perturb ONLY seq0's rows
b2 = baB[0:2 * NST].clone()
b2[0:NST] *= 2.0
oP = spec(fresh(), conv0.clone(), q2.contiguous(), b2.contiguous(),
          list(range(40, 50)), [2, 4])
per_hv = "".join(
    "o" if torch.equal(oA[5 + t, hv], oP[5 + t, hv]) else "X"
    for hv in range(HV) for t in [0])
print("M2 seq1 t0 stable under seq0 input x2 (o=stable):", per_hv)

# ---- M3: seq0 slot perturbation -------------------------------------------
oQ = spec(fresh(), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
          baB[0:2 * NST].contiguous(), list(range(70, 75)) + S1, [2, 4])
per_hv3 = "".join(
    "o" if torch.equal(oA[5 + t, hv], oQ[5 + t, hv]) else "X"
    for hv in range(HV) for t in [0])
print("M3 seq1 t0 stable under seq0 slot move (o=stable):", per_hv3)

# ---- M4: nsd perturbation, seq1 fixed inputs/slots/position ---------------
oN2 = spec(fresh(), conv0.clone(),
           qkvzB[NST:3 * NST].contiguous(), baB[NST:3 * NST].contiguous(),
           S1 + list(range(50, 55)), [4, 1])
oN4 = spec(fresh(), conv0.clone(),
           qkvzB[NST:5 * NST].contiguous()[:4 * NST - NST].contiguous(),
           baB[NST:5 * NST].contiguous()[:4 * NST - NST].contiguous(),
           S1 + list(range(50, 65)), [4, 1, 2, 3])
per_hv4 = "".join(
    "o" if torch.equal(oN2[NST + t, hv], oN4[NST + t, hv]) else "X"
    for hv in range(HV) for t in [0])
print("M4 seq1 t0 stable under nsd 2->4 (o=stable):", per_hv4)

# ---- M5: run-to-run stability of wrong bytes ------------------------------
runs = [spec(fresh(), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
             baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
        for _ in range(3)]
for hv in (10, 15):
    st = "".join(
        "o" if torch.equal(runs[0][5, hv], r[5, hv]) else "X" for r in runs)
    print("M5 hv=%d t0 run stability:" % hv, st)
print("PROBE7_DONE")
