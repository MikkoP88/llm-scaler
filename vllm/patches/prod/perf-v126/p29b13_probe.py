#!/usr/bin/env python3
"""p29b13_probe.py — v126 P29B13: forensic post-processing of the dbg
records (v130 .so, no rebuild needed).

Established by p29b11: first REAL divergence (glb_t excluded — batch
seq1 runs global tokens NST..2NST-1 vs solo 0..NST-1 by construction)
is the OWN CONV COMPUTE at t=0/1, with byte-identical inputs. This
probe classifies the corrupted values:

  A. determinism controls — solo-vs-solo and batch-vs-batch record
     diffs (is either leg itself nondeterministic?)
  B. seq-crossing       — does batch seq1's record equal batch seq0's
     record for the same t? (seq base/address crossing)
  C. token-crossing     — does batch seq1's t-record equal some OTHER
     t's record (batch or solo)? (window/checkpoint overlap in t)
  D. contamination      — exact-value search of each corrupted conv0
     value across ALL solo records (any hv/tid/t)
  E. per-hv/tid grids   — structure of the conv0/s2_0/q_lo mismatch sets

Config: T3 (zero pool + zero qkvz, nsd=2, slots 40..49, accs [2,4],
HV=24, e4m3), the exculpated geometry from p29b8-p29b12.
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

import custom_esimd_kernels_vllm  # noqa
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
    op_dbg(qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
           out, z, tok, acc, n, NST, H, HV, K, V, SCALE, dbg)
    torch.xpu.synchronize()
    return out, dbg


def fp8(t):
    return t.to(DT).contiguous()


def leg(pool_src, q_src, rows, accs):
    return spec_dbg(fp8(pool_src), conv0.clone(),
                    q_src[0:len(accs) * NST].contiguous(),
                    baB[0:len(accs) * NST].contiguous(),
                    rows, accs)


ZP = torch.zeros_like(pool16)          # zero pool
ZQ = torch.zeros_like(qkvzB)           # zero qkvz
R1 = list(range(40, 50))               # batch slots
S1 = list(range(45, 50))               # solo slots (= batch seq1 slots)

# --- A. determinism controls -------------------------------------------
oS1, dS1 = leg(ZP, ZQ, S1, [4])
oS2, dS2 = leg(ZP, ZQ, S1, [4])
oB1, dB1 = leg(ZP, ZQ, R1, [2, 4])
oB2, dB2 = leg(ZP, ZQ, R1, [2, 4])
print("[A] solo determinism  :", "STABLE" if torch.equal(dS1, dS2) else "FLAPS")
print("[A] batch determinism :", "STABLE" if torch.equal(dB1, dB2) else "FLAPS")
print("[A] solo out band     :", "".join(
    "o" if torch.equal(oS1[0, hv], oS2[0, hv]) else "X"
    for hv in range(HV)))
print("[A] batch out band    :", "".join(
    "o" if torch.equal(oB1[NST, hv], oB2[NST, hv]) else "X"
    for hv in range(HV)))

dB, dS = dB1, dS1
# seq1 rows in batch = NST..2NST-1; drop glb_t (legitimately differs)
FLD = [i for i in range(NF) if i not in (0,)]

# --- B. seq-crossing: batch seq1 record == batch seq0 record? ----------
crossB = 0
for t in range(NST):
    a = dB[NST + t][:, :, FLD]
    b = dB[t][:, :, FLD]
    crossB += (a == b).all(dim=2).sum().item()
tot = NST * HV * 64
print("[B] seq1==seq0 records: %d/%d" % (crossB, tot))

# --- C/D. value forensics on conv0 (field 3) at t=0 --------------------
t = 0
A = dB[NST + t][:, :, 3]   # (HV, 64) batch seq1 conv0
B = dS[t][:, :, 3]         # solo conv0
mm = (A != B)
print("[D] conv0 t0 mismatches: %d (hv map: %s)" % (
    mm.sum().item(),
    "".join("#" if mm[hv].any() else "." for hv in range(HV))))
n_tok = n_hv = n_else = n_unmatched = 0
solo_all = dS[:, :, :, 3].flatten()          # every solo conv0 value
batch_all = dB[:, :, :, 3].flatten()
for hv in range(HV):
    for tid in range(64):
        if not mm[hv, tid]:
            continue
        v = A[hv, tid].item()
        hits_s = (dS[:, :, :, 3] == v)
        hits_b = (dB[:, :, :, 3] == v)
        hits_b[NST + t, hv, tid] = False        # exclude self-echo
        hs = hits_s.any().item()
        hb = hits_b.any().item()
        # where does the value come from?
        src = "none"
        if hb:
            rr, hh, uu = hits_b.nonzero()[0].tolist()
            src = "batch[row=%d(%s t%d),hv=%d,tid=%d]" % (
                rr, "s0" if rr < NST else "s1", rr % NST, hh, uu)
            n_tok += 1
        elif hs:
            tt, hh, uu = hits_s.nonzero()[0].tolist()
            src = "solo[t=%d,hv=%d,tid=%d]" % (tt, hh, uu)
            n_hv += 1
        else:
            n_unmatched += 1
        if n_tok + n_hv + n_unmatched <= 12:
            print("    hv=%2d tid=%2d batch=%+.6f solo=%+.6f src=%s" % (
                hv, tid, v, B[hv, tid].item(), src))
print("[D] src tally: batch-echo=%d solo-echo=%d unmatched=%d" % (
    n_tok, n_hv, n_unmatched))

# --- E. mismatch grids for the causal fields at t=0 --------------------
for f, name in ((4, "s2_0"), (3, "conv0"), (18, "convhi"), (5, "q_lo")):
    g = (dB[NST + t][:, :, f] != dS[t][:, :, f])
    print("[E] %s t0 grid (24 hv rows x 64 tid):" % name)
    for hv in range(HV):
        print("    hv%02d %s" % (hv, "".join(
            "#" if g[hv, tid] else "." for tid in range(64))))
print("PROBE13_DONE")
