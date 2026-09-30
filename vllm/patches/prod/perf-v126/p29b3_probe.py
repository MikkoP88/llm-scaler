#!/usr/bin/env python3
"""p29b3_probe.py — v126 P29B3: localize the nsd>1 spec-kernel race.

Standing facts: (a) two IDENTICAL nsd=4 batch runs from pristine clones
produce different outputs (nondeterminism => race/UB); (b) valid same-slot
solo comparison fails at req=1 (acc=4, init_col=3), t=0, ~14 heads.

This probe diffs EVERYTHING between two identical batch runs:
  - which output rows (req, t) differ
  - which SSM pool slots differ  (vs pristine too -> OOB write check)
  - which CONV pool slots differ (vs pristine too -> OOB write check)
plus VALID same-slot solo comparisons per request (L1 form, all four reqs,
report first failing t per req), and acc-matrix nsd=2 legs with the SOLO
REUSING THE BATCH POSITION'S SLOT IDS (the p29b2 C/D legs were confounded:
different slot ids => different seed state => incomparable).
"""
import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5
NSD = 4
SPEC_SLOTS = list(range(40, 40 + NSD * NST))
SCALE = float(K ** -0.5)
DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa

torch.manual_seed(2929)
cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
conv0 = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05

torch.manual_seed(31337)
qkvzB = torch.randn(NSD * NST, DIM, dtype=torch.float16, device=DEV) * 0.1
baB = torch.randn(NSD * NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5


def fresh(dt):
    return pool16.clone() if dt == torch.float16 \
        else pool16.to(dt).contiguous()


def bv(t):
    return t.view(torch.int16) if t.dtype == torch.float16 \
        else t.view(torch.uint8)


def spec(dt, pool, conv, qkvz, ba, rows_idx, accepted):
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
    return out, pool, conv


def slots_of(pool, conv, ref_pool, ref_conv, label):
    """Row indexes whose bytes differ from the reference snapshot."""
    ps = ((bv(pool) != bv(ref_pool)).any(dim=(1, 2, 3))
          .nonzero().flatten().tolist())
    cs = ((conv != ref_conv).any(dim=(1, 2))
          .nonzero().flatten().tolist())
    return ps, cs


for name, dt in DTYPES:
    print("=== %s ===" % name)
    # --- two identical batch runs, pristine starts -------------------------
    o1, p1, c1 = spec(dt, fresh(dt), conv0.clone(), qkvzB, baB,
                      SPEC_SLOTS, [2, 4, 1, 3])
    o2, p2, c2 = spec(dt, fresh(dt), conv0.clone(), qkvzB, baB,
                      SPEC_SLOTS, [2, 4, 1, 3])
    print("  run_det:", torch.equal(o1, o2))
    if not torch.equal(o1, o2):
        d = (o1 != o2).any(dim=(1, 2)).nonzero().flatten().tolist()
        print("    out rows differing:", d)
        ps, cs = slots_of(p1, c1, p2, c2, "run2")
        print("    pool slots differing:", ps)
        print("    conv slots differing:", cs)
    # --- OOB: what changed vs pristine --------------------------------------
    pr = fresh(dt)
    ps, cs = slots_of(p1, c1, pr, conv0, "pristine")
    oob = [s for s in ps if s not in SPEC_SLOTS]
    ooC = [s for s in cs if s not in SPEC_SLOTS]
    print("  pool changed slots=%s OOB=%s" % (len(ps), oob))
    print("  conv changed slots=%s OOB=%s" % (len(cs), ooC))

    # --- VALID same-slot solos (L1 form), report per req + first bad t -----
    for i, acc_i in enumerate([2, 4, 1, 3]):
        slots_i = SPEC_SLOTS[i * NST:(i + 1) * NST]
        rows = qkvzB[i * NST:(i + 1) * NST].contiguous()
        bas = baB[i * NST:(i + 1) * NST].contiguous()
        oS, pS, cS = spec(dt, fresh(dt), conv0.clone(), rows, bas,
                          slots_i, [acc_i])
        o_eq = torch.equal(oS, o1[i * NST:(i + 1) * NST])
        sl = torch.tensor(slots_i, device=DEV)
        p_eq = torch.equal(bv(pS[sl]), bv(p1[sl]))
        c_eq = torch.equal(cS[sl], c1[sl])
        first_t = -1
        if not o_eq:
            for t in range(NST):
                if not torch.equal(oS[t:t + 1], o1[i * NST + t:i * NST + t + 1]):
                    first_t = t
                    break
        print("  req%d(acc=%d): out=%s pool=%s conv=%s first_bad_t=%d"
              % (i, acc_i, o_eq, p_eq, c_eq, first_t))

    # --- acc-matrix nsd=2, solo reuses BATCH-POSITION slot ids -------------
    for a in (1, 2, 4):
        s0 = list(range(40, 45))
        s1 = list(range(45, 50))
        oB, _, _ = spec(dt, fresh(dt), conv0.clone(), qkvzB, baB,
                        s0 + s1, [a, a])
        r0 = qkvzB[0:NST].contiguous()
        b0 = baB[0:NST].contiguous()
        oS0, _, _ = spec(dt, fresh(dt), conv0.clone(), r0, b0, s0, [a])
        r1 = qkvzB[NST:2 * NST].contiguous()
        b1 = baB[NST:2 * NST].contiguous()
        oS1, _, _ = spec(dt, fresh(dt), conv0.clone(), r1, b1, s1, [a])
        print("  nsd2 acc=%d: seq0==solo0 %s seq1==solo1 %s seq0==seq1 %s"
              % (a, torch.equal(oB[0:5], oS0), torch.equal(oB[5:10], oS1),
                 torch.equal(oB[0:5], oB[5:10])))

print("PROBE3_DONE")
