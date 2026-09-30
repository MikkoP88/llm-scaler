#!/usr/bin/env python3
"""p29b2_probe.py — v126 P29B follow-up: localize the solo-vs-batch spec
mismatch (L1 rev2 fails req=1 for ALL dtypes incl. fp16, conv rows equal,
ssm pool + outputs differ). Runs against the idle lane container (p29c
boot stopped at the P29B gate; serve never started).

Probes:
  A  batch-vs-batch (pristine clones)  -> is the nsd=4 batch itself
     deterministic? (intra-batch race would flunk this)
  B  solo-vs-solo for req1             -> solo deterministic?
  C  POSITION vs REQUEST: nsd=2 orders (req0,req1) and (req1,req0), each
     request on its own distinct slots; compare each seq position against
     the same request's solo. Failure following POSITION (seq1 wrong in
     both orders) = kernel seq>=1 bug; following REQUEST = harness/data.
  D  identical twins: nsd=2 with req1's inputs on both positions -> seq0
     vs seq1 outputs+pool must be bitwise identical (and equal to solo).
  E  token localization on the failing pair (req1): first diverging
     output row t and first diverging pool slot t; for the first bad
     pool slot, report which V-rows (vi0=2*tid) differ -> tid coverage
     pattern.
All legs per dtype fp16/e4m3/e5m2.
"""
import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa: E402

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

REQ_QKVZ = [qkvzB[i * NST:(i + 1) * NST].contiguous() for i in range(4)]
REQ_BA = [baB[i * NST:(i + 1) * NST].contiguous() for i in range(4)]
ACC = [2, 4, 1, 3]


def fresh(dt):
    return pool16.clone() if dt == torch.float16 \
        else pool16.to(dt).contiguous()


def bv(t):
    return t.view(torch.int16) if t.dtype == torch.float16 \
        else t.view(torch.uint8)


def spec(dt, pool, conv, reqs, slots, accs):
    """reqs: list of request ids; slots: flat list (len(reqs)*NST)."""
    n = len(reqs)
    qkvz = torch.cat([REQ_QKVZ[r] for r in reqs]).contiguous()
    ba = torch.cat([REQ_BA[r] for r in reqs]).contiguous()
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(slots, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accs, dtype=torch.int32, device=DEV)
    esimd_gdn_conv_fused_seq_spec(
        qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
        out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out


for name, dt in DTYPES:
    print("=== %s ===" % name)
    S40 = list(range(40, 60))       # batch order (0,1,2,3)
    S01 = list(range(40, 50))       # nsd2 (req0, req1)
    S10 = list(range(50, 60))       # nsd2 (req1, req0) — req1 on 50..54
    S11 = list(range(60, 70))       # twins: req1 on 60..64 (seq0), 65..69

    # A: batch determinism
    oA1 = spec(dt, fresh(dt), conv0.clone(), [0, 1, 2, 3], S40, ACC)
    oA2 = spec(dt, fresh(dt), conv0.clone(), [0, 1, 2, 3], S40, ACC)
    print("  A batch_det      :", torch.equal(oA1, oA2))

    # B: solo determinism (req1)
    oB1 = spec(dt, fresh(dt), conv0.clone(), [1], list(range(45, 50)), [4])
    oB2 = spec(dt, fresh(dt), conv0.clone(), [1], list(range(45, 50)), [4])
    print("  B solo_det(req1) :", torch.equal(oB1, oB2))

    # solos for C
    oS0 = spec(dt, fresh(dt), conv0.clone(), [0], list(range(40, 45)), [2])
    oS1 = oB1

    # C: position vs request (nsd=2)
    o01 = spec(dt, fresh(dt), conv0.clone(), [0, 1], S01, [2, 4])
    o10 = spec(dt, fresh(dt), conv0.clone(), [1, 0], S10, [4, 2])
    print("  C (req0,req1): seq0==solo0", torch.equal(o01[0:5], oS0),
          " seq1==solo1", torch.equal(o01[5:10], oS1))
    print("  C (req1,req0): seq0==solo1", torch.equal(o10[0:5], oS1),
          " seq1==solo0", torch.equal(o10[5:10], oS0))

    # D: identical twins (req1 inputs both positions)
    oT = spec(dt, fresh(dt), conv0.clone(), [1, 1], S11, [4, 4])
    print("  D twins: seq0==seq1", torch.equal(oT[0:5], oT[5:10]),
          " seq0==solo1", torch.equal(oT[0:5], oS1),
          " seq1==solo1", torch.equal(oT[5:10], oS1))

    # E: token localization for req1 batch-vs-solo
    oB = oA1
    for t in range(NST):
        eq_o = torch.equal(oS1[t:t + 1], oB[5 + t:6 + t])
        print("  E req1 t=%d out_eq=%s" % (t, eq_o))
        if not eq_o:
            d = (oS1[t].float() - oB[5 + t].float()).abs()
            nz = (d > 0).nonzero()
            hv_set = sorted(set(nz[:, 0].tolist()))
            vrows = sorted(set(nz[:, 1].tolist()))
            print("     bad hv=%s vrow_range=[%d..%d] count=%d"
                  % (hv_set[:6], vrows[0], vrows[-1], len(nz)))
            break

print("PROBE_DONE")
