#!/usr/bin/env python3
"""p29fix_verify.py — v126 P29B-FIX verification against the PRODUCTION op
(esimd_gdn_conv_fused_seq_spec) with the v131 .so installed.

Gate: after the ring-row snapshot fix, a batch launch and per-sequence solo
launchs (both from pristine clones) must be BITWISE identical in outputs,
SSM pool rows and conv rows — at every geometry, dtype and input class —
because every rollback read now returns deterministic pre-round bytes.

Legs:
  A. forensic nsd=2/HV=24 config (the raced one; 48 wgs): zero and random
     inputs, dtypes fp16/e4m3/e5m2, 10 reps each (flap hunting)
  B. nsd=4/HV=24 (96 wgs, accs 2/4/1/3 — every init_col), 3 reps
  C. nsd=1/HV=24 regression (single sequence must stay bitwise)
  D. nsd=2/HV=16 (double_v false, 32 wgs — was passing, must stay)
  E. solo-vs-solo determinism control (same leg twice, bitwise)

Run inside lsv-test with the v131 .so, serve DOWN.
"""
import sys

import torch

DEV = "xpu"
K = V = 128
H = 8
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]

results = []


def record(leg, ok, detail=""):
    results.append((leg, ok))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", leg,
                           (" — " + detail) if detail else ""))


def build_params(seed, HV):
    DIM = 2 * H * K + 2 * HV * V
    torch.manual_seed(seed)
    cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
    cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
    A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
    dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
    conv = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
    pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05
    ntok = 8 * NST
    qkvz = torch.randn(ntok, DIM, dtype=torch.float16, device=DEV) * 0.1
    ba = torch.randn(ntok, 2 * HV, dtype=torch.float16, device=DEV) * 0.5
    return DIM, (cw, cb, A_log16, dt_bias, conv, pool16, qkvz, ba)


def fresh_pool(pool16, dt):
    return pool16.clone() if dt == torch.float16 \
        else pool16.to(dt).contiguous()


def bytes_view(t):
    return t.view(torch.uint8) if t.dtype != torch.float16 \
        else t.view(torch.int16)


def run_spec(op, cw, cb, A_log, dt_bias, dt, pool, conv, qkvz, ba,
             rows_idx, accepted, HV):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    op(qkvz, conv, cw, cb, idx, A_log, dt_bias, ba, pool,
       out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out


def solo_vs_batch(tag, HV, nsd, accepted, reps, zero, seed):
    DIM, (cw, cb, A_log, dt_bias, conv0, pool16, qkvzB, baB) = \
        build_params(seed, HV)
    if zero:
        pool16 = torch.zeros_like(pool16)
        qkvzB = torch.zeros_like(qkvzB)
    slots = list(range(40, 40 + nsd * NST))
    worst = ""
    ok_all = True
    for rep in range(reps):
        for name, dt in DTYPES:
            pool_b = fresh_pool(pool16, dt)
            conv_b = conv0.clone()
            outB = run_spec(op, cw, cb, A_log, dt_bias, dt, pool_b, conv_b,
                            qkvzB[:nsd * NST].contiguous(),
                            baB[:nsd * NST].contiguous(), slots, accepted, HV)
            for i in range(nsd):
                pool_s = fresh_pool(pool16, dt)
                conv_s = conv0.clone()
                outS = run_spec(op, cw, cb, A_log, dt_bias, dt, pool_s, conv_s,
                                qkvzB[i * NST:(i + 1) * NST].contiguous(),
                                baB[i * NST:(i + 1) * NST].contiguous(),
                                slots[i * NST:(i + 1) * NST],
                                [accepted[i]], HV)
                sl = torch.tensor(slots[i * NST:(i + 1) * NST], device=DEV)
                o_eq = torch.equal(outS, outB[i * NST:(i + 1) * NST])
                p_eq = torch.equal(bytes_view(pool_s[sl]),
                                   bytes_view(pool_b[sl]))
                c_eq = torch.equal(conv_s[sl], conv_b[sl])
                if not (o_eq and p_eq and c_eq):
                    ok_all = False
                    worst = "rep=%d %s seq=%d out=%s pool=%s conv=%s" % (
                        rep, name, i, o_eq, p_eq, c_eq)
                    break
            if not ok_all:
                break
        if not ok_all:
            break
    record(tag, ok_all, worst if worst else "%d reps x %d dtypes bitwise" %
           (reps, len(DTYPES)))


print("=== P29B-FIX production-op verification (v131 .so) ===")
from custom_esimd_kernels_vllm import (  # noqa: E402
    esimd_gdn_conv_fused_seq_spec as op,
)

# A. forensic nsd=2/HV=24 (48 wgs — the raced geometry)
solo_vs_batch("A1 nsd=2 HV=24 zero 10x", 24, 2, [2, 4], 10, True, 2929)
solo_vs_batch("A2 nsd=2 HV=24 rand 10x", 24, 2, [2, 4], 10, False, 31337)

# B. nsd=4 (96 wgs) — every init_col exercised
solo_vs_batch("B1 nsd=4 HV=24 zero 3x", 24, 4, [2, 4, 1, 3], 3, True, 4242)
solo_vs_batch("B2 nsd=4 HV=24 rand 3x", 24, 4, [2, 4, 1, 3], 3, False, 5150)

# C. nsd=1 regression (single sequence)
solo_vs_batch("C1 nsd=1 HV=24 rand 3x", 24, 1, [4], 3, False, 6161)

# D. nsd=2/HV=16 (double_v false, 32 wgs)
solo_vs_batch("D1 nsd=2 HV=16 zero 3x", 16, 2, [2, 4], 3, True, 7171)
solo_vs_batch("D2 nsd=2 HV=16 rand 3x", 16, 2, [2, 4], 3, False, 8188)

# E. determinism control: batch-vs-batch, raced geometry, all dtypes
DIM, (cw, cb, A_log, dt_bias, conv0, pool16, qkvzB, baB) = \
    build_params(9999, 24)
slots = list(range(40, 50))
det_ok = True
for name, dt in DTYPES:
    o1 = run_spec(op, cw, cb, A_log, dt_bias, dt, fresh_pool(pool16, dt),
                  conv0.clone(), qkvzB[:10].contiguous(),
                  baB[:10].contiguous(), slots, [2, 4], 24)
    o2 = run_spec(op, cw, cb, A_log, dt_bias, dt, fresh_pool(pool16, dt),
                  conv0.clone(), qkvzB[:10].contiguous(),
                  baB[:10].contiguous(), slots, [2, 4], 24)
    if not torch.equal(o1, o2):
        det_ok = False
record("E batch-vs-batch determinism", det_ok,
       "all dtypes" if det_ok else "nondeterministic")

ok = all(o for _, o in results)
print("P29FIX_VERDICT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
