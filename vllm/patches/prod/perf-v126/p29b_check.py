#!/usr/bin/env python3
"""p29b_check.py — v126 P29B (rev2): op-level correctness of the P29A ESIMD
kernels (fp8 SSM register chain + single-conversion e4m3 load) and of the
S>1 spec-grid semantics the P29C gate extension relies on.

REV2 — window 1 verdict was FAIL, root-caused to THIS harness, not the
kernels (kernel source gdn_conv_fused_seq_spec.h:218-261):
  init_col = max(num_accepted[seq]-1, 0); token 0 seeds BOTH the conv chain
  (s0/s1/s2) and the SSM state from pool slot indices[init_col] — READ
  BEFORE any store. Window 1 ran the batch call FIRST, then cloned the
  POST-BATCH pool (and shared the mutated conv) for each solo: token 0
  re-read checkpoint slots the batch had overwritten => MISMATCH for ALL
  dtypes INCLUDING fp16, which runs P29A-dead code (kFp8Chain=false) — a
  real kernel regression cannot explain an fp16 mismatch. L3's det check
  shared the mutated conv across runs the same way; L4 filled slot 55 but
  ran accepted=[2] => init_col=1 => the kernel read slot 56 and the NaN
  bytes were never loaded.
Rev2 fixes: every run (batch, solo, control, re-run) starts from PRISTINE
pool + conv clones; L4 uses accepted=[1] => init_col=0 => indices[0]=55 is
on the load path.

  L1 SOLO_VS_BATCH (spec op, per dtype fp16/e4m3/e5m2): nsd=4 requests,
     nst=5, accepted=[2,4,1,3] (init_cols 1/3/0/2 — every token-0 seed
     column exercised), distinct slots 40..59. Batch vs 4 solos, each from
     identical pristine state. GATE: outputs bitwise equal per request AND
     final pool rows AND final conv rows at the request's slots bitwise
     equal (grid dim0 only adds independent work-groups; proves S>1 + fp8
     chain isolation + read-before-write seed semantics).
  L2 FP8_VS_FP16 (spec op): same pristine-state inputs, fp8 pool vs fp16
     pool (fresh conv per run). GATE: finite, cosine > 0.99 per batch,
     max|diff| <= 0.25 * (1 + max|out16|).
  L3 SEQ_OP_SMOKE (non-spec seq op, fp8 pools): scattered distinct slots;
     pool + conv rows written == slot set, others untouched; re-run from
     pristine clones bitwise deterministic.
  L4 E4M3_NAN_PARITY: accepted=[1], indices[0]=55 pre-filled with bytes
     0x7F/0xFF => token 0 loads them through esimd_e4m3_load => outputs
     must be NaN (c10 parity — the v125 load returned finite 480.0). fp16
     control with finite pool stays finite.

Run inside a lane container with the v129 .so installed, serve DOWN.
"""
import sys

import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5                                    # production MTP x4 -> W=5
NSD = 4
ACCEPTED = [2, 4, 1, 3]
SPEC_SLOTS = list(range(40, 40 + NSD * NST))   # distinct per column
SCALE = float(K ** -0.5)
DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]

results = []


def record(leg, ok, detail=""):
    results.append((leg, ok))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", leg,
                           (" — " + detail) if detail else ""))


def params(seed):
    torch.manual_seed(seed)
    cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
    cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
    A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
    dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
    conv = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
    pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05
    return cw, cb, A_log16, dt_bias, conv, pool16


def fresh_pool(pool16, dt):
    """Pristine pool from the same source bytes for every run."""
    return pool16.clone() if dt == torch.float16 \
        else pool16.to(dt).contiguous()


def run_spec(op, dt, pool, conv, qkvz, ba, rows_idx, accepted):
    """One esimd_gdn_conv_fused_seq_spec call; caller passes PRISTINE pool
    + conv clones; returns (out, z)."""
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    op(qkvz, conv, cw_g[0], cw_g[1], idx, cw_g[2], cw_g[3], ba, pool,
       out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out, z


def bytes_view(t):
    return t.view(torch.uint8) if t.dtype != torch.float16 \
        else t.view(torch.int16)


print("=== P29B op-level correctness (v129 kernels, harness rev2) ===")
from custom_esimd_kernels_vllm import (  # noqa: E402
    esimd_gdn_conv_fused_seq,
    esimd_gdn_conv_fused_seq_spec,
)

cw_g = None
cw, cb, A_log16, dt_bias, conv0, pool16 = params(2929)
cw_g = (cw, cb, A_log16, dt_bias)

torch.manual_seed(31337)
qkvzB = torch.randn(NSD * NST, DIM, dtype=torch.float16, device=DEV) * 0.1
baB = torch.randn(NSD * NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5

# ------------------------------- L1 ---------------------------------------
for name, dt in DTYPES:
    pool_b = fresh_pool(pool16, dt)
    conv_b = conv0.clone()
    outB, _ = run_spec(esimd_gdn_conv_fused_seq_spec, dt, pool_b, conv_b,
                       qkvzB, baB, SPEC_SLOTS, ACCEPTED)
    bad = -1
    for i in range(NSD):
        pool_s = fresh_pool(pool16, dt)
        conv_s = conv0.clone()
        rows = qkvzB[i * NST:(i + 1) * NST].contiguous()
        bas = baB[i * NST:(i + 1) * NST].contiguous()
        slots_i = SPEC_SLOTS[i * NST:(i + 1) * NST]
        outS, _ = run_spec(esimd_gdn_conv_fused_seq_spec, dt, pool_s, conv_s,
                           rows, bas, slots_i, [ACCEPTED[i]])
        sl = torch.tensor(slots_i, device=DEV)
        o_eq = torch.equal(outS, outB[i * NST:(i + 1) * NST])
        p_eq = torch.equal(bytes_view(pool_s[sl]), bytes_view(pool_b[sl]))
        c_eq = torch.equal(conv_s[sl], conv_b[sl])
        if not (o_eq and p_eq and c_eq):
            bad = i
            break
    record("L1 solo_vs_batch spec %s" % name, bad < 0,
           "out+pool+conv bitwise" if bad < 0
           else "MISMATCH req=%d (out=%s pool=%s conv=%s)"
                % (bad, o_eq, p_eq, c_eq))

# ------------------------------- L2 ---------------------------------------
outs16 = None
for name, dt in [("e4m3", torch.float8_e4m3fn),
                 ("e5m2", torch.float8_e5m2)]:
    pool8 = fresh_pool(pool16, dt)
    conv8 = conv0.clone()
    out8, _ = run_spec(esimd_gdn_conv_fused_seq_spec, dt, pool8, conv8,
                       qkvzB, baB, SPEC_SLOTS, ACCEPTED)
    pool16r = fresh_pool(pool16, torch.float16)
    conv16 = conv0.clone()
    out16, _ = run_spec(esimd_gdn_conv_fused_seq_spec, torch.float16,
                        pool16r, conv16, qkvzB, baB, SPEC_SLOTS, ACCEPTED)
    outs16 = out16
    a, b = out8.float().flatten(), out16.float().flatten()
    finite = torch.isfinite(a).all().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    mx = (a - b).abs().max().item()
    bound = 0.25 * (1 + b.abs().max().item())
    record("L2 fp8_vs_fp16 spec %s" % name,
           finite and cos > 0.99 and mx <= bound,
           "cos=%.5f maxdiff=%.5f bound=%.5f" % (cos, mx, bound))

# ------------------------------- L3 ---------------------------------------
SEQ_SLOTS = [3, 17, 99, 201, 300, 411, 128, 7]
n_dec = len(SEQ_SLOTS)
torch.manual_seed(4242)
qkvzD = torch.randn(n_dec, DIM, dtype=torch.float16, device=DEV) * 0.1
baD = torch.randn(n_dec, 2 * HV, dtype=torch.float16, device=DEV) * 0.5


def run_seq(dt, pool, conv):
    out = torch.zeros(n_dec, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n_dec, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(SEQ_SLOTS, dtype=torch.int32, device=DEV)
    esimd_gdn_conv_fused_seq(qkvzD, conv, cw, cb, idx, A_log16, dt_bias,
                             baD, pool, idx, out, z, n_dec, H, HV, K, V,
                             SCALE)
    torch.xpu.synchronize()
    return out, z, pool, conv


for name, dt in [("e4m3", torch.float8_e4m3fn),
                 ("e5m2", torch.float8_e5m2)]:
    pool1, conv1 = fresh_pool(pool16, dt), conv0.clone()
    out1, _, _, _ = run_seq(dt, pool1, conv1)
    pool1a, conv1a = fresh_pool(pool16, dt), conv0.clone()
    changed = ((bytes_view(pool1) != bytes_view(pool1a)).any(dim=(1, 2, 3))
               .nonzero().flatten().tolist())
    rows_ok = set(changed) == set(SEQ_SLOTS)
    finite = torch.isfinite(out1.float()).all().item()
    out2, _, pool2, conv2 = run_seq(dt, fresh_pool(pool16, dt), conv0.clone())
    det = (torch.equal(out1, out2)
           and torch.equal(bytes_view(pool1), bytes_view(pool2))
           and torch.equal(conv1, conv2))
    record("L3 seq_smoke %s" % name, rows_ok and finite and det,
           "rows=%s det=%s" % (rows_ok, det))

# ------------------------------- L4 ---------------------------------------
slot = 55
torch.manual_seed(99)
qkvz1 = torch.randn(NST, DIM, dtype=torch.float16, device=DEV) * 0.1
ba1 = torch.randn(NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5
# accepted=[1] => accepted_prev=0 => init_col=0 => token 0 loads indices[0]
slots_L4 = [slot, slot + 1, slot + 2, slot + 3, slot + 4]
poolN = fresh_pool(pool16, torch.float8_e4m3fn)
pb = poolN.view(torch.uint8)
for code in (0x7F, 0xFF):
    pb[slot].fill_(code)
outN, _ = run_spec(esimd_gdn_conv_fused_seq_spec, torch.float8_e4m3fn,
                   poolN, conv0.clone(), qkvz1, ba1, slots_L4, [1])
nan8 = torch.isnan(outN.float()).any().item()
poolC = fresh_pool(pool16, torch.float16)
outC, _ = run_spec(esimd_gdn_conv_fused_seq_spec, torch.float16,
                   poolC, conv0.clone(), qkvz1, ba1, slots_L4, [1])
fin16 = torch.isfinite(outC.float()).all().item()
record("L4 e4m3 NaN parity", nan8 and fin16,
       "0x7F/0xFF@indices[0]->NaN=%s fp16 finite=%s" % (nan8, fin16))

ok = all(o for _, o in results)
print("P29B_VERDICT: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
