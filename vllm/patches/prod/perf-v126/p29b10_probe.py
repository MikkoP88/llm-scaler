#!/usr/bin/env python3
"""p29b10_probe.py — v126 P29B10: double_v geometry discriminator.

All HV=24 failures implicate hv>=8. HV=24 forces double_v (24 > 32/2):
each V tid owns a FULL head via lo+hi 64-chunks (conv_result_hi, hi
checkpoint stores, hi SLM store). HV=16 takes the NON-double path
(v_tid/2 mapping, single chunk per tid) -- the production 2TP geometry
variant. If nsd=2 batch-vs-solo PASSES at HV=16, the defect is confined
to the double_v code paths; if it FAILS, it is geometry-independent
(scheduling/SLM-level).
Also re-checks HV=16 with hv bands and zero-input legs.
"""
import torch

K = V = 128
H = 8
DEV = "xpu"
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DT = torch.float8_e4m3fn

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa


def run(HV):
    DIM = 2 * H * K + 2 * HV * V
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
        idx = torch.tensor(rows_idx, dtype=torch.int32,
                           device=DEV).reshape(n, NST)
        tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
        acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
        esimd_gdn_conv_fused_seq_spec(
            qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
            out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
        torch.xpu.synchronize()
        return out

    def fp8(t):
        return t.to(DT).contiguous()

    S1 = list(range(45, 50))
    oB = spec(fp8(pool16), conv0.clone(), qkvzB[0:2 * NST].contiguous(),
              baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
    oS = spec(fp8(pool16), conv0.clone(), qkvzB[NST:2 * NST].contiguous(),
              baB[NST:2 * NST].contiguous(), S1, [4])
    per_hv = "".join(
        "o" if torch.equal(oB[NST, hv], oS[0, hv]) else "X"
        for hv in range(HV))
    allT = "".join(
        "o" if torch.equal(oB[NST + t], oS[t]) else "X" for t in range(NST))
    # zero-input leg
    oBz = spec(fp8(torch.zeros_like(pool16)), conv0.clone(),
               torch.zeros_like(qkvzB)[0:2 * NST].contiguous(),
               baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
    oSz = spec(fp8(torch.zeros_like(pool16)), conv0.clone(),
               torch.zeros_like(qkvzB)[NST:2 * NST].contiguous(),
               baB[NST:2 * NST].contiguous(), S1, [4])
    per_hv_z = "".join(
        "o" if torch.equal(oBz[NST, hv], oSz[0, hv]) else "X"
        for hv in range(HV))
    print("HV=%2d  t0-band %s   by-token %s   zero-input-band %s"
          % (HV, per_hv, allT, per_hv_z))


run(16)
run(24)
print("PROBE10_DONE")
