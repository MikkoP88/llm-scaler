#!/usr/bin/env python3
"""p29b8_probe.py — v126 P29B8: input-nulling discriminators (e4m3).

Race confirmed; novel values; invariant to other seqs' data. Two nulling
tests isolate WHICH input channel carries the corruption into (seq>=1,
hv>=8) work-groups at t=0:
  T1  ALL-ZERO POOL: every possible h-load address reads 0.
      batch==solo  -> h-load path is the vector (address or coherence);
      still differ -> SLM q/k/v or scalar inputs.
  T2  ALL-ZERO QKVZ: conv_result -> silu(bias) constants; SLM carries
      constants; q/k/v math constant. batch==solo -> the staging was the
      data-dependent vector; still differ -> h/scalar path.
  T3  zero pool AND zero qkvz: full-constant inputs except h chaining.
  T4  sanity: with null inputs the SOLO must equal a batch where nsd=1
      (validates the null tests themselves).
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


def fp8(t):
    return t.to(DT).contiguous()


S1 = list(range(45, 50))
R1 = qkvzB[NST:2 * NST].contiguous()
B1 = baB[NST:2 * NST].contiguous()


def run_case(zero_pool, zero_qkvz, tag):
    pool_src = torch.zeros_like(pool16) if zero_pool else pool16
    qkv_src = torch.zeros_like(qkvzB) if zero_qkvz else qkvzB
    oB = spec(fp8(pool_src), conv0.clone(), qkv_src[0:2 * NST].contiguous(),
              baB[0:2 * NST].contiguous(), list(range(40, 50)), [2, 4])
    oS = spec(fp8(pool_src), conv0.clone(), qkv_src[NST:2 * NST].contiguous(),
              baB[NST:2 * NST].contiguous(), S1, [4])
    per_hv = "".join(
        "o" if torch.equal(oB[5, hv], oS[0, hv]) else "X"
        for hv in range(HV))
    # T4 sanity: solo twice
    oS2 = spec(fp8(pool_src), conv0.clone(), qkv_src[NST:2 * NST].contiguous(),
               baB[NST:2 * NST].contiguous(), S1, [4])
    print("%-28s seq1-t0 vs solo: %s   solo_det=%s"
          % (tag, per_hv, torch.equal(oS, oS2)))


run_case(True, False, "T1 zero-pool")
run_case(False, True, "T2 zero-qkvz")
run_case(True, True, "T3 zero-pool+zero-qkvz")
print("PROBE8_DONE")
