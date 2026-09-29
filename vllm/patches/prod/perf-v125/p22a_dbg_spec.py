#!/usr/bin/env python3
"""p22a_dbg_spec.py — empirical localization of the e4m3 spec-op defect.
Runs INSIDE lsv-test with the P22A .so swapped in (serve must be down).

Ground-truth facts used:
  z_out = RAW copy of the qkvz z-slice (kernel: block_load(z_off) ->
  block_store(z_dst), no activation) -> comparable EXACTLY per token row.
  out   = zero-filled before call -> any all-zero 64-wide half = unwritten.
  pool  = init'd deterministically -> NaN count + slot map per variant.

Variants sweep the two index tensors + num_accepted:
  A: all tokens same slot 7 (v1/v2 smoke shape)
  B: distinct save slots per token (7,8,9,10,11)
  C: nst sweep 1..5 (same-slot)
  D: num_accepted sweep 1..5 (same-slot, nst=5)
Prints a compact report per variant.
"""
import torch

K = V = 128
H, HV = 8, 24
NSD = 1
NUM_CACHE = 32
DEV = "xpu"
DIM = 2 * H * K + 2 * HV * V
Z_BASE = 2 * H * K + HV * V
scale = float(K ** -0.5)

torch.manual_seed(1234)
qkvz = torch.randn(8, DIM, dtype=torch.float16, device=DEV) * 0.1
ba = torch.randn(8, 2 * HV, dtype=torch.float16, device=DEV) * 0.5
conv_weight = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
conv_bias = torch.zeros(DIM, dtype=torch.float16, device=DEV)
A_log = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
conv0 = torch.randn(NUM_CACHE, 3, DIM, dtype=torch.float16, device=DEV) * 0.01

z_expected = qkvz[:, Z_BASE:].clone()  # raw copy contract

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec


def run(nst, slots, accepted, dt):
    rows = NSD * nst
    # token_indx = packed-row POSITIONS (kernel indexes qkvz+ba+out+z rows
    # by it); spec_state_indices = pool SLOTS. v1 dbg passed slots as
    # token_indx -> every out/z write landed at row `slot`, past the end.
    spec_idx = torch.tensor([slots[:nst]], dtype=torch.int32, device=DEV)
    tok_idx = torch.arange(rows, dtype=torch.int32, device=DEV)
    nacc = torch.tensor([accepted], dtype=torch.int32, device=DEV)

    pool = torch.zeros(NUM_CACHE, HV, V, K, dtype=dt, device=DEV)
    canary = torch.full((4096,), 7.0, dtype=torch.float16, device=DEV)
    out = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
    conv = conv0.clone()

    ptrs = (pool.data_ptr(), out.data_ptr(), z.data_ptr(), canary.data_ptr())

    esimd_gdn_conv_fused_seq_spec(
        qkvz[:rows], conv, conv_weight, conv_bias, spec_idx,
        A_log, dt_bias, ba[:rows], pool, out, z,
        tok_idx, nacc, NSD, nst,
        H, HV, K, V, scale)
    torch.xpu.synchronize()

    # out coverage: nonzero element counts per 64-wide half of V
    nz_lo = (out[:, :, :64] != 0).sum().item()
    nz_hi = (out[:, :, 64:] != 0).sum().item()
    # z exact-copy contract
    zd = (z.float() - z_expected[:rows].view(rows, HV, V).float()).abs()
    zbad = (zd > 1e-6)
    zcnt = zbad.sum().item()
    zloc = "-"
    if zcnt:
        idx = zbad.nonzero()[0].tolist()
        zloc = "row%d,hv%d,col%d d=%.4g" % (idx[0], idx[1], idx[2],
                                            zd[ idx[0], idx[1], idx[2]].item())
    # pool NaN map
    pf = pool.to(torch.float16).float()
    nancnt = torch.isnan(pf).sum().item()
    nanloc = "-"
    if nancnt:
        m = torch.isnan(pf)
        idx = m.nonzero()[0].tolist()
        nanloc = "slot%d,hv%d,v%d,k%d" % tuple(idx)
        # per-slot NaN counts
        per = m.sum(dim=(1, 2, 3))
        nanloc += " slots=" + str({i: int(per[i]) for i in range(NUM_CACHE) if per[i]})
    canary_bad = (canary != 7.0).sum().item()
    # pool bytes actually written (pool inits to zeros; e5m2 writes are
    # invisible to a NaN-only scan)
    pnz = int((pool.view(torch.uint8) != 0).sum().item())
    # divergence probe: decode saved state, magnitude + big-value count
    pv = pool.to(torch.float16).float()
    pmax = pv.abs().max().item()
    pbig = int((pv.abs() > 10).sum().item())
    print("  nst=%d slots=%s acc=%d %s | out nz lo/hi=%d/%d | zbad=%d @%s | poolNaN=%d @%s | pnz=%d pmax=%.4g pbig=%d | canary=%d"
          % (nst, slots[:nst], accepted, str(dt).split(".")[-1],
             nz_lo, nz_hi, zcnt, zloc, nancnt, nanloc, pnz, pmax, pbig,
             canary_bad))
    print("    ptrs pool=%x out=%x z=%x canary=%x" % ptrs)
    return out, z, pool


print("=== A/B: nst=5, same-slot vs distinct slots, e4m3 vs e5m2 ===")
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    run(5, [7] * 5, 3, dt)                            # A same-slot (smoke shape)
    run(5, [7, 8, 9, 10, 11], 3, dt)                  # B distinct saves

print("=== C: nst sweep (same-slot, acc=nst) ===")
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    for nst in (1, 2, 3, 5):
        run(nst, [7] * nst, nst, dt)

print("=== D: accepted sweep (nst=5 same-slot) ===")
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    for acc in (1, 2, 3, 4, 5):
        run(5, [7] * 5, acc, dt)

print("DBG_DONE")
