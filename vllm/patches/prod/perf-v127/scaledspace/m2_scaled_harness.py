#!/usr/bin/env python3
"""m2_scaled_harness.py — v127 M2 validation harness for the scaled-space
e4m3 spec kernel (custom_esimd_kernels_lgrf.so with
esimd_gdn_conv_fused_seq_spec_scaled).

Runs INSIDE a throwaway GPU container (never on the lane):
  docker run --rm --device /dev/dri -v /root/build/v132_so:/so \
    -v /root/build/v127_stage/scaledspace:/h \
    intel/omix:0.1.0-devel-ubuntu24.04 \
    bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; \
             /opt/venv/bin/python /h/m2_scaled_harness.py /so/custom_esimd_kernels_lgrf.so'

STORAGE CONVENTION (M1 == M2): a scaled-space pool stores
e4m3(h * scale[hv,k]) bytes. Every scaled-kernel run therefore starts
from a bridge-encoded pool; the raw-e4m3 production op starts from
e4m3(h) bytes; the fp16 reference from h itself.

GATES
  A  identity-at-unit-scale: scaled(scales=1) must be BITWISE identical
     to the production esimd_gdn_conv_fused_seq_spec on an identically
     initialized e4m3 pool (outputs, z_out, and final pool bytes).
     Proves the transplant; neutralizes the scale.
  B  boundary idempotence: every pool byte after a scaled run decodes
     (*inv_scale) and re-encodes (*scale, RNE) to the same byte — up to
     fp32 boundary flips, which must be < 1e-5 of elements and within
     one e4m3 grid step. Proves the stored-space convention end-to-end.
  C  end-to-end superiority: on magnitude-stratified states (per-feature
     runmax spanning ~1e-3..30 — the P23F regime), the scaled run's
     output error vs the fp16-pool truth must be <= the raw-e4m3 run's
     error (mean AND max, STRICT). On uniform states both formats use
     the same relative grid, so the unif leg asserts PARITY within a
     5 % sampling band (a strict <= there is a seed coin-flip).
  D  storage superiority (M1-compat core): decoding a bridge-encoded
     pool (*inv_scale) is closer to the fp16 truth than decoding a raw
     e4m3 pool, on stratified states (mean AND max, STRICT) — the P23F
     fix, measured at the storage layer; unif leg = parity band as in C.

Exit 0 iff all gates pass. Prints per-gate verdicts + a speed spot.
"""
import sys
import time

import torch

SO = sys.argv[1] if len(sys.argv) > 1 else "/so/custom_esimd_kernels_lgrf.so"
torch.ops.load_library(SO)
OP = torch.ops.custom_esimd_kernels_vllm
DEV = torch.device("xpu")

# Qwen3.5/3.6 TP=2 geometry accepted by the kernel's TORCH_CHECK.
H, HV, K, V = 8, 16, 128, 128
NSD, NST = 2, 4          # spec decode sequences x spec tokens (MTPx4)
DIM = 2 * H * K + HV * V  # sequential qkvz width
NTOK = NSD * NST
NSS = NTOK + 4           # pool slots (distinct per-column ids + slack)
SCALE = 1.0 / (H ** 0.5)

gates = {}


def make_inputs(seed, stratified):
    g = torch.Generator(device="cpu").manual_seed(seed)
    qkvz = torch.randn(NTOK, DIM, generator=g).to(torch.float16)
    conv_state = torch.randn(NSS, 3, DIM, generator=g).to(torch.float16)
    conv_weight = (torch.randn(DIM, 4, generator=g) * 0.25).to(torch.float16)
    conv_bias = (torch.randn(DIM, generator=g) * 0.1).to(torch.float16)
    A_log = torch.randn(HV, generator=g).to(torch.float16)
    dt_bias = torch.randn(HV, generator=g).to(torch.float16)
    ba = torch.randn(NTOK, 2 * HV, generator=g).to(torch.float16)
    spec_idx = torch.arange(NTOK, dtype=torch.int32)
    token_indx = torch.arange(NTOK, dtype=torch.int32)
    accepted = torch.full((NSD,), NST, dtype=torch.int32)
    if not stratified:
        ssm = torch.randn(NSS, HV, V, K, generator=g).to(torch.float16)
    else:
        # P23F regime: per-(hv,k) feature magnitudes span ~1e-3..30 —
        # small features flush under raw e4m3, every feature sits
        # in-range once scaled. Shape [1,HV,1,K] broadcasts (hv,k) across
        # (nss,v); [HV,1,1,K] would right-align HV onto NSS.
        mag = torch.pow(10.0, torch.rand(1, HV, 1, K, generator=g) * 4.0 - 3.0)
        ssm = (torch.randn(NSS, HV, V, K, generator=g) * mag).to(torch.float16)
    return dict(
        qkvz=qkvz, conv_state=conv_state, conv_weight=conv_weight,
        conv_bias=conv_bias, A_log=A_log, dt_bias=dt_bias, ba=ba,
        spec_idx=spec_idx, token_indx=token_indx, accepted=accepted, ssm=ssm)


def run(op_name, ssm_pool, scales=None):
    x = make_inputs(SEED, STRAT)  # same seed => identical non-pool inputs
    out = torch.zeros(NTOK, HV, V, dtype=torch.float16)
    zout = torch.zeros(NTOK, HV, V, dtype=torch.float16)
    kw = dict(
        qkvz=x["qkvz"].to(DEV), conv_state=x["conv_state"].to(DEV),
        conv_weight=x["conv_weight"].to(DEV), conv_bias=x["conv_bias"].to(DEV),
        spec_state_indices=x["spec_idx"].to(DEV), A_log=x["A_log"].to(DEV),
        dt_bias=x["dt_bias"].to(DEV), ba=x["ba"].to(DEV),
        ssm_state=ssm_pool,
        output=out, z_out=zout, token_indx=x["token_indx"].to(DEV),
        num_accepted_tokens=x["accepted"].to(DEV),
        num_spec_decodes=NSD, num_spec_tokens=NST,
        H=H, HV=HV, K=K, V=V, scale=SCALE)
    if scales is not None:
        kw["ssm_scales"] = scales
    getattr(OP, op_name)(**kw)
    torch.xpu.synchronize()
    return out, zout, ssm_pool


def derive_scales(pool_fp16):
    """M0 rule: scale[hv,k] = 0.98 * 448 / runmax[hv,k] (fp32)."""
    runmax = pool_fp16.abs().amax(dim=(0, 2)).float().clamp_min(1e-30)
    return (0.98 * 448.0 / runmax).contiguous()


def encode_bridge(h_fp16, scales):
    """Bridge write semantics (M1 == M2): stored = e4m3(h * scale)."""
    sc = scales.view(1, HV, 1, K)
    return (h_fp16.float() * sc).to(torch.float8_e4m3fn)


def decode_bridge(pool_e4m3, scales):
    """Bridge read semantics: h = fp32(stored) * (1/scale)."""
    inv = (1.0 / scales).view(1, HV, 1, K)
    return pool_e4m3.float() * inv


print(f"[m2] device={DEV} so={SO}")
for SEED, STRAT in ((1234, False), (777, True)):
    tag = "strat" if STRAT else "unif"
    pool_fp16 = make_inputs(SEED, STRAT)["ssm"]
    scales = derive_scales(pool_fp16)
    sc_dev, ones_dev = scales.to(DEV), torch.ones_like(scales).to(DEV)

    # ---- GATE A: identity at unit scale (uniform seed) ----
    if not STRAT:
        p_raw = pool_fp16.to(DEV).to(torch.float8_e4m3fn).clone()
        p_scaled = pool_fp16.to(DEV).to(torch.float8_e4m3fn).clone()
        o1, z1, _ = run("esimd_gdn_conv_fused_seq_spec", p_raw)
        o2, z2, _ = run("esimd_gdn_conv_fused_seq_spec_scaled", p_scaled,
                        ones_dev)
        a_out = bool(torch.equal(o1, o2) and torch.equal(z1, z2))
        a_pool = bool(torch.equal(p_raw.view(torch.uint8),
                                  p_scaled.view(torch.uint8)))
        gates["A_identity"] = a_out and a_pool
        print(f"[m2] A identity seed={SEED}: out/z={a_out} pool={a_pool}")

    # ---- GATE B: boundary idempotence on the scaled run's pool ----
    p_sc = encode_bridge(pool_fp16, scales).to(DEV).clone()
    o_sc, _, _ = run("esimd_gdn_conv_fused_seq_spec_scaled", p_sc, sc_dev)
    p_cpu = p_sc.cpu()  # analysis host-side; kernel pool stays on DEV
    dec = decode_bridge(p_cpu, scales)
    re_enc = (dec * scales.view(1, HV, 1, K)).to(torch.float8_e4m3fn)
    mism = (re_enc.view(torch.uint8) != p_cpu.view(torch.uint8))
    n_mism = int(mism.sum())
    frac = n_mism / mism.numel()
    if n_mism == 0:
        b_ok = True
    else:
        # fp32 decode*inv*re-encode may flip an RNE boundary when a value
        # sits within ~2 ulp_fp32 of a grid midpoint; accept only
        # one-grid-step adjacency at a rate < 1e-5.
        d = (re_enc.float() - p_cpu.float()).abs()
        step = p_cpu.float().abs().clamp_min(2.0 ** -9) * (2.0 ** -3) * 1.5
        b_ok = bool((d[mism] <= step[mism]).all())
    gates[f"B_idempotence_{tag}"] = b_ok and frac < 1e-5
    print(f"[m2] B idempotence seed={SEED}: mism={n_mism} ({frac:.2e}) "
          f"ulp-adjacent={b_ok}")

    # ---- GATE C: end-to-end superiority vs raw e4m3, fp16 truth ----
    p_fp16 = pool_fp16.to(DEV).clone()
    p_raw = pool_fp16.to(DEV).to(torch.float8_e4m3fn).clone()
    p_sc2 = encode_bridge(pool_fp16, scales).to(DEV).clone()
    o_ref, _, _ = run("esimd_gdn_conv_fused_seq_spec", p_fp16)
    o_raw, _, _ = run("esimd_gdn_conv_fused_seq_spec", p_raw)
    o_sc2, _, _ = run("esimd_gdn_conv_fused_seq_spec_scaled", p_sc2, sc_dev)
    err_raw = (o_raw.float() - o_ref.float()).abs()
    err_sc = (o_sc2.float() - o_ref.float()).abs()
    # Strat = the P23F regime: STRICT superiority is the project's claim.
    # Unif = same relative grid for both formats: parity within a 5 %
    # sampling band (quantization-error mean over 16k elements has ~1 %
    # SE; a strict <= there is a seed coin-flip, not a gate).
    band = 1.0 if STRAT else 1.05
    c_mean = bool(err_sc.mean() <= err_raw.mean() * band)
    c_max = bool(err_sc.max() <= err_raw.max() * band)
    gates[f"C_superiority_{tag}"] = c_mean and c_max
    print(f"[m2] C seed={SEED} tag={tag} band={band}: "
          f"mean scaled={err_sc.mean():.5f} raw={err_raw.mean():.5f} ok={c_mean} | "
          f"max scaled={err_sc.max():.4f} raw={err_raw.max():.4f} ok={c_max}")

    # ---- GATE D: storage superiority (P23F fix, storage layer) ----
    # Fresh encodes — the GATE-C runs mutated p_raw/p_sc2 in place.
    truth = pool_fp16.float()
    d_raw = (pool_fp16.to(torch.float8_e4m3fn).float() - truth).abs()
    d_sc = (decode_bridge(encode_bridge(pool_fp16, scales), scales)
            - truth).abs()
    d_mean = bool(d_sc.mean() <= d_raw.mean() * band)
    d_max = bool(d_sc.max() <= d_raw.max() * band)
    gates[f"D_storage_{tag}"] = d_mean and d_max
    print(f"[m2] D storage seed={SEED} tag={tag} band={band}: "
          f"mean scaled={d_sc.mean():.6f} raw={d_raw.mean():.6f} ok={d_mean} | "
          f"max scaled={d_sc.max():.4f} raw={d_raw.max():.4f} ok={d_max}")

# ---- speed spot (informational; formal benches need a quiet window) ----
pool_fp16 = make_inputs(42, True)["ssm"]
scales = derive_scales(pool_fp16)
p_raw = pool_fp16.to(DEV).to(torch.float8_e4m3fn).clone()
p_sc = encode_bridge(pool_fp16, scales).to(DEV).clone()
sc_dev = scales.to(DEV)
for name, op, pool, sc in (("raw", "esimd_gdn_conv_fused_seq_spec", p_raw, None),
                           ("scaled", "esimd_gdn_conv_fused_seq_spec_scaled",
                            p_sc, sc_dev)):
    SEED, STRAT = 42, True
    for _ in range(20):
        run(op, pool, sc)
    t0 = time.perf_counter()
    for _ in range(200):
        run(op, pool, sc)
    torch.xpu.synchronize()
    print(f"[m2] speed spot {name}: {(time.perf_counter()-t0)/200*1e6:.0f} us/launch")

ok = all(gates.values())
print("[m2] GATES " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                               for k, v in gates.items()))
print("M2_HARNESS_" + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
