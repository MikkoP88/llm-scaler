#!/usr/bin/env python3
"""p22c1_traj.py (v125 P22-C C1) — recurrent trajectory differential at
op level. WHY: single-call op tests (p22a_smoke T0-T3) pass for both
fp8 dtypes, yet the e4m3ns SERVE leg collapsed (repetition loops) while
B2's wheel-kernel e4m3 legs stayed coherent. A single call cannot catch
error COMPOUNDING through the state recurrence:
    S(t) = S(t-1) * decay(t) + outer(k_t, v_t)     [per head, VxK]
With an fp8 pool the state is dequantized->computed->requantized EVERY
decode step; a per-step error eps can stay bounded (|noise floor|),
random-walk (eps*sqrt(t)), or amplify (eps*t^alpha, alpha>0.5) — only a
multi-step trajectory with state carried in the pool tensor can tell.

Design (mirrors the production call site gdn_linear_attn.py + smoke):
  T successive single-token calls of esimd_gdn_conv_fused_seq, pool
  carried IN-PLACE (production solo decode = 1 token/call under
  cudagraph). Inputs and initial pool image are IDENTICAL across cells;
  the ONLY variable is pool dtype:
    A16 fp16 pool (reference — the coherent-in-production semantics)
    A4  e4m3 pool (collapsed in serve)
    A5  e5m2 pool (serve quality still unmeasured)
  Initial fp8 pool = quant(A16 init), so the seeds differ by exactly the
  quantization noise floor (measured and reported).
  Params stay fp16 in ALL cells (A_log16/dt_bias as the call site feeds
  them) — pool dtype is isolated as the single variable.

Reported per cell vs A16: d_state(t), d_out(t) (relative inf-norm),
reference drift, floor; growth class at t=64; NaN/degenerate scan.
Section 2 repeats the experiment on the SPEC op (nsd=1, nst=5, cycling
num_accepted 1..4) — the kernel the certified spec-4 lane itself runs.

PASS criteria (script exit 0): determinism control exact; no NaNs; no
AMPLIFIED/DEGENERATE verdict on either fp8 dtype in either section.
Note: BOUNDED/RANDOM-WALK with d_state ~<10x floor = benign -> the
serve-level collapse then points AWAY from pool dtype (H2 direction).
"""
import sys
import torch

K = V = 128
H, HV = 8, 24
DEV = "xpu"
T_SEQ = 64
T_SPEC = 32
SLOT = 0
NC = 32  # pool slots (only SLOT used)

VERDICTS = []


def rel(a, b):
    """relative inf-norm distance a vs b (float views)."""
    af, bf = a.float(), b.float()
    num = (af - bf).abs().max().item()
    den = max(af.abs().max().item(), 1e-6)
    return num / den


def make_streams(steps, seed):
    """one qkvz/ba row per recurrent step, production scales (smoke).
    Global manual_seed — the pattern the smoke proved in this container
    (torch.Generator(device='xpu') is unproven here; don't risk it)."""
    DIM = 2 * H * K + 2 * HV * V
    torch.manual_seed(seed)
    qkvz = (torch.randn(steps, DIM, dtype=torch.float16, device=DEV) * 0.1)
    ba = (torch.randn(steps, 2 * HV, dtype=torch.float16, device=DEV) * 0.5)
    cw = (torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1)
    cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
    A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
    dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
    init_conv = torch.randn(NC, 3, 2 * 8 * K + 2 * HV * V,
                            dtype=torch.float16, device=DEV) * 0.01
    init_ssm16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05
    return qkvz, ba, cw, cb, A_log16, dt_bias, init_conv, init_ssm16


def classify(d0, d_last, floor):
    # T=64: benign random walk ends at ~8x floor (sqrt(64)); linear
    # amplification ends at ~64x. 16x separates them with margin.
    if d_last > 16 * max(d0, floor) and d_last > 20 * floor:
        return "AMPLIFIED"
    if d_last > 2.5 * max(d0, floor):
        return "RANDOM-WALK"
    return "BOUNDED"


# ---------------------------------------------------------------- section 1
def seq_trajectory(dt, name, qkvz, ba, cw, cb, A_log16, dt_bias,
                   init_conv, init_ssm16, ref=None):
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq
    conv = init_conv.clone()
    pool = init_ssm16.clone() if dt == torch.float16 else init_ssm16.to(dt)
    idx = torch.full((1,), SLOT, dtype=torch.int32, device=DEV)
    scale = float(K ** -0.5)
    out = torch.zeros(1, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(1, HV, V, dtype=torch.float16, device=DEV)
    snaps = []
    douts = []
    for t in range(T_SEQ):
        esimd_gdn_conv_fused_seq(
            qkvz[t:t + 1].contiguous(), conv, cw, cb, idx,
            A_log16, dt_bias, ba[t:t + 1].contiguous(), pool, idx,
            out, z, 1, H, HV, K, V, scale)
        torch.xpu.synchronize()
        snaps.append(pool[SLOT].float().clone())
        douts.append(out.float().clone())
    st = torch.stack(snaps)          # [T, HV, V, K]
    ot = torch.stack(douts)          # [T, HV, V]
    nan = bool(torch.isnan(st).any() or torch.isnan(ot).any())
    mx = st.abs().max().item()
    line = f"[seq {name}] max|S|={mx:.4f} nan={nan}"
    if ref is not None:
        rs, ro = ref
        d0 = rel(st[0], rs[0])
        floor = rel(init_ssm16[SLOT].to(dt).float(), init_ssm16[SLOT].float())
        dstate = [rel(st[i], rs[i]) for i in (1, 8, 32, 63)]
        d_out63 = rel(ot[63], ro[63])
        drift = rel(rs[63], rs[0])
        cls = classify(d0, dstate[3], floor)
        VERDICTS.append((f"seq {name}", cls, nan))
        line += (f" floor={floor:.5f} d_state(t1,t8,t32,t64)="
                 f"{dstate[0]:.5f},{dstate[1]:.5f},{dstate[2]:.5f},{dstate[3]:.5f}"
                 f" d_out(64)={d_out63:.5f} ref_drift={drift:.4f} -> {cls}")
    print(line)
    return (st, ot)


# ---------------------------------------------------------------- section 2
def spec_trajectory(dt, name, qkvz, ba, cw, cb, A_log16, dt_bias,
                    init_conv, init_ssm16, ref=None):
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec
    conv = init_conv.clone()
    pool = init_ssm16.clone() if dt == torch.float16 else init_ssm16.to(dt)
    nsd, nst = 1, 5
    rows = nsd * nst
    scale = float(K ** -0.5)
    ssi = torch.full((nsd, nst), SLOT, dtype=torch.int32, device=DEV)
    tix = torch.arange(rows, dtype=torch.int32, device=DEV)
    out = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
    snaps = []
    for t in range(T_SPEC):
        na = (t % 4) + 1  # cycle acceptance 1..4 (spec depth 4)
        num_acc = torch.tensor([na], dtype=torch.int32, device=DEV)
        esimd_gdn_conv_fused_seq_spec(
            qkvz[t * rows:(t + 1) * rows].contiguous(), conv, cw, cb, ssi,
            A_log16, dt_bias, ba[t * rows:(t + 1) * rows].contiguous(),
            pool, out, z, tix, num_acc, nsd, nst, H, HV, K, V, scale)
        torch.xpu.synchronize()
        snaps.append(pool[SLOT].float().clone())
    st = torch.stack(snaps)
    nan = bool(torch.isnan(st).any())
    mx = st.abs().max().item()
    line = f"[spec {name}] max|S|={mx:.4f} nan={nan}"
    if ref is not None:
        rs = ref
        d0 = rel(st[0], rs[0])
        floor = rel(init_ssm16[SLOT].to(dt).float(), init_ssm16[SLOT].float())
        dstate = [rel(st[i], rs[i]) for i in (1, 8, 31)]
        cls = classify(d0, dstate[2], floor)
        VERDICTS.append((f"spec {name}", cls, nan))
        line += (f" floor={floor:.5f} d_state(t1,t8,t32)="
                 f"{dstate[0]:.5f},{dstate[1]:.5f},{dstate[2]:.5f} -> {cls}")
    print(line)
    return st


def main():
    print("loading custom_esimd_kernels_vllm ...")
    import custom_esimd_kernels_vllm  # noqa: F401
    print("module import OK")

    # ---- determinism control: fp16 cell twice, must match exactly
    qkvz, ba, cw, cb, A_log16, dt_bias, ic, is16 = make_streams(T_SEQ, seed=4242)
    a = seq_trajectory(torch.float16, "det-a", qkvz, ba, cw, cb, A_log16,
                       dt_bias, ic, is16)
    b = seq_trajectory(torch.float16, "det-b", qkvz, ba, cw, cb, A_log16,
                       dt_bias, ic, is16)
    det = (a[0] - b[0]).abs().max().item()
    print(f"[det] cross-run max|dS|={det:.8f} [{'PASS' if det == 0 else 'FAIL'}]")
    VERDICTS.append(("determinism", "PASS" if det == 0 else "FAIL", det != 0))

    # ---- section 1: seq trajectories
    ref = a
    s4 = seq_trajectory(torch.float8_e4m3fn, "e4m3", qkvz, ba, cw, cb,
                        A_log16, dt_bias, ic, is16, ref=ref)
    s5 = seq_trajectory(torch.float8_e5m2, "e5m2", qkvz, ba, cw, cb,
                        A_log16, dt_bias, ic, is16, ref=ref)

    # ---- section 2: spec trajectories (fresh streams, 5 rows per call)
    qkvz2, ba2, cw2, cb2, Al2, dtb2, ic2, is2 = make_streams(T_SPEC * 5, seed=777)
    r16 = spec_trajectory(torch.float16, "fp16", qkvz2, ba2, cw2, cb2,
                          Al2, dtb2, ic2, is2)
    spec_trajectory(torch.float8_e4m3fn, "e4m3", qkvz2, ba2, cw2, cb2,
                    Al2, dtb2, ic2, is2, ref=r16)
    spec_trajectory(torch.float8_e5m2, "e5m2", qkvz2, ba2, cw2, cb2,
                    Al2, dtb2, ic2, is2, ref=r16)

    print("---- verdicts ----")
    bad = 0
    for tag, cls, nan in VERDICTS:
        ok = (cls in ("PASS", "BOUNDED", "RANDOM-WALK")) and not nan
        bad += 0 if ok else 1
        print(f"  {tag:14s} {cls:12s} nan={nan} [{'PASS' if ok else 'FAIL'}]")
    print("TRAJ_FAILS=%d" % bad)
    if bad:
        sys.exit(1)
    print("P22C1_TRAJ_PASS")


if __name__ == "__main__":
    main()
