#!/usr/bin/env python3
"""p22a_smoke.py (v2) — op-level fp8 correctness gate for the rebuilt
custom_esimd_kernels_lgrf .so (v125 P22A). Runs INSIDE lsv-test (GPU)
in a FRESH process right after the .so swap, BEFORE any serve leg.

Scope = every ESIMD surface this lane's model can actually reach
(qwen3.8-27b = sequential layout, TP2 -> H=8, HV=24, K=V=128):
  esimd_gdn_conv_fused_seq       non-spec decode (WG64 dispatch)
  esimd_gdn_conv_fused_seq_spec  spec-1 decode (rollback states)

DROPPED v1 cells: esimd_gdn_conv_fused (interleaved-GQA, Qwen3-Next-
80B). v1 smoke proved that kernel is structurally H<=4 only — WG_SIZE=32
hardcoded, double_v = (HV > (32-4*H)/2), useful = 4*H + HV threads; at
our H=8, 4*H=32 already exhausts the WG and z_out rows are never written
(identical failures for e4m3 AND e5m2 AND fp16 pools = dtype-independent,
pre-existing, NOT a P22A artifact). Our model never dispatches it
(gqa_interleaved_layout=False; vLLM spec pin H==8/HV==24 sequential).

Per (dtype in {e4m3, e5m2}):
  T0 fp16 control      — same op run twice on fp16 pools: cross-run
                         consistency of the harness itself.
  T1 zero-state parity — fp16 zero pool vs fp8 zero pool: outputs must
                         match atol 1e-3 (0 encodes exactly in fp8).
  T2 exact-load identity — both kernels load THE SAME values (fp16 pool
                         holds the dequantized fp8 image; fp8 pool the
                         fp8 bytes) -> outputs must match atol 1e-3.
  T3 NaN/bounds scan   — outputs, z, conv_state, updated pool image.
out/z buffers are ZERO-FILLED before every call: any region the kernel
leaves unwritten compares equal (0==0) instead of diffing empty-garbage
(lesson from v1: torch.empty + partial-write = spurious cross-run diff).

PASS = every check; any FAIL exits nonzero (runner aborts + restores).
"""
import sys
import torch

K = V = 128
ATOL = 1e-3
DEV = "xpu"

FAILS = []


def check(label, ref, test, atol):
    d = (ref.float() - test.float()).abs().max().item()
    nan = torch.isnan(test.float()).any().item()
    ok = (not nan) and d < atol
    print("  %-38s max_diff=%.6f nan=%s [%s]" % (label, d, nan, "PASS" if ok else "FAIL"))
    if not ok:
        FAILS.append(label)


def check_sane(label, t):
    nan = torch.isnan(t.float()).any().item()
    mx = t.float().abs().max().item()
    ok = (not nan) and mx < 100.0
    print("  %-38s max_abs=%.4f nan=%s [%s]" % (label, mx, nan, "PASS" if ok else "FAIL"))
    if not ok:
        FAILS.append(label)


def make_inputs(rows, H, HV, seed):
    HPG = HV // H
    DIM = 2 * H * K + 2 * HV * V  # causal conv spans the full q|k|v|z row
    QKVZ_DIM = DIM
    torch.manual_seed(seed)
    qkvz_seq = torch.randn(rows, QKVZ_DIM, dtype=torch.float16, device=DEV) * 0.1
    ba_seq = torch.randn(rows, 2 * HV, dtype=torch.float16, device=DEV) * 0.5
    conv_weight = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
    conv_bias_zeros = torch.zeros(DIM, dtype=torch.float16, device=DEV)
    A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
    dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
    return qkvz_seq, ba_seq, conv_weight, conv_bias_zeros, A_log16, dt_bias


def make_pools(num_cache, HV, dt, seed):
    torch.manual_seed(seed)
    init_conv = torch.randn(num_cache, 3, 2 * 8 * K + 2 * HV * V,
                            dtype=torch.float16, device=DEV) * 0.01
    init_ssm_q = (torch.randn(num_cache, HV, V, K,
                              dtype=torch.float16, device=DEV) * 0.05).to(dt).to(torch.float16)
    init_ssm8 = init_ssm_q.to(dt)
    return init_conv, init_ssm_q, init_ssm8


def run_seq_case(N, H, HV, dt, dt_name, seed):
    num_cache = 32
    HPG = HV // H
    qkvz_seq, ba_seq, cw, cb, A_log16, dt_bias = make_inputs(N, H, HV, seed)
    init_conv, init_ssm_q, init_ssm8 = make_pools(num_cache, HV, dt, seed + 1000)
    idx = torch.arange(N, dtype=torch.int32, device=DEV)
    scale = float(K ** -0.5)

    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq

    def call(pool):
        conv = init_conv.clone()
        out = torch.zeros(N, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros(N, HV, V, dtype=torch.float16, device=DEV)
        esimd_gdn_conv_fused_seq(
            qkvz_seq, conv, cw, cb, idx,
            A_log16, dt_bias, ba_seq, pool, idx, out, z,
            N, H, HV, K, V, scale)
        torch.xpu.synchronize()
        return out, z, conv, pool

    print("== seq N=%d H=%d HV=%d %s ==" % (N, H, HV, dt_name))

    # T0: fp16-vs-fp16 cross-run control (harness determinism)
    z16a = torch.zeros(num_cache, HV, V, K, dtype=torch.float16, device=DEV)
    o_a, z_a, c_a, p_a = call(z16a)
    o_b, z_b, c_b, p_b = call(z16a)
    check("T0 core_attn_out (fp16 ctrl)", o_a, o_b, ATOL)
    check("T0 z_out (fp16 ctrl)", z_a, z_b, ATOL)

    # T1: zero-state parity (fp16 zeros vs fp8 zeros)
    z8 = torch.zeros(num_cache, HV, V, K, dtype=dt, device=DEV)
    o8, z8o, c8, p8 = call(z8)
    check("T1 core_attn_out", o_a, o8, ATOL)
    check("T1 z_out", z_a, z8o, ATOL)
    check("T1 conv_state", c_a, c8, ATOL)
    # T1b: SAVE-path differential — fp16 run's saved pool vs fp8 run's
    # saved pool decoded. Catches store-encode corruption directly (the
    # 0xF887FFFF bug decoded to 80/88 vs true ~0.25 and hid under the
    # sanity bound; loads never see it in a single-call op).
    check("T1b pool image (saves fp16 vs fp8)", p_b, p8.to(torch.float16), 0.1)

    # T2: exact-load identity (dequantized fp8 image vs fp8 bytes)
    q16 = init_ssm_q.clone()
    q8 = init_ssm8.clone()
    o16, z16o, c16, p16 = call(q16)
    o8, z8o, c8, p8 = call(q8)
    check("T2 core_attn_out", o16, o8, ATOL)
    check("T2 z_out", z16o, z8o, ATOL)
    check("T2 conv_state", c16, c8, ATOL)

    # T3: updated-pool image sane
    check_sane("T3 updated pool image", p8.to(torch.float16))


def run_spec_case(dt, dt_name, seed):
    """Spec-1 differential mirroring the production call site
    (gdn_linear_attn.py): nsd=1 row of nst=5 spec tokens, rollback
    state handling by the kernel."""
    H, HV = 8, 24
    nsd, nst = 1, 5
    rows = nsd * nst
    num_cache = 32
    qkvz_seq, ba_seq, cw, cb, A_log16, dt_bias = make_inputs(rows, H, HV, seed)
    init_conv, init_ssm_q, init_ssm8 = make_pools(num_cache, HV, dt, seed + 1000)
    slot = 7
    # token_indx = PACKED-ROW POSITIONS 0..rows-1. Kernel contract (verified
    # against gdn_conv_fused_seq_spec.h + the production call site, which
    # asserts qkvz.size(0) == nsd*nst): global_t = token_indx[...] indexes
    # qkvz AND ba input rows AND output/z_out rows of the packed tensors.
    # spec_state_indices carries the pool SLOTS — a different tensor for a
    # different purpose. Passing the slot here (v1/v2 bug) made every out/z
    # write land at out row `slot` -> past the buffer end -> stomp.
    spec_state_indices = torch.full((nsd, nst), slot, dtype=torch.int32, device=DEV)
    token_indx = torch.arange(rows, dtype=torch.int32, device=DEV)
    num_accepted = torch.tensor([3], dtype=torch.int32, device=DEV)
    scale = float(K ** -0.5)

    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec

    def call(pool):
        conv = init_conv.clone()
        out = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros(rows, HV, V, dtype=torch.float16, device=DEV)
        esimd_gdn_conv_fused_seq_spec(
            qkvz_seq, conv, cw, cb, spec_state_indices,
            A_log16, dt_bias, ba_seq, pool, out, z,
            token_indx, num_accepted, nsd, nst,
            H, HV, K, V, scale)
        torch.xpu.synchronize()
        return out, z, conv, pool

    print("== spec nsd=%d nst=%d H=%d HV=%d %s ==" % (nsd, nst, H, HV, dt_name))

    # T1: zero-state parity
    z16 = torch.zeros(num_cache, HV, V, K, dtype=torch.float16, device=DEV)
    z8 = torch.zeros(num_cache, HV, V, K, dtype=dt, device=DEV)
    o16, zo16, c16, p16 = call(z16)
    o8, zo8, c8, p8 = call(z8)
    check("T1 core_attn_out", o16, o8, ATOL)
    check("T1 z_out", zo16, zo8, ATOL)
    check("T1 conv_state", c16, c8, ATOL)
    # T1b: SAVE-path differential (see run_seq_case)
    check("T1b pool image (saves fp16 vs fp8)", p16, p8.to(torch.float16), 0.1)

    # T2: exact-load identity
    q16 = init_ssm_q.clone()
    q8 = init_ssm8.clone()
    o16, zo16, c16, p16 = call(q16)
    o8, zo8, c8, p8 = call(q8)
    check("T2 core_attn_out", o16, o8, ATOL)
    check("T2 z_out", zo16, zo8, ATOL)
    check("T2 conv_state", c16, c8, ATOL)

    # T3: outputs + updated pool sane
    check_sane("T3 core_attn_out", o8)
    check_sane("T3 z_out", zo8)
    check_sane("T3 updated pool image", p8.to(torch.float16))


def main():
    print("loading custom_esimd_kernels_vllm ...")
    import custom_esimd_kernels_vllm  # noqa: F401
    from custom_esimd_kernels_vllm import (  # noqa: F401
        esimd_gdn_conv_fused_seq,
        esimd_gdn_conv_fused_seq_spec,
    )
    print("module import OK")

    # seq op: production shape, N sweep (N=1 solo decode .. N=8 batch)
    for dt, name in ((torch.float8_e4m3fn, "e4m3"), (torch.float8_e5m2, "e5m2")):
        for N in (1, 4, 8):
            run_seq_case(N, 8, 24, dt, name, seed=42 + N)
        # spec op: production spec-1 shape
        run_spec_case(dt, name, seed=99)

    print("SMOKE_FAILS=%d" % len(FAILS))
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("P22A_SMOKE_PASS")


if __name__ == "__main__":
    main()
