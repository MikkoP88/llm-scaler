#!/usr/bin/env python3
"""Standalone GDN spec-path OOB repro (KNOWN_ISSUES #11 wedge root cause).

Calls torch.ops._xpu_C.gdn_attention directly with the EXACT metadata shapes
gdn_attn.py produces for ragged speculative steps:

  - Case F  (full 5-token single request) = regression reference; runs FIRST
             and is saved for bit-comparison across wheels (pre/post fix).
  - Case R  (single request, k=4, final verify step clamped to 1 token):
             num_spec_decodes=1, num_spec_tokens=5, spec_token=1.
             gdn_attn.py: spec_token_indx = arange(min(1*5, 1)) = [0]   (truncated!)
  - Case R2 (two requests, accepted lens 5 and 2):
             num_spec_decodes=2, spec_token=7, spec_query_start_loc=[0,5,7].

The spec kernels historically walked a RECTANGULAR batch_id*5+t_local index
over the raggedly-truncated spec_token_indx and over the spec_token-sized
q/k/v/b/a scratch -> OOB reads + OOB writes. Observed on the pre-fix wheel:
Case R kills the XPU device on the FIRST call (UR_RESULT_ERROR_DEVICE_LOST /
xe CCS engine reset) -- the serve wedge, reproduced deterministically. On a
fixed wheel the phantom-token walks are skipped and all cases pass cleanly.

To make any phantom writes DETERMINISTIC (not just a fault), spec_token_indx
is a narrow() view of a 64-int buffer whose tail beyond the true token count
holds a sentinel row id pointing INTO a large zero-filled buffer that backs
the outputs. canary != 0 after the call  ==>  the bug fired without faulting
(detectable on wheels where the OOB stays inside the allocator slab).

Model shapes mirror qwen3.8-27b GDN layers per TP=2 rank:
k=8,v=24 heads of 128/128, qkvz 8192, ba 48, conv [5120,4].

Exit code 0 = no phantom writes / no device faults (fixed wheel); 1 = bug.
"""
import sys
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)

torch.manual_seed(1234)
DEV = torch.device("xpu")

# ---- model geometry (per-rank, tp=2), as the serve passes them -----------
NK, NV, HK, HV = 16, 48, 128, 128      # global head counts + dims
TP = 2
QKVZ = (NK // TP) * (2 * HK + 2 * (HV * NV // NK))   # 8 * 1024 = 8192
BA = 2 * NV // TP                                     # 48
CONV_DIM = (NK // TP) * HK + (NK // TP) * HK + (NV // TP) * HV  # 5120
WIDTH = 4
NST = 5                                # num_spec_tokens = k+1 (k=4)
SLOTS = 16

DTYPE = torch.bfloat16

def make_states():
    conv_state = (torch.randn(SLOTS, WIDTH - 1, CONV_DIM, dtype=torch.float32) * 0.05).to(DTYPE).to(DEV)
    ssm_state = (torch.randn(SLOTS, NV // TP, HV, HK, dtype=torch.float32) * 0.05).to(DEV)
    return conv_state, ssm_state

def make_params():
    conv_weights = (torch.randn(CONV_DIM, WIDTH, dtype=torch.float32) * 0.2).to(DTYPE).to(DEV)
    A_log = (torch.randn(NV // TP, dtype=torch.float32) * 0.3 - 1.0).to(DEV)
    dt_bias = (torch.randn(NV // TP, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    return conv_weights, A_log, dt_bias

def call_op(core_attn_out, z, qkvz, ba, conv_state, ssm_state, conv_weights, A_log, dt_bias,
            num_spec_decodes, spec_qsl, spec_token_indx, spec_state_indices,
            num_accepted, num_actual_tokens):
    torch.ops._xpu_C.gdn_attention(
        core_attn_out, z, qkvz, ba,
        NK, NV, HK, HV,
        conv_state=conv_state, ssm_state=ssm_state,
        conv_weights=conv_weights, conv_bias=None, activation="silu",
        A_log=A_log, dt_bias=dt_bias,
        num_prefills=0, num_decodes=0, num_spec_decodes=num_spec_decodes,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=spec_qsl,
        spec_token_indx=spec_token_indx,
        spec_state_indices_tensor=spec_state_indices,
        num_accepted_tokens=num_accepted,
        num_actual_tokens=num_actual_tokens,
        tp_size=TP, reorder_input=True,
    )

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/out"
    import os
    os.makedirs(outdir, exist_ok=True)

    conv_weights, A_log, dt_bias = make_params()

    # Big zero backing buffers: rows of core_attn_out shape. Sentinel row id
    # 40000 points into them. Phantom writes via a garbage global_t land at
    # core_attn_out/z + SENT_ROW*(NV/TP*HV); by backing the outputs with
    # narrow() views of large buffers we keep those landings INSIDE our own
    # allocation (deterministic detection instead of an unrelated victim).
    CANARY_ROWS = 50000
    SENT_ROW = 40000
    print(f"[repro] qkvz={QKVZ} ba={BA} conv={CONV_DIM}x{WIDTH} dtype={DTYPE}")
    print(f"[repro] canary/out buffers: {CANARY_ROWS} rows, sentinel row {SENT_ROW}")

    results = {}

    # ---------------- Case F: full 5-token batch (regression reference) -----
    # Runs FIRST so the bit-compare reference is saved even on a wheel that
    # dies on the ragged cases below.
    conv_stateF, ssm_stateF = make_states()
    qkvzF = (torch.randn(NST, QKVZ, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    baF = (torch.randn(NST, BA, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    outF = torch.zeros(NST, NV // TP, HV, dtype=DTYPE, device=DEV)
    zF = torch.zeros_like(outF)
    tiF = torch.arange(NST, dtype=torch.int32, device=DEV)
    qslF = torch.tensor([0, NST], dtype=torch.int32, device=DEV)
    ssiF = torch.arange(NST, dtype=torch.int32, device=DEV).view(1, NST)
    accF = torch.tensor([NST], dtype=torch.int32, device=DEV)
    call_op(outF, zF, qkvzF, baF, conv_stateF, ssm_stateF,
            conv_weights, A_log, dt_bias, 1, qslF, tiF, ssiF, accF, NST)
    torch.xpu.synchronize()
    fin = bool(torch.isfinite(outF.float()).all() and torch.isfinite(zF.float()).all())
    results["F_finite"] = fin
    print(f"[F] full batch: finite={fin} out[0,0,:4]={outF[0,0,:4].float().tolist()}")
    torch.save({"out": outF.cpu(), "z": zF.cpu(),
                "qkvz": qkvzF.cpu(), "ba": baF.cpu(),
                "conv_state": conv_stateF.cpu(), "ssm_state": ssm_stateF.cpu(),
                "conv_weights": conv_weights.cpu(), "A_log": A_log.cpu(),
                "dt_bias": dt_bias.cpu()},
               f"{outdir}/full_ref.pt")
    print(f"[F] reference tensors saved to {outdir}/full_ref.pt")

    # If a reference from another wheel exists, bit-compare against it now.
    ref_path = f"{outdir}/full_ref.prev.pt"
    if os.path.exists(ref_path):
        prev = torch.load(ref_path, map_location="cpu", weights_only=True)
        same = (torch.equal(prev["out"], outF.cpu()) and
                torch.equal(prev["z"], zF.cpu()))
        results["F_bitidentical_to_prev_wheel"] = same
        print(f"[F] bit-identical to previous wheel: {same}")

    # ---------------- Case R: single request, 1 real token (ragged) ---------
    conv_state, ssm_state = make_states()
    qkvz = (torch.randn(1, QKVZ, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    ba = (torch.randn(1, BA, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    # token_indx buffer: [0]=0 (real token), tail = sentinel rows
    ti_buf = torch.zeros(64, dtype=torch.int32, device=DEV)
    ti_buf[1:] = SENT_ROW
    spec_ti = ti_buf.narrow(0, 0, 1)                  # the truncated arange(1)
    spec_qsl = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    spec_ssi = torch.arange(NST, dtype=torch.int32, device=DEV).view(1, NST)
    num_acc = torch.tensor([1], dtype=torch.int32, device=DEV)

    big_out = torch.zeros(CANARY_ROWS + 64, NV // TP, HV, dtype=DTYPE, device=DEV)
    core_attn_out = big_out.narrow(0, 0, 1)           # the [1,24,128] "output"
    z_big = torch.zeros_like(big_out)
    z = z_big.narrow(0, 0, 1)

    n_corrupt = 0
    try:
        for it in range(50):
            core_attn_out.zero_(); z.zero_()
            big_out.zero_(); z_big.zero_()
            call_op(core_attn_out, z, qkvz, ba, conv_state, ssm_state,
                    conv_weights, A_log, dt_bias,
                    1, spec_qsl, spec_ti, spec_ssi, num_acc, 1)
            torch.xpu.synchronize()
            nz = int((big_out[1:] != 0).sum()) + int((z_big[1:] != 0).sum())
            if nz:
                n_corrupt += 1
                rows = torch.nonzero(big_out.amax(dim=(1, 2)) != 0)
                if n_corrupt == 1:
                    print(f"[R] first corruption at iter {it}: {nz} elems; rows written: {rows[:8].flatten().tolist()}")
    except RuntimeError as e:
        # Pre-fix wheel: the OOB q/k/v/b/a writes fault the CCS engine.
        print(f"[R] RuntimeError at iter {it}: {e}")
        print("[R] ==> DEVICE FAULT (the serve-wedge mechanism), counted as corruption")
        n_corrupt += 1
    results["R"] = n_corrupt
    print(f"[R] ragged single (50 iters): {n_corrupt}/50 phantom-write iterations")

    # ---------------- Case R2: two requests, lens 5 and 2 -------------------
    conv_state, ssm_state = make_states()
    qkvz2 = (torch.randn(7, QKVZ, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    ba2 = (torch.randn(7, BA, dtype=torch.float32) * 0.1).to(DTYPE).to(DEV)
    big_out2 = torch.zeros(CANARY_ROWS + 64, NV // TP, HV, dtype=DTYPE, device=DEV)
    z_big2 = torch.zeros_like(big_out2)
    core2 = big_out2.narrow(0, 0, 7)
    z2 = z_big2.narrow(0, 0, 7)
    ti_buf2 = torch.zeros(64, dtype=torch.int32, device=DEV)
    ti_buf2[:7] = torch.arange(7, dtype=torch.int32)
    ti_buf2[7:] = SENT_ROW
    spec_ti2 = ti_buf2.narrow(0, 0, 7)                # true spec_token = 7
    spec_qsl2 = torch.tensor([0, 5, 7], dtype=torch.int32, device=DEV)
    spec_ssi2 = torch.arange(2 * NST, dtype=torch.int32, device=DEV).view(2, NST)
    num_acc2 = torch.tensor([5, 2], dtype=torch.int32, device=DEV)

    n2 = 0
    try:
        for it in range(50):
            big_out2.zero_(); z_big2.zero_()
            call_op(core2, z2, qkvz2, ba2, conv_state, ssm_state,
                    conv_weights, A_log, dt_bias,
                    2, spec_qsl2, spec_ti2, spec_ssi2, num_acc2, 7)
            torch.xpu.synchronize()
            nz = int((big_out2[7:] != 0).sum()) + int((z_big2[7:] != 0).sum())
            if nz:
                n2 += 1
                if n2 == 1:
                    print(f"[R2] first corruption at iter {it}: {nz} elems beyond row 7")
    except RuntimeError as e:
        print(f"[R2] RuntimeError at iter {it}: {e}")
        print("[R2] ==> DEVICE FAULT (the serve-wedge mechanism), counted as corruption")
        n2 += 1
    results["R2"] = n2
    print(f"[R2] ragged n=2 (50 iters): {n2}/50 phantom-write iterations")

    # ---------------- verdict ------------------------------------------------
    buggy = (results.get("R", 0) > 0) or (results.get("R2", 0) > 0)
    print(f"[verdict] phantom writes: R={results.get('R', 0)}/50 R2={results.get('R2', 0)}/50 -> "
          + ("BUGGY WHEEL (OOB writes reproduced)" if buggy else "CLEAN (no OOB writes)"))
    sys.exit(1 if buggy else 0)

if __name__ == "__main__":
    main()
