#!/usr/bin/env python3
"""v88 — int32 pool-offset overflow repro (lsv-test wedge root cause).

Replays the EXACT faulting prefill GDN call captured by v86
(gdn_capture_last_548.pt: layer-16 linear_attn, non_spec_ssi=[4173],
qsl=[0,8], 8 tokens) against a standalone mamba pool that reproduces the
serve geometry byte-for-byte:

  - one fp16 backing buffer of POOL_ROWS x 524288 elements (padded 1-MiB
    pages, stride(0)=524288 — the value the v87 probe measured live)
  - conv view  (POOL_ROWS, 7, 5120)      stride (524288, 5120, 1)  at page offset 0
  - ssm view   (POOL_ROWS, 24, 128, 128) stride (524288, 16384, 128, 1) at +71680 B

chunk_causal_conv1d_xe2.hpp:186/547 computes the state pointer as
    conv_states + states_id * conv_states_stride_0      (int * int)
which overflows signed int32 for states_id >= 4097 (4096 -> INT32_MIN).

Cases (in order; a fault may kill the device so the money shot runs early):
  LO   ssi=100   — in-range baseline; outputs saved for pre/post bit-compare
  HI   ssi=4173  — the exact killing request's layer-16 index
  BND  ssi=4096  — the 2^31 boundary (wraps to INT32_MIN)

Detection per case: pool rows sentinel-filled; after the call the target row
must have changed (conv state written) with finite core_attn_out, and no
OTHER row may change. On a buggy wheel HI/BND either raise
RuntimeError/XPU device loss (unmapped wrapped target — the serve wedge) or
silently leave the target row untouched (mapped wrapped target — the silent
corruption variant). Exit 0 = clean, 1 = bug fired.

Usage: repro_v88_int32.py <capture.pt> <outdir> [--pool-rows N] [--skip-hi]
"""
import os
import sys

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)

DEV = torch.device("xpu")
SENT = 12345.0
POOL_ROWS = 8192          # > 4173, keeps the index legal like the 8050-row serve pool
PAGE = 524288             # stride(0) in elements — measured live by the v87 probe
CONV_OFF_EL = 0
SSM_OFF_EL = 35840        # 71,680 B / 2 — same-page offset measured by v87


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap_path = args[0] if args else "/root/gdn_cap_input.pt"
    outdir = args[1] if len(args) > 1 else "/root/v88out"
    skip_hi = "--skip-hi" in sys.argv
    rows = POOL_ROWS
    for a in sys.argv[1:]:
        if a.startswith("--pool-rows"):
            rows = int(a.split("=", 1)[1])
    os.makedirs(outdir, exist_ok=True)

    cap = torch.load(cap_path, map_location="cpu", weights_only=True)
    qkvz = cap["qkvz"].to(DEV)
    ba = cap["ba"].to(DEV)
    conv_weights = cap["conv_weights"].to(DEV)
    A_log = cap["A_log"].to(DEV)
    dt_bias = cap["dt_bias"].to(DEV)
    has_init = cap["has_initial_state"].to(DEV)
    conv_bias = cap.get("conv_bias")
    conv_bias = conv_bias.to(DEV) if torch.is_tensor(conv_bias) else None
    qsl = cap["non_spec_qsl"].to(DEV).contiguous()
    ti = cap.get("non_spec_ti")
    ti = ti.to(DEV).contiguous() if torch.is_tensor(ti) else None  # prefill passes None
    NK = int(cap["num_k_heads"]); NV = int(cap["num_v_heads"])
    HK = int(cap["head_k_dim"]); HV = int(cap["head_v_dim"])
    TP = int(cap["tp_size"])
    reorder = not bool(cap["gqa_interleaved_layout"])
    act = str(cap["activation"])
    out_shape = tuple(cap["core_attn_out_shape"])
    ntok = int(cap["num_actual_tokens"])
    dtype = qkvz.dtype
    print(f"[v88] capture={os.path.basename(cap_path)} dtype={dtype} out={out_shape} "
          f"NK={NK} NV={NV} HK={HK} HV={HV} tp={TP} act={act} reorder={reorder} ntok={ntok}")
    print(f"[v88] pool: {rows} rows x {PAGE} el ({rows*PAGE*2/2**30:.2f} GiB fp16 backing)")

    backing = torch.empty(rows * PAGE, dtype=dtype, device=DEV)
    conv_states = backing.as_strided((rows, 7, 5120), (PAGE, 5120, 1), CONV_OFF_EL)
    ssm_states = backing.as_strided((rows, 24, 128, 128), (PAGE, 16384, 128, 1), SSM_OFF_EL)
    print(f"[v88] conv_states {tuple(conv_states.shape)} stride {conv_states.stride()} | "
          f"ssm {tuple(ssm_states.shape)} stride {ssm_states.stride()}")

    def run_case(name: str, ssi_val: int, save_ref: bool = False) -> bool:
        backing.fill_(SENT)
        core = torch.zeros(out_shape, dtype=dtype, device=DEV)
        z = torch.zeros(out_shape, dtype=dtype, device=DEV)
        ssi = torch.tensor([ssi_val], dtype=torch.int32, device=DEV)
        try:
            torch.ops._xpu_C.gdn_attention(
                core, z, qkvz, ba, NK, NV, HK, HV,
                conv_state=conv_states, ssm_state=ssm_states,
                conv_weights=conv_weights, conv_bias=conv_bias, activation=act,
                A_log=A_log, dt_bias=dt_bias,
                num_prefills=1, num_decodes=0, num_spec_decodes=0,
                has_initial_state=has_init,
                non_spec_query_start_loc=qsl,
                non_spec_token_indx=ti,
                non_spec_state_indices_tensor=ssi,
                spec_query_start_loc=None, spec_token_indx=None,
                spec_state_indices_tensor=None,
                num_accepted_tokens=None,
                num_actual_tokens=ntok,
                tp_size=TP, reorder_input=reorder,
            )
            torch.xpu.synchronize()
        except RuntimeError as e:
            print(f"[{name}] RuntimeError: {str(e)[:300]}")
            print(f"[{name}] ==> DEVICE FAULT (the serve-wedge mechanism)")
            return False
        # which rows changed?
        conv2d = conv_states.reshape(rows, -1)
        row_changed = (conv2d != torch.tensor(SENT, dtype=dtype, device=DEV)).any(dim=1)
        changed = torch.nonzero(row_changed).flatten().tolist()
        finite = bool(torch.isfinite(core.float()).all() and torch.isfinite(z.float()).all())
        ok = changed == [ssi_val] and finite
        print(f"[{name}] ssi={ssi_val} rows_changed={changed[:8]}{'...' if len(changed) > 8 else ''} "
              f"(n={len(changed)}) finite={finite} -> {'OK' if ok else 'BUG: write misdirected'}")
        if save_ref:
            torch.save({"core": core.cpu(), "z": z.cpu()}, f"{outdir}/ref100.pt")
            print(f"[{name}] reference outputs saved to {outdir}/ref100.pt")
        return ok

    lo_ok = run_case("LO", 100, save_ref=True)

    # bit-compare against a previous wheel's reference if present
    prev = f"{outdir}/ref100.prev.pt"
    if os.path.exists(prev) and lo_ok:
        p = torch.load(prev, map_location="cpu", weights_only=True)
        cur = torch.load(f"{outdir}/ref100.pt", map_location="cpu", weights_only=True)
        same = torch.equal(p["core"], cur["core"]) and torch.equal(p["z"], cur["z"])
        print(f"[LO] bit-identical to previous wheel reference: {same}")

    verdict = {"LO": lo_ok}
    if not skip_hi:
        verdict["HI"] = run_case("HI", 4173)
        if verdict["HI"]:
            verdict["BND"] = run_case("BND", 4096)
    clean = all(verdict.values())
    print(f"[verdict] {verdict} -> " + ("CLEAN WHEEL" if clean else "BUGGY WHEEL (int32 overflow reproduced)"))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
