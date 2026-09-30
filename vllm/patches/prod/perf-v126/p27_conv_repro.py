#!/usr/bin/env python3
"""p27_conv_repro.py — v126 P27 runtime conviction test for the fp8-state
quality root cause.

HYPOTHESIS (convicted at source level): gdn_attention drives BOTH
gdn::causal_conv1d (conv_state) and gdn::gated_delta_rule (ssm_state) with
the SAME state-index tensors (gdn_attn_interface.cpp:310/346). The fp8 SSM
bridges remap those indices to compact ssm_run rows while passing the REAL
conv pool => the conv kernel reads/writes conv rows at COMPACT ids instead
of the requests' real slots => cross-request state crosstalk under
concurrency (P23D/P23F).

TEST: 2-sequence prefill batch, real slots [5, 9], fp16 pools (isolates
index semantics from fp8 math; the OLD wheel accepts fp16 everywhere).
  A (control): indices [5, 9] against the real pools.
  B (bridge) : ssm gathered to rows [0, 1], indices [0, 1], conv pool REAL.
CONVICTION: B writes conv rows 0 and 1 (not 5 and 9).
"""
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers _xpu_C.gdn_attention)

torch.manual_seed(126)
torch.set_grad_enabled(False)

NK, NV, DK, DV, W, TP = 2, 2, 4, 4, 4, 1
KTP, VTP = NK // TP, NV // TP
CONV_DIM = KTP * (2 * DK + DV * (NV // NK))          # 24
QKVZ_DIM = KTP * (2 * DK + 2 * DV * (NV // NK))      # 32
BA_DIM = KTP * (2 * (NV // NK))                      # 4
P = 16                                                # pool slots
SEQ_LENS = [3, 2]
T = sum(SEQ_LENS)
SLOTS = [5, 9]

dev = torch.device("xpu")


def fresh_pools():
    conv = torch.randn(P, W - 1, CONV_DIM, dtype=torch.float16, device=dev)
    ssm = torch.randn(P, VTP, DV, DK, dtype=torch.float16, device=dev)
    return conv, ssm


def run(conv, ssm, indices, label):
    core = torch.zeros(T, VTP, DV, dtype=torch.float16, device=dev)
    z = torch.zeros(T, VTP, DV, dtype=torch.float16, device=dev)
    qkvz = torch.randn(T, QKVZ_DIM, dtype=torch.float16, device=dev) * 0.05
    ba = torch.randn(T, BA_DIM, dtype=torch.float16, device=dev) * 0.05
    cw = torch.randn(CONV_DIM, W, dtype=torch.float16, device=dev) * 0.05
    cb = torch.zeros(CONV_DIM, dtype=torch.float16, device=dev)
    A_log = torch.zeros(VTP, dtype=torch.float32, device=dev)
    dtb = torch.ones(VTP, dtype=torch.float16, device=dev) * 0.01
    qsl = torch.tensor([0] + list(torch.tensor(SEQ_LENS).cumsum(0)),
                       dtype=torch.int32, device=dev)
    idx = torch.tensor(indices, dtype=torch.int32, device=dev)
    his = torch.zeros(len(SEQ_LENS), dtype=torch.bool, device=dev)
    torch.ops._xpu_C.gdn_attention(
        core, z, qkvz, ba, NK, NV, DK, DV,
        conv_state=conv, ssm_state=ssm,
        conv_weights=cw, conv_bias=cb, activation="silu",
        A_log=A_log, dt_bias=dtb,
        num_prefills=len(SEQ_LENS), num_decodes=0, num_spec_decodes=0,
        has_initial_state=his,
        non_spec_query_start_loc=qsl,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=idx,
        spec_query_start_loc=None, spec_token_indx=None,
        spec_state_indices_tensor=None, num_accepted_tokens=None,
        num_actual_tokens=T, tp_size=TP, reorder_input=True,
    )
    torch.xpu.synchronize()
    return label


# ---- variant A: real indices against real pools ---------------------------
convA, ssmA = fresh_pools()
convA0, ssmA0 = convA.clone(), ssmA.clone()
run(convA, ssmA, SLOTS, "A")
convA_diff = (convA != convA0).any(dim=(1, 2)).nonzero().flatten().tolist()
ssmA_diff = (ssmA != ssmA0).any(dim=(1, 2, 3)).nonzero().flatten().tolist()
print("A (real idx %s): conv rows written = %s ; ssm rows written = %s"
      % (SLOTS, convA_diff, ssmA_diff))

# ---- variant B: bridge-simulated (remapped ssm, REAL conv pool) -----------
convB, ssmB = fresh_pools()
convB0 = convB.clone()
ssm_run = ssmB[SLOTS].contiguous()          # compact gather (rows 0,1)
run(convB, ssm_run, [0, 1], "B")            # indices REMAPPED, conv REAL
convB_diff = (convB != convB0).any(dim=(1, 2)).nonzero().flatten().tolist()
print("B (bridge sim):  conv rows written = %s ; ssm_run rows written = %s"
      % (convB_diff, [0, 1]))

verdict_a = sorted(convA_diff) == SLOTS
verdict_b = sorted(convB_diff) == [0, 1]
print("A_writes_real_slots=%s" % verdict_a)
print("B_writes_compact_rows(conv crosstalk)=%s" % verdict_b)
print("P27_RUNTIME_VERDICT: %s"
      % ("CONVICTED" if (verdict_a and verdict_b) else "NOT-CONVICTED"))
