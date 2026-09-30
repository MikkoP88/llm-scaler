#!/usr/bin/env python3
"""p28_dual_check.py — v126 P28 op-level verification of the DUAL bridge.

Validates the bridge MECHANISM (gather both pools -> run with remapped
indices -> scatter both home) against the real SYCL op, with an fp8 ssm
pool + fp16 conv pool — the production fp8-state posture:

  LEG 1 (ns/prefill route, unique branch): 2-seq prefill, slots [5, 9].
       Dual: gather ssm[[5,9]]->fp16 + conv[[5,9]], indices [0,1], run,
       scatter both home. PASS = conv rows written {5,9} (v124 bug wrote
       {0,1}), ssm rows changed within {5,9} only, rows 0/1 untouched.

  LEG 2 (spec/decode route, static flat branch): 2 spec requests,
       num_spec=4 (W=5), DISTINCT per-rollback columns — req0 slots
       [5,6,7,8,9], req1 [10,11,12,13,14]. Flat gather 10 rows, remap
       arange(10).reshape(2,5), num_accepted [3,2]. Both kernels
       checkpoint the rolling state at EVERY column t_local (conv:
       causal_conv1d.hpp:827-845; ssm: gated_delta_rule.hpp:527-540).
       PASS = changed-row sets IDENTICAL to the CONTROL leg (same inputs,
       real indices against real fp16 pools) for both pools, and no row
       outside {5..14} changes.

Inputs are O(1) (not 0.05-scaled): state updates must survive fp8
quantization to be observable in pool bytes. DK=DV=32 satisfies the
chunked spec kernel's head_k_dim % sub_group_size == 0.

Run inside a v1.2.25-raw-lineage container AFTER patch_v126_dualbridge.py.
"""
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers _xpu_C.gdn_attention)

torch.manual_seed(126)
torch.set_grad_enabled(False)

NK, NV, DK, DV, TP = 2, 2, 32, 32, 1
NUM_SPEC = 4                                   # production MTP x4
W = NUM_SPEC + 1
KTP, VTP = NK // TP, NV // TP
CONV_DIM = KTP * (2 * DK + DV * (NV // NK))
QKVZ_DIM = KTP * (2 * DK + 2 * DV * (NV // NK))
BA_DIM = KTP * (2 * (NV // NK))
P = 32                                          # pool slots
FP8 = torch.float8_e4m3fn

dev = torch.device("xpu")


def fresh_pools():
    conv = torch.randn(P, W - 1, CONV_DIM, dtype=torch.float16, device=dev)
    ssm = (
        torch.randn(P, VTP, DV, DK, dtype=torch.float16, device=dev)
        .to(FP8)
        .contiguous()
    )
    return conv, ssm


def common_inputs(T):
    core = torch.zeros(T, VTP, DV, dtype=torch.float16, device=dev)
    z = torch.zeros(T, VTP, DV, dtype=torch.float16, device=dev)
    qkvz = torch.randn(T, QKVZ_DIM, dtype=torch.float16, device=dev)
    ba = torch.randn(T, BA_DIM, dtype=torch.float16, device=dev)
    cw = torch.randn(CONV_DIM, W, dtype=torch.float16, device=dev)
    cb = torch.zeros(CONV_DIM, dtype=torch.float16, device=dev)
    A_log = torch.zeros(VTP, dtype=torch.float32, device=dev)
    dtb = torch.ones(VTP, dtype=torch.float16, device=dev) * 0.5
    return core, z, qkvz, ba, cw, cb, A_log, dtb


def run_ns(conv_arg, ssm_arg, indices, T, seq_lens, has_init):
    core, z, qkvz, ba, cw, cb, A_log, dtb = common_inputs(T)
    qsl = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)),
                       dtype=torch.int32, device=dev)
    idx = torch.tensor(indices, dtype=torch.int32, device=dev)
    his = torch.tensor(has_init, dtype=torch.bool, device=dev)
    torch.ops._xpu_C.gdn_attention(
        core, z, qkvz, ba, NK, NV, DK, DV,
        conv_state=conv_arg, ssm_state=ssm_arg,
        conv_weights=cw, conv_bias=cb, activation="silu",
        A_log=A_log, dt_bias=dtb,
        num_prefills=len(seq_lens), num_decodes=0, num_spec_decodes=0,
        has_initial_state=his,
        non_spec_query_start_loc=qsl,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=idx,
        spec_query_start_loc=None, spec_token_indx=None,
        spec_state_indices_tensor=None, num_accepted_tokens=None,
        num_actual_tokens=T, tp_size=TP, reorder_input=True,
    )
    torch.xpu.synchronize()


def run_spec(conv_arg, ssm_arg, remap, n, accepted):
    T = n * W
    core, z, qkvz, ba, cw, cb, A_log, dtb = common_inputs(T)
    qsl = torch.tensor([i * W for i in range(n + 1)], dtype=torch.int32, device=dev)
    tok = torch.arange(T, dtype=torch.int32, device=dev)
    acc = torch.tensor(accepted, dtype=torch.int32, device=dev)
    torch.ops._xpu_C.gdn_attention(
        core, z, qkvz, ba, NK, NV, DK, DV,
        conv_state=conv_arg, ssm_state=ssm_arg,
        conv_weights=cw, conv_bias=cb, activation="silu",
        A_log=A_log, dt_bias=dtb,
        num_prefills=0, num_decodes=0, num_spec_decodes=n,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=None,
        spec_query_start_loc=qsl, spec_token_indx=tok,
        spec_state_indices_tensor=remap, num_accepted_tokens=acc,
        num_actual_tokens=T, tp_size=TP, reorder_input=True,
    )
    torch.xpu.synchronize()


def rows_changed_conv(conv, conv0):
    return set((conv != conv0).any(dim=(1, 2)).nonzero().flatten().tolist())


def rows_changed_ssm_bytes(ssm, ssm0):
    return set(
        (ssm.view(torch.uint8) != ssm0.view(torch.uint8))
        .any(dim=(1, 2, 3))
        .nonzero()
        .flatten()
        .tolist()
    )


# ============================ LEG 1: ns route ===============================
SLOTS1 = [5, 9]
conv1, ssm1 = fresh_pools()
conv1_0, ssm1_0 = conv1.clone(), ssm1.clone()
# dual bridge: gather BOTH pools by the unique slot set, remap to [0, 1]
uniq1 = torch.tensor(SLOTS1, dtype=torch.int64, device=dev)
ssm_run1 = ssm1.index_select(0, uniq1).to(torch.float16).contiguous()
conv_run1 = conv1.index_select(0, uniq1).contiguous()
run_ns(conv_run1, ssm_run1, [0, 1], T=5, seq_lens=[3, 2], has_init=[False, False])
ssm1.view(torch.uint8).index_copy_(
    0, uniq1, ssm_run1.to(FP8).view(torch.uint8)
)
conv1.index_copy_(0, uniq1, conv_run1)
torch.xpu.synchronize()

c1 = rows_changed_conv(conv1, conv1_0)
s1 = rows_changed_ssm_bytes(ssm1, ssm1_0)
leg1_pass = c1 == {5, 9} and s1 <= {5, 9} and len(s1) > 0
print("LEG1 ns-route dual: conv rows written = %s ; ssm rows written = %s"
      % (sorted(c1), sorted(s1)))
print("LEG1_PASS=%s (expect conv {5,9} not {0,1}; ssm within {5,9}, rows 0/1 untouched)" % leg1_pass)

# ========================== LEG 2: spec route ===============================
# CONTROL: same inputs, real indices against real fp16 pools -> ground-truth
# changed-row sets.
SPEC_SLOTS = [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]  # distinct per column
n2 = 2
allowed2 = set(range(5, 15))

torch.manual_seed(2828)
convC = torch.randn(P, W - 1, CONV_DIM, dtype=torch.float16, device=dev)
ssmC = torch.randn(P, VTP, DV, DK, dtype=torch.float16, device=dev)
convC0, ssmC0 = convC.clone(), ssmC.clone()
idxC = torch.tensor(SPEC_SLOTS, dtype=torch.int32, device=dev)
run_spec(convC, ssmC, idxC, n2, accepted=[3, 2])
ctrl_conv = rows_changed_conv(convC, convC0)
ctrl_ssm = rows_changed_ssm_bytes(ssmC, ssmC0)
print("LEG2 control (real idx, fp16 pools): conv rows = %s ; ssm rows = %s"
      % (sorted(ctrl_conv), sorted(ctrl_ssm)))

# DUAL: fp8 ssm pool + fp16 conv pool, static flat remap (arange rows).
conv2, ssm2 = fresh_pools()
conv2_0, ssm2_0 = conv2.clone(), ssm2.clone()
flat_idx = idxC.reshape(-1)
uniq2 = flat_idx.to(torch.int64)
remap2 = (
    torch.arange(n2 * W, dtype=torch.int32, device=dev).reshape(n2, W)
)
ssm_run2 = ssm2.index_select(0, uniq2).to(torch.float16).contiguous()
conv_run2 = conv2.index_select(0, uniq2).contiguous()
run_spec(conv_run2, ssm_run2, remap2, n2, accepted=[3, 2])
ssm2.view(torch.uint8).index_copy_(
    0, uniq2, ssm_run2.to(FP8).view(torch.uint8)
)
conv2.index_copy_(0, uniq2, conv_run2)
torch.xpu.synchronize()

c2 = rows_changed_conv(conv2, conv2_0)
s2 = rows_changed_ssm_bytes(ssm2, ssm2_0)
leg2_pass = (
    c2 == ctrl_conv
    and s2 == ctrl_ssm
    and c2 <= allowed2
    and s2 <= allowed2
    and len(c2) > 0
    and len(s2) > 0
)
print("LEG2 spec-route dual: conv rows written = %s ; ssm rows written = %s"
      % (sorted(c2), sorted(s2)))
print("LEG2_PASS=%s (row sets must EQUAL control; nothing outside 5..14)" % leg2_pass)

print("P28_DUAL_BRIDGE_VERDICT: %s"
      % ("PASS" if (leg1_pass and leg2_pass) else "FAIL"))
