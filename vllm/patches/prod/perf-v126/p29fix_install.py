#!/usr/bin/env python3
"""p29fix_install.py — v126 P29B-FIX integrator (host-side, idempotent).

Root cause (proven, p29b11/p29b13): the spec kernel's rollback reads race
with sibling work-groups of the same sequence. Ring position init_col is
read at work-group start and rewritten at t==init_col; the row's q/k dims
are written by wg hv==0 and each V dim by wg hv', so with nsd*HV
work-groups (48 at nsd=2/HV=24) exceeding the ~32 HW residency slots a
late-starting wg reads sibling post-checkpoint bytes (exact-zero conv
windows under zero input; corrupted dims [0,3072) = retired writers).

Fix: pre-launch snapshot of the ring rows. The wrapper gathers the conv
rows and SSM pool rows (by ring position, -1 clamped to 0) via
at::index_select on the same in-order XPU stream; the kernel reads the
init conv window and the token-0 SSM row from the snapshot (never written
in-kernel), keeps t>=1 SSM reads on the live pool (own prior store,
program ordered — snapshot would be stale there), and leaves all
checkpoint stores unchanged.

This script:
  1. backs up the pristine tree files ONCE to /root/build/p29fix_backup/
  2. copies the fixed production + dbg headers into csrc/xpu/esimd_kernels/
  3. replaces the WHOLE production wrapper esimd_gdn_conv_fused_seq_spec
     and the WHOLE dbg wrapper esimd_gdn_conv_fused_seq_spec_dbg in
     csrc/xpu/esimd_kernel_lgrf.sycl with snapshot-building versions
     (span-located by symbol; guarded by 'p_conv_snap' — a string that
     actually occurs in the inserted text)
  4. hard-verifies occurrence counts

Idempotent: re-running is a no-op.
"""
import os
import shutil
import sys

TREE = "/root/llm-scaler/vllm/custom-esimd-kernels-vllm"
SRC = "/root/build"
SYCL = TREE + "/csrc/xpu/esimd_kernel_lgrf.sycl"
BK = "/root/build/p29fix_backup"
FIX_MARK = "v126 P29B-FIX"

PROD_HDR = "p29fix_gdn_conv_fused_seq_spec.h"
DBG_HDR = "p29fix_gdn_conv_fused_seq_spec_dbg.h"

PROD_WRAPPER = '''at::Tensor esimd_gdn_conv_fused_seq_spec(
    at::Tensor qkvz,
    at::Tensor conv_state,
    at::Tensor conv_weight,
    at::Tensor conv_bias,
    at::Tensor spec_state_indices,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor ba,
    at::Tensor ssm_state,
    at::Tensor output,
    at::Tensor z_out,
    at::Tensor token_indx,
    at::Tensor num_accepted_tokens,
    int64_t num_spec_decodes,
    int64_t num_spec_tokens,
    int64_t H,
    int64_t HV,
    int64_t K,
    int64_t V,
    double scale)
{
    auto& dpcpp_queue = get_device_queue(qkvz);

    auto* p_qkvz = reinterpret_cast<const fp16*>(qkvz.data_ptr());
    auto* p_cstate = reinterpret_cast<fp16*>(conv_state.data_ptr());
    auto* p_cweight = reinterpret_cast<const fp16*>(conv_weight.data_ptr());
    auto* p_cbias = reinterpret_cast<const fp16*>(conv_bias.data_ptr());
    auto* p_spec_idx = spec_state_indices.data_ptr<int>();
    auto* p_alog = reinterpret_cast<const fp16*>(A_log.data_ptr());
    auto* p_dtbias = reinterpret_cast<const fp16*>(dt_bias.data_ptr());
    auto* p_ba = reinterpret_cast<const fp16*>(ba.data_ptr());
    void* p_sstate_raw = ssm_state.data_ptr();  // v125 P22A
    auto* p_out = reinterpret_cast<fp16*>(output.data_ptr());
    auto* p_zout = reinterpret_cast<fp16*>(z_out.data_ptr());
    auto* p_token_indx = token_indx.data_ptr<int>();
    auto* p_accepted = num_accepted_tokens.data_ptr<int>();

    // v126 P29B-FIX: snapshot the ring rows BEFORE launch. The kernel's
    // rollback reads (conv window at ring position init_col; SSM init row
    // at token 0) target rows the kernel itself rewrites during the walk,
    // and the writers are SIBLING work-groups of the same sequence (hv==0
    // owns the q/k dims, hv' owns V dim hv'), so a live read observes a
    // scheduling-dependent mix of pre/post-checkpoint bytes once
    // nsd*HV work-groups exceed HW residency (~32 slots; 48 at nsd=2,
    // HV=24). Gather by ring position (snapshot row r == seq*NST + col);
    // -1 slots clamp to row 0 and are never dereferenced (the kernel's
    // value guards stay). index_select runs on the same in-order XPU
    // stream as the launch, so ordering is guaranteed.
    auto snap_rows = spec_state_indices.reshape({-1})
                         .to(at::kLong).clamp_min(0);
    auto conv_snap = at::index_select(
        conv_state.reshape({conv_state.size(0), -1}), 0, snap_rows)
                         .contiguous();
    auto ssm_snap = at::index_select(
        ssm_state.reshape({ssm_state.size(0), -1}), 0, snap_rows)
                        .contiguous();
    auto* p_conv_snap = reinterpret_cast<const fp16*>(conv_snap.data_ptr());
    void* p_ssm_snap_raw = ssm_snap.data_ptr();
    const int64_t conv_snap_stride0 = conv_snap.size(1);
    const int64_t ssm_snap_stride0 = ssm_snap.size(1);

    // v125 P22A: dispatch on SSM state element type (fp16 default;
    // fp8 pools run the fp8-native kernels; conv_state stays fp16).
    const at::ScalarType ssm_dt = ssm_state.scalar_type();
    if (ssm_dt == at::kFloat8_e4m3fn) {
    gdn_conv_fused_seq_spec_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e4m3_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<esimd_e4m3_state_t*>(p_ssm_snap_raw),
        ssm_snap_stride0,
        dpcpp_queue);
    } else if (ssm_dt == at::kFloat8_e5m2) {
    gdn_conv_fused_seq_spec_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e5m2_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<esimd_e5m2_state_t*>(p_ssm_snap_raw),
        ssm_snap_stride0,
        dpcpp_queue);
    } else {
    gdn_conv_fused_seq_spec_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<fp16*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<fp16*>(p_ssm_snap_raw), ssm_snap_stride0,
        dpcpp_queue);
    }
    return output;
}
'''

DBG_WRAPPER = '''// v126 P29B-DIAG: instrumented wrapper — same launch, plus a per
// (seq,t,hv,tid) 20-slot fp32 record. Diagnostic only.
// v126 P29B-FIX revision: snapshots the ring rows exactly like the
// production wrapper so the records reflect the fixed dataflow.
at::Tensor esimd_gdn_conv_fused_seq_spec_dbg(
    at::Tensor qkvz,
    at::Tensor conv_state,
    at::Tensor conv_weight,
    at::Tensor conv_bias,
    at::Tensor spec_state_indices,
    at::Tensor A_log,
    at::Tensor dt_bias,
    at::Tensor ba,
    at::Tensor ssm_state,
    at::Tensor output,
    at::Tensor z_out,
    at::Tensor token_indx,
    at::Tensor num_accepted_tokens,
    int64_t num_spec_decodes,
    int64_t num_spec_tokens,
    int64_t H,
    int64_t HV,
    int64_t K,
    int64_t V,
    double scale,
    at::Tensor dbg)
{
    auto& dpcpp_queue = get_device_queue(qkvz);
    auto* p_qkvz = reinterpret_cast<const fp16*>(qkvz.data_ptr());
    auto* p_cstate = reinterpret_cast<fp16*>(conv_state.data_ptr());
    auto* p_cweight = reinterpret_cast<const fp16*>(conv_weight.data_ptr());
    auto* p_cbias = reinterpret_cast<const fp16*>(conv_bias.data_ptr());
    auto* p_spec_idx = spec_state_indices.data_ptr<int>();
    auto* p_alog = reinterpret_cast<const fp16*>(A_log.data_ptr());
    auto* p_dtbias = reinterpret_cast<const fp16*>(dt_bias.data_ptr());
    auto* p_ba = reinterpret_cast<const fp16*>(ba.data_ptr());
    void* p_sstate_raw = ssm_state.data_ptr();
    auto* p_out = reinterpret_cast<fp16*>(output.data_ptr());
    auto* p_zout = reinterpret_cast<fp16*>(z_out.data_ptr());
    auto* p_token_indx = token_indx.data_ptr<int>();
    auto* p_accepted = num_accepted_tokens.data_ptr<int>();
    auto* p_dbg = dbg.data_ptr<float>();

    // v126 P29B-FIX: same pre-launch ring-row snapshot as production.
    auto snap_rows = spec_state_indices.reshape({-1})
                         .to(at::kLong).clamp_min(0);
    auto conv_snap = at::index_select(
        conv_state.reshape({conv_state.size(0), -1}), 0, snap_rows)
                         .contiguous();
    auto ssm_snap = at::index_select(
        ssm_state.reshape({ssm_state.size(0), -1}), 0, snap_rows)
                        .contiguous();
    auto* p_conv_snap = reinterpret_cast<const fp16*>(conv_snap.data_ptr());
    void* p_ssm_snap_raw = ssm_snap.data_ptr();
    const int64_t conv_snap_stride0 = conv_snap.size(1);
    const int64_t ssm_snap_stride0 = ssm_snap.size(1);

    const at::ScalarType ssm_dt = ssm_state.scalar_type();
    if (ssm_dt == at::kFloat8_e4m3fn) {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e4m3_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<esimd_e4m3_state_t*>(p_ssm_snap_raw),
        ssm_snap_stride0,
        p_dbg, dpcpp_queue);
    } else if (ssm_dt == at::kFloat8_e5m2) {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e5m2_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<esimd_e5m2_state_t*>(p_ssm_snap_raw),
        ssm_snap_stride0,
        p_dbg, dpcpp_queue);
    } else {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<fp16*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap_stride0,
        reinterpret_cast<fp16*>(p_ssm_snap_raw), ssm_snap_stride0,
        p_dbg, dpcpp_queue);
    }
    return output;
}
'''


def replace_span(txt, start_sym, new_text):
    """Replace the whole function definition starting at start_sym (which
    must be unique and must NOT be a prefix of a longer symbol) up to and
    including its closing '    return output;\\n}'."""
    n = txt.count(start_sym)
    if n != 1:
        print("SPAN-ANCHOR BAD for %r: %d occurrences" % (start_sym, n))
        sys.exit(1)
    i = txt.find(start_sym)
    endmark = "\n    return output;\n}\n"
    j = txt.find(endmark, i)
    if j < 0:
        print("SPAN-END MISSING for %r" % start_sym)
        sys.exit(1)
    j += len(endmark)
    return txt[:i] + new_text + txt[j:]


# 1. one-time backup of the pre-fix tree state
os.makedirs(BK, exist_ok=True)
for rel in ("csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h",
            "csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_dbg.h",
            "csrc/xpu/esimd_kernel_lgrf.sycl"):
    dst = os.path.join(BK, rel.replace("/", "__") + ".pre_p29fix")
    if not os.path.exists(dst):
        shutil.copyfile(os.path.join(TREE, rel), dst)
        print("backed up:", dst)

# 2. headers (idempotent copies)
shutil.copyfile(os.path.join(SRC, PROD_HDR),
                TREE + "/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h")
shutil.copyfile(os.path.join(SRC, DBG_HDR),
                TREE + "/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_dbg.h")
print("copied fixed headers (prod + dbg)")

# 3. wrappers — whole-function replacement, guarded by a real symbol
with open(SYCL) as f:
    txt = f.read()
if "p_conv_snap" in txt:
    print("already: sycl wrappers carry P29B-FIX")
else:
    if '#include "esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"' not in txt:
        print("ANCHOR MISSING: dbg include (run p29dbg_install.py first)")
        sys.exit(1)
    txt = replace_span(
        txt, "at::Tensor esimd_gdn_conv_fused_seq_spec_dbg(", DBG_WRAPPER)
    txt = replace_span(
        txt, "at::Tensor esimd_gdn_conv_fused_seq_spec(", PROD_WRAPPER)
    with open(SYCL, "w") as f:
        f.write(txt)
    print("patched: sycl wrappers (prod + dbg) with P29B-FIX")

# 4. hard verification
with open(SYCL) as f:
    txt = f.read()
checks = {
    "prod host calls": txt.count("gdn_conv_fused_seq_spec_host(") == 3,
    "dbg host calls": txt.count("gdn_conv_fused_seq_spec_dbg_host(") == 3,
    "prod wrapper def": txt.count("at::Tensor esimd_gdn_conv_fused_seq_spec(") == 1,
    "dbg wrapper def": txt.count("at::Tensor esimd_gdn_conv_fused_seq_spec_dbg(") == 1,
    "snap builds": txt.count("at::index_select(") == 4,
    "fix marks": txt.count(FIX_MARK) >= 2,
}
for k, ok in checks.items():
    print("verify %-18s: %s" % (k, "OK" if ok else "FAIL"))
if not all(checks.values()):
    sys.exit(1)

for rel in ("csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec.h",
            "csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"):
    with open(os.path.join(TREE, rel)) as f:
        h = f.read()
    need = ("conv_snap_ptr" in h and "ssm_snap_ptr" in h
            and FIX_MARK in h)
    print("verify %-46s: %s" % (rel, "OK" if need else "FAIL"))
    if not need:
        sys.exit(1)

print("P29FIX_INSTALL_DONE")
