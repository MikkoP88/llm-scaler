#!/usr/bin/env python3
"""p29dbg_install.py — v126 P29B-DIAG integrator (host-side, idempotent).

Stages the instrumented spec-kernel variant into the ESIMD tree:
  1. copy gdn_conv_fused_seq_spec_dbg.h into csrc/xpu/esimd_kernels/
  2. include + wrapper esimd_gdn_conv_fused_seq_spec_dbg in
     csrc/xpu/esimd_kernel_lgrf.sycl (after the production wrapper)
  3. declaration in include/kernel_ops.h
  4. TORCH_LIBRARY m.def/m.impl in csrc/xpu/torch_extension_lgrf.cc
Diagnostic-only; production kernels and ops are byte-untouched. Every edit
is guarded by the marker 'v126 P29B-DIAG' so re-runs are no-ops.
"""
import shutil
import sys

TREE = "/root/llm-scaler/vllm/custom-esimd-kernels-vllm"
SRC = "/root/build/gdn_conv_fused_seq_spec_dbg.h"
MARK = "v126 P29B-DIAG"


def patch(path, anchor, addition, tag, probe):
    with open(path) as f:
        txt = f.read()
    # Guard on a symbol that actually exists in the inserted text — a
    # marker+tag conjunct is useless when the tag never occurs in the
    # addition (that bug triple-patched decl/op and quad-registered the op).
    if probe in txt:
        print("already:", tag)
        return
    if anchor not in txt:
        print("ANCHOR MISSING for", tag, "in", path)
        sys.exit(1)
    txt = txt.replace(anchor, addition, 1)
    with open(path, "w") as f:
        f.write(txt)
    print("patched:", tag)


# 1. header copy
dst = TREE + "/csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"
shutil.copyfile(SRC, dst)
print("copied dbg header ->", dst)

# 2a. include in the SYCL unit
patch(
    TREE + "/csrc/xpu/esimd_kernel_lgrf.sycl",
    '#include "esimd_kernels/gdn_conv_fused_seq_spec.h"\n',
    '#include "esimd_kernels/gdn_conv_fused_seq_spec.h"\n'
    '#include "esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"'
    '  // v126 P29B-DIAG\n',
    "dbg_include",
    '#include "esimd_kernels/gdn_conv_fused_seq_spec_dbg.h"')

# 2b. wrapper after the production spec wrapper (anchor = its final lines)
WRAPPER = '''
// v126 P29B-DIAG: instrumented wrapper — same launch, plus a per
// (seq,t,hv,tid) 20-slot fp32 record. Diagnostic only.
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

    const at::ScalarType ssm_dt = ssm_state.scalar_type();
    if (ssm_dt == at::kFloat8_e4m3fn) {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e4m3_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_dbg, dpcpp_queue);
    } else if (ssm_dt == at::kFloat8_e5m2) {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<esimd_e5m2_state_t*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_dbg, dpcpp_queue);
    } else {
    gdn_conv_fused_seq_spec_dbg_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), reinterpret_cast<fp16*>(p_sstate_raw), p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_dbg, dpcpp_queue);
    }
    return output;
}
'''

with open(TREE + "/csrc/xpu/esimd_kernel_lgrf.sycl") as f:
    txt = f.read()
if "esimd_gdn_conv_fused_seq_spec_dbg(" in txt:
    print("already: dbg_wrapper")
else:
    # anchor: end of the production spec wrapper
    anchor = "    }\n    return output;\n}\n"
    idx = txt.find("at::Tensor esimd_gdn_conv_fused_seq_spec(")
    end = txt.find(anchor, idx)
    if idx < 0 or end < 0:
        print("ANCHOR MISSING for dbg_wrapper")
        sys.exit(1)
    insert_at = end + len(anchor)
    txt = txt[:insert_at] + WRAPPER + txt[insert_at:]
    with open(TREE + "/csrc/xpu/esimd_kernel_lgrf.sycl", "w") as f:
        f.write(txt)
    print("patched: dbg_wrapper")

# 3. kernel_ops.h declaration
patch(
    TREE + "/include/kernel_ops.h",
    "    int64_t H, int64_t HV, int64_t K, int64_t V,\n"
    "    double scale);\n",
    "    int64_t H, int64_t HV, int64_t K, int64_t V,\n"
    "    double scale);\n\n"
    "// v126 P29B-DIAG: instrumented spec variant (diagnostic only)\n"
    "at::Tensor esimd_gdn_conv_fused_seq_spec_dbg(\n"
    "    at::Tensor qkvz,\n"
    "    at::Tensor conv_state, at::Tensor conv_weight, at::Tensor conv_bias,\n"
    "    at::Tensor spec_state_indices,\n"
    "    at::Tensor A_log, at::Tensor dt_bias,\n"
    "    at::Tensor ba, at::Tensor ssm_state,\n"
    "    at::Tensor output, at::Tensor z_out,\n"
    "    at::Tensor token_indx, at::Tensor num_accepted_tokens,\n"
    "    int64_t num_spec_decodes, int64_t num_spec_tokens,\n"
    "    int64_t H, int64_t HV, int64_t K, int64_t V,\n"
    "    double scale, at::Tensor dbg);\n",
    "dbg_decl",
    "at::Tensor esimd_gdn_conv_fused_seq_spec_dbg(")

# 4. torch library registration
patch(
    TREE + "/csrc/xpu/torch_extension_lgrf.cc",
    '  m.impl("esimd_gdn_conv_fused_seq_spec", torch::kXPU,\n'
    '         &esimd_gdn_conv_fused_seq_spec);\n',
    '  m.impl("esimd_gdn_conv_fused_seq_spec", torch::kXPU,\n'
    '         &esimd_gdn_conv_fused_seq_spec);\n\n'
    '  // v126 P29B-DIAG: instrumented spec op (diagnostic only)\n'
    '  m.def("esimd_gdn_conv_fused_seq_spec_dbg(Tensor qkvz, "\n'
    '        "Tensor(a!) conv_state, Tensor conv_weight, Tensor conv_bias, "\n'
    '        "Tensor spec_state_indices, "\n'
    '        "Tensor A_log, Tensor dt_bias, "\n'
    '        "Tensor ba, Tensor(b!) ssm_state, "\n'
    '        "Tensor(c!) output, Tensor(d!) z_out, "\n'
    '        "Tensor token_indx, Tensor num_accepted_tokens, "\n'
    '        "int num_spec_decodes, int num_spec_tokens, "\n'
    '        "int H, int HV, int K, int V, float scale, "\n'
    '        "Tensor(e!) dbg) -> Tensor(c!)");\n'
    '  m.impl("esimd_gdn_conv_fused_seq_spec_dbg", torch::kXPU,\n'
    '         &esimd_gdn_conv_fused_seq_spec_dbg);\n',
    "dbg_op",
    'm.def("esimd_gdn_conv_fused_seq_spec_dbg')

print("P29DBG_INSTALL_DONE")
