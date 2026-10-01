#!/usr/bin/env python3
"""m2_install.py — v127 M2: integrate the scaled-space e4m3 spec kernel
into the custom-esimd-kernels-vllm tree (build tree ONLY; the certified
production .so in lsv-test is never touched).

Additive and idempotent — every edit is guarded by a `v127 M2` marker:
  1. csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_scaled.h  (new file,
     copied from the staged source of truth next to this script)
  2. esimd_kernel_lgrf.sycl   — include line + wrapper fn
     esimd_gdn_conv_fused_seq_spec_scaled (mirrors the production spec
     wrapper incl. the v126 P29B-FIX ring snapshot; adds the ssm_scales
     tensor and an e4m3-only dispatch)
  3. include/kernel_ops.h     — declaration
  4. torch_extension_lgrf.cc  — TORCH_LIBRARY def/impl

Production symbols and kernels are byte-untouched: the original
kernels/entries are never edited, only new ones appended.

Usage: python3 m2_install.py /root/llm-scaler/vllm/custom-esimd-kernels-vllm
"""
import os
import shutil
import sys

TREE = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/llm-scaler/vllm/custom-esimd-kernels-vllm"
MARKER = "v127 M2"
HERE = os.path.dirname(os.path.abspath(__file__))

SYCL = os.path.join(TREE, "csrc/xpu/esimd_kernel_lgrf.sycl")
KOPS = os.path.join(TREE, "include/kernel_ops.h")
BIND = os.path.join(TREE, "csrc/xpu/torch_extension_lgrf.cc")
KHDR_DST = os.path.join(TREE, "csrc/xpu/esimd_kernels/gdn_conv_fused_seq_spec_scaled.h")
KHDR_SRC = os.path.join(HERE, "gdn_conv_fused_seq_spec_scaled.h")

INCLUDE_LINE = '#include "esimd_kernels/gdn_conv_fused_seq_spec_scaled.h"  // v127 M2\n'

WRAPPER = r'''
// v127 M2: scaled-space e4m3 speculative wrapper — same launch as the
// production spec wrapper (incl. the v126 P29B-FIX pre-launch ring
// snapshot), with per-(hv,k) scales applied inside the kernel at the
// pool boundary. Pools written by this op store e4m3(h*scale) bytes and
// are byte-compatible with the M1 bridge. e4m3 pools only.
at::Tensor esimd_gdn_conv_fused_seq_spec_scaled(
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
    at::Tensor ssm_scales)          // v127 M2: [HV, K] fp32, contiguous
{
    auto& dpcpp_queue = get_device_queue(qkvz);
    TORCH_CHECK(ssm_state.scalar_type() == at::kFloat8_e4m3fn,
        "esimd_gdn_conv_fused_seq_spec_scaled: e4m3 SSM pool required, got ",
        ssm_state.scalar_type());
    // Metadata checks only — value checks (finite, >0) run at calibration
    // time in M0; .item() here would sync the device on the forward path
    // (READOUT LAW).
    TORCH_CHECK(ssm_scales.scalar_type() == at::kFloat &&
                ssm_scales.dim() == 2 &&
                ssm_scales.size(0) == HV && ssm_scales.size(1) == K &&
                ssm_scales.is_contiguous(),
        "ssm_scales must be contiguous [HV, K] fp32; finite/positive is "
        "guaranteed by calibration (M0), not checked per-launch");

    auto* p_qkvz    = reinterpret_cast<const fp16*>(qkvz.data_ptr());
    auto* p_cstate  = reinterpret_cast<fp16*>(conv_state.data_ptr());
    auto* p_cweight = reinterpret_cast<const fp16*>(conv_weight.data_ptr());
    auto* p_cbias   = reinterpret_cast<const fp16*>(conv_bias.data_ptr());
    auto* p_spec_idx = spec_state_indices.data_ptr<int>();
    auto* p_alog    = reinterpret_cast<const fp16*>(A_log.data_ptr());
    auto* p_dtbias  = reinterpret_cast<const fp16*>(dt_bias.data_ptr());
    auto* p_ba      = reinterpret_cast<const fp16*>(ba.data_ptr());
    auto* p_sstate  = reinterpret_cast<esimd_e4m3_state_t*>(ssm_state.data_ptr());
    auto* p_out     = reinterpret_cast<fp16*>(output.data_ptr());
    auto* p_zout    = reinterpret_cast<fp16*>(z_out.data_ptr());
    auto* p_token_indx = token_indx.data_ptr<int>();
    auto* p_accepted = num_accepted_tokens.data_ptr<int>();
    auto* p_scales  = ssm_scales.data_ptr<float>();

    // v126 P29B-FIX snapshot (byte gather; stored-space rows decode with
    // inv_scale inside the kernel — snapshot stays a raw byte copy).
    auto snap_rows = spec_state_indices.reshape({-1})
                         .to(at::kLong).clamp_min(0);
    auto conv_snap = at::index_select(
        conv_state.reshape({conv_state.size(0), -1}), 0, snap_rows)
                         .contiguous();
    auto ssm_snap = at::index_select(
        ssm_state.reshape({ssm_state.size(0), -1}), 0, snap_rows)
                        .contiguous();
    auto* p_conv_snap = reinterpret_cast<const fp16*>(conv_snap.data_ptr());
    auto* p_ssm_snap = reinterpret_cast<esimd_e4m3_state_t*>(ssm_snap.data_ptr());

    gdn_conv_fused_seq_spec_scaled_host(
        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,
        p_alog, p_dtbias, p_ba, ba.stride(0), p_sstate, p_out, p_zout,
        p_token_indx, p_accepted, (int)num_spec_decodes,
        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,
        (float)scale, conv_state.stride(0), ssm_state.stride(0),
        p_conv_snap, conv_snap.size(1),
        p_ssm_snap, ssm_snap.size(1),
        p_scales, dpcpp_queue);
    return output;
}
'''

KOPS_DECL = r'''
// v127 M2: scaled-space e4m3 spec variant (per-(hv,k) scales at the
// pool boundary; pools store e4m3(h*scale) bytes, M1-bridge compatible)
at::Tensor esimd_gdn_conv_fused_seq_spec_scaled(
    at::Tensor qkvz,
    at::Tensor conv_state, at::Tensor conv_weight, at::Tensor conv_bias,
    at::Tensor spec_state_indices,
    at::Tensor A_log, at::Tensor dt_bias,
    at::Tensor ba, at::Tensor ssm_state,
    at::Tensor output, at::Tensor z_out,
    at::Tensor token_indx, at::Tensor num_accepted_tokens,
    int64_t num_spec_decodes, int64_t num_spec_tokens,
    int64_t H, int64_t HV, int64_t K, int64_t V,
    double scale, at::Tensor ssm_scales);
'''

BINDING = r'''
  // v127 M2: scaled-space e4m3 spec op
  m.def("esimd_gdn_conv_fused_seq_spec_scaled(Tensor qkvz, "
        "Tensor(a!) conv_state, Tensor conv_weight, Tensor conv_bias, "
        "Tensor spec_state_indices, "
        "Tensor A_log, Tensor dt_bias, "
        "Tensor ba, Tensor(b!) ssm_state, "
        "Tensor(c!) output, Tensor(d!) z_out, "
        "Tensor token_indx, Tensor num_accepted_tokens, "
        "int num_spec_decodes, int num_spec_tokens, "
        "int H, int HV, int K, int V, float scale, "
        "Tensor ssm_scales) -> Tensor(c!)");
  m.impl("esimd_gdn_conv_fused_seq_spec_scaled", torch::kXPU,
         &esimd_gdn_conv_fused_seq_spec_scaled);
'''


def insert_once(path, insertion, label):
    """Ensure `insertion` appears EXACTLY ONCE in the file. Collapses
    accidental duplicates (keeps the first). Returns True if a duplicate
    was collapsed."""
    text = open(path, encoding="utf-8").read()
    n = text.count(insertion)
    if n <= 1:
        return False
    first = text.find(insertion)
    text = text[:first + len(insertion)] + \
        text[first + len(insertion):].replace(insertion, "")
    open(path, "w", encoding="utf-8").write(text)
    print(f"  {label}: collapsed {n - 1} duplicate(s)")
    return True


def patch(path, anchor, insertion, label):
    """Insert `insertion` right after the line containing `anchor`.
    Guard = exact content match (not markers/labels — the v132 rerun
    showed those never match), with duplicate collapse."""
    insert_once(path, insertion, label)
    text = open(path, encoding="utf-8").read()
    if insertion in text:
        print(f"  {label}: already present, skip")
        return
    idx = text.find(anchor)
    if idx < 0:
        raise SystemExit(f"ABORT: anchor not found in {path}: {anchor!r}")
    eol = text.index("\n", idx) + 1
    open(path, "w", encoding="utf-8").write(text[:eol] + insertion + text[eol:])
    print(f"  {label}: inserted after {anchor.strip()[:60]!r}")


def main() -> int:
    print(f"M2_INSTALL tree={TREE}")
    # 1. kernel header
    shutil.copyfile(KHDR_SRC, KHDR_DST)
    print("  header: gdn_conv_fused_seq_spec_scaled.h copied "
          f"({os.path.getsize(KHDR_DST)} bytes)")
    # 2. .sycl: include after the production spec include
    patch(SYCL,
          '#include "esimd_kernels/gdn_conv_fused_seq_spec.h"',
          INCLUDE_LINE, "sycl_include")
    # 3. .sycl: wrapper appended at end of file (exactly once)
    insert_once(SYCL, WRAPPER, "sycl_wrapper")
    text = open(SYCL, encoding="utf-8").read()
    if WRAPPER not in text:
        open(SYCL, "a", encoding="utf-8").write(WRAPPER)
        print("  sycl_wrapper: appended")
    # 4. kernel_ops.h declaration after the END of the dbg declaration
    #    (anchor on its closing line — the decl spans multiple lines)
    patch(KOPS,
          "double scale, at::Tensor dbg);",
          KOPS_DECL, "kernel_ops_decl")
    # 5. pybind: insert INSIDE the TORCH_LIBRARY_FRAGMENT, right after the
    #    last existing m.impl (rfind("}") would anchor past PyInit and put
    #    the binding out of scope — v132 build failure, fixed). Exactly once.
    text = open(BIND, encoding="utf-8").read()
    if BINDING in text and \
            text.find("PyMODINIT_FUNC") < text.find(BINDING):
        start = text.find(BINDING)
        text = text[:start] + text[start + len(BINDING):]
        print("  binding: misplaced block stripped for repair")
    if BINDING not in text:
        anchor = "&esimd_gdn_conv_fused_seq_spec_dbg);"
        idx = text.find(anchor)
        if idx < 0:
            raise SystemExit("ABORT: fragment impl anchor not found")
        eol = text.index("\n", idx) + 1
        text = text[:eol] + BINDING + text[eol:]
        print("  binding: inserted after last fragment m.impl")
    else:
        print("  binding: already present, skip")
    open(BIND, "w", encoding="utf-8").write(text)
    insert_once(BIND, BINDING, "binding_dedupe")
    # proof
    for p in (SYCL, KOPS, BIND, KHDR_DST):
        n = open(p, encoding="utf-8").read().count("v127 M2")
        print(f"  marker count {os.path.basename(p)}: {n}")
    print("M2_INSTALL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
