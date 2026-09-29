#!/usr/bin/env python3
"""p22a_kernel_patch.py — v125 P22A: fp8-native SSM state in the ESIMD GDN
kernels (custom-esimd-kernels-vllm, the lgrf unit).

Surface A (ESIMD fused/seq/spec kernels) currently hardcodes fp16* SSM
state pointers. This patch templates every kernel/host on StateT and adds
e4m3fn/e5m2 load/store primitives so an fp8 SSM pool can run ESIMD-native
(decode + solo spec-1), removing the stage-A bridge's gather/remap/scatter
and its ~5x state-op penalty. conv_state stays fp16 everywhere (by design:
it is tiny and delta-coded). All math stays float; fp8 only frames the
state load/quantize boundary (64 elems = one 64B LSC block per half-row).

Conversion semantics (derived + verified vs c10, see PHASES P22):
  e5m2 load : byte<<8 — exact fp16 widening incl. subnormals
  e5m2 store: RNE truncation to top 8 bits + NaN-wraparound clamp (w>0xFF
              -> 0xFF; documented deviation vs c10: overflow -> inf not NaN)
  e4m3 load : exact incl. subnormals via subtract-hidden-unit trick
              (subnormal pre = (1+m/8)*2^-6, subtract 2^-6 -> m*2^-9);
              NaN codes 0x7F/0xFF decode to 480.0 (documented, never
              written by store)
  e4m3 store: c10-exact SIMD port (fp8_max 1087<<20, denorm magic-add
              141<<23, normal RNE via mant_odd + 0xC407FFFF wrap)

Idiom discipline: only repo-proven ESIMD primitives (convert<T>(),
bit_cast_view<T>().read(), 2-arg merge, elementwise integer simd ops — all
from fp8_GEMV_v2.h fp8_dequant :16-46, minus its subnormal flush which is
correct for weights but WRONG for state: these conversions are exact).

Edits (all additive or type-only; fp16 path compiles to identical code):
  utils.h     : tag types + 4 conversion helpers
  gdn_conv_fused.h        : fp8 overload pair; kernel_v9 + host templated
  gdn_conv_fused_seq.h    : fp8 _seq overload pair; kernel / large_h /
                            dispatch / host templated
  gdn_conv_fused_seq_spec.h: update_seq / spec_kernel / host templated
  esimd_kernel_lgrf.sycl  : the three wrappers dispatch on ssm dtype

Idempotent via 'v125 P22A' marker.
"""
import re
import sys

ROOT = "/root/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/xpu"
F = {
    "utils": ROOT + "/esimd_kernels/utils.h",
    "fused": ROOT + "/esimd_kernels/gdn_conv_fused.h",
    "seq": ROOT + "/esimd_kernels/gdn_conv_fused_seq.h",
    "spec": ROOT + "/esimd_kernels/gdn_conv_fused_seq_spec.h",
    "sycl": ROOT + "/esimd_kernel_lgrf.sycl",
}
src = {k: open(p).read() for k, p in F.items()}

if any("v125 P22A" in s for s in src.values()):
    print("ALREADY_APPLIED")
    sys.exit(0)


def sub(key, old, new, n):
    """Replace with exact-count assertion; abort (no partial write) on drift."""
    c = src[key].count(old)
    assert c == n, "%s: anchor count %d != %d for %r" % (key, c, n, old[:60])
    src[key] = src[key].replace(old, new)


# ============================ utils.h ============================
# append after the final `using namespace sycl;` (newline-agnostic)
UTILS_ADD = '''
/* ---- v125 P22A: fp8-native SSM state element types + conversions ----
 * State rows are K=128 elems = 128 fp8 bytes: two 64B LSC blocks per row
 * (lo at +0, hi at +64 elems); torch base >=512B aligned so 64B lsc_block
 * loads are safe. Math stays float; these frame only the load/quantize
 * boundary. Idioms proven by fp8_GEMV_v2.h fp8_dequant — except that its
 * fp8 subnormal FLUSH is a weight-path simplification we must NOT apply
 * to state: these conversions are exact for every representable value. */
struct esimd_e5m2_state_t { unsigned char bits; };
struct esimd_e4m3_state_t { unsigned char bits; };

ESIMD_INLINE simd<float, 64> esimd_e5m2_load(simd<uint8_t, 64> b) {
    simd<uint16_t, 64> w = convert<uint16_t>(b);
    w = w << 8;                                    /* exact: e5m2 == top 8 fp16 bits */
    simd<fp16, 64> h = w.template bit_cast_view<fp16>().read();
    return simd<float, 64>(h);
}

ESIMD_INLINE simd<uint8_t, 64> esimd_e5m2_store(simd<float, 64> v) {
    simd<fp16, 64> h(v);
    simd<uint16_t, 64> w = h.template bit_cast_view<uint16_t>().read();
    w = w + (0x7Fu + ((w >> 8) & 1u));             /* RNE truncation to top 8 bits */
    w = w >> 8;
    w.merge(simd<uint16_t, 64>(0xFFu), w > 0xFFu); /* NaN-wraparound clamp */
    return convert<uint8_t>(w);
}

ESIMD_INLINE simd<float, 64> esimd_e4m3_load(simd<uint8_t, 64> b) {
    simd<uint16_t, 64> u16 = convert<uint16_t>(b);
    simd<uint16_t, 64> sgn = (u16 >> 7) & 1;
    simd<uint16_t, 64> e4  = (u16 >> 3) & 0xF;
    simd<uint16_t, 64> m3  = u16 & 0x7;
    simd<uint16_t, 64> pre = ((e4 + 8) << 10) | (m3 << 7);  /* normal: 2^(e4-7)*(1+m/8) */
    simd<uint16_t, 64> sub = (9u << 10) | (m3 << 7);        /* subnormal pre: (1+m/8)*2^-6 */
    pre.merge(sub, e4 == 0);
    simd<fp16, 64> pre_h = pre.template bit_cast_view<fp16>().read();
    simd<float, 64> mag(pre_h);
    simd<float, 64> corrected = mag - 0.015625f;  /* hidden 2^-6 -> m*2^-9, exact */
    mag.merge(corrected, e4 == 0);
    simd<fp16, 64> res_h(mag);                    /* <=480, all exact in fp16 */
    simd<uint16_t, 64> res_bits = res_h.template bit_cast_view<uint16_t>().read();
    res_bits |= sgn << 15;
    return simd<float, 64>(res_bits.template bit_cast_view<fp16>().read());
}

ESIMD_INLINE simd<uint8_t, 64> esimd_e4m3_store(simd<float, 64> v) {
    simd<uint32_t, 64> f_bits = v.template bit_cast_view<uint32_t>().read();
    simd<uint32_t, 64> sign = f_bits & 0x80000000u;
    f_bits = f_bits ^ sign;
    auto big  = f_bits >= (1087u << 20);          /* >=480.0 -> NaN code 0x7F */
    auto tiny = f_bits < (121u << 23);            /* < 2^-6 -> denorm magic-add */
    simd<float, 64> fav = f_bits.template bit_cast_view<float>().read();
    simd<uint32_t, 64> magic_bits(141u << 23);
    simd<float, 64> magic = magic_bits.template bit_cast_view<float>().read();
    simd<float, 64> fsum = fav + magic;           /* float add performs the RNE */
    simd<uint32_t, 64> den =
        fsum.template bit_cast_view<uint32_t>().read() - (141u << 23);
    simd<uint32_t, 64> mant_odd = (f_bits >> 20) & 1u;
    simd<uint32_t, 64> norm = f_bits + 0xC407FFFFu;  /* ((7-127)<<23)+0x7FFFF */
    norm += mant_odd;
    norm = norm >> 20;
    simd<uint32_t, 64> res32 = norm;
    res32.merge(den, tiny);
    res32.merge(simd<uint32_t, 64>(0x7Fu), big);
    res32 |= sign >> 24;
    return convert<uint8_t>(res32);
}
'''
_u = src["utils"].rstrip("\n")
assert _u.endswith("using namespace sycl;"), "utils.h: trailing using-anchor not found"
src["utils"] = _u + "\n" + UTILS_ADD
# utils.h has NO include guard and is pulled in by every kernel header ->
# redefinition errors; pragma once is required (fix2 lesson)
src["utils"] = "#pragma once\n" + src["utils"]


def fp8_primitives(load_name, store_name, marker_comment):
    return (
        "\n/* v125 P22A: fp8-native state primitives (%s) */\n" % marker_comment
        + "ESIMD_INLINE simd<float, 64> %s(const esimd_e5m2_state_t* ptr) {\n" % load_name
        + "    return esimd_e5m2_load(xmem::lsc_block_load<unsigned char, 64,\n"
        + "        xmem::lsc_data_size::default_size,\n"
        + "        xmem::cache_hint::streaming, xmem::cache_hint::cached>(\n"
        + "        reinterpret_cast<const unsigned char*>(ptr)));\n"
        + "}\n\n"
        + "ESIMD_INLINE void %s(esimd_e5m2_state_t* ptr, simd<float, 64> val) {\n" % store_name
        + "    xmem::lsc_block_store<unsigned char, 64,\n"
        + "        xmem::lsc_data_size::default_size,\n"
        + "        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(\n"
        + "        reinterpret_cast<unsigned char*>(ptr), esimd_e5m2_store(val));\n"
        + "}\n\n"
        + "ESIMD_INLINE simd<float, 64> %s(const esimd_e4m3_state_t* ptr) {\n" % load_name
        + "    return esimd_e4m3_load(xmem::lsc_block_load<unsigned char, 64,\n"
        + "        xmem::lsc_data_size::default_size,\n"
        + "        xmem::cache_hint::streaming, xmem::cache_hint::cached>(\n"
        + "        reinterpret_cast<const unsigned char*>(ptr)));\n"
        + "}\n\n"
        + "ESIMD_INLINE void %s(esimd_e4m3_state_t* ptr, simd<float, 64> val) {\n" % store_name
        + "    xmem::lsc_block_store<unsigned char, 64,\n"
        + "        xmem::lsc_data_size::default_size,\n"
        + "        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(\n"
        + "        reinterpret_cast<unsigned char*>(ptr), esimd_e4m3_store(val));\n"
        + "}\n"
    )


# ====================== gdn_conv_fused.h ======================
sub(
    "fused",
    """ESIMD_INLINE void lsc_store_state_64(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}
""",
    """ESIMD_INLINE void lsc_store_state_64(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}
""" + fp8_primitives("lsc_load_state_64", "lsc_store_state_64", "overload set"),
    1,
)
sub(
    "fused",
    "ESIMD_INLINE void gdn_conv_fused_kernel_v9(",
    "template <typename StateT>\nESIMD_INLINE void gdn_conv_fused_kernel_v9(",
    1,
)
sub("fused", "    fp16* __restrict__ ssm_state_ptr,", "    StateT* __restrict__ ssm_state_ptr,", 1)
sub("fused", "        fp16* sstate_base = ssm_state_ptr +", "        StateT* sstate_base = ssm_state_ptr +", 1)
for i in range(4):
    sub(
        "fused",
        "        fp16* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        "        StateT* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        1,
    )
sub(
    "fused",
    "inline void gdn_conv_fused_host(",
    "template <typename StateT>\ninline void gdn_conv_fused_host(",
    1,
)
# host param (kernel param is the __restrict__ variant, already replaced)
sub("fused", "    fp16* ssm_state_ptr,", "    StateT* ssm_state_ptr,", 1)
sub(
    "fused",
    "            gdn_conv_fused_kernel_v9(",
    "            gdn_conv_fused_kernel_v9<StateT>(",
    1,
)

# ====================== gdn_conv_fused_seq.h ======================
sub(
    "seq",
    """ESIMD_INLINE void lsc_store_state_64_seq(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}
""",
    """ESIMD_INLINE void lsc_store_state_64_seq(fp16* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<fp16, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<fp16, 64>(val));
}
""" + fp8_primitives("lsc_load_state_64_seq", "lsc_store_state_64_seq", "overload set, _seq"),
    1,
)
sub(
    "seq",
    "template<int WG_SIZE>\nESIMD_INLINE void gdn_conv_fused_seq_kernel(",
    "template<int WG_SIZE, typename StateT>\nESIMD_INLINE void gdn_conv_fused_seq_kernel(",
    1,
)
# main kernel :118 + large_h :485
sub("seq", "    fp16* __restrict__ ssm_state_ptr,", "    StateT* __restrict__ ssm_state_ptr,", 2)
# main kernel :299 + large_h :587
sub("seq", "        fp16* sstate_base = ssm_state_ptr +", "        StateT* sstate_base = ssm_state_ptr +", 2)
# sr0/sr1 appear in VPT4 (:307/:308), VPT2 (:356/:357) and large_h (:591/:592)
for i in (0, 1):
    sub(
        "seq",
        "fp16* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        "StateT* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        3,
    )
for i in (2, 3):
    sub(
        "seq",
        "fp16* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        "StateT* sr%d = sstate_base + (int64_t)(vi0 + %d) * gdn_K;" % (i, i),
        1,
    )
sub(
    "seq",
    "ESIMD_INLINE void gdn_conv_fused_seq_kernel_large_h(",
    "template <typename StateT>\nESIMD_INLINE void gdn_conv_fused_seq_kernel_large_h(",
    1,
)
sub(
    "seq",
    "template<int WG_SIZE>\ninline void gdn_conv_fused_seq_dispatch(",
    "template<int WG_SIZE, typename StateT>\ninline void gdn_conv_fused_seq_dispatch(",
    1,
)
sub(
    "seq",
    "    fp16* ssm_state_ptr, const int* ssm_state_indices_ptr,",
    "    StateT* ssm_state_ptr, const int* ssm_state_indices_ptr,",
    1,
)
sub(
    "seq",
    "            gdn_conv_fused_seq_kernel<WG_SIZE>(",
    "            gdn_conv_fused_seq_kernel<WG_SIZE, StateT>(",
    1,
)
sub(
    "seq",
    "inline void gdn_conv_fused_seq_host(",
    "template <typename StateT>\ninline void gdn_conv_fused_seq_host(",
    1,
)
sub("seq", "    fp16* ssm_state_ptr,\n", "    StateT* ssm_state_ptr,\n", 1)
sub("seq", "        gdn_conv_fused_seq_dispatch<32>(", "        gdn_conv_fused_seq_dispatch<32, StateT>(", 1)
sub("seq", "        gdn_conv_fused_seq_dispatch<64>(", "        gdn_conv_fused_seq_dispatch<64, StateT>(", 1)

# ====================== gdn_conv_fused_seq_spec.h ======================
sub(
    "spec",
    "template <int WG_SIZE>\nESIMD_INLINE simd<float, 2> gdn_spec_update_seq(",
    "template <int WG_SIZE, typename StateT>\n"
    "ESIMD_INLINE simd<float, 2> gdn_spec_update_seq(",
    1,
)
# update_seq :31 + host :405
sub("spec", "    fp16* ssm_state_ptr,", "    StateT* ssm_state_ptr,", 2)
sub("spec", "    fp16* state_base = nullptr;", "    StateT* state_base = nullptr;  // v125 P22A: fp8-capable SSM state", 1)
sub("spec", "    fp16* save_base = nullptr;", "    StateT* save_base = nullptr;", 1)
# gdn_spec_update_seq body row pointers (state_base pair :82-83, save_base
# pair :109-110 pre-patch) — fix2: these four were missed in rev1
sub("spec", "fp16* sr0 = state_base +", "StateT* sr0 = state_base +", 1)
sub("spec", "fp16* sr1 = state_base +", "StateT* sr1 = state_base +", 1)
sub("spec", "fp16* sr0 = save_base +", "StateT* sr0 = save_base +", 1)
sub("spec", "fp16* sr1 = save_base +", "StateT* sr1 = save_base +", 1)
sub(
    "spec",
    "template <int WG_SIZE>\nESIMD_INLINE void gdn_conv_fused_seq_spec_kernel(",
    "template <int WG_SIZE, typename StateT>\n"
    "ESIMD_INLINE void gdn_conv_fused_seq_spec_kernel(",
    1,
)
sub("spec", "    fp16* __restrict__ ssm_state_ptr,", "    StateT* __restrict__ ssm_state_ptr,", 1)
sub(
    "spec",
    "        simd<float, 2> o_acc = gdn_spec_update_seq<WG_SIZE>(",
    "        simd<float, 2> o_acc = gdn_spec_update_seq<WG_SIZE, StateT>(",
    1,
)
sub(
    "spec",
    "inline void gdn_conv_fused_seq_spec_host(",
    "template <typename StateT>\ninline void gdn_conv_fused_seq_spec_host(",
    1,
)
sub(
    "spec",
    "            gdn_conv_fused_seq_spec_kernel<WG_SIZE>(",
    "            gdn_conv_fused_seq_spec_kernel<WG_SIZE, StateT>(",
    1,
)

# ====================== esimd_kernel_lgrf.sycl ======================
pat = re.compile(
    r"auto\* p_sstate\s+= reinterpret_cast<fp16\*>\(ssm_state\.data_ptr\(\)\);"
)
n = len(pat.findall(src["sycl"]))
assert n == 3, "sycl: p_sstate decls found %d != 3" % n
src["sycl"] = pat.sub(
    "void* p_sstate_raw = ssm_state.data_ptr();  // v125 P22A", src["sycl"]
)


def sycl_dispatch(fn, args_mid, args_tail, indent="    "):
    """Build the 3-way dtype dispatch around a host call."""
    def call(cast):
        return (
            "%s%s(\n" % (indent, fn)
            + args_mid.replace("p_sstate,", "%s," % cast)
            + args_tail
        )
    return (
        "%s// v125 P22A: dispatch on SSM state element type (fp16 default;\n"
        "%s// fp8 pools run the fp8-native kernels; conv_state stays fp16).\n"
        "%sconst at::ScalarType ssm_dt = ssm_state.scalar_type();\n"
        "%sif (ssm_dt == at::kFloat8_e4m3fn) {\n"
        "%s"
        "%s} else if (ssm_dt == at::kFloat8_e5m2) {\n"
        "%s"
        "%s} else {\n"
        "%s"
        "%s}\n"
    ) % (
        indent, indent, indent, indent,
        call("reinterpret_cast<esimd_e4m3_state_t*>(p_sstate_raw)"),
        indent,
        call("reinterpret_cast<esimd_e5m2_state_t*>(p_sstate_raw)"),
        indent,
        call("reinterpret_cast<fp16*>(p_sstate_raw)"),
        indent,
    )


# wrapper 1 + 2: identical arg blocks, distinct function names
for fn in ("gdn_conv_fused_host", "gdn_conv_fused_seq_host"):
    OLD = (
        "    %s(\n"
        "        p_qkvz, qkvz_stride0, p_cstate, p_cweight, p_cbias, p_csidx,\n"
        "        p_alog, p_dtbias, p_ba, ba_stride0,\n"
        "        p_sstate, p_ssidx, p_out, p_zout,\n"
        "        (int)N, (int)H, (int)HV, (int)K, (int)V,\n"
        "        (float)scale, conv_stride0, ssm_stride0,\n"
        "        dpcpp_queue);\n"
    ) % fn
    NEW = sycl_dispatch(
        fn,
        args_mid=(
            "        p_qkvz, qkvz_stride0, p_cstate, p_cweight, p_cbias, p_csidx,\n"
            "        p_alog, p_dtbias, p_ba, ba_stride0,\n"
            "        p_sstate, p_ssidx, p_out, p_zout,\n"
            "        (int)N, (int)H, (int)HV, (int)K, (int)V,\n"
            "        (float)scale, conv_stride0, ssm_stride0,\n"
            "        dpcpp_queue);\n"
        ),
        args_tail="",
    )
    sub("sycl", OLD, NEW, 1)

# wrapper 3 (spec)
OLD3 = (
    "    gdn_conv_fused_seq_spec_host(\n"
    "        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,\n"
    "        p_alog, p_dtbias, p_ba, ba.stride(0), p_sstate, p_out, p_zout,\n"
    "        p_token_indx, p_accepted, (int)num_spec_decodes,\n"
    "        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,\n"
    "        (float)scale, conv_state.stride(0), ssm_state.stride(0),\n"
    "        dpcpp_queue);\n"
)
NEW3 = sycl_dispatch(
    "gdn_conv_fused_seq_spec_host",
    args_mid=(
        "        p_qkvz, qkvz.stride(0), p_cstate, p_cweight, p_cbias, p_spec_idx,\n"
        "        p_alog, p_dtbias, p_ba, ba.stride(0), p_sstate, p_out, p_zout,\n"
        "        p_token_indx, p_accepted, (int)num_spec_decodes,\n"
        "        (int)num_spec_tokens, (int)H, (int)HV, (int)K, (int)V,\n"
        "        (float)scale, conv_state.stride(0), ssm_state.stride(0),\n"
        "        dpcpp_queue);\n"
    ),
    args_tail="",
)
sub("sycl", OLD3, NEW3, 1)

# ====================== write + post-conditions ======================
for k, s in src.items():
    assert "v125 P22A" in s, k + ": marker missing"
assert src["fused"].count("esimd_e5m2_state_t") >= 2
assert src["seq"].count("esimd_e4m3_state_t") >= 2
assert "gdn_conv_fused_kernel_v9<StateT>" in src["fused"]
assert "gdn_conv_fused_seq_kernel<WG_SIZE, StateT>" in src["seq"]
assert "gdn_spec_update_seq<WG_SIZE, StateT>" in src["spec"]
assert "gdn_conv_fused_seq_spec_kernel<WG_SIZE, StateT>" in src["spec"]
assert src["sycl"].count("at::kFloat8_e4m3fn") == 3
assert src["sycl"].count("at::kFloat8_e5m2") == 3
assert "reinterpret_cast<fp16*>(ssm_state.data_ptr())" not in src["sycl"]

for k, p in F.items():
    open(p, "w").write(src[k])
    print("PATCHED", p)
print("P22A_KERNEL_PATCHED")
