#!/usr/bin/env python3
"""p29a_kernel_patch.py — v126 P29A: remove the extra quantization
roundtrips from the ESIMD fp8 SSM-state path (directive: "remove all
possible extra quantization roundtrips").

Audit result (whole esimd_kernels tree):
- gdn_conv_fused_seq.h / gdn_conv_fused.h have NO token loops: one state
  load + one store per kernel call = pool residency, nothing to chain.
- The ONLY pool-roundtrip chain is gdn_spec_update_seq in
  gdn_conv_fused_seq_spec.h: per draft token it LOADS the prev state from
  pool slot[t-1] (dequant) and STORES to slot[t] (quant) — W loads + W
  requantizing round-trips per verify round, where the conv chain (s0/s1/s2)
  is already register-chained.
- e5m2 load/store are already minimal (3-op bit lift / RNE truncate).
- e4m3 LOAD wastes three fp16<->f32 conversions and maps NaN codes 0x7F/0xFF
  to finite 480.0 (diverges from torch c10 overflow->NaN semantics).

EDIT 1 (utils.h): single-conversion e4m3 load — magnitude converts once,
sign ORs into the fp32 bit pattern, NaN codes -> qNaN (exact c10 parity;
matches the bridged c10 reference class that passed the P28 quality gate).

EDIT 2-6 (gdn_conv_fused_seq_spec.h): fp8 REGISTER CHAIN — the SSM state
carries across draft tokens in fp32 registers owned by the caller
(ch_h0_lo..ch_h1_hi); token 0 of each verify round dequantizes the rollback
state from the pool exactly as before, tokens 1..W-1 REUSE the registers:
W-1 loads removed, zero requantization feedback (checkpoints still quantize
once per column — required for rollback). fp16 is NEVER chained
(kFp8Chain = sizeof(StateT)==1, constexpr): identical reload dataflow, so
the production fp16 lane stays bit-identical.

Idempotent (marker "v126 P29A"), per-file backups *.v129prebak.
"""
import pathlib
import shutil
import sys

TREE = pathlib.Path(
    "/root/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels"
)
MARKER = "v126 P29A"

# ---------------------------- EDIT 1: utils.h -------------------------------
UTILS = TREE / "utils.h"

UTILS_OLD = """ESIMD_INLINE simd<float, 64> esimd_e4m3_load(simd<uint8_t, 64> b) {
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
}"""

UTILS_NEW = """/* v126 P29A: single-conversion e4m3 load. The v125 load took THREE
 * fp16<->f32 round trips (value detour + fp16 sign re-injection); here the
 * magnitude converts once, the sign bit ORs straight into the fp32 pattern,
 * and NaN codes 0x7F/0xFF decode to qNaN — exact c10 parity (the old load
 * returned finite 480.0 for the overflow byte, diverging from torch
 * .to(float8_e4m3fn) overflow->NaN semantics). */
ESIMD_INLINE simd<float, 64> esimd_e4m3_load(simd<uint8_t, 64> b) {
    simd<uint16_t, 64> u16 = convert<uint16_t>(b);
    simd<uint16_t, 64> e4  = (u16 >> 3) & 0xFu;
    simd<uint16_t, 64> m3  = u16 & 0x7u;
    simd<uint16_t, 64> pre = ((e4 + 8u) << 10) | (m3 << 7);  /* normal: 2^(e4-7)*(1+m/8) */
    simd<uint16_t, 64> sub = (9u << 10) | (m3 << 7);        /* subnormal pre: (1+m/8)*2^-6 */
    pre.merge(sub, e4 == 0u);
    simd<fp16, 64> pre_h = pre.template bit_cast_view<fp16>().read();
    simd<float, 64> f(pre_h);
    simd<float, 64> corrected = f - 0.015625f;  /* hidden 2^-6 -> m*2^-9, exact */
    f.merge(corrected, e4 == 0u);
    simd<uint32_t, 64> fb = f.template bit_cast_view<uint32_t>().read();
    fb |= convert<uint32_t>(u16 >> 7) << 31;    /* sign straight into fp32 bits */
    simd<float, 64> out = fb.template bit_cast_view<float>().read();
    simd<uint32_t, 64> qnan_bits(0x7FC00000u);
    simd<float, 64> qnan = qnan_bits.template bit_cast_view<float>().read();
    out.merge(qnan, (u16 & 0x7Fu) == 0x7Fu);    /* NaN codes -> qNaN (c10 parity) */
    return out;
}"""

# ------------------- EDITS 2-6: gdn_conv_fused_seq_spec.h -------------------
SPEC = TREE / "gdn_conv_fused_seq_spec.h"

SIG_OLD = """    int gdn_K,
    int gdn_V,
    float attn_scale)
{
    float q_inv ="""

SIG_NEW = """    int gdn_K,
    int gdn_V,
    float attn_scale,
    simd<float, 64>& h0_lo,
    simd<float, 64>& h0_hi,
    simd<float, 64>& h1_lo,
    simd<float, 64>& h1_hi,
    bool h_chained)
{
    float q_inv ="""

LOAD_OLD = """    simd<float, 64> h0_lo(0.0f), h0_hi(0.0f);
    simd<float, 64> h1_lo(0.0f), h1_hi(0.0f);
    if (state_base != nullptr) {
        StateT* sr0 = state_base + (int64_t)(vi0 + 0) * gdn_K;
        StateT* sr1 = state_base + (int64_t)(vi0 + 1) * gdn_K;
        h0_lo = lsc_load_state_64_seq(sr0);
        h0_hi = lsc_load_state_64_seq(sr0 + 64);
        h1_lo = lsc_load_state_64_seq(sr1);
        h1_hi = lsc_load_state_64_seq(sr1 + 64);
    }"""

LOAD_NEW = """    /* v126 P29A: fp8 register chain — h* are caller-owned carriers. The
     * first token of a verify round dequantizes the pool state exactly as
     * before; later draft tokens REUSE the fp32 registers instead of
     * round-tripping through the pool (W-1 loads + all requantization
     * feedback removed; checkpoints still quantize once per column for
     * rollback). fp16 is never chained: identical reload dataflow, so the
     * production fp16 lane stays bit-identical. */
    if (!h_chained) {
        h0_lo = simd<float, 64>(0.0f);
        h0_hi = simd<float, 64>(0.0f);
        h1_lo = simd<float, 64>(0.0f);
        h1_hi = simd<float, 64>(0.0f);
        if (state_base != nullptr) {
            StateT* sr0 = state_base + (int64_t)(vi0 + 0) * gdn_K;
            StateT* sr1 = state_base + (int64_t)(vi0 + 1) * gdn_K;
            h0_lo = lsc_load_state_64_seq(sr0);
            h0_hi = lsc_load_state_64_seq(sr0 + 64);
            h1_lo = lsc_load_state_64_seq(sr1);
            h1_hi = lsc_load_state_64_seq(sr1 + 64);
        }
    }"""

LOOP_OLD = """    for (int t = 0; t < num_spec_tokens; ++t) {"""

LOOP_NEW = """    /* v126 P29A: fp8 SSM-state register chain carriers — persist across the
     * draft-token loop; kFp8Chain is constexpr so fp16 instantiations
     * never chain (legacy reload dataflow, bit-identical). */
    constexpr bool kFp8Chain = (sizeof(StateT) == 1);
    simd<float, 64> ch_h0_lo(0.0f), ch_h0_hi(0.0f);
    simd<float, 64> ch_h1_lo(0.0f), ch_h1_hi(0.0f);
    for (int t = 0; t < num_spec_tokens; ++t) {"""

SAVEIDX_OLD = """        const int save_state_idx = spec_state_indices_ptr[state_row + t];"""

SAVEIDX_NEW = """        const int save_state_idx = spec_state_indices_ptr[state_row + t];
        /* v126 P29A: chain from the previous draft token — its save slot IS
         * this token's prev slot (t-1), per the allocator's distinct
         * per-column ids. Token 0 always loads the rollback state from the
         * pool; a missing prev slot falls back to the legacy load path. */
        const bool ch_chained = kFp8Chain && t > 0 && prev_state_idx >= 0;"""

CALL_OLD = """            prev_state_idx, save_state_idx, tid, hv, HV, gdn_K, gdn_V,
            attn_scale);"""

CALL_NEW = """            prev_state_idx, save_state_idx, tid, hv, HV, gdn_K, gdn_V,
            attn_scale, ch_h0_lo, ch_h0_hi, ch_h1_lo, ch_h1_hi, ch_chained);"""

EDITS = [
    (UTILS, "e4m3_load rewrite", [(UTILS_OLD, UTILS_NEW)]),
    (SPEC, "spec register chain", [
        (SIG_OLD, SIG_NEW),
        (LOAD_OLD, LOAD_NEW),
        (LOOP_OLD, LOOP_NEW),
        (SAVEIDX_OLD, SAVEIDX_NEW),
        (CALL_OLD, CALL_NEW),
    ]),
]


def main() -> int:
    for path, label, pairs in EDITS:
        text = path.read_text()
        if MARKER in text:
            print("[p29a] %s: marker already present — skip (%s)"
                  % (path.name, label))
            continue
        bak = path.with_suffix(path.suffix + ".v129prebak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print("[p29a] backup %s" % bak)
        for old, new in pairs:
            n = text.count(old)
            if n != 1:
                print("[p29a] ABORT %s: anchor x%d (need 1) for %s"
                      % (path.name, n, old.splitlines()[0][:60]))
                return 1
            text = text.replace(old, new, 1)
        path.write_text(text)
        print("[p29a] %s: %d edit(s) applied (%s); markers=%d"
              % (path.name, len(pairs), label, text.count(MARKER)))
    # post-verify
    ok = True
    for path, _, _ in EDITS:
        t = path.read_text()
        c = t.count(MARKER)
        print("[p29a] verify %s markers=%d" % (path.name, c))
        ok = ok and c >= 1
    print("[p29a] P29A_KERNEL_PATCH_%s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
