#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
using namespace sycl;

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
    simd<uint32_t, 64> norm = f_bits + 0xF887FFFFu;  /* ((7-127)<<23)+0x7FFFF */
    norm += mant_odd;
    norm = norm >> 20;
    simd<uint32_t, 64> res32 = norm;
    res32.merge(den, tiny);
    res32.merge(simd<uint32_t, 64>(0x7Fu), big);
    res32 |= sign >> 24;
    return convert<uint8_t>(res32);
}
