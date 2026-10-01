#pragma once

/*
 * v127 M2 — scaled-space e4m3 speculative GDN kernel.
 *
 * Sibling of gdn_conv_fused_seq_spec.h; everything except the SSM pool
 * I/O boundary is intentionally line-identical. Include AFTER
 * gdn_conv_fused_seq.h (shares its scalar helpers, dot128, and SLM
 * layout constants) and AFTER utils.h (e4m3 conversion primitives).
 *
 * WHY: raw e4m3 SSM pools fail P23D/P23F not because 8 bits are too few
 * but because the GDN state features span a dynamic range far wider than
 * e4m3's fixed grid — small-magnitude features land in the subnormal
 * tail (or flush to zero) while large ones are fine. Per-feature scaling
 * maps each feature's live range onto the full e4m3 grid:
 *
 *     stored byte  = e4m3( h * scale[hv][k] )     (scatter)
 *     h (register) = e4m3_load( byte ) * 1/scale  (gather)
 *
 * scale[hv][k] = 0.98 * 448 / runmax[hv][k] from live telemetry (M0
 * calibrate_offline.py / P29M collector). Relative quantization error
 * of in-range values is UNCHANGED (identical grid, scaled); the win is
 * that every feature now sits in-range by construction. The arithmetic
 * itself stays fp32 in registers — identical to the production fp8
 * lane; only the pool boundary changes.
 *
 * M1 (python bridge: gather→mul, mul→scatter around the existing
 * kernel) and M2 implement the SAME storage semantics; M2 removes the
 * two extra device roundtrips and the separate broadcast kernels by
 * applying the scale inside the native kernel. Pools written by M1 are
 * readable by M2 and vice versa (both store e4m3(h*scale) bytes).
 *
 * Scope: e4m3 pools ONLY (fp16 pools have no quantization boundary and
 * keep using the production kernel). The register chain (v126 P29A) is
 * preserved: h* stay fp32 across draft tokens; scale/inv_scale apply
 * exactly once at the pool boundary — first load (token 0, from the
 * P29B-FIX pre-launch snapshot) and each checkpoint save.
 *
 * ROLLBACK SAFETY: the wrapper-side ring snapshot gathers RAW stored
 * bytes (byte copy), so snapshot rows decode with the same inv_scale —
 * no wrapper change beyond passing the scales tensor through.
 */

/* ---- M2 scaled pool primitives (self-contained; utils.h conversions) ---- */
ESIMD_INLINE simd<float, 64> m2_scaled_load_64(
    const esimd_e4m3_state_t* ptr, const simd<float, 64>& inv_scale) {
    return esimd_e4m3_load(xmem::lsc_block_load<unsigned char, 64,
               xmem::lsc_data_size::default_size,
               xmem::cache_hint::streaming, xmem::cache_hint::cached>(
               reinterpret_cast<const unsigned char*>(ptr))) *
           inv_scale;
}

ESIMD_INLINE void m2_scaled_store_64(
    esimd_e4m3_state_t* ptr, simd<float, 64> val,
    const simd<float, 64>& scale) {
    xmem::lsc_block_store<unsigned char, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        reinterpret_cast<unsigned char*>(ptr),
        esimd_e4m3_store(val * scale));
}

template <int WG_SIZE>
ESIMD_INLINE simd<float, 2> gdn_spec_update_seq_scaled(
    const simd<float, 64>& q_lo,
    const simd<float, 64>& q_hi,
    const simd<float, 64>& k_lo,
    const simd<float, 64>& k_hi,
    const simd<float, 2>& v_f32,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_offset,
    esimd_e4m3_state_t* prev_read_base,
    esimd_e4m3_state_t* ssm_state_ptr,
    int64_t ssm_stride0,
    int save_state_idx,
    int tid,
    int hv,
    int HV,
    int gdn_K,
    int gdn_V,
    float attn_scale,
    simd<float, 64>& h0_lo,
    simd<float, 64>& h0_hi,
    simd<float, 64>& h1_lo,
    simd<float, 64>& h1_hi,
    bool h_chained,
    const simd<float, 64>& sc_lo,
    const simd<float, 64>& sc_hi,
    const simd<float, 64>& isc_lo,
    const simd<float, 64>& isc_hi)
{
    float q_inv =
        1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
    float k_inv =
        1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
    simd<float, 64> qn_lo = q_lo * (q_inv * attn_scale);
    simd<float, 64> qn_hi = q_hi * (q_inv * attn_scale);
    simd<float, 64> kn_lo = k_lo * k_inv;
    simd<float, 64> kn_hi = k_hi * k_inv;

    const float A_log_val = gdn_load_fp16_scalar_seq(A_log_ptr, hv);
    const float dt_bias_val = gdn_load_fp16_scalar_seq(dt_bias_ptr, hv);
    const float neg_exp_A = -esimd_expf_seq(A_log_val);
    const int b_col = hv;
    const int a_col = HV + hv;
    const float b_val = gdn_load_fp16_scalar_seq(ba_ptr, ba_offset + b_col);
    const float a_val = gdn_load_fp16_scalar_seq(
        ba_ptr, ba_offset + a_col);
    const float x_gate = a_val + dt_bias_val;
    const float sp =
        (x_gate > 20.0f) ? x_gate : esimd_logf_seq(1.0f + esimd_expf_seq(x_gate));
    const float exp_g = esimd_expf_seq(neg_exp_A * sp);
    const float beta = 1.0f / (1.0f + esimd_expf_seq(-b_val));

    /* v126 P29B-FIX read-source selection, unchanged (see spec header).
     * prev_read_base is null when the prev value is -1 (no state). */
    const int vi0 = tid * 2;
    esimd_e4m3_state_t* save_base = nullptr;
    if (save_state_idx >= 0) {
        save_base = ssm_state_ptr +
            (int64_t)save_state_idx * ssm_stride0 +
            (int64_t)hv * gdn_V * gdn_K;
    }

    /* v126 P29A register chain, unchanged — first token of a verify
     * round dequantizes the pool (NOW with inv_scale); later draft
     * tokens reuse the fp32 carriers, so the scale applies exactly once
     * per boundary crossing, never compounding. */
    if (!h_chained) {
        h0_lo = simd<float, 64>(0.0f);
        h0_hi = simd<float, 64>(0.0f);
        h1_lo = simd<float, 64>(0.0f);
        h1_hi = simd<float, 64>(0.0f);
        if (prev_read_base != nullptr) {
            esimd_e4m3_state_t* sr0 = prev_read_base + (int64_t)(vi0 + 0) * gdn_K;
            esimd_e4m3_state_t* sr1 = prev_read_base + (int64_t)(vi0 + 1) * gdn_K;
            h0_lo = m2_scaled_load_64(sr0, isc_lo);
            h0_hi = m2_scaled_load_64(sr0 + 64, isc_hi);
            h1_lo = m2_scaled_load_64(sr1, isc_lo);
            h1_hi = m2_scaled_load_64(sr1 + 64, isc_hi);
        }
    }

    h0_lo *= exp_g;
    h0_hi *= exp_g;
    h1_lo *= exp_g;
    h1_hi *= exp_g;

    const float kv0 = gdn_dot128_seq(h0_lo, h0_hi, kn_lo, kn_hi);
    const float kv1 = gdn_dot128_seq(h1_lo, h1_hi, kn_lo, kn_hi);
    const float d0 = (v_f32[0] - kv0) * beta;
    const float d1 = (v_f32[1] - kv1) * beta;
    h0_lo += d0 * kn_lo;
    h0_hi += d0 * kn_hi;
    h1_lo += d1 * kn_lo;
    h1_hi += d1 * kn_hi;

    simd<float, 2> result;
    result[0] = gdn_dot128_seq(h0_lo, h0_hi, qn_lo, qn_hi);
    result[1] = gdn_dot128_seq(h1_lo, h1_hi, qn_lo, qn_hi);

    if (save_base != nullptr) {
        esimd_e4m3_state_t* sr0 = save_base + (int64_t)(vi0 + 0) * gdn_K;
        esimd_e4m3_state_t* sr1 = save_base + (int64_t)(vi0 + 1) * gdn_K;
        m2_scaled_store_64(sr0, h0_lo, sc_lo);
        m2_scaled_store_64(sr0 + 64, h0_hi, sc_hi);
        m2_scaled_store_64(sr1, h1_lo, sc_lo);
        m2_scaled_store_64(sr1 + 64, h1_hi, sc_hi);
    }
    return result;
}

template <int WG_SIZE>
ESIMD_INLINE void gdn_conv_fused_seq_spec_scaled_kernel(
    const fp16* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* __restrict__ conv_state_ptr,
    const fp16* __restrict__ conv_weight_ptr,
    const fp16* __restrict__ conv_bias_ptr,
    const int* __restrict__ spec_state_indices_ptr,
    const fp16* __restrict__ A_log_ptr,
    const fp16* __restrict__ dt_bias_ptr,
    const fp16* __restrict__ ba_ptr,
    int64_t ba_stride0,
    esimd_e4m3_state_t* __restrict__ ssm_state_ptr,
    fp16* __restrict__ output_ptr,
    fp16* __restrict__ z_out_ptr,
    const int* __restrict__ token_indx_ptr,
    const int* __restrict__ num_accepted_tokens_ptr,
    int num_spec_decodes,
    int num_spec_tokens,
    int H,
    int HV,
    int gdn_K,
    int gdn_V,
    float attn_scale,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    const fp16* __restrict__ conv_snap_ptr,
    int64_t conv_snap_stride0,
    esimd_e4m3_state_t* __restrict__ ssm_snap_ptr,
    int64_t ssm_snap_stride0,
    const float* __restrict__ ssm_scales_ptr,   // [HV, K] fp32
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);
    if (seq_idx >= num_spec_decodes) {
        return;
    }

    /* M2: per-(hv, k) scale slice, loaded ONCE per work-group — every
     * state row this WG touches shares the same K-lane layout, so the
     * lo/hi 64-lane slices are work-group constants. inv computed
     * in-register with the same IEEE divide the harness uses
     * (1.0f / s), keeping kernel and reference bitwise-comparable. */
    const float* sc_row = ssm_scales_ptr + (int64_t)hv * gdn_K;
    const simd<float, 64> sc_lo = block_load<float, 64>(sc_row + 0);
    const simd<float, 64> sc_hi = block_load<float, 64>(sc_row + 64);
    const simd<float, 64> isc_lo = simd<float, 64>(1.0f) / sc_lo;
    const simd<float, 64> isc_hi = simd<float, 64>(1.0f) / sc_hi;

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;
    const int num_v_threads = WG_SIZE - 4 * H;
    const bool double_v = HV > num_v_threads / 2;
    const bool v_oob = tid >= 4 * H &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));

    const int dim = 2 * H * gdn_K + HV * gdn_V;
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;
    const int state_row = seq_idx * num_spec_tokens;

    int qkvz_offset = 0;
    int qkvz_offset_hi = 0;
    int chunk_start = 0;
    int chunk_start_hi = 0;
    if (tid < 2 * H) {
        const int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        const int k_tid = tid - 2 * H;
        const int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        const int v_hv = tid - 4 * H;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        const int v_tid = tid - 4 * H;
        const int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }
    if (v_oob) {
        qkvz_offset = v_base;
        qkvz_offset_hi = v_base + 64;
        chunk_start = v_base;
        chunk_start_hi = v_base + 64;
    }

    const int accepted_prev = num_accepted_tokens_ptr[seq_idx] - 1;
    const int init_col = accepted_prev > 0 ? accepted_prev : 0;
    const int init_state_idx = spec_state_indices_ptr[state_row + init_col];
    /* v126 P29B-FIX: init conv window from the pre-launch ring snapshot
     * (see the production spec header for the sibling-writer race). */
    const fp16* init_conv_state = nullptr;
    if (init_state_idx >= 0) {
        init_conv_state = conv_snap_ptr +
            (int64_t)(state_row + init_col) * conv_snap_stride0;
    }
    simd<float, 64> s0(0.0f), s1(0.0f), s2(0.0f);
    if (init_conv_state != nullptr) {
        s0 = block_load<fp16, 64>(init_conv_state + 0 * dim + chunk_start);
        s1 = block_load<fp16, 64>(init_conv_state + 1 * dim + chunk_start);
        s2 = block_load<fp16, 64>(init_conv_state + 2 * dim + chunk_start);
    }
    simd<float, 64> s0_hi(0.0f), s1_hi(0.0f), s2_hi(0.0f);
    if (double_v && tid >= 4 * H && !v_oob &&
        init_conv_state != nullptr) {
        s0_hi = block_load<fp16, 64>(
            init_conv_state + 0 * dim + chunk_start_hi);
        s1_hi = block_load<fp16, 64>(
            init_conv_state + 1 * dim + chunk_start_hi);
        s2_hi = block_load<fp16, 64>(
            init_conv_state + 2 * dim + chunk_start_hi);
    }

    /* v126 P29A register chain carriers — the scaled pool is e4m3
     * (1 byte), so the chain is always active in this kernel. */
    simd<float, 64> ch_h0_lo(0.0f), ch_h0_hi(0.0f);
    simd<float, 64> ch_h1_lo(0.0f), ch_h1_hi(0.0f);
    for (int t = 0; t < num_spec_tokens; ++t) {
        const int global_t = token_indx_ptr[state_row + t];
        const int prev_col = t == 0
            ? init_col
            : t - 1;
        const int prev_state_idx =
            spec_state_indices_ptr[state_row + prev_col];
        const int save_state_idx = spec_state_indices_ptr[state_row + t];
        const bool ch_chained = t > 0 && prev_state_idx >= 0;
        /* v126 P29B-FIX read-source split: token 0 from the snapshot,
         * tokens t>0 from the live row this WG stored at t-1. Both hold
         * stored-space bytes, so both decode with inv_scale. */
        esimd_e4m3_state_t* prev_read_base = nullptr;
        if (prev_state_idx >= 0) {
            if (t == 0) {
                prev_read_base = ssm_snap_ptr +
                    (int64_t)(state_row + prev_col) * ssm_snap_stride0 +
                    (int64_t)hv * gdn_V * gdn_K;
            } else {
                prev_read_base = ssm_state_ptr +
                    (int64_t)prev_state_idx * ssm_stride0 +
                    (int64_t)hv * gdn_V * gdn_K;
            }
        }

        const fp16* qkvz_row =
            qkvz_ptr + (int64_t)global_t * qkvz_stride0;
        simd<fp16, 64> x_fp16 = block_load<fp16, 64>(
            qkvz_row + qkvz_offset);
        simd<float, 64> x_f32 = x_fp16;
        simd<fp16, 256> w_raw = block_load<fp16, 256>(
            conv_weight_ptr + (int64_t)chunk_start * 4);
        simd<float, 64> conv_result =
            s0 * w_raw.select<64, 4>(0) + s1 * w_raw.select<64, 4>(1) +
            s2 * w_raw.select<64, 4>(2) + x_f32 * w_raw.select<64, 4>(3) +
            (simd<float, 64>)block_load<fp16, 64>(
                conv_bias_ptr + chunk_start);
        conv_result = conv_result /
            (1.0f + sycl::ext::intel::esimd::exp(-conv_result));

        simd<fp16, 64> x_fp16_hi;
        simd<float, 64> conv_result_hi(0.0f);
        if (double_v && tid >= 4 * H && !v_oob) {
            x_fp16_hi = block_load<fp16, 64>(qkvz_row + qkvz_offset_hi);
            simd<float, 64> x_f32_hi = x_fp16_hi;
            simd<fp16, 256> w_raw_hi = block_load<fp16, 256>(
                conv_weight_ptr + (int64_t)chunk_start_hi * 4);
            conv_result_hi =
                s0_hi * w_raw_hi.select<64, 4>(0) +
                s1_hi * w_raw_hi.select<64, 4>(1) +
                s2_hi * w_raw_hi.select<64, 4>(2) +
                x_f32_hi * w_raw_hi.select<64, 4>(3) +
                (simd<float, 64>)block_load<fp16, 64>(
                    conv_bias_ptr + chunk_start_hi);
            conv_result_hi = conv_result_hi /
                (1.0f + sycl::ext::intel::esimd::exp(-conv_result_hi));
        }

        // Checkpoint conv window — unchanged (conv_state stays fp16).
        if (save_state_idx >= 0) {
            fp16* save_state =
                conv_state_ptr + (int64_t)save_state_idx * conv_stride0;
            if (hv == 0 && tid < 4 * H) {
                block_store<fp16, 64>(
                    save_state + 0 * dim + chunk_start,
                    simd<fp16, 64>(s1));
                block_store<fp16, 64>(
                    save_state + 1 * dim + chunk_start,
                    simd<fp16, 64>(s2));
                block_store<fp16, 64>(
                    save_state + 2 * dim + chunk_start,
                    x_fp16);
            }
            if (!v_oob && tid >= 4 * H) {
                const int v_tid = tid - 4 * H;
                if (double_v && v_tid == hv) {
                    block_store<fp16, 64>(
                        save_state + 0 * dim + chunk_start,
                        simd<fp16, 64>(s1));
                    block_store<fp16, 64>(
                        save_state + 1 * dim + chunk_start,
                        simd<fp16, 64>(s2));
                    block_store<fp16, 64>(
                        save_state + 2 * dim + chunk_start,
                        x_fp16);
                    block_store<fp16, 64>(
                        save_state + 0 * dim + chunk_start_hi,
                        simd<fp16, 64>(s1_hi));
                    block_store<fp16, 64>(
                        save_state + 1 * dim + chunk_start_hi,
                        simd<fp16, 64>(s2_hi));
                    block_store<fp16, 64>(
                        save_state + 2 * dim + chunk_start_hi,
                        x_fp16_hi);
                } else if (!double_v && v_tid / 2 == hv) {
                    block_store<fp16, 64>(
                        save_state + 0 * dim + chunk_start,
                        simd<fp16, 64>(s1));
                    block_store<fp16, 64>(
                        save_state + 1 * dim + chunk_start,
                        simd<fp16, 64>(s2));
                    block_store<fp16, 64>(
                        save_state + 2 * dim + chunk_start,
                        x_fp16);
                }
            }
        }

        const int q_tid_lo = 2 * i_h;
        if (tid == q_tid_lo) {
            slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
        }
        if (tid == q_tid_lo + 1) {
            slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);
        }
        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo) {
            slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
        }
        if (tid == k_tid_lo + 1) {
            slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);
        }
        if (!v_oob && tid >= 4 * H) {
            const int v_tid = tid - 4 * H;
            if (double_v) {
                if (v_tid == hv) {
                    slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
                    slm_block_store<float, 64>(
                        SLM_V_SEQ + 256, conv_result_hi);
                }
            } else {
                const int v_hv = v_tid / 2;
                if (v_hv == hv) {
                    slm_block_store<float, 64>(
                        SLM_V_SEQ + (v_tid & 1) * 256, conv_result);
                }
            }
        }

        barrier();

        const int vi0 = tid * 2;
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);
        simd<float, 2> v_f32 =
            slm_block_load<float, 2>(SLM_V_SEQ + vi0 * (int)sizeof(float));

        const int64_t ba_offset = (int64_t)global_t * ba_stride0;
        simd<float, 2> o_acc = gdn_spec_update_seq_scaled<WG_SIZE>(
            q_lo, q_hi, k_lo, k_hi, v_f32, A_log_ptr, dt_bias_ptr,
            ba_ptr, ba_offset, prev_read_base, ssm_state_ptr, ssm_stride0,
            save_state_idx, tid, hv, HV, gdn_K, gdn_V,
            attn_scale, ch_h0_lo, ch_h0_hi, ch_h1_lo, ch_h1_hi, ch_chained,
            sc_lo, sc_hi, isc_lo, isc_hi);

        fp16* out = output_ptr + (int64_t)global_t * HV * gdn_V +
            (int64_t)hv * gdn_V + vi0;
        block_store<fp16, 2>(out, simd<fp16, 2>(o_acc));

        if (tid < 2) {
            const int z_off = z_base + hv * gdn_V + tid * 64;
            simd<fp16, 64> z_data =
                block_load<fp16, 64>(qkvz_row + z_off);
            fp16* z_dst = z_out_ptr + (int64_t)global_t * HV * gdn_V +
                (int64_t)hv * gdn_V + tid * 64;
            block_store<fp16, 64>(z_dst, z_data);
        }

        s0 = s1;
        s1 = s2;
        s2 = x_f32;
        if (double_v && tid >= 4 * H && !v_oob) {
            s0_hi = s1_hi;
            s1_hi = s2_hi;
            s2_hi = x_fp16_hi;
        }

        barrier();
    }
}

inline void gdn_conv_fused_seq_spec_scaled_host(
    const fp16* qkvz_ptr,
    int64_t qkvz_stride0,
    fp16* conv_state_ptr,
    const fp16* conv_weight_ptr,
    const fp16* conv_bias_ptr,
    const int* spec_state_indices_ptr,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_stride0,
    esimd_e4m3_state_t* ssm_state_ptr,
    fp16* output_ptr,
    fp16* z_out_ptr,
    const int* token_indx_ptr,
    const int* num_accepted_tokens_ptr,
    int num_spec_decodes,
    int num_spec_tokens,
    int H,
    int HV,
    int K,
    int V,
    float scale,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    const fp16* conv_snap_ptr,
    int64_t conv_snap_stride0,
    esimd_e4m3_state_t* ssm_snap_ptr,
    int64_t ssm_snap_stride0,
    const float* ssm_scales_ptr,
    sycl::queue& q)
{
    TORCH_CHECK(H == 8 && (HV == 16 || HV == 24) && K == 128 && V == 128,
        "gdn_conv_fused_seq_spec_scaled supports H=8, HV=16/24, K=V=128; "
        "got H=", H, " HV=", HV, " K=", K, " V=", V);
    TORCH_CHECK(num_spec_decodes > 0 && num_spec_tokens > 0,
        "speculative GDN dimensions must be positive");

    constexpr int WG_SIZE = 64;
    sycl::nd_range<3> range(
        sycl::range<3>(num_spec_decodes, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_spec_scaled_kernel<WG_SIZE>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, spec_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, output_ptr, z_out_ptr, token_indx_ptr,
                num_accepted_tokens_ptr, num_spec_decodes, num_spec_tokens,
                H, HV, K, V, scale, conv_stride0, ssm_stride0,
                conv_snap_ptr, conv_snap_stride0,
                ssm_snap_ptr, ssm_snap_stride0,
                ssm_scales_ptr, ndi);
        });
    });
}
