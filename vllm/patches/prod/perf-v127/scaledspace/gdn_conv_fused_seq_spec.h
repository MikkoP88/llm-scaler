#pragma once

/*
 * Speculative variant of the sequential Qwen3.5/Qwen3.6 GDN kernel.
 *
 * The ordinary fused kernel maps one work-group to one token and cannot model
 * speculative rollback: every draft token has its own conv/SSM cache slot and
 * the initial slot depends on the number of tokens accepted in the previous
 * step. This variant maps one work-group to one speculative sequence and
 * walks all of its tokens in order, keeping the same state semantics as the
 * vllm-xpu speculative GDN kernels while avoiding the intermediate q/k/v/b/a
 * buffers and the separate conv and delta-rule launches.
 *
 * This implementation is intentionally scoped to the Qwen3.5/3.6 TP=2
 * geometries (H=8, HV=16/24, K=V=128, WG_SIZE=64). The host dispatcher
 * rejects unsupported geometries rather than silently selecting a slower or
 * incorrect layout.
 *
 * v126 P29B-FIX — rollback reads are served from pre-launch snapshots.
 * The ring calling convention makes the init row a save row: ring position
 * init_col is read at work-group start and rewritten at t==init_col. The
 * writers of that row are SIBLING work-groups of the same sequence (wg
 * hv==0 stores the q/k dims via tids 0..31; wg hv' stores V dim hv'), so
 * once nsd*HV work-groups exceed HW residency (~32 slots; 48 at
 * nsd=2/HV=24) a late-starting work-group reads sibling post-checkpoint
 * bytes instead of the pre-round state — observed as exact-zero conv
 * windows under zero input and corrupted dims [0, 3072) matching the
 * retired-writer split. The wrapper now gathers the ring rows (conv state
 * and SSM pool, by ring position) into scratch tensors before launch;
 * this kernel reads the init conv window and the token-0 SSM row from
 * those snapshots (never written in-kernel => race-free by construction)
 * while checkpoint stores keep targeting the live tensors. Later tokens'
 * SSM reads stay on the live pool: ring position t-1 was stored one
 * iteration earlier by the SAME work-group's own hv slice (program
 * ordered), and redirecting them to the snapshot would return pre-round
 * state instead of the post-t-1 state. On paths where the race never
 * fired, the snapshot bytes equal what the live read returned, so solo /
 * nsd=1 / fp16 outputs are bit-identical to the previous build.
 */

template <int WG_SIZE, typename StateT>
ESIMD_INLINE simd<float, 2> gdn_spec_update_seq(
    const simd<float, 64>& q_lo,
    const simd<float, 64>& q_hi,
    const simd<float, 64>& k_lo,
    const simd<float, 64>& k_hi,
    const simd<float, 2>& v_f32,
    const fp16* A_log_ptr,
    const fp16* dt_bias_ptr,
    const fp16* ba_ptr,
    int64_t ba_offset,
    StateT* prev_read_base,
    StateT* ssm_state_ptr,
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
    bool h_chained)
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

    /* v126 P29B-FIX: the caller selects the read source — the token-0
     * rollback read comes from the pre-launch snapshot (the init row is
     * rewritten in-launch by sibling work-groups), later tokens re-read
     * the row this same work-group stored at t-1 on the live pool.
     * prev_read_base is null when the prev value is -1 (no state). */
    const int vi0 = tid * 2;
    StateT* save_base = nullptr;
    if (save_state_idx >= 0) {
        save_base = ssm_state_ptr +
            (int64_t)save_state_idx * ssm_stride0 +
            (int64_t)hv * gdn_V * gdn_K;
    }

    /* v126 P29A: fp8 register chain — h* are caller-owned carriers. The
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
        if (prev_read_base != nullptr) {
            StateT* sr0 = prev_read_base + (int64_t)(vi0 + 0) * gdn_K;
            StateT* sr1 = prev_read_base + (int64_t)(vi0 + 1) * gdn_K;
            h0_lo = lsc_load_state_64_seq(sr0);
            h0_hi = lsc_load_state_64_seq(sr0 + 64);
            h1_lo = lsc_load_state_64_seq(sr1);
            h1_hi = lsc_load_state_64_seq(sr1 + 64);
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
        StateT* sr0 = save_base + (int64_t)(vi0 + 0) * gdn_K;
        StateT* sr1 = save_base + (int64_t)(vi0 + 1) * gdn_K;
        lsc_store_state_64_seq(sr0, h0_lo);
        lsc_store_state_64_seq(sr0 + 64, h0_hi);
        lsc_store_state_64_seq(sr1, h1_lo);
        lsc_store_state_64_seq(sr1 + 64, h1_hi);
    }
    return result;
}

template <int WG_SIZE, typename StateT>
ESIMD_INLINE void gdn_conv_fused_seq_spec_kernel(
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
    StateT* __restrict__ ssm_state_ptr,
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
    StateT* __restrict__ ssm_snap_ptr,
    int64_t ssm_snap_stride0,
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);
    if (seq_idx >= num_spec_decodes) {
        return;
    }

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
    /* v126 P29B-FIX: the init conv window comes from the pre-launch ring
     * snapshot, not the live conv pool — ring position init_col is
     * rewritten at t==init_col and its dims belong to sibling work-groups
     * (see the file header). Snapshot row state_row+init_col carries
     * exactly the pre-round bytes for init_state_idx; the value guard
     * stays so -1 still means "no state" (zero window). */
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

    /* v126 P29A: fp8 SSM-state register chain carriers — persist across the
     * draft-token loop; kFp8Chain is constexpr so fp16 instantiations
     * never chain (legacy reload dataflow, bit-identical). */
    constexpr bool kFp8Chain = (sizeof(StateT) == 1);
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
        /* v126 P29A: chain from the previous draft token — its save slot IS
         * this token's prev slot (t-1), per the allocator's distinct
         * per-column ids. Token 0 always loads the rollback state from the
         * pool; a missing prev slot falls back to the legacy load path. */
        const bool ch_chained = kFp8Chain && t > 0 && prev_state_idx >= 0;
        /* v126 P29B-FIX: token 0 reads its rollback state from the
         * pre-launch snapshot (init row == in-launch save row of sibling
         * work-groups). Tokens t>0 read the live pool row this same
         * work-group stored at t-1 (own hv slice, program ordered) — the
         * snapshot would be stale there, so it is only used at t==0. */
        StateT* prev_read_base = nullptr;
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

        // Each value-head work-group owns its V slice. Only hv=0 writes the
        // replicated Q/K slices, so every checkpoint slot has one writer.
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
        simd<float, 2> o_acc = gdn_spec_update_seq<WG_SIZE, StateT>(
            q_lo, q_hi, k_lo, k_hi, v_f32, A_log_ptr, dt_bias_ptr,
            ba_ptr, ba_offset, prev_read_base, ssm_state_ptr, ssm_stride0,
            save_state_idx, tid, hv, HV, gdn_K, gdn_V,
            attn_scale, ch_h0_lo, ch_h0_hi, ch_h1_lo, ch_h1_hi, ch_chained);

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

template <typename StateT>
inline void gdn_conv_fused_seq_spec_host(
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
    StateT* ssm_state_ptr,
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
    StateT* ssm_snap_ptr,
    int64_t ssm_snap_stride0,
    sycl::queue& q)
{
    TORCH_CHECK(H == 8 && (HV == 16 || HV == 24) && K == 128 && V == 128,
        "gdn_conv_fused_seq_spec supports H=8, HV=16/24, K=V=128; got H=",
        H, " HV=", HV, " K=", K, " V=", V);
    TORCH_CHECK(num_spec_decodes > 0 && num_spec_tokens > 0,
        "speculative GDN dimensions must be positive");

    constexpr int WG_SIZE = 64;
    sycl::nd_range<3> range(
        sycl::range<3>(num_spec_decodes, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_spec_kernel<WG_SIZE, StateT>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, spec_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, output_ptr, z_out_ptr, token_indx_ptr,
                num_accepted_tokens_ptr, num_spec_decodes, num_spec_tokens,
                H, HV, K, V, scale, conv_stride0, ssm_stride0,
                conv_snap_ptr, conv_snap_stride0,
                ssm_snap_ptr, ssm_snap_stride0, ndi);
        });
    });
}
