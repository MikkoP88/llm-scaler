FROM llm-scaler-vllm-adv:v20
# llm-scaler-vllm-adv:v21 - honor --kv-cache-dtype for the DFlash draft KV
# pool (VRAM parity with nospec for spec cells). Three overlays:
#
#   1. vllm/v1/spec_decode/dflash.py: _create_draft_vllm_config no longer
#      rewrites turboquant_* -> "auto" for the draft engine
#      (VLLM_DFLASH_TQ_DRAFT_KV, default 1; =0 restores v20 behavior).
#      DFlash drafting attention needs NON-causal metadata (every draft
#      query row attends the full stored context), so use_non_causal is
#      set False for TQ drafts (TQ does not declare supports_non_causal;
#      the causal flag travels on the metadata instead). Effect: the
#      drafter's KV pool is TQ-compressed like the target's -- with
#      tq4nc, spec decode at --max-model-len 262144 needed 13.09 GiB KV
#      vs 6.67 GiB available pre-v21 (est. max len 126976); now it boots
#      like the nospec bar cell.
#      v21b: VLLM_DFLASH_DRAFT_KV_DTYPE overrides the draft pool dtype
#      independently of the target (recommended for turboquant_4bit_nc
#      targets: turboquant_k8v4 -- 4-bit draft KV costs acceptance
#      E[len] 1.77 vs 2.01; k8v4 draft is near-lossless at 2.07-2.16).
#
#   2. vllm/v1/attention/backends/turboquant_attn.py: TurboQuantMetadata
#      carries the causal flag from CommonAttentionMetadata; new
#      TurboQuantMetadataBuilder.build_for_drafting override (a genuinely
#      mixed draft batch is forced onto the pure-prefill continuation
#      loop); _prefill_attention serves non-causal draft steps through
#      the multi-query kernel (one shared KV pass for all q_len rows
#      instead of q_len synthetic-decode full-context rescans); the
#      synthetic-decode fallback keeps a non-causal variant (full-extent
#      seq_lens). Target-side paths (build / build_for_cudagraph_capture)
#      unchanged: metadata without a causal attr defaults to causal.
#
#   3. vllm/v1/attention/ops/triton_turboquant_decode.py:
#      triton_turboquant_mq_decode_attention gains non_causal=False
#      (kernel constexpr NON_CAUSAL: q0 = full seq_len, all rows attend
#      the whole stored context, no causal ramp). Default False = the
#      v19/v20 causal verify behavior, bit-identical.
#
#   4. vllm/v1/worker/gpu_model_runner.py (v21c): _get_attention_kv_cache_shape
#      derives the TurboQuant slot size from the GROUP'S OWN spec
#      (TQFullAttentionSpec.tq_slot_size / page_size_bytes) instead of the
#      runner's cache_dtype. The DFlash drafter registers draft-dtype TQ
#      groups into the runner's group list; with a draft dtype different
#      from the target's (draft-KV policy), the dtype-string derivation
#      computed the wrong slot and pool init crashed with
#      "shape '[14256, 128, 4, 134]' is invalid for input of size ...".
#      Same-dtype pairings produce identical shapes (spec authors the same
#      slot), so v20-equivalent behavior is unchanged.
#
# No other deltas vs the validated v20 image.

COPY dflash.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py
COPY turboquant_attn.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py
COPY triton_turboquant_decode.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py
COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    grep -q 'VLLM_DFLASH_TQ_DRAFT_KV' $SP/vllm/v1/spec_decode/dflash.py && \
    grep -q 'VLLM_DFLASH_DRAFT_KV_DTYPE' $SP/vllm/v1/spec_decode/dflash.py && \
    grep -q '_TQ4NC_DRAFT_PARITY_MINLEN' $SP/vllm/v1/spec_decode/dflash.py && \
    grep -q 'use_non_causal=False' $SP/vllm/v1/spec_decode/dflash.py && \
    grep -q 'def build_for_drafting' $SP/vllm/v1/attention/backends/turboquant_attn.py && \
    grep -q 'non_causal=(not causal)' $SP/vllm/v1/attention/backends/turboquant_attn.py && \
    grep -q 'def get_kv_cache_block_dim' $SP/vllm/v1/attention/backends/turboquant_attn.py && \
    grep -q 'NON_CAUSAL' $SP/vllm/v1/attention/ops/triton_turboquant_decode.py && \
    grep -q 'non_causal: bool = False' $SP/vllm/v1/attention/ops/triton_turboquant_decode.py && \
    grep -q 'tq_slot_size' $SP/vllm/v1/worker/gpu_model_runner.py && \
    find $SP/vllm/v1/spec_decode $SP/vllm/v1/attention/backends $SP/vllm/v1/attention/ops $SP/vllm/v1/worker -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -m py_compile \
      $SP/vllm/v1/spec_decode/dflash.py \
      $SP/vllm/v1/attention/backends/turboquant_attn.py \
      $SP/vllm/v1/attention/ops/triton_turboquant_decode.py \
      $SP/vllm/v1/worker/gpu_model_runner.py && \
    python -c "from vllm.v1.spec_decode.dflash import DFlashProposer; from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadataBuilder, TurboQuantAttentionImpl; from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_mq_decode_attention; import inspect; assert hasattr(TurboQuantMetadataBuilder, 'build_for_drafting'); assert 'non_causal' in inspect.signature(triton_turboquant_mq_decode_attention).parameters; print('v21 imports OK')"
