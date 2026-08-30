FROM llm-scaler-vllm-adv:v25
# llm-scaler-vllm-adv:v26 - GDN spec-kernel ragged-batch fix: the ACTUAL
# >=32k MTP wedge root cause (KNOWN_ISSUES #11), fixed in the GPU kernels.
#
# Root cause (empirically confirmed 2026-08-30, see repro_gdn_spec_oob.py):
#   vllm-xpu-kernels' two spec kernels,
#     csrc/xpu/gdn_attn/causal_conv1d.hpp  :: causal_conv1d_spec_kernel
#     csrc/xpu/gdn_attn/gated_delta_rule.hpp :: gated_delta_rule_spec_kernel
#   walked a RECTANGULAR  batch_id*num_spec_tokens + t_local  index over
#   spec_token_indx and over the q/k/v/b/a scratch that gdn_attn_interface
#   allocates at the ACTUAL ragged size {spec_token, ...}. But the producer
#   (vllm/v1/attention/backends/gdn_attn.py) deliberately builds RAGGED spec
#   metadata:
#     - final verify steps near the generation-end budget clamp (draft_len
#       < k+1), e.g. a 1-of-5 tail: spec_token_indx = arange(min(1*5, 1)) = [0]
#     - drafter replay steps process only the accepted tokens
#       (num_decode_draft_tokens = accepted-1; with E[len] ~ 1.9 that is
#       ~80% of all drafter steps)
#     - cudagraph padding rows flatten the tail of spec_query_start_loc.
#   On every such step the kernels read spec_token_indx OUT OF BOUNDS
#   (garbage global_t -> wild writes through core_attn_out/z) and write
#   q/k/v/b/a rows PAST their spec_token-sized allocations. Standalone repro
#   on the v22 wheel: ONE call with the 1-of-5 verify-tail shape faults the
#   device (UR_RESULT_ERROR_DEVICE_LOST; dmesg: "xe ... Engine reset:
#   engine_class=ccs" + Xe device coredump) - the serve wedge, reproduced
#   deterministically with no serve in the loop. The oneCCL/barrier work in
#   v23-v25 treated a real but secondary queue-clogging effect; the faulting
#   OOB writes are the primary cause, which is why no host-side arm could
#   stop it.
#
# Fix (gdn_spec_fix.patch, 3 files, bit-identical for full batches):
#   both spec kernels now honor the per-request ranges the interface NOTE
#   always claimed they did:
#     token_start  = spec_query_start_loc[b]
#     token_count  = spec_query_start_loc[b+1] - token_start
#     walk t_local in [0, token_count), index token_start + t_local
#   plus an upper clamp  init_col <= num_spec_tokens-1  for the
#   num_accepted_tokens-selected initial state column. For full batches
#   spec_query_start_loc[b] == b*num_spec_tokens, so indices are unchanged.
#
# Image contents: v25 (both env-gated overlays kept - they are orthogonal
# queue-hygiene) + force-reinstalled vllm-xpu-kernels wheel built from the
# patched tree + the three fixed kernel sources refreshed in the provenance
# tree /llm-scaler/vllm/vllm-xpu-kernels.
#
# The wheel is NOT committed to the repo; stage it next to this Dockerfile
# before building (see README.md "Build"):
#   cp /root/build/wheels/vllm_xpu_kernels-*.whl .

COPY vllm_xpu_kernels-*.whl /tmp/
COPY causal_conv1d.hpp /tmp/fixed/causal_conv1d.hpp
COPY gated_delta_rule.hpp /tmp/fixed/gated_delta_rule.hpp
COPY gdn_attn_interface.cpp /tmp/fixed/gdn_attn_interface.cpp

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    pip install --force-reinstall --no-deps /tmp/vllm_xpu_kernels-*.whl && \
    TREE=/llm-scaler/vllm/vllm-xpu-kernels/csrc/xpu/gdn_attn && \
    cp /tmp/fixed/causal_conv1d.hpp $TREE/causal_conv1d.hpp && \
    cp /tmp/fixed/gated_delta_rule.hpp $TREE/gated_delta_rule.hpp && \
    cp /tmp/fixed/gdn_attn_interface.cpp $TREE/gdn_attn_interface.cpp && \
    grep -q 'llm-scaler fix (KNOWN_ISSUES #11' $TREE/causal_conv1d.hpp && \
    grep -q 'llm-scaler fix (KNOWN_ISSUES #11' $TREE/gated_delta_rule.hpp && \
    grep -q 'llm-scaler fix, KNOWN_ISSUES #11' $TREE/gdn_attn_interface.cpp && \
    test "$(grep -c 'token_start + t_local' $TREE/causal_conv1d.hpp)" = "3" && \
    test "$(grep -c 'token_start + t_local' $TREE/gated_delta_rule.hpp)" = "1" && \
    test "$(grep -c 'init_col > num_spec_tokens - 1' $TREE/causal_conv1d.hpp)" = "1" && \
    test "$(grep -c 'init_col > num_spec_tokens - 1' $TREE/gated_delta_rule.hpp)" = "1" && \
    python -c "import torch, vllm_xpu_kernels._xpu_C, importlib.metadata as md; v=md.version(\"vllm-xpu-kernels\"); assert v.endswith(\"d20260830\"), v; assert hasattr(torch.ops._xpu_C, \"gdn_attention\"); print(\"v26 wheel\", v, \"+ gdn_attention OK\")" && \
    rm -f /tmp/vllm_xpu_kernels-*.whl && rm -rf /tmp/fixed && \
    find $SP/vllm_xpu_kernels -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null && \
    echo "v26 overlay done"
