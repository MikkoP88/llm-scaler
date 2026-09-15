# Dockerfile.v10 — llm-scaler crashfix-v57: mamba deferred-copy buffer
# isolation.
#
# FROM v1.2.9 (crashfix-v55.2, sha256:ddc6f15a7061…): all v55 fences +
# progressive wait — plus, in this layer:
#
#   patch_v57_mamba_buf_isolation.py — v57: the v32 deferred mamba
#   postprocess and preprocess_mamba both used ONE shared
#   MambaCopyBuffers instance inside a single async-scheduling
#   execute_model call (entry-time deferred producer fills/launches,
#   then _update_states' preprocess_mamba OVERWRITES the same gpu
#   metadata slots milliseconds later; the first kernel queued behind
#   the previous step's tail can read the second producer's src/dst/
#   sizes -> wrong-block mamba copies -> corrupted GDN state -> kernel
#   spin -> num_accepted_tokens_event/async_copy_ready_event never
#   signal -> v55 ASYNC-EVENT-STALL wedge + engine death). Fix: private
#   MambaCopyBuffers instance for the deferred producer. Zero hot-path
#   cost, zero stream-order change, zero numerics change.
#
# Validated by Phase D A/B (e4m3+mtp4 lane): control boot (unpatched,
# no DEBUG) wedged under sustain pressure; v57 boot clean under
# identical pressure.

FROM llm-scaler-exp:v1.2.9
COPY patch_v57_mamba_buf_isolation.py /w/v57/
RUN python3 /w/v57/patch_v57_mamba_buf_isolation.py --sp /opt/venv/lib/python3.12/site-packages \
 && grep -q "_get_mamba_copy_bufs_deferred" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py \
 && grep -q "_get_mamba_copy_bufs_deferred()" /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py \
 && rm -rf /w/v57 \
 && echo BAKE_V57_OK
