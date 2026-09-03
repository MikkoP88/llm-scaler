# llm-scaler v32 hot-patch: align-mode accepted-count drain -> async + deferred postprocess
# Target: vllm/v1/worker/gpu_model_runner.py (in-container)
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
src = open(P).read()

old1 = """        if self.cache_config.mamba_cache_mode == "align":
            for i, num_tokens in enumerate(
                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
            ):
                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
            mamba_utils.postprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.input_batch,
                self.requests,
                self.mamba_state_idx,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                self._get_mamba_copy_bufs(),
            )
"""
new1 = """        if self.cache_config.mamba_cache_mode == "align":
            # llm-scaler v32: async accepted-count D2H + deferred postprocess
            # (was: blocking .cpu().numpy() drain + immediate postprocess).
            # The drain was the per-step serialization point (57% of worker
            # host time, py-spy gpu_model_runner:1489). Semantics preserved:
            # counts land in the pinned cpu tensor asynchronously; postprocess
            # runs at the next _prepare_inputs right after event.synchronize()
            # and the condense remap (the non-align transport), still strictly
            # before preprocess_mamba of that step.
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
            self._v32_pending_postprocess_so = scheduler_output
"""

old2 = """            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
"""
new2 = """            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
            # llm-scaler v32: deferred align-mode postprocess. Counts were
            # synced + remapped above (current request order); original call
            # site was the blocking branch of _update_states_after_model_execute.
            if getattr(self, "_v32_pending_postprocess_so", None) is not None:
                _v32_so = self._v32_pending_postprocess_so
                self._v32_pending_postprocess_so = None
                mamba_utils.postprocess_mamba(
                    _v32_so,
                    self.kv_cache_config,
                    self.input_batch,
                    self.requests,
                    self.mamba_state_idx,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    self._get_mamba_copy_bufs(),
                )
        else:
"""

assert src.count(old1) == 1, f"anchor1 count={src.count(old1)}"
assert src.count(old2) == 1, f"anchor2 count={src.count(old2)}"
src = src.replace(old1, new1).replace(old2, new2)
open(P, "w").write(src)
py_compile.compile(P, doraise=True)
print("V32PATCH OK: align drain -> async copy + deferred postprocess")
