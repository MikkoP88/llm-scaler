# llm-scaler v32 hot-patch v2: align-mode accepted-count drain -> async + deferred postprocess
# v1 lesson: consuming at _prepare_inputs (post _update_states) sees N+1 request set
#   (KeyError on newly scheduled reqs) and advanced num_computed_tokens.
# v2: consume at execute_model ENTRY (before _update_states) where req_ids and
#   req_state.num_computed_tokens still hold the exact N-view the original
#   blocking postprocess saw; per-request arithmetic uses N-time snapshots.
import py_compile

GM = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
MU = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

src = open(GM).read()

# --- site A: align branch -> async copy + N-view snapshot stash ---
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
            # llm-scaler v32: async accepted-count D2H + deferred postprocess.
            # The blocking .cpu().numpy() drain was the per-step serialization
            # point (57% of worker host time, py-spy gpu_model_runner:1489).
            # Counts travel via the pinned tensor + event (non-align
            # transport); the postprocess runs at the NEXT execute_model
            # entry, before _update_states, where req_ids/num_computed still
            # hold this step's view. N-view per-request inputs are snapshotted
            # here; consumption skips requests that finished or whose mamba
            # state pointer moved (preempt/resume) in between.
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
            _v32_nst = scheduler_output.num_scheduled_tokens
            _v32_nsd = scheduler_output.scheduled_spec_decode_tokens
            _v32_midx = self.mamba_state_idx
            _v32_reqs = self.requests
            self._v32_pending_pp = [
                (
                    _rid,
                    _v32_reqs[_rid].num_computed_tokens,
                    _v32_nst[_rid],
                    len(_v32_nsd.get(_rid, [])),
                    _v32_midx.get(_rid, -1),
                )
                for _rid in list(self.input_batch.req_ids)[:num_reqs]
            ]
"""

# --- site B: consume at execute_model entry ---
old2 = """            deferred_state_corrections_fn = self._update_states(scheduler_output)
"""
new2 = """            _v32_pp = getattr(self, "_v32_pending_pp", None)
            if _v32_pp is not None:
                self._v32_pending_pp = None
                # counts landed asynchronously during the previous step;
                # one full engine round-trip of host work has elapsed.
                self.num_accepted_tokens_event.synchronize()
                mamba_utils.run_deferred_postprocess(self, _v32_pp)
            deferred_state_corrections_fn = self._update_states(scheduler_output)
"""

assert src.count(old1) == 1, f"anchorA={src.count(old1)}"
assert src.count(old2) == 1, f"anchorB={src.count(old2)}"
src = src.replace(old1, new1).replace(old2, new2)
open(GM, "w").write(src)
py_compile.compile(GM, doraise=True)

# --- site C: the deferred runner, appended to mamba_utils.py ---
helper = '''

def run_deferred_postprocess(model_runner, rows):
    """llm-scaler v32: align-mode postprocess deferred from the previous
    step's _update_states_after_model_execute.

    Counts are on host (async pinned copy, event already synced by the
    caller, still in PREVIOUS-step order -- the _prepare_inputs remap has
    not run yet at the execute_model entry consumption point). Per-request
    arithmetic uses the N-step snapshots stashed at deferral. Rows whose
    request finished/dropped, or whose mamba state pointer moved since
    (preempt/resume re-migration), are skipped: their copies are either
    useless (dead state) or obsolete (handled by the migration path).
    """
    copy_bufs = model_runner._get_mamba_copy_bufs()
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    block_size = mamba_spec.block_size
    cpu = model_runner.input_batch.num_accepted_tokens_cpu
    copy_bufs.offset = 0
    for i, (rid, computed, sched, draft, src) in enumerate(rows):
        if rid not in model_runner.requests:
            continue
        if src < 0 or model_runner.mamba_state_idx.get(rid, -1) != src:
            continue
        running = computed + sched - draft
        newc = running + int(cpu[i]) - 1
        aligned = newc // block_size * block_size
        if aligned >= running:
            bias = aligned - running
            dest = aligned // block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                model_runner.kv_cache_config,
                model_runner.model.get_mamba_state_copy_func(),
                mamba_group_ids,
                src,
                dest,
                bias,
                model_runner.requests[rid],
                model_runner.compilation_config.static_forward_context,
            )
            if src == dest:
                cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
'''
mu = open(MU).read()
assert "run_deferred_postprocess" not in mu
mu = mu + helper
open(MU, "w").write(mu)
py_compile.compile(MU, doraise=True)
print("V32PATCH2 OK: async counts + deferred postprocess at execute_model entry")
