# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        # llm-scaler v42c: honor DFLASH2_EMIT_K on the async path.
        # The placeholder list width IS the scheduled spec width
        # under async scheduling; everything downstream (spec slot
        # count, num_output_placeholders accounting, engine-side
        # -1 padding in update_draft_token_ids_in_output) derives
        # from the scheduled width, so capping here is
        # self-consistent end-to-end. Default: stock width.
        _emit_k = self.num_spec_tokens
        _env_k = os.environ.get("DFLASH2_EMIT_K", "")
        if _env_k:
            _emit_k = int(_env_k)
            if not 1 <= _emit_k <= self.num_spec_tokens:
                raise ValueError(
                    f"DFLASH2_EMIT_K={_emit_k} out of range 1.."
                    f"{self.num_spec_tokens} (num_spec_tokens)"
                )
        self._spec_token_placeholders: list[int] = [-1] * _emit_k
        if _emit_k < self.num_spec_tokens:
            logger.info(
                "DFlash2 v42c async cap: scheduling %d of %d spec "
                "slots per step (async scheduling preserved)",
                _emit_k,
                self.num_spec_tokens,
            )

        # llm-scaler v50 F8: runaway-request guard (crash 3; redo capture
        # 2026-09-06, salvage_010017). Placeholders accrue at
        # (num_sampled_tokens_per_step + spec width) per scheduled step and
        # resolve at the same rate while the request's outputs are being
        # applied (emission + rejection legs in _update_request_with_output
        # and update_from_output). A monotonic climb means outputs stopped
        # being processed while scheduling continued — the usage-less
        # zombie stream-end vehicle: the client stream ends silently, no
        # abort reaches the engine (async_llm aborts only on
        # CancelledError/GeneratorExit), the request decodes alone, and the
        # placeholder-inflated cached num_output_tokens overruns worker-side
        # persistent-batch alignment -> SYCL vectorized gather OOB -> fatal
        # worker death (observed ~19k placeholders / ~2.4k steps).
        # Force-finish long before that point. Set
        # VLLM_V50_PLACEHOLDER_LIMIT=0 to disable.
        self._v50_ph_limit = int(
            os.environ.get("VLLM_V50_PLACEHOLDER_LIMIT", "64"))
        if self._v50_ph_limit > 0:
            logger.info(
                "llm-scaler v50 F8: runaway-request guard armed "
                "(placeholder limit=%d, spec width=%d)",
                self._v50_ph_limit, len(self._spec_token_placeholders))

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            _v50_reason = None
            if (self._v50_ph_limit > 0
                    and request.num_output_placeholders > self._v50_ph_limit):
                _v50_reason = (
                    f"placeholder-overflow {request.num_output_placeholders}"
                    f" > {self._v50_ph_limit}")
            elif (self._v50_ph_limit > 0
                    and request.num_output_tokens > (
                        request.max_tokens + self.num_spec_tokens + 8)):
                _v50_reason = (
                    f"real-count {request.num_output_tokens} > max_tokens"
                    f" {request.max_tokens} + {self.num_spec_tokens + 8}")
            if _v50_reason is not None:
                logger.warning(
                    "llm-scaler v50 F8: runaway-request guard fired for %s"
                    " (%s): num_output_placeholders=%d, num_output_tokens=%d,"
                    " max_tokens=%d, num_computed_tokens=%d — outputs stopped"
                    " resolving while scheduled (crash-3 zombie signature);"
                    " force-finishing FINISHED_ABORTED",
                    request.request_id, _v50_reason,
                    request.num_output_placeholders,
                    request.num_output_tokens,
                    request.max_tokens, request.num_computed_tokens)
                self.finish_requests(request.request_id,
                                     RequestStatus.FINISHED_ABORTED)
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped