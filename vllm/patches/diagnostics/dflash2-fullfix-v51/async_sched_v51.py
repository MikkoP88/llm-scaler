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

        # llm-scaler v51 F8-detect: the v50 F8 runaway guard is DEMOTED to
        # detect-only. Root cause of the crash-3 zombie class (usage-less
        # silent stream end leaking generate()) is now fixed at the source
        # by F9 (async_llm abort-on-any-exit + serving-layer aclose
        # companions), so force-finishing requests here is no longer the
        # defense — and force-finishing CAN cut a real valid request whose
        # output leg stalls transiently (user directive 2026-09-06: remove
        # guards that can cut valid requests). A firing in detect mode means
        # an F9 gap: investigate, do not paper over.
        # VLLM_V51_F8_MODE: detect (default) = log only; kill = v50
        # behavior (A/B validation only); off = fully inert.
        self._v51_f8_mode = os.environ.get("VLLM_V51_F8_MODE", "detect")
        if self._v51_f8_mode not in ("detect", "kill", "off"):
            raise ValueError(
                f"VLLM_V51_F8_MODE={self._v51_f8_mode!r} invalid "
                "(must be detect, kill, or off)")
        self._v50_ph_limit = int(
            os.environ.get("VLLM_V50_PLACEHOLDER_LIMIT", "64"))
        self._v51_f8_logged: set[str] = set()
        logger.info(
            "llm-scaler v51: F8 guard mode=%s (placeholder limit=%d, "
            "spec width=%d) — F9 (serving-layer abort) is the primary "
            "defense; F8 firings indicate an F9 gap",
            self._v51_f8_mode, self._v50_ph_limit,
            len(self._spec_token_placeholders))

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
                if self._v51_f8_mode == "off":
                    continue
                if (request.request_id not in self._v51_f8_logged
                        or self._v51_f8_mode == "kill"):
                    logger.warning(
                        "llm-scaler v51 F8-detect: runaway signature for %s"
                        " (%s): num_output_placeholders=%d, "
                        "num_output_tokens=%d, max_tokens=%d, "
                        "num_computed_tokens=%d — outputs stopped resolving"
                        " while scheduled; F9 should have aborted this "
                        "request at the serving layer; a firing here is an"
                        " F9 GAP signal%s",
                        request.request_id, _v50_reason,
                        request.num_output_placeholders,
                        request.num_output_tokens,
                        request.max_tokens, request.num_computed_tokens,
                        (" (force-finishing FINISHED_ABORTED)"
                         if self._v51_f8_mode == "kill" else
                         " (detect-only: NOT finishing — valid requests"
                         " must never be cut)"))
                    self._v51_f8_logged.add(request.request_id)
                if self._v51_f8_mode == "kill":
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
