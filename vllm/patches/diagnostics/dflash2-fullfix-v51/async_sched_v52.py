# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
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
        #
        # llm-scaler v52 (crash-4 fix, F8v2, 2026-09-06 15:32): the crash
        # FALSIFIED the v51 premise above — a LIVE-client request's engine
        # outputs stopped resolving (a genuine F9 gap, root cause still
        # open), placeholders climbed to 72, and the overshoot drove
        # num_new_tokens to -7, killing both TP workers via the runner
        # assert (negative-clamped separately in scheduler.py). F8v2
        # re-arms the guard WITH A GRACE PERIOD so it can never cut a
        # valid request whose output leg only stalled transiently:
        #   v52 (DEFAULT) = park a timer at detection; if the request
        #     recovers (placeholders drain below the limit — outputs
        #     resumed), the timer is cancelled and the request continues
        #     untouched; only after VLLM_V52_F8_GRACE seconds (default
        #     60) of NO resolution is it force-finished FINISHED_ABORTED
        #     (the v50-validated finish path). A scheduled request whose
        #     outputs have not resolved for limit/8+ consecutive steps is
        #     pathological regardless — client backpressure never pauses
        #     scheduler-side resolution.
        #   detect = log only (v51 behavior)   kill = v50 behavior (A/B)
        #   off = fully inert.
        self._v51_f8_mode = os.environ.get("VLLM_V51_F8_MODE", "v52")
        if self._v51_f8_mode not in ("v52", "detect", "kill", "off"):
            raise ValueError(
                f"VLLM_V51_F8_MODE={self._v51_f8_mode!r} invalid "
                "(must be v52, detect, kill, or off)")
        self._v50_ph_limit = int(
            os.environ.get("VLLM_V50_PLACEHOLDER_LIMIT", "64"))
        self._v52_f8_grace = float(
            os.environ.get("VLLM_V52_F8_GRACE", "60"))
        if self._v52_f8_grace < 0:
            raise ValueError(
                f"VLLM_V52_F8_GRACE={self._v52_f8_grace} must be >= 0")
        self._v51_f8_logged: set[str] = set()
        self._v52_parked: dict[str, float] = {}
        # llm-scaler v52 (crash-4 R2 root fix): terminal EngineCoreOutputs
        # that must reach the client for requests force-finished while NOT
        # part of a scheduled step (F8v2 reaping). In async scheduling
        # update_from_output — the only normal delivery path for a final
        # output — runs per executed step and only emits for requests in
        # num_scheduled_tokens (an already-finished request is skipped by
        # its `request.is_finished(): continue` guard), so a reaped zombie
        # would otherwise NEVER terminate its client stream (crash-3
        # "usage-less silent stream end", reproduced 2026-09-06 16:14).
        # Entries are emitted at the next update_from_output OR flushed at
        # the engine idle transition (core.py _v52_flush_pending_finishes).
        self._v52_term_pending: dict[str, tuple[int, str]] = {}
        # llm-scaler v52b (crash-4 root-cause telemetry): per-request
        # emission-strike counters; see update_from_output wrap below.
        self._v52b_strikes: dict[str, int] = {}
        self._v52b_logged: set[str] = set()
        logger.info(
            "llm-scaler v52: F8 guard mode=%s (placeholder limit=%d, "
            "spec width=%d, f8v2 grace=%.0fs) — F8v2 force-finish only "
            "after grace without resolution; F9 remains the primary "
            "defense",
            self._v51_f8_mode, self._v50_ph_limit,
            len(self._spec_token_placeholders), self._v52_f8_grace)

    def update_from_output(self, scheduler_output, model_runner_output):
        # llm-scaler v52 (crash-4 root-cause telemetry, F8v2 companion):
        # per-request emission counter at the scheduler/engine boundary.
        # The crash-4 zombie signature (num_output_placeholders grows +8
        # per step while the request stays scheduled and computed) REQUIRES
        # that update_from_output receive zero new tokens for a scheduled
        # decode request, step after step. The base method reads
        # model_runner_output.sampled_token_ids[req_index] per request, so
        # this counter sees exactly what token resolution will see:
        #   n == 0  -> the worker emitted an EMPTY token list for this
        #              request: prime suspect is per-request emission
        #              bookkeeping in the dflash spec path (H1)
        #   n == -1 -> request absent from req_id_to_index: the runner
        #              never registered it (H2)
        # The client/serving layer is EXCLUDED by construction: the
        # placeholder decrement happens here scheduler-side. Detect-only
        # (F8v2 remains the actor); warns once per episode at 4 consecutive
        # zero steps (~half the F8 placeholder threshold); auto-clears on
        # recovery; wrapped so telemetry can never take the engine down.
        try:
            _sti = model_runner_output.sampled_token_ids
            _idx = model_runner_output.req_id_to_index
            if _sti and _idx:
                for _rid, _nsched in scheduler_output.num_scheduled_tokens.items():
                    _req = self.requests.get(_rid)
                    if (_req is None or _req.is_finished()
                            or _req.is_prefill_chunk):
                        continue
                    _n = len(_sti[_idx[_rid]]) if _rid in _idx else -1
                    if _n > 0:
                        if _rid in self._v52b_strikes:
                            self._v52b_strikes.pop(_rid, None)
                            self._v52b_logged.discard(_rid)
                        continue
                    _s = self._v52b_strikes.get(_rid, 0) + 1
                    self._v52b_strikes[_rid] = _s
                    if _s >= 4 and _rid not in self._v52b_logged:
                        self._v52b_logged.add(_rid)
                        logger.warning(
                            "llm-scaler v52b ZERO-EMISSION: %s scheduled "
                            "(%d tokens this step) but emission counter "
                            "n=%d for %d consecutive steps "
                            "(placeholders=%d, num_output_tokens=%d, "
                            "num_computed_tokens=%d) — H1 worker-emitted-0 "
                            "if n==0 / H2 not-in-req_id_to_index if n==-1; "
                            "root-cause target locked",
                            _rid, _nsched, _n, _s,
                            _req.num_output_placeholders,
                            _req.num_output_tokens,
                            _req.num_computed_tokens)
        except Exception:
            pass
        ret = super().update_from_output(scheduler_output,
                                         model_runner_output)
        # llm-scaler v52 (crash-4 R2 root fix): attach the pending
        # terminal outputs (F8v2 reaping) to this step's client buckets so
        # they ride the normal delivery; anything still pending when the
        # engine goes idle is flushed by core.py _v52_flush_pending_finishes.
        if self._v52_term_pending:
            for _ci, _eco in self._v52_take_term_outputs().items():
                if _ci in ret:
                    if ret[_ci].outputs:
                        ret[_ci].outputs.extend(_eco.outputs)
                    else:
                        ret[_ci].outputs = _eco.outputs
                else:
                    ret[_ci] = _eco
        return ret

    def _v52_take_term_outputs(self) -> dict[int, EngineCoreOutputs]:
        # llm-scaler v52 (crash-4 R2 root fix): build (and consume) the
        # terminal EngineCoreOutputs for force-finished-but-unscheduled
        # requests. Mirrors the failed-KV-load finish pattern in
        # Scheduler.update_from_output (empty token list + finish_reason).
        if not self._v52_term_pending:
            return {}
        out: dict[int, EngineCoreOutputs] = {}
        for _rid, (_ci, _reason) in self._v52_term_pending.items():
            eco = out.setdefault(_ci, EngineCoreOutputs(outputs=[]))
            eco.outputs.append(EngineCoreOutput(
                request_id=_rid,
                new_token_ids=[],
                finish_reason=_reason,
            ))
        self._v52_term_pending.clear()
        return out

    def flush_pending_finishes(self) -> dict[int, EngineCoreOutputs]:
        """llm-scaler v52 (crash-4 R2 root fix): terminal EngineCoreOutputs
        for scheduler-finished-but-unscheduled requests (F8v2 zombie
        reaping). Called by EngineCore at the idle transition and on
        non-executed steps; see core.py _v52_flush_pending_finishes."""
        return self._v52_take_term_outputs()

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
                _rid = request.request_id
                # v52: dedup on the parked map (timer started once, never
                # reset on re-detection, so the grace always expires).
                _already_parked = (
                    self._v51_f8_mode == "v52" and _rid in self._v52_parked)
                if (request.request_id not in self._v51_f8_logged
                        or self._v51_f8_mode == "kill"
                        or (self._v51_f8_mode == "v52"
                            and not _already_parked)):
                    if self._v51_f8_mode == "kill":
                        _suffix = " (force-finishing FINISHED_ABORTED)"
                    elif self._v51_f8_mode == "v52":
                        _suffix = (
                            f" (F8v2: parked, grace="
                            f"{self._v52_f8_grace:.0f}s — force-finish only"
                            " if outputs never resume)")
                    else:
                        _suffix = (
                            " (detect-only: NOT finishing — valid requests"
                            " must never be cut)")
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
                        _suffix)
                    self._v51_f8_logged.add(request.request_id)
                if self._v51_f8_mode == "v52" and not _already_parked:
                    self._v52_parked[_rid] = time.monotonic()
                elif self._v51_f8_mode == "kill":
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

        # llm-scaler v52 (crash-4 fix, F8v2 maintenance): reap parked
        # zombies once the grace expires; cancel the timer if the request
        # recovered (outputs resumed -> placeholders drained) or finished
        # legitimately. Runs every scheduling step; the map is empty in
        # normal operation. This pass — not the scheduled-loop above — is
        # authoritative, because a Fix-A-skipped zombie (negative
        # num_new_tokens) no longer appears in num_scheduled_tokens.
        if self._v52_parked:
            _now = time.monotonic()
            for _rid in list(self._v52_parked):
                _req = self.requests.get(_rid)
                if (_req is None or _req.is_finished()
                        or _req.num_output_placeholders
                        <= self._v50_ph_limit):
                    self._v52_parked.pop(_rid, None)
                    logger.info(
                        "llm-scaler v52 F8v2: %s recovered — grace timer "
                        "cancelled (finished=%s, placeholders=%s)",
                        _rid,
                        "n/a" if _req is None else _req.is_finished(),
                        "n/a" if _req is None
                        else _req.num_output_placeholders)
                    continue
                if _now - self._v52_parked[_rid] >= self._v52_f8_grace:
                    self._v52_parked.pop(_rid, None)
                    logger.warning(
                        "llm-scaler v52 F8v2: FORCE-FINISH %s "
                        "(FINISHED_ABORTED) after %.0fs grace with no "
                        "output resolution (placeholders=%d, "
                        "num_output_tokens=%d, max_tokens=%d, "
                        "num_computed_tokens=%d) — F9 gap reaped; engine "
                        "stays up",
                        _rid, self._v52_f8_grace,
                        _req.num_output_placeholders,
                        _req.num_output_tokens, _req.max_tokens,
                        _req.num_computed_tokens)
                    # llm-scaler v52 (crash-4 R2 root fix): stash the
                    # terminal (client_index, finish_reason) so the
                    # client stream actually terminates. finish_requests
                    # returns [(req_id, client_index)]; the finish
                    # reason is read AFTER finishing per the failed-KV
                    # delivery pattern (scheduler.py:1494-1506) — the
                    # local _req reference survives _free_request.
                    _fin = self.finish_requests(
                        _rid, RequestStatus.FINISHED_ABORTED)
                    if _fin:
                        self._v52_term_pending[_rid] = (
                            _fin[0][1], _req.get_finished_reason())

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
