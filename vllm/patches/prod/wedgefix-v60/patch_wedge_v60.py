#!/usr/bin/env python3
"""llm-scaler v60 wedge patcher (WEDGEFIX-A/B/C).

Root-cause evidence 2026-09-21 (CC-intensive usage crash loop):
  - NOT OOM: zero oom-kill records, host 376G/6G used.
  - 'Killed' = v55.3 fast-clean SIGKILL after device-dead-stream wedge
    (VLLM_V55_DECODE_TIMEOUT_S=30), bracketed by xe ccs/bcs engine
    resets on BOTH tiles; GPU devcoredumps captured (WD_crash_*).
  - Wedge lives in the MTP spec-drafter path (f15b ring: drafter ARs
    numel=25600 all DONE then silence; fork KNOWN_ISSUES #11 >=32k ctx).
  - AMPLIFIER: AsyncScheduler was ON in every boot ('Asynchronous
    scheduling is enabled', vllm.py:920) despite the §24 K 'async OFF'
    label — the fork defaults it on; xpu.py only forced it off for PP>1.
  - MITIGATION: VLLM_XPU_SPEC_DRAFT_BARRIER (v25 pre-drafter drain,
    gated by MIN_CTX) was default-OFF since v37 for +13.7% perf under
    short-context posture; CC long-context traffic re-opened the wedge.
  - C-GAP (stage4 soak, 13:45:33): v60-A's async flip silently disabled
    the AsyncScheduler zombie reapers (v52m zero-emission strike-out +
    F8v2 force-finish — async_scheduler.py only, zero marks in the sync
    sched/scheduler.py). A crash-4 NaN-zombie request (onset ~11k ctx
    under 12k-30k tool-call soak) stayed scheduled with empty emissions
    until its poisoned rows hit an unclamped worker gather:
    IndexKernelUtils.h:63 'vectorized gather kernel index out of
    bounds' SYCL assert storm -> VllmWorker-1 native abort ->
    EngineDeadError (0 engine resets; v55.3 did NOT fire; evidence
    serve_v60_soak_oob.log). WEDGEFIX-C ports the v52m semantics to
    the sync scheduler.

PATCHES (defaults baked; env kill-switches preserved):
  A (xpu.py):            async_scheduling DEFAULT DISABLED on XPU
                         (opt-in only via VLLM_XPU_ALLOW_ASYNC=1).
  B (gpu_model_runner):  _SPEC_DRAFT_BARRIER default 0->1,
                         _SPEC_DRAFT_BARRIER_MIN_CTX default 0->0
                         (every step; drill3 wedge fired at ~4k ctx,
                         so the 8192 gate was inert there). Explicit
                         env still overrides.
  C (sched/scheduler.py): sync-path zombie guard — v60c poisoned-row
                         strike-out (VLLM_V60_ZOMBIE_STRIKES, default 1
                         = immediate; the fatal gather fires the step
                         after the first sentinel row) + runaway
                         output-count guard (VLLM_V60_RUNAWAY_GUARD,
                         default on). FINISHED_ABORTED delivered
                         client-visibly in the same step's buckets.
  D (gpu_model_runner): WEDGEFIX-E (v60g) no-draft degenerate-row
                         sanitize — plain-decode steps without draft
                         proposals reach sampling with UNMATERIALIZED
                         deferred state (v60f instrumentation 2026-09-21:
                         deferred logits zeros; sample_hidden_states NaN;
                         lm_head recompute over them is still NaN — v60e
                         proven insufficient and superseded). The XPU
                         topk/topp op deterministically returns token 0
                         for NaN rows (offline repro3: only degenerate
                         rows do this) -> strict-JSON FSM rejects token 0
                         -> FINISHED_ERROR -> deterministic HTTP 500 on
                         every structured-output request hitting a
                         no-draft step (XGrammar-2 regression found by
                         stage4-T2 after the async flip; the async lane
                         never hit it because async always runs the
                         spec/verify pipeline). Plain completions on the
                         same step class silently sampled token 0.
                         Root (mechanism level): the fork's nospec
                         FULL_DECODE_ONLY graph never materializes its
                         hidden-state output into the deferred
                         execute_model_state (gpu_model_runner.py:4732
                         gather reads unmaterialized graph rows; xj
                         extract; flash_attn.py:17 documents the same
                         nospec-graph fragility class). Graph wiring is
                         not patchable from Python; E makes the damage
                         non-fatal instead: AFTER the grammar bitmask
                         apply, NaN rows are set to 0.0 so the sampler
                         picks uniformly over GRAMMAR-ALLOWED tokens
                         (the -inf mask survives) — schema-legal token,
                         no FSM reject, no 500, XGrammar-2 crash-free.
                         (VLLM_V60_NOSPEC_LOGITS_FIX=0 disables.) First
                         10 firings log 'NO-DRAFT DEGENERATE-ROW
                         SANITIZED'. Decode-only steps (all scheduled
                         tokens == 1) are touched; prefills stay stock.

Speculative decoding (MTP x4) remains fully supported on the sync
scheduler. STALFIX A/B/C and XGrammar-2 (0.2.7) untouched.

Usage:  python3 patch_wedge_v60.py [--check|--revert|--apply]
Idempotent; anchor-count==1 asserted; backups <file>.v60bak;
py_compile after write with auto-restore on failure.
"""

import os
import py_compile
import shutil
import sys

SP = "/opt/venv/lib/python3.12/site-packages/vllm"
F_XPU = f"{SP}/platforms/xpu.py"
F_GMR = f"{SP}/v1/worker/gpu_model_runner.py"
F_SCHED = f"{SP}/v1/core/sched/scheduler.py"

MARK = "llm-scaler v60"

# ---------------------------------------------------------------- A: xpu.py
A_OLD = """\
        # Disable async scheduling when pipeline parallelism is enabled on XPU
        if (
            parallel_config.pipeline_parallel_size > 1
        ):
            scheduler_config = vllm_config.scheduler_config
            scheduler_config.async_scheduling = False
"""

A_NEW = """\
        # llm-scaler v60 (wedge fix): Asynchronous scheduling is DEFAULT
        # DISABLED on XPU. The AsyncScheduler event pipeline amplifies the
        # spec-drafter device-wedge class (crashfix-v58 / GSD-12919:
        # 0 tok/s -> v55.3 fast-clean SIGKILL; xe ccs/bcs engine resets on
        # both tiles; evidence 2026-09-21 killan_evidence*.log). Async OFF
        # + draft barrier = certified crash-free posture. Spec decode (MTP)
        # stays fully supported on the sync scheduler. Opt back in ONLY
        # for research with VLLM_XPU_ALLOW_ASYNC=1 — NEVER in production
        # images (no image may boot with 'Asynchronous scheduling is
        # enabled').
        if (
            parallel_config.pipeline_parallel_size > 1
            or os.environ.get("VLLM_XPU_ALLOW_ASYNC", "0") != "1"
        ):
            scheduler_config = vllm_config.scheduler_config
            scheduler_config.async_scheduling = False
"""

# ------------------------------------------------- B: gpu_model_runner.py
B_OLD = """\
_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "0") == "1"
)
_SPEC_DRAFT_BARRIER_MIN_CTX = int(
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX", "0")
)
"""

B_NEW = """\
# llm-scaler v60 (wedge fix): barrier DEFAULT ON, every step — CC
# traffic re-opened the KNOWN_ISSUES #11 drafter wedge (2026-09-21: xe
# engine resets + 0 tok/s kills under Claude Code intensive usage; the
# observed wedge fired at ~4k ctx, so the first-cut MIN_CTX=8192 gate
# was INERT at the real wedge site — drill3 died at computed=3994).
# MIN_CTX default 0 = unconditional pre-drafter drain. Explicit
# VLLM_XPU_SPEC_DRAFT_BARRIER=0 restores the v37 posture.
_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "1") == "1"
)
_SPEC_DRAFT_BARRIER_MIN_CTX = int(
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER_MIN_CTX", "0")
)
"""

# ------------------------------------------------ C: sched/scheduler.py
C_OLD = """\
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
"""

C_NEW = """\
    def _v60c_zombie_init(self) -> None:
        # llm-scaler v60c (sync zombie guard): port of the async v52m
        # strike-out to the SYNC scheduler, MTP-aware. v60-A turned
        # async scheduling OFF, which silently disabled the
        # AsyncScheduler zombie reapers (v52m strike-out + F8v2
        # force-finish live in async_scheduler.py only) — the 2026-09-21
        # stage4 soaks (13:45:33 AND 14:24:35, both DETERMINISTIC at
        # computed=11264 on the first 3-chunk tool-call request) lost
        # VllmWorker-1 to the crash-4 NaN-zombie class: poisoned sampler
        # rows reached an unclamped worker gather ->
        # IndexKernelUtils.h:63 'vectorized gather kernel index out of
        # bounds' SYCL assert -> native worker abort -> EngineDeadError
        # (serve_v60_soak_oob.log / v60_soak_crash2; 0 engine resets,
        # v55.3 never fired). On the MTP lane the zombie's RAW rows stay
        # NON-EMPTY (empty-row detection logged nothing in crash-2) —
        # the signature is the v52n one: healthy rows never contain ids
        # >= vocab_size; -1 rejection tails are normal padded transport.
        # A NaN-zombie can never emit a valid token again (5/5
        # async-era episodes, zero recoveries), so poisoned/empty-row
        # steps abort the request client-visibly while the engine stays
        # up. DEFAULT STRIKES=1 (immediate): crash-3 (14:39 soak, C-v2
        # lane) proved the fatal worker gather fires on the step AFTER
        # the first sentinel row — POISONED-ROW strike-1 was logged,
        # then the worker died before strike-2/3 could accrue. The
        # strike-out rides the SAME update_from_output that first saw
        # the sentinel, so the zombie is never scheduled again and the
        # poisoned gather never runs. Structured-output requests are
        # exempt (grammar-masked bonus rows can legitimately resolve
        # empty). VLLM_V60_ZOMBIE_STRIKES=0 disables.
        import os as _os
        self._v60c_strikes_n = int(
            _os.environ.get("VLLM_V60_ZOMBIE_STRIKES", "1"))
        self._v60c_strikes: dict[str, int] = {}
        self._v60c_logged: set[str] = set()
        self._v60c_runaway = (
            _os.environ.get("VLLM_V60_RUNAWAY_GUARD", "1") == "1")
        _v60c_vocab = 0
        try:
            _mc = getattr(self, "vllm_config", None)
            if _mc is not None:
                _v60c_vocab = _mc.model_config.get_vocab_size()
        except Exception:
            _v60c_vocab = 0
        self._v60c_vocab = _v60c_vocab
        logger.warning(
            "llm-scaler v60c SYNC ZOMBIE GUARD armed (strikes=%d, "
            "vocab=%d, runaway=%s)",
            self._v60c_strikes_n, self._v60c_vocab, self._v60c_runaway)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # llm-scaler v60c wrap: strike counting BEFORE the original
        # bookkeeping (read-only over sampled_token_ids), aborts AFTER
        # it so the terminal FINISHED_ABORTED output is appended to the
        # SAME returned dict and rides this step to the client (sync
        # delivery: EngineCore sends the returned dict immediately).
        if not hasattr(self, "_v60c_strikes_n"):
            self._v60c_zombie_init()
        _v60c_doomed: list[tuple[str, str]] = []
        try:
            _sti = model_runner_output.sampled_token_ids
            _idx = model_runner_output.req_id_to_index
            _spec_w = getattr(self, "num_spec_tokens", 0) or 0
            _vocab = self._v60c_vocab
            if _sti and _idx:
                for _rid, _nsched in (
                        scheduler_output.num_scheduled_tokens.items()):
                    _req = self.requests.get(_rid)
                    if (_req is None or _req.is_finished()
                            or _req.is_prefill_chunk
                            or getattr(_req, "use_structured_output",
                                       False)):
                        continue
                    _row = list(_sti[_idx[_rid]]) if _rid in _idx else None
                    if _row:
                        _poisoned = _vocab > 0 and any(
                            isinstance(_t, int) and _t >= _vocab
                            for _t in _row)
                    else:
                        _poisoned = True  # empty row or unregistered
                    if not _poisoned:
                        if _rid in self._v60c_strikes:
                            self._v60c_strikes.pop(_rid, None)
                            self._v60c_logged.discard(_rid)
                        continue
                    _s = self._v60c_strikes.get(_rid, 0) + 1
                    self._v60c_strikes[_rid] = _s
                    if (self._v60c_strikes_n > 0
                            and _s >= self._v60c_strikes_n):
                        _v60c_doomed.append((_rid, "poisoned-row"))
                    if _s >= 1 and _rid not in self._v60c_logged:
                        self._v60c_logged.add(_rid)
                        logger.warning(
                            "llm-scaler v60c POISONED-ROW: %s scheduled "
                            "(%d tokens this step, row_len=%d, "
                            "row_head=%s) strike %d "
                            "(num_output_tokens=%d, num_computed_tokens=%d)"
                            " — NaN-zombie sentinel signature on the "
                            "sync path",
                            _rid, _nsched, len(_row) if _row else 0,
                            [_t for _t in (_row or [])][:6], _s,
                            _req.num_output_tokens,
                            _req.num_computed_tokens)
            if self._v60c_runaway:
                for _rid in scheduler_output.num_scheduled_tokens:
                    _req = self.requests.get(_rid)
                    if _req is None or _req.is_finished():
                        continue
                    if (_req.num_output_tokens
                            > _req.max_tokens + _spec_w + 8):
                        _v60c_doomed.append((_rid, "runaway"))
        except Exception:
            pass
        ret = self._v60c_update_from_output_orig(
            scheduler_output, model_runner_output)
        for _rid, _cause in _v60c_doomed:
            try:
                self._v60c_strike_out(_rid, _cause, ret)
            except Exception:
                pass
        return ret

    def _v60c_strike_out(
        self, _rid: str, _cause: str,
        ret: dict[int, EngineCoreOutputs],
    ) -> None:
        # llm-scaler v60c: FINISHED_ABORTED for a zombie; terminal
        # output injected into THIS step's client buckets. Idempotent,
        # never raises — a failed/vanished request is dropped and the
        # strike state is consumed here so retries are not duplicated.
        _s = self._v60c_strikes.pop(_rid, 0)
        _req = self.requests.get(_rid)
        self._v60c_logged.discard(_rid)
        if _req is None or _req.is_finished():
            return
        logger.warning(
            "llm-scaler v60c STRIKE-OUT (%s): %s — %d consecutive "
            "poisoned/empty sampled rows while scheduled (crash-4 "
            "NaN-zombie terminal state, sync path) — FINISHED_ABORTED "
            "delivered client-visibly, engine stays up "
            "(num_output_tokens=%d, num_computed_tokens=%d)",
            _cause, _rid, _s,
            _req.num_output_tokens, _req.num_computed_tokens)
        _fin = self.finish_requests([_rid],
                                    RequestStatus.FINISHED_ABORTED)
        # finish_requests returns [(req_id, client_index), ...]
        for _rid2, _ci in _fin:
            _reason = _req.get_finished_reason()
            eco = ret.get(_ci)
            if eco is None:
                eco = EngineCoreOutputs(outputs=[])
                ret[_ci] = eco
            eco.outputs.append(EngineCoreOutput(
                request_id=_rid, new_token_ids=[],
                finish_reason=_reason))

    def _v60c_update_from_output_orig(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
"""

# ------------------------------------------------- D: gpu_model_runner.py
# WEDGEFIX-E (v60g): no-draft plain-decode steps reach sampling with
# UNMATERIALIZED deferred state. v60f instrumentation (2026-09-21,
# serve_full.log 468-492): healthy spec steps (#6 amax=8656, #7
# min=-inf-from-grammar amax=220) then the first no-draft step: deferred
# logits zeros; sample_hidden_states NaN; the v60e lm_head recompute over
# those hidden states is STILL NaN (PREOP #8 nan_any=[True] amax=[1]) ->
# XPU topk/topp op returns token 0 for degenerate rows (offline repro3:
# only all--inf/NaN rows do this) -> backend_xgrammar.py:158
# 'Failed to advance FSM for tokens 0' -> scheduler 'grammar rejected
# tokens [0]' -> FINISHED_ERROR -> deterministic HTTP 500. Mechanism:
# execute_model's deferred tlogits stage (gpu_model_runner.py ~4732)
# gathers hidden_states[logits_indices] where hidden_states is the
# nospec FULL_DECODE_ONLY graph output — those rows are never
# materialized for this step class (fork graph wiring; not patchable
# from Python). E makes the damage non-fatal at the sampling boundary:
# AFTER apply_grammar_bitmask, NaN rows -> 0.0 so the sampler chooses
# uniformly over grammar-ALLOWED tokens (-inf mask survives) -> legal
# token, no FSM reject, no 500.
D_MARK = "llm-scaler v60g (WEDGEFIX-E)"
D_OLD = """\
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
"""

D_NEW = """\
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # llm-scaler v60g (WEDGEFIX-E): no-draft plain-decode steps on the
        # MTP lane (scheduler schedules a bare 1-token step with no draft
        # proposals, e.g. after a rejection-tail step) reach sampling with
        # UNMATERIALIZED deferred state: the nospec FULL_DECODE_ONLY graph
        # never writes the hidden rows the deferred tlogits gather reads,
        # so the logits row is zeros-or-NaN and the XPU topk/topp op
        # deterministically returns token 0 for degenerate rows (offline
        # repro3). Strict-JSON grammars reject token 0 (backend_xgrammar
        # 'Failed to advance FSM' -> 'grammar rejected tokens [0]' ->
        # FINISHED_ERROR -> HTTP 500, deterministic on every
        # structured-output request whose lifetime contains a no-draft
        # step); plain completions silently sample token 0 on the same
        # step class. The spec/verify path (rejection_sampler) is
        # unaffected — async-mode batteries never saw this because async
        # always runs the spec pipeline. v60e's lm_head recompute over
        # sample_hidden_states was PROVEN INSUFFICIENT (those hidden
        # states are NaN too; serve_full.log 468-492) and is superseded
        # by this sanitize. Fix: AFTER the grammar bitmask apply, replace
        # NaN entries with 0.0 so the sampler picks uniformly over
        # GRAMMAR-ALLOWED tokens (the -inf mask survives the replace):
        # schema-legal token, no FSM reject, no 500, XGrammar-2
        # crash-free. Decode-only steps (all scheduled tokens == 1) are
        # touched; prefills stay stock. VLLM_V60_NOSPEC_LOGITS_FIX=0
        # disables.
        if (spec_decode_metadata is None
                and logits is not None
                and logits.dtype.is_floating_point):
            _nst = scheduler_output.num_scheduled_tokens
            if _nst and all(_v == 1 for _v in _nst.values()):
                import os as _v60g_os
                if (_v60g_os.environ.get(
                        "VLLM_V60_NOSPEC_LOGITS_FIX", "1") == "1"):
                    _v60g_nan = torch.isnan(logits).any(dim=-1)
                    if bool(_v60g_nan.any()):
                        _v60g_rows = logits[_v60g_nan]
                        logits[_v60g_nan] = torch.where(
                            torch.isnan(_v60g_rows),
                            torch.zeros_like(_v60g_rows),
                            _v60g_rows)
                        self._v60g_fired = getattr(self, "_v60g_fired", 0) + 1
                        if self._v60g_fired <= 10:
                            logger.warning(
                                "llm-scaler v60g NO-DRAFT DEGENERATE-ROW "
                                "SANITIZED #%d rows=%d/%d — NaN logits "
                                "from unmaterialized nospec graph output "
                                "replaced with uniform over "
                                "grammar-allowed tokens (root: fork "
                                "nospec FULL_DECODE_ONLY output wiring, "
                                "wedgefix-v60 E)",
                                self._v60g_fired, int(_v60g_nan.sum()),
                                int(logits.shape[0]))
"""

# Per-needle marks (a file can hold several needles; each is skipped
# individually once applied — the old file-level MARK check would
# short-circuit D because B already marks gpu_model_runner.py).
PATCHES = [
    ("A", F_XPU, "WEDGEFIX-A async-default-OFF (v60)",
     [(A_OLD, A_NEW, "llm-scaler v60 (wedge fix): Asynchronous scheduling")]),
    ("B", F_GMR, "WEDGEFIX-B draft-barrier-ON (v60)",
     [(B_OLD, B_NEW, "llm-scaler v60 (wedge fix): barrier DEFAULT ON")]),
    ("C", F_SCHED, "WEDGEFIX-C sync-zombie-guard (v60c)",
     [(C_OLD, C_NEW, "llm-scaler v60c (sync zombie guard)")]),
    ("D", F_GMR, "WEDGEFIX-E no-draft degenerate-row sanitize (v60g)",
     [(D_OLD, D_NEW, D_MARK)]),
]


def die(msg: str) -> None:
    print(f"[v60] FAIL: {msg}")
    sys.exit(1)


def check_apply(revert: bool) -> None:
    for tag, path, label, pairs in PATCHES:
        try:
            src = open(path, encoding="utf-8").read()
        except OSError as e:
            die(f"{label}: cannot read {path}: {e}")
        bak = path + ".v60bak"
        if revert:
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print(f"[v60] {tag} REVERTED {path}")
            else:
                print(f"[v60] {tag} no-backup (unchanged) {path}")
            continue
        applied_any = False
        ok = True
        for old, new, nmark in pairs:
            if nmark in src:
                print(f"[v60] {tag} ALREADY-APPLIED ({nmark[:34]}…)")
                continue
            n = src.count(old)
            if n != 1:
                print(f"[v60] {tag} ANCHOR-COUNT {n} (need 1) in {path}")
                ok = False
                continue
            if not os.path.exists(bak):
                shutil.copyfile(path, bak)
            src = src.replace(old, new)
            applied_any = True
        if not ok:
            die(f"{label}: anchor check failed")
        if not applied_any:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            shutil.copyfile(bak, path)
            die(f"{label}: py_compile failed (restored): {e}")
        print(f"[v60] {tag} APPLIED {path}")
    print("[v60] pass complete")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--check":
        # check-only: report anchor state without writing
        for tag, path, label, pairs in PATCHES:
            try:
                src = open(path, encoding="utf-8").read()
            except OSError as e:
                die(f"{label}: {e}")
            for old, new, nmark in pairs:
                state = "ALREADY" if nmark in src else (
                    "ANCHOR-OK" if src.count(old) == 1 else "ANCHOR-BAD")
                print(f"[v60] {tag} {state} {path}")
        sys.exit(0)
    check_apply(revert=(mode == "--revert"))
