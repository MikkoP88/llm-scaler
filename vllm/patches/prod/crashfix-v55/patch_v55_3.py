#!/usr/bin/env python3
"""llm-scaler v55.3 patcher (crashfix-v58 wedge class, boot-order: AFTER
patch_f15b.py, BEFORE patch_v58p1.py is fine either way).

Problem (crashfix-v58 RCA, closed): the all-sessions wedge is a default-XPU-
stream device-side deadlock on both TP ranks. num_accepted_tokens_event is
recorded by step N's _update_states_after_model_execute and never fires;
step N+1's host code waits on it at execute_model entry (v32 deferred) and
in _update_states, while TP0's copy path simultaneously waits on
async_copy_ready_event (copy stream gated behind the dead default stream).
Host event protocol proven sound; nothing host-side will ever fire these
events again.

v55 (crashfix-v55) already bounds every such wait, but prod boots set
VLLM_V55_EVENT_TIMEOUT_S=600 (deliberately above the ~140s 128k-prefill
TTFT to avoid false kills) — so the 300s sample_tokens RPC timeout and the
step watchdog always win the race, producing a DIRTY teardown that leaks
TP workers and stretches outage to the full watchdog cycle.

Fix, three parts:

P1  Adaptive bound, keyed to the RECORDING step's token count (not the
    waiting step's): step_tokens <= VLLM_V55_DECODE_MAX_TOKENS (4096)
    -> VLLM_V55_DECODE_TIMEOUT_S (30), else the existing long bound
    (600 in prod). Decode-shaped forward is ~1-2s wall; 30s is a 15x
    margin. Chunked-prefill steps (mnbt 8192) and 128k prefill steps stay
    on the long bound, so no false kills. The shape is stashed at record
    time (self._v55_rec_step_tokens / async_output._v55_step_tokens) and
    read at every wait site, because the waiting step N+1 may be a tiny
    decode step waiting on an event launched by a big step N (and vice
    versa) — keying on the waiter would false-kill long-prefill steps.

P2  Fast-clean death: on timeout, append crash context (what / bound /
    recording-step shape / f15b ring tail) to /root/v55_3_crash.log, then
    SIGKILL the whole serving process group (EngineCore + both TP workers
    + API server die instantly; no dirty teardown, no leaked workers),
    fallback os._exit(86). The lane watchdog sees health-000 / dead
    container and relaunches through repro_bootV1212.sh. Container-side
    stderr is broken on this image — hence the dedicated file.

P3  _f15b.py dump-budget fix: the f15b watchdog burned all MAX_DUMPS=5 on
    post-warmup IDLE false positives (engine idle = no marks = "stall",
    wait=None). Now only dumps taken while a v55 wait is active
    (_wait_what is not None) consume the budget; idle dumps are
    rate-limited to one per 30 min and never count.

Idempotent: skips if the v55.3 marker is present. --check applies in
memory, ast.parses both results, writes nothing. --revert restores the
.bak_v55_3 snapshots. Backup taken only if not already present.
"""

import argparse
import ast
import os
import shutil
import sys

GMR = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"
F15B = "/opt/venv/lib/python3.12/site-packages/vllm/_f15b.py"
MARK = "llm-scaler v55.3"


def die(msg):
    print(f"[v55.3] FAIL: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- GMR P2+P1
CRASH_FN = '''\
def _v55_3_crash(what, bound, step_tokens):
    # llm-scaler v55.3: fast-clean death. The event's producing stream is
    # device-dead (crashfix-v58 wedge class); waiting longer only hands
    # the kill to the 300s RPC timeout, whose dirty teardown leaks TP
    # workers. Write the crash context (stderr is broken in-container),
    # then SIGKILL the serving process group so the lane watchdog can
    # relaunch within one cycle.
    try:
        import time as _t
        with open("/root/v55_3_crash.log", "a") as f:
            f.write(
                f"[{_t.strftime('%Y-%m-%d %H:%M:%S')}] v55.3 KILL "
                f"pid={os.getpid()} wait={what!r} bound={bound:.0f}s "
                f"rec_step_tokens={step_tokens!r}\\n")
            try:
                from vllm import _f15b as _f
                with _f._lock:
                    _items = list(_f._ring)
                _now = _t.monotonic()
                f.write("  f15b ring tail:\\n")
                for _tt, _n, _d, _e in _items[-24:]:
                    _st = ""
                    if _e is not None:
                        try:
                            _st = "DONE" if _e.query() else "PENDING"
                        except Exception:
                            _st = "ERR"
                    f.write(
                        f"  [{_now - _tt:8.1f}s ago] {_n} {_d} {_st}\\n")
            except Exception as _ex:
                f.write(f"  f15b ring unavailable: {_ex}\\n")
    except Exception:
        pass
    import signal
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:
        pass
    os._exit(86)


'''

OLD_WRAPPER_DEF = '''\
def _v55_wait_event(evt, what: str) -> None:
    # llm-scaler f15b: wait-state visibility (dump names the wait)
    from vllm import _f15b as _f15b_mod
    _f15b_mod.enter_wait(what)
    try:
        return _v55_wait_event_inner(evt, what)
    finally:
        _f15b_mod.exit_wait()


def _v55_wait_event_inner(evt, what: str) -> None:
    try:
        _tmo = float(os.environ.get("VLLM_V55_EVENT_TIMEOUT_S", "120"))
    except ValueError:
        _tmo = 120.0
    if not evt.query():
        # v55.2 progressive cadence: the flat 50ms poll quantized every
        # hot-path wait (regression: ~+40ms/step, genspeed -37%). Normal
        # async completion lands in <20ms — poll at 0.5ms there; back off
        # only once the wait is suspicious. Bound unchanged (crash-2).
        _dl = time.monotonic() + _tmo
        _t0 = time.monotonic()
        while not evt.query():
            _now = time.monotonic()
            if _now > _dl:
                raise RuntimeError(
                    f"llm-scaler v55 ASYNC-EVENT-STALL: {what} not "
                    f"signaled within {_tmo:.0f}s — async producer dead "
                    f"(crash-2 wedge class); failing fast instead of "
                    f"wedging the RPC thread until the step watchdog")
            _el = _now - _t0
            if _el < 0.02:
                time.sleep(0.0005)
            elif _el < 0.2:
                time.sleep(0.005)
            elif _el < 2.0:
                time.sleep(0.05)
            else:
                time.sleep(0.25)
    evt.synchronize()
'''

NEW_WRAPPER_DEF = CRASH_FN + '''\
def _v55_wait_event(evt, what: str, step_tokens=None) -> None:
    # llm-scaler f15b: wait-state visibility (dump names the wait)
    from vllm import _f15b as _f15b_mod
    _f15b_mod.enter_wait(what)
    try:
        return _v55_wait_event_inner(evt, what, step_tokens)
    finally:
        _f15b_mod.exit_wait()


def _v55_wait_event_inner(evt, what: str, step_tokens=None) -> None:
    # llm-scaler v55.3: adaptive bound keyed to the RECORDING step's
    # token count (stashed at record time by the callers). The wedge
    # class fires during decode-shaped steps; a 600s bound there loses
    # the race to the 300s sample_tokens RPC timeout, whose dirty
    # teardown leaks TP workers. Prefill-shaped steps (mnbt 8192 chunks,
    # 128k TTFT ~140s) keep the long bound — no false kills.
    try:
        _tmo = float(os.environ.get("VLLM_V55_EVENT_TIMEOUT_S", "120"))
    except ValueError:
        _tmo = 120.0
    try:
        _dtmo = float(os.environ.get("VLLM_V55_DECODE_TIMEOUT_S", "30"))
    except ValueError:
        _dtmo = 30.0
    try:
        _dmax = float(os.environ.get("VLLM_V55_DECODE_MAX_TOKENS", "4096"))
    except ValueError:
        _dmax = 4096.0
    if step_tokens is not None and step_tokens <= _dmax:
        _tmo = _dtmo
    if not evt.query():
        # v55.2 progressive cadence: the flat 50ms poll quantized every
        # hot-path wait (regression: ~+40ms/step, genspeed -37%). Normal
        # async completion lands in <20ms — poll at 0.5ms there; back off
        # only once the wait is suspicious. Bound unchanged (crash-2).
        _dl = time.monotonic() + _tmo
        _t0 = time.monotonic()
        while not evt.query():
            _now = time.monotonic()
            if _now > _dl:
                _v55_3_crash(what, _tmo, step_tokens)
                raise RuntimeError(
                    f"llm-scaler v55.3 ASYNC-EVENT-STALL: {what} not "
                    f"signaled within {_tmo:.0f}s (recording step "
                    f"{step_tokens} tok) — async producer dead "
                    f"(crash-2 wedge class); fast-clean kill issued")
            _el = _now - _t0
            if _el < 0.02:
                time.sleep(0.0005)
            elif _el < 0.2:
                time.sleep(0.005)
            elif _el < 2.0:
                time.sleep(0.05)
            else:
                time.sleep(0.25)
    evt.synchronize()
'''

# record sites: stash the recording step's shape (align + non-align)
OLD_REC_ALIGN = '''\
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
            _v32_nst = scheduler_output.num_scheduled_tokens
'''
NEW_REC_ALIGN = '''\
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
            # llm-scaler v55.3: stash the RECORDING step's shape — the
            # wait bound must key to the step that launched this event,
            # not the (later, differently-shaped) step that waits on it.
            self._v55_rec_step_tokens = (
                scheduler_output.total_num_scheduled_tokens)
            _v32_nst = scheduler_output.num_scheduled_tokens
'''

OLD_REC_NONALIGN = '''\
        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
'''
NEW_REC_NONALIGN = '''\
        else:
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()
            # llm-scaler v55.3: recording-step shape stash (non-align).
            self._v55_rec_step_tokens = (
                scheduler_output.total_num_scheduled_tokens)
'''

# wait site 1: execute_model entry (v32 deferred postprocess)
OLD_WAIT_DEFERRED = '''\
                _v55_wait_event(
                    self.num_accepted_tokens_event,
                    "num_accepted_tokens_event (deferred postprocess)")
'''
NEW_WAIT_DEFERRED = '''\
                _v55_wait_event(
                    self.num_accepted_tokens_event,
                    "num_accepted_tokens_event (deferred postprocess)",
                    step_tokens=getattr(self, "_v55_rec_step_tokens", None))
'''

# wait site 2: _update_states
OLD_WAIT_UPD = '''\
            _v55_wait_event(
                self.num_accepted_tokens_event,
                "num_accepted_tokens_event (_update_states)")
'''
NEW_WAIT_UPD = '''\
            _v55_wait_event(
                self.num_accepted_tokens_event,
                "num_accepted_tokens_event (_update_states)",
                step_tokens=getattr(self, "_v55_rec_step_tokens", None))
'''

# wait site 3: async-output-copy get_output — TWO identical occurrences:
# AsyncGPUModelRunnerOutput.get_output (:346, generation path — this lane)
# and AsyncGPUPoolingModelRunnerOutput.get_output (:568, pooling path —
# attr not attached there, getattr -> None -> long bound, safe).
OLD_WAIT_COPY = '''\
        _v55_wait_event(
            self.async_copy_ready_event,
            "async_output_copy event")
'''
NEW_WAIT_COPY = '''\
        _v55_wait_event(
            self.async_copy_ready_event,
            "async_output_copy event",
            step_tokens=getattr(self, "_v55_step_tokens", None))
'''

# wrapper construction: attach this step's shape to the wrapper
OLD_WRAP_CTOR = '''\
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
            )
            # llm-scaler v52o (amends v52e): attach step-accurate refs for
'''
NEW_WRAP_CTOR = '''\
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
            )
            # llm-scaler v55.3: the recording step's shape for the copy
            # event's bounded wait in get_output (this step recorded it).
            async_output._v55_step_tokens = (
                scheduler_output.total_num_scheduled_tokens)
            # llm-scaler v52o (amends v52e): attach step-accurate refs for
'''

GMR_EDITS = [
    ("wrapper def + crash fn", OLD_WRAPPER_DEF, NEW_WRAPPER_DEF, 1),
    ("record stash (align)", OLD_REC_ALIGN, NEW_REC_ALIGN, 1),
    ("record stash (non-align)", OLD_REC_NONALIGN, NEW_REC_NONALIGN, 1),
    ("wait: deferred postprocess", OLD_WAIT_DEFERRED, NEW_WAIT_DEFERRED, 1),
    ("wait: _update_states", OLD_WAIT_UPD, NEW_WAIT_UPD, 1),
    # both get_output classes share the identical wait block; the pooling
    # wrapper never gets the attr attached -> getattr None -> long bound
    ("wait: async_output_copy", OLD_WAIT_COPY, NEW_WAIT_COPY, 2),
    ("wrapper ctor shape attach", OLD_WRAP_CTOR, NEW_WRAP_CTOR, 1),
]

# ---------------------------------------------------------------- f15b P3
OLD_F15B_GLOBALS = '''\
_wait_what = None
_dump_count = 0
'''
NEW_F15B_GLOBALS = '''\
_wait_what = None
_dump_count = 0
_idle_dump_last = 0.0  # llm-scaler v55.3: idle-dump rate limit (wall)
'''

OLD_F15B_DUMP = '''\
def _dump(reason):
    global _dump_count
    _dump_count += 1
'''
NEW_F15B_DUMP = '''\
def _dump(reason):
    global _dump_count, _idle_dump_last
    # llm-scaler v55.3: only dumps taken while a v55 wait is ACTIVE
    # (_wait_what set) are wedge evidence and consume the budget. Idle
    # stalls (post-warmup quiescence, no engine steps -> no marks) burned
    # all 5 dumps in boot V1212 pre-cert; they are now rate-limited to
    # one per 30 min and never count.
    if _wait_what is None:
        if time.time() - _idle_dump_last < 1800:
            return
        _idle_dump_last = time.time()
    else:
        _dump_count += 1
'''

OLD_F15B_GATE = '''\
        if stall > STALL:
            if _dump_count < MAX_DUMPS:
                _dump(f"stall {stall:.0f}s")
'''
NEW_F15B_GATE = '''\
        if stall > STALL:
            # llm-scaler v55.3: idle stalls bypass the wedge budget.
            if _dump_count < MAX_DUMPS or _wait_what is None:
                _dump(f"stall {stall:.0f}s")
'''

F15B_EDITS = [
    ("idle rate-limit global", OLD_F15B_GLOBALS, NEW_F15B_GLOBALS, 1),
    ("dump budget fix", OLD_F15B_DUMP, NEW_F15B_DUMP, 1),
    ("watchdog gate fix", OLD_F15B_GATE, NEW_F15B_GATE, 1),
]


def apply_edits(path, edits, text):
    for name, old, new, expect in edits:
        n = text.count(old)
        if n != expect:
            die(f"{path}: anchor {name!r} matched {n} times (need exactly {expect})")
        text = text.replace(old, new)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="apply in memory + ast.parse, write nothing")
    ap.add_argument("--revert", action="store_true",
                    help="restore .bak_v55_3 snapshots")
    args = ap.parse_args()

    for path in (GMR, F15B):
        if not os.path.exists(path):
            die(f"missing target {path}")

    if args.revert:
        for path in (GMR, F15B):
            bak = path + ".bak_v55_3"
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print(f"[v55.3] reverted {path}")
            else:
                print(f"[v55.3] no backup for {path}")
        return

    texts = {}
    for path in (GMR, F15B):
        with open(path, encoding="utf-8") as f:
            texts[path] = f.read()

    already = MARK in texts[GMR] and "_idle_dump_last" in texts[F15B]
    if already:
        print(f"[v55.3] already applied ({MARK} present) — nothing to do")
        return

    new_gmr = apply_edits(GMR, GMR_EDITS, texts[GMR])
    new_f15b = apply_edits(F15B, F15B_EDITS, texts[F15B])

    for path, txt in ((GMR, new_gmr), (F15B, new_f15b)):
        try:
            ast.parse(txt)
        except SyntaxError as e:
            die(f"{path}: patched text fails ast.parse: {e}")

    if args.check:
        print(f"[v55.3] CHECK OK: {len(GMR_EDITS)} GMR edits + "
              f"{len(F15B_EDITS)} f15b edits anchored, ast.parse clean, "
              f"nothing written")
        return

    for path, txt in ((GMR, new_gmr), (F15B, new_f15b)):
        bak = path + ".bak_v55_3"
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        n = txt.count(MARK)
        print(f"[v55.3] patched {path} ({n} markers)")

    print("[v55.3] OK — requires engine restart to take effect")


if __name__ == "__main__":
    main()
