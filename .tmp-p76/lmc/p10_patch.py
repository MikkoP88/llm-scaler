#!/usr/bin/env python3
"""p10_patch.py — apply/revert the P10 num_accepted_tokens direct-copy
fallback in the container's gpu_model_runner.py (v138d/P10, 2026-10-10).

Problem (measured on live lane .65, 2026-10-10): long-context chunked
prefill (>=16k, multi-step past the MNBT 8192 first step) parks the
worker 23-33 s at the two num_accepted_tokens_event entry waits, per
chunk-step. Store cadence proof (16k axis @1024): first 8192-step 3.7 s,
then 23 s and 26 s gaps, then five 1.05 s chunks, then ~33 s at the
prefill->decode boundary. Waits resolve just under the 30 s decode
bound; ASYNC-EVENT-STALL never fires (over-bound = os._exit(86) with no
log line — also never fired).

Fix: keep the v55.3 bounded-poll + crash semantics, but after 250 ms of
polling spawn ONE helper thread that performs a direct blocking copy of
the tiny counts slice into the pinned CPU tensor and sets a flag; the
main thread proceeds the moment either the event or the flag lands. A
dead stream still fast-kills at the original bound (no new wedge class;
a stuck helper thread is a leaked daemon, never a blocker). Concurrent
writes of identical bytes into the pinned tensor are unobservable.

Usage: p10_patch.py apply|revert
"""
import sys
import py_compile

PATH = "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py"

HELPER = '''

def _p10_wait_counts(mr, what: str, step_tokens=None) -> None:
    # llm-scaler v138d/P10 (2026-10-10): long-ctx chunked prefill parks
    # the worker ~23-33s at the num_accepted_tokens entry waits — the
    # event completes but its visibility lands absurdly late while the
    # engine round-trip idles. Needed semantics only: the pinned
    # num_accepted_tokens_cpu tensor must hold the recording step's
    # counts. Poll the event briefly (normal path <20ms, unchanged);
    # if late, run ONE bounded direct blocking copy of the tiny counts
    # buffer in a helper thread (never blocks this loop) and proceed the
    # moment it lands. v55.3 crash bound preserved verbatim: a truly
    # dead stream still fast-kills — no new wedge exposure.
    evt = mr.num_accepted_tokens_event
    if evt is None or evt.query():
        return
    _dl = time.monotonic() + 0.25
    while time.monotonic() < _dl:
        if evt.query():
            return
        time.sleep(0.005)
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
    done = threading.Event()

    def _direct_copy():
        try:
            mr.input_batch.num_accepted_tokens_cpu_tensor.copy_(
                mr.num_accepted_tokens.gpu)
        except Exception:
            pass
        done.set()

    _t = threading.Thread(target=_direct_copy, daemon=True)
    _t.start()
    _t0 = time.monotonic()
    while True:
        if evt.query() or done.is_set():
            try:
                logger.warning(
                    "[P10] counts wait resolved via %s after %.3fs (%s)",
                    "event" if evt.query() else "direct copy",
                    time.monotonic() - _t0, what)
            except Exception:
                pass
            return
        if time.monotonic() - _t0 > _tmo:
            _v55_3_crash(what, _tmo, step_tokens)
            raise RuntimeError(
                f"llm-scaler v138d ASYNC-EVENT-STALL: {what} not "
                f"signaled within {_tmo:.0f}s (direct copy also failed) "
                f"— async producer dead (crash-2 wedge class); "
                f"fast-clean kill issued")
        time.sleep(0.05)
'''

SITE_A_OLD = """            _v55_wait_event(
                self.num_accepted_tokens_event,
                "num_accepted_tokens_event (_update_states)",
                step_tokens=getattr(self, "_v55_rec_step_tokens", None))"""
SITE_A_NEW = """            _p10_wait_counts(
                self, "num_accepted_tokens_event (_update_states)",
                step_tokens=getattr(self, "_v55_rec_step_tokens", None))"""

SITE_B_OLD = """                _v55_wait_event(
                    self.num_accepted_tokens_event,
                    "num_accepted_tokens_event (deferred postprocess)",
                    step_tokens=getattr(self, "_v55_rec_step_tokens", None))"""
SITE_B_NEW = """                _p10_wait_counts(
                    self, "num_accepted_tokens_event (deferred postprocess)",
                    step_tokens=getattr(self, "_v55_rec_step_tokens", None))"""

MARKER = "def _p10_wait_counts("


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    src = open(PATH).read()
    if mode == "apply":
        if MARKER in src:
            raise SystemExit("ALREADY_PATCHED")
        if src.count(SITE_A_OLD) != 1 or src.count(SITE_B_OLD) != 1:
            raise SystemExit("CALL_SITES_NOT_FOUND a=%d b=%d" % (
                src.count(SITE_A_OLD), src.count(SITE_B_OLD)))
        anchor = "import torch\nimport torch.distributed\n"
        if src.count(anchor) < 1:
            raise SystemExit("ANCHOR_NOT_FOUND")
        src = src.replace(anchor, anchor.rstrip("\n") + "\n" + HELPER, 1)
        src = src.replace(SITE_A_OLD, SITE_A_NEW, 1)
        src = src.replace(SITE_B_OLD, SITE_B_NEW, 1)
    elif mode == "revert":
        if MARKER not in src:
            raise SystemExit("NOT_PATCHED")
        start = src.index("\ndef _p10_wait_counts(")
        end = src.index("\nimport torch.nn as nn\n", start)
        src = src[:start] + "\nimport torch.nn as nn" + src[end + len("\nimport torch.nn as nn\n"):]
        src = src.replace(SITE_A_NEW, SITE_A_OLD, 1)
        src = src.replace(SITE_B_NEW, SITE_B_OLD, 1)
    else:
        raise SystemExit("MODE?")
    open(PATH, "w").write(src)
    py_compile.compile(PATH, doraise=True)
    print("PATCH_%s_OK" % mode.upper())


if __name__ == "__main__":
    main()
