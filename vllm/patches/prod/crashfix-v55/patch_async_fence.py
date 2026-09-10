#!/usr/bin/env python3
"""patch_async_fence.py — llm-scaler v55 (crashfix-v55), patch 1 of 2.

Two fences against the 2026-09-10 crash pair, both in
vllm/v1/worker/gpu_model_runner.py:

1. BOUNDED ASYNC-EVENT WAITS. The async-scheduling machinery synchronizes
   torch events at four sites with an UNBOUNDED .synchronize():

     - WorkerAsyncOutputCopy.get_output        (async_copy_ready_event)
     - <draft async output>.get_output         (async_copy_ready_event)
     - _update_states                          (num_accepted_tokens_event)
     - execute_model deferred postprocess      (num_accepted_tokens_event)

   Crash-2 (fp8_e5m2 + mtp k=3, solo request @17,182 computed tokens):
   the flight recorders of BOTH ranks end with a cleanly completed
   `propose end` (fr_521/fr_527 t=2430.27 s == 15:44:30, freeze 15:44:59),
   with no collective in flight — the accepted-counts event of the verify
   step simply never signaled. The unbounded synchronize wedged the
   worker's RPC thread; `sample_tokens` timed out; the scheduler could
   not reach ANY guard (it was blocked on the RPC); 600 s step watchdog
   killed both ranks. dmesg shows NO GPU fault in the window: host-side
   deadlock on a dead event, not device corruption. Crash-1 surfaced at
   the same two event sites with UR_RESULT_ERROR_DEVICE_LOST — same
   machinery, messenger not killer.

   Fix: _v55_wait_event() polls evt.query() with a 50 ms cadence and
   raises a named RuntimeError after VLLM_V55_EVENT_TIMEOUT_S (default
   120 s). Engine dies in seconds with the culprit event named, instead
   of silent service loss for 5+ minutes.

2. GAP-FENCE on the v52c DISCARD-GAP probe. Crash-2's precursor
   (15:41:33): worker num_tokens=3117 vs scheduler optimistic 2048
   (gap=1069, just past a 2048-block boundary — same onset geometry as
   the crash-4 onsets 2085/4172). The request was discarded and
   re-prefilled over stale worker state; the engine wedged ~3 min later.
   v52c was detect-only. v55 escalates on the LATCHING signature only:
   a non-decreasing gap series over 3 consecutive discard observations
   (all >= 8) means no chunk progress — healthy multi-chunk prefills
   decrease every chunk (routine one-shot gaps are large: gap =
   remaining prompt; validation run 1 observed a benign 2005 and
   convicted the initial |gap|>=256 immediate arm as a false trip) ->
   logger.error + raise. Per-request KV/mamba state after such a
   divergence is untrustworthy (v52e: NaN logits); fast-fail beats
   wedge. VLLM_V55_GAP_FENCE=0 reverts to detect-only.

Patcher protocol (flash-fp8-fanout-v54 style): exact anchors must match
the expected count, py_compile gates the write, idempotent (skips files
already carrying the v55 markers).
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "_v55_wait_event"

# ---------------------------------------------------------------- helpers


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, src: str) -> None:
    p.write_text(src, encoding="utf-8", newline="\n")


def _compile_gate(p: Path) -> None:
    py_compile.compile(str(p), doraise=True)


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR-FAIL {what}: expected 1 occurrence, found {n}")
    return src.replace(old, new, 1)


# ---------------------------------------------------------------- patch 1a
# Module-level bounded-wait helper, inserted after `import numpy as np`.

HELPER_ANCHOR = "import numpy as np\n"

HELPER_CODE = '''import numpy as np

# llm-scaler v55 (crashfix-v55): bounded wait for async events. An event
# that is never recorded must kill the engine within seconds, with the
# culprit named — not wedge the RPC thread until the step watchdog
# (crash-2, 2026-09-10 15:44:30: last completed op on BOTH ranks was a
# clean `propose end`, num_accepted_tokens_event never signaled,
# sample_tokens RPC timed out, 600 s watchdog kill; dmesg clean = no
# device fault, host-side deadlock on a dead event).
def _v55_wait_event(evt, what: str) -> None:
    try:
        _tmo = float(os.environ.get("VLLM_V55_EVENT_TIMEOUT_S", "120"))
    except ValueError:
        _tmo = 120.0
    if not evt.query():
        _dl = time.monotonic() + _tmo
        while not evt.query():
            if time.monotonic() > _dl:
                raise RuntimeError(
                    f"llm-scaler v55 ASYNC-EVENT-STALL: {what} not "
                    f"signaled within {_tmo:.0f}s — async producer dead "
                    f"(crash-2 wedge class); failing fast instead of "
                    f"wedging the RPC thread until the step watchdog")
            time.sleep(0.05)
    evt.synchronize()
'''

# ---------------------------------------------------------------- patch 1b
# The four unbounded synchronize sites.

SYNC_COPY_OLD = "        self.async_copy_ready_event.synchronize()\n"
SYNC_COPY_NEW = (
    "        _v55_wait_event(\n"
    "            self.async_copy_ready_event,\n"
    '            "async_output_copy event")\n'
)
# expected occurrences: get_output of both async-output classes
SYNC_COPY_COUNT = 2

SYNC_UPD_OLD = """        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
"""
SYNC_UPD_NEW = """        if self.num_accepted_tokens_event is not None:
            _v55_wait_event(
                self.num_accepted_tokens_event,
                "num_accepted_tokens_event (_update_states)")
"""

SYNC_PP_OLD = """                # one full engine round-trip of host work has elapsed.
                self.num_accepted_tokens_event.synchronize()
"""
SYNC_PP_NEW = """                # one full engine round-trip of host work has elapsed.
                _v55_wait_event(
                    self.num_accepted_tokens_event,
                    "num_accepted_tokens_event (deferred postprocess)")
"""

# ---------------------------------------------------------------- patch 2a
# Per-request gap history recorded alongside the v52c probe.

HIST_INIT_OLD = """                _seen = getattr(self, "_v52c_seen", None)
                if _seen is None:
                    _seen = self._v52c_seen = {}
"""
HIST_INIT_NEW = """                _seen = getattr(self, "_v52c_seen", None)
                if _seen is None:
                    _seen = self._v52c_seen = {}
                if getattr(self, "_v55_gap_hist", None) is None:
                    self._v55_gap_hist = {}
"""

HIST_RECORD_OLD = """                    _gap = int(_gaps[_i])
                    if _seen.get(_rid2) == _gap:
                        continue
                    _seen[_rid2] = _gap
                    logger.warning(
"""
HIST_RECORD_NEW = """                    _gap = int(_gaps[_i])
                    # v55: record EVERY discard observation (the v52c
                    # _seen dedup would starve the latching detector on
                    # a frozen gap). Gap semantics learned on validation
                    # run 1 (2026-09-10 18:11:56, warmup p5): on a
                    # discard step, gap = prompt_len - (computed + this
                    # chunk's budget share) = REMAINING PROMPT — large
                    # values are ROUTINE for any multi-chunk prefill
                    # co-batched with decodes (observed |gap| 2005).
                    # The zombie signature is a NON-DECREASING series:
                    # no chunk progress across 3+ observations.
                    # v55.1: probe demoted WARNING->DEBUG — routine
                    # prefills fire it per chunk (8 WARNINGs in 3 healthy
                    # validation cycles, run 4); the GAP-FENCE escalation
                    # below is the anomaly signal that matters.
                    _gh = self._v55_gap_hist.get(_rid2)
                    if _gh is None:
                        _gh = self._v55_gap_hist[_rid2] = []
                    _gh.append(abs(_gap))
                    if len(_gh) > 16:
                        del _gh[0]
                    if _seen.get(_rid2) == _gap:
                        continue
                    _seen[_rid2] = _gap
                    logger.debug(
"""

# ---------------------------------------------------------------- patch 2b
# Escalation after the detect-only try/except.

FENCE_OLD = """        except Exception:
            pass

        # Sync num_accepted_tokens from CPU (set by
"""
FENCE_NEW = """        except Exception:
            pass

        # llm-scaler v55 GAP-FENCE (crash-4/crash-2 zombie onset): the
        # v52c probe above is detect-only. Gap semantics: on a discard
        # step gap = REMAINING PROMPT beyond this chunk's budget — large
        # one-shot values are ROUTINE multi-chunk prefill (validation
        # run 1 observed |gap| 2005 on a healthy warmup p5 and the
        # initial |gap|>=256 arm was a false trip, removed). The zombie
        # signature — crash-2 precursor (15:41:33) and the v52e NaN
        # verdict class — is a NON-DECREASING gap series: no chunk
        # progress across 3+ consecutive discard observations (healthy
        # prefills decrease every chunk). Fence: 3 non-decreasing
        # observations, all >= 8 -> logger.error + raise (per-request
        # KV/mamba state is untrustworthy; a fast, loud, named death
        # beats the silent 10-minute crash-2 wedge). Disarm per boot
        # with VLLM_V55_GAP_FENCE=0 (probe warning stays).
        if os.environ.get("VLLM_V55_GAP_FENCE", "1") != "0":
            _v55_live = set(self.input_batch.req_ids)
            _v55_hist = getattr(self, "_v55_gap_hist", None) or {}
            for _rid3 in list(_v55_hist):
                if _rid3 not in _v55_live:
                    _v55_hist.pop(_rid3, None)
            for _rid3, _series in _v55_hist.items():
                if (len(_series) >= 3 and _series[-1] >= 8
                        and _series[-1] >= _series[-2] >= _series[-3]):
                    _why = (f"non-decreasing gap series "
                            f"{_series[-3:]} (no chunk progress across "
                            f"3+ discard steps; crash-2 precursor class)")
                    logger.error(
                        "llm-scaler v55 GAP-FENCE: %s — %s; worker "
                        "num_tokens diverged from scheduler accounting "
                        "(crash-4 zombie class); aborting before the "
                        "async machinery wedges", _rid3, _why)
                    raise RuntimeError(
                        f"llm-scaler v55 GAP-FENCE: request {_rid3}: "
                        f"{_why} — per-request KV/mamba state "
                        f"untrustworthy; fast-fail (crash-2 wedge "
                        f"prevention)")

        # Sync num_accepted_tokens from CPU (set by
"""


def patch_gmr(path: Path) -> None:
    src = _read(path)
    if MARKER in src:
        print(f"  {path.name}: already patched (v55 marker present), skip")
        return
    if "import time\n" not in src or "import os\n" not in src:
        raise SystemExit("ANCHOR-FAIL: time/os imports not both present")

    n = src.count(SYNC_COPY_OLD)
    if n != SYNC_COPY_COUNT:
        raise SystemExit(
            f"ANCHOR-FAIL async_copy sync: expected {SYNC_COPY_COUNT}, "
            f"found {n}")
    src = src.replace(SYNC_COPY_OLD, SYNC_COPY_NEW)

    src = _replace_once(src, SYNC_UPD_OLD, SYNC_UPD_NEW, "update_states sync")
    src = _replace_once(src, SYNC_PP_OLD, SYNC_PP_NEW, "deferred-pp sync")
    src = _replace_once(src, HELPER_ANCHOR, HELPER_CODE, "helper insert")
    src = _replace_once(src, HIST_INIT_OLD, HIST_INIT_NEW, "hist init")
    src = _replace_once(src, HIST_RECORD_OLD, HIST_RECORD_NEW, "hist record")
    src = _replace_once(src, FENCE_OLD, FENCE_NEW, "gap fence")

    _write(path, src)
    _compile_gate(path)
    print(f"  {path.name}: patched OK "
          f"(2x copy-sync + 2x counts-sync bounded, gap fence armed)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sp", default="/opt/venv/lib/python3.12/site-packages",
        help="site-packages root containing the vllm package")
    args = ap.parse_args()
    gmr = Path(args.sp) / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    if not gmr.is_file():
        raise SystemExit(f"target missing: {gmr}")
    print(f"v55 async fence: {gmr}")
    patch_gmr(gmr)
    print("V55_ASYNC_FENCE_OK")


if __name__ == "__main__":
    sys.exit(main())
