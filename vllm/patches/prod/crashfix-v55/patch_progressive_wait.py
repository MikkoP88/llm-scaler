#!/usr/bin/env python3
"""patch_progressive_wait.py — v56 micro-patcher (crashfix-v55 follow-up).

2026-09-11 regression fix: v1.2.8's _v55_wait_event used a flat
"time.sleep(0.05)" poll loop. The pre-v55 code blocked in
".synchronize()" (efficient driver wait); the flat 50 ms sleep quantized
EVERY not-yet-signaled hot-path event wait to 50 ms granularity. Three of
the four fence sites run per engine step (WorkerAsyncOutputCopy x2,
deferred postprocess num_accepted_tokens_event) — measured effect ~+40 ms
per decode step: warm A/B on a healthy node gave v1.2.8 genspeed -37%,
ctxscan -14..-44% vs v1.2.7 (reg_ab harness, 2026-09-11 05:16-05:43).

Fix: progressive cadence with identical semantics and identical 120 s
bound (VLLM_V55_EVENT_TIMEOUT_S). 0.5 ms polls while the event is
expected to land (< 20 ms — the normal async-completion window), 5 ms to
200 ms, 50 ms to 2 s, 250 ms beyond (a > 2 s wait is already patho-
logical; only the deadline matters). Normal-path inflation <= 0.5 ms.

Idempotent: skips if the progressive cadence is already present.
"""

import argparse
import hashlib
import sys
from pathlib import Path

NEW = '''def _v55_wait_event(evt, what: str) -> None:
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
    evt.synchronize()'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", required=True,
                    help="site-packages root containing vllm/")
    args = ap.parse_args()
    gmr = Path(args.sp) / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    if not gmr.exists():
        print(f"[v56] FATAL: {gmr} not found")
        return 1
    src = gmr.read_text(encoding="utf-8")
    if "_el < 0.02" in src:
        print(f"[v56] already progressive — skip ({gmr})")
        return 0
    a = src.find("def _v55_wait_event(evt, what: str) -> None:")
    if a < 0:
        print(f"[v56] FATAL: _v55_wait_event not found in {gmr}")
        return 1
    b = src.find("    evt.synchronize()", a)
    if b < 0:
        print("[v56] FATAL: end anchor not found")
        return 1
    b += len("    evt.synchronize()")
    src2 = src[:a] + NEW + src[b:]
    gmr.write_text(src2, encoding="utf-8")
    md5 = hashlib.md5(gmr.read_bytes()).hexdigest()[:8]
    print(f"[v56] PATCHED {gmr}")
    print(f"[v56] old fn ~{b - a} chars -> new {len(NEW)} chars; gmr md5 {md5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
