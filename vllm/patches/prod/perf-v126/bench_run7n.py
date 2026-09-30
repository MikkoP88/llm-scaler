#!/usr/bin/env python3
"""bench_run7n.py — v126 ship-posture speed matrix, EXACT Run-7n
(p22c5_agg2.sh) methodology for baseline comparability.

Run-7n legs measured: agg 4x256 x2 (aggregate = toks/wall across the
4-thread join) + solo x2 at max_tokens=200 (solo tps = n/dt including
TTFT), prompt "Count from 1 to 1000.", temperature 0,
chat_template_kwargs enable_thinking=False, non-streaming.
cert16s4 baselines: agg 181.81 / 244.88, solo 75.06 / 74.89.

Usage: bench_run7n.py            (agg + solo, Run-7n exact)
       bench_run7n.py --extra8   (additionally agg 8x256 x2 for the
                                  8-stream axis seen in today's matrix)
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"


def call(mt: int, out: list) -> None:
    body = {"model": MODEL, "max_tokens": mt, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user",
                          "content": "Count from 1 to 1000."}]}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        n = r.get("usage", {}).get("completion_tokens", mt)
        out.append((n, time.perf_counter() - t0))
    except Exception as e:  # noqa: BLE001
        print(f"call_FAIL {type(e).__name__}: {e}", flush=True)
        out.append((0, -1.0))


def agg(nstreams: int, mt: int, run: int) -> None:
    outs = [[] for _ in range(nstreams)]
    ths = [threading.Thread(target=call, args=(mt, o)) for o in outs]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    recs = [o[0] for o in outs if o]
    ok = [n for n, dt in recs if dt > 0]
    toks = sum(ok)
    per = " ".join(f"{n}tok/{dt:.1f}s" for n, dt in recs)
    rate = toks / wall if wall > 0 else 0.0
    print(f"agg{nstreams}x{mt}#{run} wall={wall:.2f}s "
          f"aggregate={rate:.2f} tok/s ok_streams={len(ok)}/{nstreams} "
          f"[{per}]", flush=True)


def solo() -> None:
    o: list = []
    t0 = time.perf_counter()
    call(200, o)
    dt = time.perf_counter() - t0
    n = o[0][0] if o else 0
    print(f"solo200 n={n} dt={dt:.2f}s tps={n / dt if dt > 0 else 0:.2f}",
          flush=True)


def main() -> int:
    print("=== bench_run7n (cert16s4 methodology replica) ===", flush=True)
    for run in (1, 2):
        agg(4, 256, run)
    solo()
    solo()
    if "--extra8" in sys.argv:
        for run in (1, 2):
            agg(8, 256, run)
    print("BENCH_RUN7N_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
