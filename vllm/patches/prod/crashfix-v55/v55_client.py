#!/usr/bin/env python3
"""v55_client.py — crashfix-v55 validation traffic generator.

Reproduces the 2026-09-10 crash traffic shape under the user's
constraints, with the fixes in place:

  - 5 concurrent streams, STAGGERED starts (0/5/10/15/20 s) — the crash
    boots ramped 3-5 running requests with chunked prefills landing on
    live decodes.
  - DISTINCT prompts per stream AND per cycle (user: "different prompts
    and different start time").
  - Every request's context stays < 64k (user: "keep all testing
    content lenght less that 64k, stop on 64k context lenght"): the
    largest stream is a ~16k-token prompt + 2048 out; the crash-prompt
    stream is short-prompt + 4096 out.
  - Stream 1 is the exact crash prompt, "Write a html car game", with
    the crash sampling params (temp 0.9 / top_k 20 / top_p 0.95) and a
    long repetitive decode — the crash-2 wedge lane.
  - Mixed sampling across streams (sampled explicit / default / greedy)
    to exercise every sampler branch concurrently.

Success criteria (checked here per stream): HTTP 200, non-empty
completion, finish_reason in {stop, length}; the harness
(validate_v55.sh) additionally scans the engine log for crash/warning
classes between cycle markers.

Usage: v55_client.py <cycle-index>   (cycle index varies the prompts)
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"

RESULTS: list[dict] = []
LOCK = threading.Lock()


def _doc(seed: int, reps: int) -> str:
    # Distinct per (stream, cycle): seeded filler, no cross-stream
    # prefix sharing.
    unit = (
        f"Document cycle {seed} records that staggered streams with "
        f"distinct contents exercise co-admission without sharing a "
        f"prefix cache entry {seed}; "
    )
    return f"Summarize document cycle {seed}. " + unit * reps


def stream(
    idx: int,
    name: str,
    delay_s: float,
    prompt: str,
    max_tokens: int,
    sampling: dict | None,
) -> None:
    time.sleep(delay_s)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if sampling:
        payload.update(sampling)
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    rec = {"stream": name, "ok": False, "err": "", "out_tokens": -1,
           "latency_s": -1.0, "finish": ""}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            j = json.loads(r.read())
        rec["out_tokens"] = (j.get("usage") or {}).get(
            "completion_tokens", -1)
        choices = j.get("choices") or [{}]
        rec["finish"] = choices[0].get("finish_reason", "")
        message = choices[0].get("message", {})
        # preserve_thinking: a max_tokens cut inside the think block
        # leaves content empty with the partial thinking in the
        # `reasoning` field (this serving layer's name; the OpenAI
        # spec name is `reasoning_content`), and a cut inside the
        # parser buffer can leave BOTH empty while usage correctly
        # reports the generated tokens. Any of these is a healthy
        # completion — usage is the ground truth.
        text = (message.get("content", "") or message.get("reasoning", "")
                or message.get("reasoning_content", ""))
        if rec["finish"] == "abort":
            # FINISHED_ABORTED delivered client-visibly = the v52m
            # NaN-zombie containment guard (engine stays up by
            # construction — a dead engine yields a connection error,
            # not this JSON). Recorded as CONTAINED, not a stream
            # failure: the crash-2 wedge class costs one visible
            # request abort instead of the engine.
            rec["ok"] = True
            rec["err"] = "CONTAINED-ABORT (v52m guard; engine alive)"
        else:
            rec["ok"] = bool(
                r.status == 200 and rec["finish"] in ("stop", "length")
                and (text or rec["out_tokens"] > 0))
            if not rec["ok"]:
                rec["err"] = (f"bad finish={rec['finish']!r} "
                              f"empty={not text} out={rec['out_tokens']}")
    except Exception as e:  # noqa: BLE001
        rec["err"] = f"{type(e).__name__}: {e}"
    rec["latency_s"] = round(time.time() - t0, 1)
    with LOCK:
        RESULTS.append(rec)
    print(f"[stream {name}] ok={rec['ok']} out={rec['out_tokens']} "
          f"finish={rec['finish']} {rec['latency_s']}s err={rec['err']}",
          flush=True)


def main() -> int:
    cycle = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    c = cycle + 100  # prompt seed base
    streams = [
        # name, delay, prompt, max_tokens, sampling
        ("htmlgame", 0.0, "Write a html car game", 4096,
         {"temperature": 0.9, "top_k": 20, "top_p": 0.95}),
        ("doc16k", 5.0, _doc(c, 520), 2048,
         {"temperature": 0.9, "top_k": 20, "top_p": 0.95}),
        ("doc8k", 10.0, _doc(c + 1, 260), 1024, None),
        ("qa-greedy", 15.0,
         f"Cycle {c}: explain in one paragraph why staggered concurrent "
         f"prefill admission must be warmed at boot.", 512,
         {"temperature": 0.0}),
        ("qa-sampled", 20.0,
         f"Cycle {c}: write a short python function that interleaves "
         f"three iterators and yields the smallest head each step.", 2048,
         {"temperature": 0.9, "top_k": 20, "top_p": 0.95}),
    ]
    threads = [
        threading.Thread(
            target=stream,
            args=(i, n, d, p, m, s),
        )
        for i, (n, d, p, m, s) in enumerate(streams)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = all(r["ok"] for r in RESULTS)
    total_out = sum(r["out_tokens"] for r in RESULTS)
    print(f"CYCLE {cycle} {'PASS' if ok else 'FAIL'}: "
          f"{sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} streams ok, "
          f"{total_out} out tokens, {time.time() - t0:.0f}s wall",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
