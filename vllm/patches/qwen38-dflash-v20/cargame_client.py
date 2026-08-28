#!/usr/bin/env python3
"""llm-scaler v20 canonical car-game benchmark client (stdlib only).

User-specified canonical test:
  prompt    "Write a html car game."
  sampling  temperature 0.3, top_k 20, top_p 0.95, min_p 0.0,
            presence_penalty 0.0, repetition_penalty 1.0
  max_tokens 4096, streaming

v20 fix (CRITICAL): the v19 client counted SSE delta EVENTS as tokens. With
spec decode the detokenizer flushes multiple accepted tokens in one delta
event, so every spec cell underreported by ~E[len] (measured ~1.9x on
dflash k4; nospec cells are 1 token/event and were correct). The engine-side
SpecDecoding windows proved it: emitted = mean_accept_length * drafted/k
matched the engine rolling gen throughput, not the v19 client numbers.
This client now requests stream_options.include_usage and reports TRUE
token rates:
  tokens_true          usage.completion_tokens (server-side exact count)
  overall_true         (tokens_true - 1) / gen_time
  steady_true          steady event rate * (tokens_true / events)
                       (tokens/event is stable in steady state; cross-check
                       against the serve-log SpecDecoding windows - the
                       authoritative steady source)
Legacy event-based metrics are still reported (suffix _events) for
continuity with the v19 result tables.

Measures TTFT, overall + steady (last 50%), per-30s buckets, min full
bucket. Prints a human summary and appends a JSON line to --json.
"""

import argparse
import json
import sys
import time
import http.client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--json", default=None, help="append result JSON here")
    ap.add_argument("--tag", default="cell")
    args = ap.parse_args()

    payload = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": "Write a html car game."}],
        "max_tokens": args.max_tokens,
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    conn = http.client.HTTPConnection(args.host, args.port, timeout=600)
    t_req = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"FATAL: HTTP {resp.status}: {resp.read()[:400]!r}")
        sys.exit(2)

    stamps = []  # perf_counter per received delta EVENT (may carry >1 token)
    usage = None
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        now = time.perf_counter()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            deltas = obj.get("choices") or []
            for ch in deltas:
                if ch.get("delta", {}).get("content"):
                    stamps.append(now)
    conn.close()

    if len(stamps) < 2:
        print(f"FATAL: only {len(stamps)} events received")
        sys.exit(2)

    ttft = stamps[0] - t_req
    gen_time = stamps[-1] - stamps[0]
    n_events = len(stamps)

    # TRUE token count from server usage (see module docstring).
    n_true = usage.get("completion_tokens") if usage else None
    if not n_true:
        # server did not honor include_usage: fall back to event count with a
        # loud warning (nospec == 1 token/event; spec UNDERCOUNTS).
        print("WARN: no usage.completion_tokens - reporting EVENT rates as tokens")
        n_true = n_events
    tok_per_event = n_true / n_events if n_events else 1.0

    overall_events = (n_events - 1) / gen_time if gen_time > 0 else float("nan")
    overall_true = (n_true - 1) / gen_time if gen_time > 0 else float("nan")

    # per-30s buckets from first token (event counts, scaled by tok/event)
    t0 = stamps[0]
    buckets = {}
    for t in stamps:
        buckets[int((t - t0) // 30)] = buckets.get(int((t - t0) // 30), 0) + 1
    bucket_rates = [
        (b, cnt, cnt / 30.0, cnt * tok_per_event / 30.0)
        for b, cnt in sorted(buckets.items())
    ]

    # steady = last 50% of the generation window
    half = t0 + gen_time / 2
    late = [t for t in stamps if t >= half]
    steady_events = (
        (len(late) - 1) / (late[-1] - late[0]) if len(late) > 2 else overall_events
    )
    steady_true = steady_events * tok_per_event
    # min FULL bucket rate (partial last bucket excluded)
    full_buckets = [r for b, c, r, rt in bucket_rates if (b + 1) * 30 <= gen_time + 1]
    min_bucket = min(full_buckets) if full_buckets else float("nan")

    print(f"TAG={args.tag} events={n_events} tokens_true={n_true} "
          f"tok/event={tok_per_event:.2f}")
    print(f"TTFT={ttft:.2f}s gen_time={gen_time:.1f}s "
          f"overall_true={overall_true:.2f} tok/s "
          f"steady_true(last50%)={steady_true:.2f} tok/s "
          f"[legacy events: overall={overall_events:.2f} steady={steady_events:.2f}] "
          f"min_full_30s_bucket_events={min_bucket:.2f} tok/s")
    for b, c, r, rt in bucket_rates:
        print(f"  bucket[{b*30:>4}-{(b+1)*30:>4}s] events={c:>4} "
              f"rate_events={r:.2f} rate_true={rt:.2f} tok/s")

    if args.json:
        rec = {
            "tag": args.tag, "events": n_events, "tokens_true": n_true,
            "tok_per_event": round(tok_per_event, 3),
            "ttft_s": round(ttft, 3), "gen_time_s": round(gen_time, 2),
            "overall_tok_s": round(overall_true, 2),
            "steady_tok_s": round(steady_true, 2),
            "overall_events_s": round(overall_events, 2),
            "steady_events_s": round(steady_events, 2),
            "min_bucket_tok_s": round(min_bucket, 2),
            "bucket_rates": [
                (b * 30, c, round(r, 2), round(rt, 2)) for b, c, r, rt in bucket_rates
            ],
            "usage_completion_tokens": usage.get("completion_tokens") if usage else None,
        }
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
