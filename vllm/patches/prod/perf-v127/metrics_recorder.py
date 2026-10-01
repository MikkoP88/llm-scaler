#!/usr/bin/env python3
"""metrics_recorder.py — v127 WS-B: continuous lane telemetry recorder.

STANDING TELEMETRY DIRECTIVE (user, 2026-09-30): every phase of the v127
program must be measurable from CAPTURES, not forensics. This daemon is the
always-on layer: it scrapes the engine's /metrics every --interval seconds
and appends one JSON line per sample to a daily JSONL file. The 2026-09-30
stall had to be reconstructed from cumulative counters hours after the
fact (STALL_ROOT_CAUSE.md); with this recorder the eviction/re-prefill
storm class leaves a per-10-second fingerprint:

  kv_cache_usage_perc      -> eviction pressure cycles (sawtooth = storms)
  num_requests_waiting     -> admission backlog onset/duration
  waiting_by_reason        -> capacity vs preempted-vs-... queueing cause
  prefix hits/queries      -> cache-hit collapse when a session is evicted
  prompt_tokens_by_source  -> local_compute deltas = re-prefill volume
  TTFT buckets (vector)    -> tail growth in real time (the "hard stall")
  num_preemptions          -> must stay 0 (any nonzero = new defect class)

Host-side by design (python3 stdlib only): survives container
recreates/windows; dies with the host (re-arm after reboots — the boot
script re-arms it). Read-only against the lane: GET /metrics only.

Usage:
  nohup python3 metrics_recorder.py >> /root/build/telemetry/recorder.log 2>&1 &
  python3 metrics_recorder.py --once          # single sample to stdout
Output: /root/build/telemetry/metrics_YYYYMMDD.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
import urllib.request

# scalar metrics: name -> type
SCALARS = {
    "num_requests_running": float,
    "num_requests_waiting": float,
    "kv_cache_usage_perc": float,
    "num_preemptions_total": float,
    "prefix_cache_hits_total": float,
    "prefix_cache_queries_total": float,
    "prompt_tokens_total": float,
    "prompt_tokens_cached_total": float,
    "generation_tokens_total": float,
    "e2e_request_latency_seconds_count": float,
    "time_to_first_token_seconds_count": float,
    "time_to_first_token_seconds_sum": float,
    "inter_token_latency_seconds_count": float,
    "inter_token_latency_seconds_sum": float,
    "engine_sleep_state": float,
}
# label-scraped cumulative counters (each sample keeps the full map)
BY_SOURCE = "prompt_tokens_by_source_total"
BY_REASON = "num_requests_waiting_by_reason"
# histogram vectors worth keeping whole (tail forensics without the lane)
TTFT_BUCKETS = "time_to_first_token_seconds_bucket"
TPOT_BUCKETS = "inter_token_latency_seconds_bucket"


def parse_line(line: str, out: dict) -> None:
    """Parse one prometheus exposition line into the sample dict."""
    if not line.startswith("vllm:"):
        return
    body = line[len("vllm:"):]
    # split name{labels} value
    if "{" in body:
        name, rest = body.split("{", 1)
        labels_s, val_s = rest.split("}", 1)
    else:
        name, val_s = body.split(" ", 1) if " " in body else (body, "nan")
        labels_s = ""
    name = name.strip()
    try:
        val = float(val_s.strip().split()[0])
    except (ValueError, IndexError):
        return
    if name in SCALARS:
        out[name] = val
    elif name == BY_SOURCE and labels_s:
        src = labels_s.split('source="')[1].split('"')[0]
        out[BY_SOURCE][src] = val
    elif name == BY_REASON and labels_s and 'reason="' in labels_s:
        rsn = labels_s.split('reason="')[1].split('"')[0]
        out[BY_REASON][rsn] = val
    elif name == TTFT_BUCKETS and labels_s:
        le = labels_s.split('le="')[1].split('"')[0]
        out["ttft_bucket"][le] = val
    elif name == TPOT_BUCKETS and labels_s:
        le = labels_s.split('le="')[1].split('"')[0]
        out["tpot_bucket"][le] = val


def sample(base: str) -> dict:
    out = {BY_SOURCE: {}, BY_REASON: {}, "ttft_bucket": {}, "tpot_bucket": {}}
    with urllib.request.urlopen(base + "/metrics", timeout=8) as r:
        for raw in r.read().decode("utf-8", "replace").splitlines():
            parse_line(raw.strip(), out)
    out["ts"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--outdir", default="/root/build/telemetry")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[recorder] start host={args.host} interval={args.interval}s out={args.outdir}", flush=True)
    errs = 0
    while True:
        try:
            s = sample(args.host)
            errs = 0
            day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
            path = os.path.join(args.outdir, f"metrics_{day}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
            if args.once:
                print(json.dumps(s, indent=1))
                return 0
        except Exception as e:  # noqa: BLE001 — record and keep the loop alive
            errs += 1
            if errs <= 3 or errs % 30 == 0:
                print(f"[recorder] scrape error #{errs}: {e!r}", flush=True)
            if args.once:
                return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
