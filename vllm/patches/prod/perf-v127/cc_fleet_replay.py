#!/usr/bin/env python3
"""cc_fleet_replay.py — v127 CC-fleet long-context replay gate.

Reproduces the multi-agent "hard stalling" regime against a LIVE lane:
N staggered streams, each an agent-style conversation whose context starts
large and grows every turn. Measures per-request TTFT / TPOT / e2e, fleet
aggregate throughput, and counts re-prefill volume from the engine's
prompt_tokens_by_source counters.

Stdlib only (no requests dependency). Direct :8000 endpoint — no litellm,
no Bearer needed. Non-destructive: read-only load, lane stays up.

Usage (inside the host or any box that can reach :8000):
  python3 cc_fleet_replay.py --tag restore_leg
  python3 cc_fleet_replay.py --streams 8 --base-ctx 50000 --turns 4 --tag T-A

--model auto (default) discovers the served model id from /v1/models, so the
gate follows whatever posture the lane runs (swift posture serves under
--served-model-name qwen3.8-27b-fp8 for litellm compatibility).

PASS bar (FIX_AND_TEST_PLAN Layer 2): TTFT p99 < 20 s, no request > 45 s,
re-prefill volume ≈ first-turn totals only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request

METRIC_RE = re.compile(
    r'^vllm:(prompt_tokens_by_source_total)\{[^}]*source="([a-z_]+)"[^}]*\}\s+([0-9.e+-]+)$'
)

WORDS = (
    "kernel bridge scheduler decode prefill context token stream batch "
    "memory pool state scale format compute engine queue latency window "
    "graph capture spec draft accept cache prefix block shard tensor "
).split()


def build_context(n_tokens: int, rng: random.Random) -> str:
    """Deterministic-ish token soup; ~1.3 words ≈ 1 token for this tokenizer."""
    n_words = int(n_tokens * 1.3)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def discover_model(base: str) -> str:
    """First served model id from /v1/models — replay targets the live lane."""
    with urllib.request.urlopen(base + "/v1/models", timeout=8) as r:
        data = json.loads(r.read().decode())
    ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
    if not ids:
        raise SystemExit("no models listed at /v1/models — pass --model explicitly")
    return ids[0]


def get_metrics(base: str) -> dict:
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=8) as r:
        for raw in r.read().decode().splitlines():
            m = METRIC_RE.match(raw.strip())
            if m:
                out[m.group(2)] = float(m.group(3))
    return out


def one_request(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Streaming completion; returns ttfb, per-token deltas, e2e."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttfb = None
    deltas = []
    last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("choices") and obj["choices"][0].get("text"):
                now = time.monotonic()
                if ttfb is None:
                    ttfb = now - t0
                elif last is not None:
                    deltas.append(now - last)
                last = now
    e2e = time.monotonic() - t0
    return {
        "ttft": ttfb,
        "n_tokens": len(deltas) + (1 if ttfb is not None else 0),
        "tpot_avg": (sum(deltas) / len(deltas)) if deltas else None,
        "tpot_max": max(deltas) if deltas else None,
        "e2e": e2e,
    }


def pct(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--base-ctx", type=int, default=20000)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--grow", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--stagger", type=float, default=8.0)
    ap.add_argument("--tag", default="leg")
    ap.add_argument("--out", default="/root/build/v127_stage")
    args = ap.parse_args()

    if args.model == "auto":
        args.model = discover_model(args.host)
        print(f"[replay] target model: {args.model}")

    m0 = get_metrics(args.host)
    results = []
    lock = threading.Lock()
    errs = []

    def stream(idx: int) -> None:
        rng = random.Random(1000 + idx)
        ctx = build_context(args.base_ctx + idx * 500, rng)
        for turn in range(args.turns):
            ctx += " " + build_context(args.grow, rng)
            try:
                r = one_request(args.host, args.model, ctx, args.max_tokens)
            except Exception as e:  # noqa: BLE001 — record, keep firing
                with lock:
                    errs.append(f"stream{idx} turn{turn}: {e!r}")
                # v127 fix: a single timed-out turn (600 s urlopen ceiling under
                # live load) used to ABORT the whole stream — losing every
                # remaining turn's data. Record and continue instead.
                time.sleep(2.0)
                continue
            r.update(stream=idx, turn=turn, ctx_tokens=int(ctx.count(" ") / 1.3))
            with lock:
                results.append(r)
            # v127 fix: per-request progress line (flush) so a killed run still
            # shows where it was — the W1 baseline died inside `timeout 1500`
            # with a completely empty log because nothing printed until the end.
            print(
                f"s{idx} t{turn} ttft={(r['ttft'] or 0):.1f}s "
                f"e2e={r['e2e']:.1f}s tok={r['n_tokens']}",
                flush=True,
            )
            time.sleep(random.uniform(0.5, 2.0))

    threads = [
        threading.Thread(target=stream, args=(i,), daemon=True)
        for i in range(args.streams)
    ]
    t_start = time.monotonic()
    for i, th in enumerate(threads):
        th.start()
        time.sleep(args.stagger / max(1, args.streams))
    for th in threads:
        th.join()
    runtime = time.monotonic() - t_start

    m1 = get_metrics(args.host)
    computed = m1.get("local_compute", 0.0) - m0.get("local_compute", 0.0)
    cached = m1.get("local_cache_hit", 0.0) - m0.get("local_cache_hit", 0.0)

    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    tpots = sorted(r["tpot_avg"] for r in results if r["tpot_avg"] is not None)
    tpot_max = max((r["tpot_max"] or 0) for r in results)
    gen_total = sum(r["n_tokens"] for r in results)
    report = {
        "tag": args.tag,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "streams": args.streams,
        "turns": args.turns,
        "requests": len(results),
        "errors": errs,
        "runtime_s": round(runtime, 1),
        "ttft_p50_s": pct(ttfts, 0.50),
        "ttft_p90_s": pct(ttfts, 0.90),
        "ttft_p99_s": pct(ttfts, 0.99),
        "ttft_max_s": max(ttfts) if ttfts else None,
        "tpot_avg_ms": (pct(tpots, 0.50) or 0) * 1000,
        "tpot_max_ms": tpot_max * 1000,
        "agg_gen_tok_s": round(gen_total / runtime, 2),
        "prefill_computed_tokens": int(computed),
        "prefill_cached_tokens": int(cached),
        "cache_hit_pct": round(100 * cached / max(1.0, computed + cached), 2),
        "per_request": results,
    }

    import os

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"cc_replay_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"CC_REPLAY tag={args.tag} reqs={len(results)} errs={len(errs)}")
    print(
        f"TTFT p50/p90/p99/max s: {report['ttft_p50_s']:.2f} / "
        f"{report['ttft_p90_s']:.2f} / {report['ttft_p99_s']:.2f} / "
        f"{report['ttft_max_s']:.2f}"
    )
    print(
        f"TPOT p50 {report['tpot_avg_ms']:.1f} ms, worst-gap "
        f"{report['tpot_max_ms']:.0f} ms; agg {report['agg_gen_tok_s']} tok/s"
    )
    print(
        f"prefill: computed={int(computed)} cached={int(cached)} "
        f"({report['cache_hit_pct']}% hit)"
    )
    verdict = (
        "CC_REPLAY_PASS"
        if not errs
        and report["ttft_p99_s"] is not None
        and report["ttft_p99_s"] < 20
        and report["ttft_max_s"] < 45
        else "CC_REPLAY_FAIL"
    )
    print(f"report: {path}")
    print(verdict)
    return 0 if verdict == "CC_REPLAY_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
