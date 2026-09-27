#!/usr/bin/env python3
"""bench_v89_contention.py — v89 T1 A/B contention benchmark (CC-shaped load).

N concurrent streams, each with a DISTINCT long prompt (distinct head per
tag/round/stream defeats the prefix cache, so every leg pays full prefill —
identical conditions for A/B). Raw /v1/completions (no chat template, no
thinking) so the scheduler knobs (v63 contended budget / v64 interleave) are
the only variables. Streaming; measures per-stream TTFT, decode tok/s
(post-first-token), and aggregate tok/s per round.

Usage: bench_v89_contention.py [n_streams=8] [gen_tokens=600] [prompt_words=6000] [rounds=2] [tag=leg]
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
FILLER = (
    "The scheduler admits chunked prefill work between decode batches, so every "
    "prefill token directly pauses generation for all running streams, and the "
    "aggregate token ceiling is shared across requests in proportion to batch. "
)


def build_prompt(tag: str, rnd: int, i: int, words: int) -> str:
    head = f"bench-v89 tag={tag} round={rnd} stream={i} seq={tag}{rnd:02d}{i:02d}: "
    body = FILLER * (words // len(FILLER.split()) + 2)
    body = " ".join(body.split()[:words])
    return (
        head + body
        + "\n\nBased strictly on the text above, write a numbered list of ten distinct observations. Answer directly."
    )


def one(i: int, prompt: str, gen_tokens: int, out: list) -> None:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    t_first = None
    ct = -1
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:
                    continue
                if t_first is None:
                    ch = (j.get("choices") or [{}])[0]
                    if (ch.get("text") or "").strip() or ch.get("finish_reason"):
                        t_first = time.time()
                u = j.get("usage")
                if u and u.get("completion_tokens") is not None:
                    ct = u["completion_tokens"]
    except Exception as e:  # noqa: BLE001
        out[i] = {"err": f"{type(e).__name__}: {e}", "ttft": None, "wall": time.time() - t0, "ct": ct, "tps": None}
        return
    t_end = time.time()
    dec = (t_end - t_first) if t_first else None
    out[i] = {
        "ttft": (t_first - t0) if t_first else None,
        "wall": t_end - t0,
        "ct": ct,
        "tps": (ct / dec) if (dec and dec > 0 and ct and ct > 0) else None,
    }


def main() -> int:
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    pw = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    tag = sys.argv[5] if len(sys.argv) > 5 else "leg"

    print(f"v89-contention n={ns} gen={mt} prompt_words={pw} (~{int(pw*4//3)} tok) rounds={rounds} tag={tag}")
    agg_by_round = []
    total_errs = 0
    for rnd in range(1, rounds + 1):
        out = [None] * ns
        ts = [
            threading.Thread(target=one, args=(i, build_prompt(tag, rnd, i, pw), mt, out))
            for i in range(ns)
        ]
        t_start = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t_start
        errs = [r for r in out if r and r.get("err")]
        total_errs += len(errs)
        cts = [r["ct"] for r in out if r and r["ct"] and r["ct"] > 0]
        ttfts = sorted(r["ttft"] for r in out if r and r["ttft"])
        tps = [r["tps"] for r in out if r and r["tps"]]
        agg = sum(cts) / wall if cts and wall else 0.0
        agg_by_round.append(agg)
        print(
            f"round {rnd}: wall={wall:.1f}s ok={len(cts)}/{ns} err={len(errs)} "
            f"agg={agg:.1f} tok/s | ttft p50={ttfts[len(ttfts)//2]:.1f}s max={ttfts[-1] if ttfts else -1:.1f}s "
            f"| per-stream tps p50={statistics.median(tps):.1f} min={min(tps):.1f}"
            if tps and ttfts
            else f"round {rnd}: wall={wall:.1f}s ok={len(cts)}/{ns} err={len(errs)} agg={agg:.1f} (incomplete)"
        )
        for e in errs[:3]:
            print(f"    ERR {e['err']}")
    if agg_by_round:
        print(f"SUMMARY {tag}: agg_median={statistics.median(agg_by_round):.1f} tok/s rounds={agg_by_round}")
    return 1 if total_errs else 0


if __name__ == "__main__":
    sys.exit(main())
