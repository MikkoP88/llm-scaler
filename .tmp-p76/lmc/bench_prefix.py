#!/usr/bin/env python3
"""bench_prefix.py — long-prefix reuse A/B for LMCache evaluation.

Phases (GPU attn pool on this lane ~506k tokens; docs are ~5k tokens each —
the 1.3 chars/token generator estimate is ~4x off for word-soup content):
  cold   : 16 distinct docs, sequential -> populate caches
  warm   : same 16 docs again -> engine-internal prefix cache hits
  churn  : 112 NEW distinct docs -> ~573k tokens + orig > pool, evicts
           the originals from the GPU pool (native reuse impossible)
  rehit  : the original 16 docs again -> baseline=full recompute,
           LMCache=CPU-pool retrieve (the value axis)
Reports per-phase TTFT median/mean/min/max, wall, and (with
stream_options.include_usage) real prompt/cached token counts.
Deterministic docs (seeded), so both engine legs see byte-identical prompts.
Usage: bench_prefix.py            (all phases)
       bench_prefix.py rehit      (just the re-hit phase, post external churn)
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"

N_ORIG = 16
N_CHURN = 112
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega river "
         "mountain forest glacier desert canyon meadow ocean island valley "
         "signal pattern vector matrix tensor kernel schedule priority "
         "resource cluster network storage compute memory bandwidth latency "
         "harbor lantern compass anchor vessel cargo rudder sail mast").split()


def make_doc(doc_id: int, approx_tokens: int = 16384) -> str:
    rng = random.Random(10007 * doc_id + 13)
    # ~1.3 chars/token for this vocab; build until char target
    target_chars = int(approx_tokens * 1.3)
    parts = [f"Document {doc_id} field notes follow."]
    n = len(parts[0])
    i = 0
    while n < target_chars:
        i += 1
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16)))
        parts.append(f"[{doc_id}.{i}] {sent}.")
        n += len(parts[-1]) + 1
    return " ".join(parts)


DOCS_ORIG = [make_doc(i) for i in range(N_ORIG)]
DOCS_CHURN = [make_doc(1000 + i) for i in range(N_CHURN)]


def call(doc: str) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": doc + "\nReply with DONE."}],
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if ttft is None:
                    c = (chunk.get("choices") or [{}])[0].get("delta", {})
                    if c.get("content"):
                        ttft = time.perf_counter() - t0
                u = chunk.get("usage")
                if u:
                    usage = u
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}
    total = time.perf_counter() - t0
    return {
        "ok": True,
        "ttft": ttft if ttft is not None else total,
        "total": total,
        "prompt_tokens": usage.get("prompt_tokens", -1),
        "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", -1),
    }


def phase(name: str, docs: list[str]) -> None:
    t0 = time.perf_counter()
    recs = [call(d) for d in docs]
    wall = time.perf_counter() - t0
    ok = [r for r in recs if r["ok"]]
    if len(ok) != len(recs):
        print(f"{name}: FAIL {len(recs)-len(ok)}/{len(recs)} "
              f"{[r.get('err') for r in recs if not r['ok']][:2]}")
    tt = sorted(r["ttft"] for r in ok)
    pt = sorted(r["prompt_tokens"] for r in ok)
    cvals = [r["cached"] for r in ok if r["cached"] >= 0]
    med = tt[len(tt)//2]
    mean = sum(tt) / len(tt)
    cn = sum(1 for c in cvals if c > 0)
    print(
        f"{name}: n={len(ok)} wall={wall:.1f}s ttft_med={med:.2f}s "
        f"ttft_mean={mean:.2f}s "
        f"ttft_min={tt[0]:.2f}s ttft_max={tt[-1]:.2f}s "
        f"ptok_med={pt[len(pt)//2]} "
        f"cached_med={sorted(cvals)[len(cvals)//2] if cvals else -1} "
        f"cached_n={cn}/{len(ok)} "
        f"ttft_sum={sum(tt):.1f}s",
        flush=True)


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"=== bench_prefix (docs ~5k tok, orig={N_ORIG} churn={N_CHURN}) ===",
          flush=True)
    if only in (None, "cold"):
        phase("cold", DOCS_ORIG)
    if only in (None, "warm"):
        phase("warm", DOCS_ORIG)
    if only in (None, "churn"):
        phase("churn", DOCS_CHURN)
    if only in (None, "rehit"):
        phase("rehit", DOCS_ORIG)
    print("BENCH_PREFIX_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
