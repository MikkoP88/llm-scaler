#!/usr/bin/env python3
"""bench_longctx.py — same-session reprompt probe (the live-traffic symptom,
measured directly). For each size axis (small ~600 tok, long ~32k tok):
  t1  : fresh doc + "Reply with the single word OK."   (cold turn)
  t2  : SAME prompt + tiny delta suffix                 (reprompt turn)
Reports TTFT (stream first-token), prompt_tokens, cached_tokens (usage),
and engine prompt_tokens_total deltas (approx respin census; live traffic
adds noise). The small axis exposes LMCACHE_MIN_RETRIEVE_TOKENS gating
(<1024 retrievable = lookup skipped); the long axis exposes chunk-boundary
rehit depth (LMCACHE_CHUNK_SIZE) at real session scale.
Usage: bench_longctx.py        (both axes)
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-fp8"
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega river "
         "mountain forest glacier desert canyon meadow ocean island valley "
         "signal pattern vector matrix tensor kernel schedule priority "
         "resource cluster network storage compute memory bandwidth latency "
         "harbor lantern compass anchor vessel cargo rudder sail mast").split()


def make_doc(doc_id: int, approx_tokens: int) -> str:
    rng = random.Random(1009 * doc_id + 7)
    target_chars = int(approx_tokens * 1.3)
    out = []
    n = 0
    while n < target_chars:
        w = WORDS[rng.randrange(len(WORDS))]
        out.append(w)
        n += len(w) + 1
    return " ".join(out)


def prompt_tokens_total() -> float:
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prompt_tokens_total{"):
                return float(line.rsplit(" ", 1)[1])
    return -1.0


def turn(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 48,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if ttft is None and (delta.get("content") or ch.get("text")):
                ttft = time.time() - t0
    return {"ttft": ttft, "wall": time.time() - t0, "usage": usage}


def axis(name: str, doc_id: int, approx: int) -> None:
    doc = make_doc(doc_id, approx)
    base = doc + "\n\nReply with the single word OK."
    t2p = base + "\nAlso reply with the single word YES."
    m0 = prompt_tokens_total()
    r1 = turn(base)
    m1 = prompt_tokens_total()
    r2 = turn(t2p)
    m2 = prompt_tokens_total()
    pt = (r1["usage"] or {}).get("prompt_tokens", -1)
    pt2 = (r2["usage"] or {}).get("prompt_tokens", -1)
    ct = ((r2["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens", -1)
    ct1 = ((r1["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens", -1)
    print(f"[{name}] t1 ttft={r1['ttft']:.3f}s wall={r1['wall']:.2f}s "
          f"prompt={pt} cached={ct1}")
    print(f"[{name}] t2 ttft={r2['ttft']:.3f}s wall={r2['wall']:.2f}s "
          f"prompt={pt2} cached={ct} (rehit depth {ct}/{pt2})")
    print(f"[{name}] engine prompt_tokens_total delta: t1={m1-m0:.0f} "
          f"(excess {m1-m0-pt:.0f})  t2={m2-m1:.0f} (excess {m2-m1-pt2:.0f})")
    sys.stdout.flush()


if __name__ == "__main__":
    print(f"=== BENCH_LONGCTX start {time.strftime('%H:%M:%S')}")
    # approx param is ~4.5x off for word-soup vocab (measured 2026-10-10:
    # 600->153 actual, 32000->7049 actual) -> scale targets accordingly.
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    AXES = [
        ("small", "small~600tok", 9001, 2700),
        ("mid", "mid~8ktok", 9003, 36000),
        ("mid", "mid~16ktok", 9004, 73000),
        ("long", "long~32ktok", 9002, 145000),
    ]
    for key, name, did, approx in AXES:
        if which in ("all", key):
            axis(name, did, approx)
    print(f"=== BENCH_LONGCTX done {time.strftime('%H:%M:%S')}")
