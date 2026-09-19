#!/usr/bin/env python3
"""dt_halfprobe — §24 R half-hit probe.

65k-class prompt sharing ONLY its first ~32k tokens with the cached ctx65k
FILLER prefix, then diverging filler. Run AFTER the bench3x screen (the
ctx65k prefix must already be cached). Two submissions (half, half2).
Discriminates: dip follows hit-presence/size vs needs a near-full-length
hit. The [v24r] HIT/FREE lines in serve_full.log give the tables.
"""
import json
import time
import urllib.request

M = "http://localhost:8000"
MODEL = "qwen3.8-27b-fp8"
FILLER = ("The river carried silt and small branches past the mooring posts, and the "
          "water level changed with every lock opening upstream. ")
DIVERGE = ("Storm winds battered the breakwater and the harbor lights swung in the "
           "gusts while fishing crews secured their boats before the tide turned. ")


def prompt_half(approx_tokens, share_tokens):
    share_reps = share_tokens // 22
    rest_reps = max(0, (approx_tokens - share_tokens)) // 24
    return (FILLER * share_reps + DIVERGE * rest_reps +
            "\n\nSummarize the passage above in exactly two sentences.")


def stream_once(tag, prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1234,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        M + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    compl = 0
    ptok = -1
    resp = urllib.request.urlopen(req, timeout=900)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            u = d["usage"]
            ptok = u.get("prompt_tokens", -1)
            compl = u.get("completion_tokens", -1)
        if ttft is None and d.get("choices"):
            delta = d["choices"][0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                ttft = time.time() - t0
    dt = time.time() - t0
    dec = (compl - 1) / (dt - ttft) if ttft and compl and compl > 1 else 0.0
    print(f"BENCH {tag}: prompt_tok={ptok} compl_tok={compl} "
          f"ttft={ttft if ttft else -1:.2f}s decode={dec:.1f} tok/s "
          f"total={dt:.1f}s", flush=True)


if __name__ == "__main__":
    for tag in ("half65k", "half65kb"):
        stream_once(tag, prompt_half(65000, 32000), 256)
