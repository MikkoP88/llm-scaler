#!/usr/bin/env python3
"""dt_bench3 — dflash2 vs MTP lane benchmark (identical run on both lanes).

Per lane: streaming TTFT + decode tok/s at 2k/16k/65k contexts, 8-way
concurrent short-request aggregate throughput, and spec acceptance delta
from /metrics over the whole bench.
"""
import json
import re
import threading
import time
import urllib.request

M = "http://localhost:8000"
MODEL = "qwen3.8-27b-fp8"
FILLER = ("The river carried silt and small branches past the mooring posts, and the "
          "water level changed with every lock opening upstream. ")


def long_prompt(approx_tokens):
    reps = max(1, approx_tokens // 22)
    return FILLER * reps + "\n\nSummarize the passage above in exactly two sentences."


SHORT = ("2+2=4, 4+4=8, 8+8=16. Continue this doubling pattern for exactly "
         "twenty more steps, one per line.")


def metrics():
    txt = urllib.request.urlopen(M + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^vllm:([a-z_]+)_total\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if m and any(m.group(1).endswith(k) for k in (
                "num_accepted_tokens", "num_draft_tokens",
                "num_emitted_tokens")):
            out[m.group(1)] = float(m.group(2))
    return out


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


def conc(n, tag, max_tokens=256):
    out = [None] * n

    def worker(i):
        body = {"model": MODEL,
                "messages": [{"role": "user", "content": SHORT}],
                "max_tokens": max_tokens, "temperature": 0, "seed": 1234}
        req = urllib.request.Request(
            M + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=900))
        out[i] = (r.get("usage", {}).get("completion_tokens", 0),
                  time.time() - t)

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    total = sum(o[0] for o in out if o)
    print(f"BENCH {tag}: n={n} wall={wall:.1f}s aggregate={total / wall:.1f} "
          f"tok/s per_req={total / n} tok", flush=True)


def main():
    m0 = metrics()
    stream_once("warmup", SHORT, 8)
    stream_once("ctx2k", long_prompt(2000), 256)
    stream_once("ctx16k", long_prompt(16000), 256)
    stream_once("ctx65k", long_prompt(65000), 256)
    conc(8, "conc8")
    m1 = metrics()
    d = m1.get("spec_decode_num_draft_tokens", 0) - m0.get(
        "spec_decode_num_draft_tokens", 0)
    a = m1.get("spec_decode_num_accepted_tokens", 0) - m0.get(
        "spec_decode_num_accepted_tokens", 0)
    e = m1.get("spec_decode_num_emitted_tokens", 0) - m0.get(
        "spec_decode_num_emitted_tokens", 0)
    if d > 0:
        print(f"BENCH acceptance: {a:.0f}/{d:.0f} = {a / d:.3f} "
              f"emitted={e:.0f}")
    else:
        print("BENCH acceptance: no spec metrics")


if __name__ == "__main__":
    main()
