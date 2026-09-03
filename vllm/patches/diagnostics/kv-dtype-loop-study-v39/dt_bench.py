#!/usr/bin/env python3
"""dt_bench — per-kv-dtype performance probe (llm-scaler dtype matrix).

Measures on a running server:
  - chat TTFT/ITL (short prompt, streaming)
  - prefill TTFT at ~2k / ~16k / ~65k prompt tokens (mt=1)
  - decode tok/s + ITL p50/p95 at ~2k / ~16k / ~65k context (streaming)
Filler is deliberately VARIED (numbered markers every 12 words) to avoid
the KNOWN_ISSUES #14 synthetic-repetition sampling pathology."""
import http.client
import json
import random
import sys
import time

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
HOST, PORT = "127.0.0.1", 8000

VOCAB = (
    "harbor lantern quarry meadow cinder falcon timber granite violet "
    "compass anchor sapphire driftwood parchment kiln foundry orchard "
    "cascade willow basalt copper marble thistle kestrel rivet sailcloth "
    "tundra embers ferry bellows cobalt hazel juniper lathe mosaic "
    "obelisk prairie sandalwood trellis urn verdigris wharf yarrow "
    "almanac beehive cloister derrick espalier flint grotto hammock "
    "inkwell jetty kelp lighthouse mangrove nectarine observatory "
    "parapet quince rookery silo thatch vellum windlass apothecary "
    "brigantine carafe dugout estuary feedstock gable hopper"
).split()

SIZES = [("2k", 2700), ("16k", 21500), ("65k", 87000)]


def filler(target_words, seed):
    rng = random.Random(seed)
    parts = []
    i = 0
    while len(parts) < target_words:
        parts.append(rng.choice(VOCAB))
        if i % 12 == 0:
            parts.append("marker%06d" % i)
        i += 1
    return " ".join(parts)


def post(path, payload, timeout=1200):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("POST", path, body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    return conn.getresponse()


def completions_stream(prompt, max_tokens):
    payload = {
        "model": "qwen3.8-27b-fp8", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }
    t_req = time.perf_counter()
    resp = post("/v1/completions", payload)
    if resp.status != 200:
        return {"error": f"HTTP {resp.status}: {resp.read()[:200]!r}"}
    stamps, usage, buf = [], None, b""
    ttft = None
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
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
            ch = obj.get("choices") or []
            if ch and (ch[0].get("text") or "") != "":
                if ttft is None:
                    ttft = now - t_req
                stamps.append(now)
    total = time.perf_counter() - t_req
    comp = usage["completion_tokens"] if usage else len(stamps)
    ptok = usage["prompt_tokens"] if usage else -1
    if len(stamps) >= 3:
        diffs = sorted(
            stamps[j + 1] - stamps[j] for j in range(len(stamps) - 1))
        itl50 = diffs[len(diffs) // 2] * 1000
        itl95 = diffs[min(len(diffs) - 1, int(len(diffs) * 0.95))] * 1000
    else:
        itl50 = itl95 = -1
    decode_toks = max(comp, 1)
    decode_time = max(total - (ttft or 0), 1e-6)
    return {
        "ptok": ptok, "ttft_s": round(ttft or total, 3),
        "total_s": round(total, 2), "comp": comp,
        "decode_tok_s": round(decode_toks / decode_time, 2),
        "itl_ms_p50": round(itl50, 1), "itl_ms_p95": round(itl95, 1),
    }


def chat_ttft():
    stamps, ttft = [], None
    payload = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user",
                      "content": "Explain the difference between TCP and UDP."}],
        "max_tokens": 256, "temperature": 0, "stream": True,
    }
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", payload)
    if resp.status != 200:
        return {"error": f"HTTP {resp.status}"}
    buf = b""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip().startswith(b"data:") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
    return {"ttft_s": round(ttft or -1, 3),
            "n_chunks": len(stamps)}


print(f"=== chat short ===", flush=True)
r = chat_ttft()
print(f"CHAT[{TAG}]: {json.dumps(r)}", flush=True)

for name, words in SIZES:
    p = ("Summarize the following document.\n"
         + filler(words, seed=hash(name) % 1000))
    print(f"=== prefill {name} (mt=1) ===", flush=True)
    t0 = time.perf_counter()
    resp = post("/v1/completions", {
        "model": "qwen3.8-27b-fp8", "prompt": p,
        "max_tokens": 1, "temperature": 0})
    body = json.loads(resp.read())
    tt = time.perf_counter() - t0
    ptok = body.get("usage", {}).get("prompt_tokens", -1)
    print(f"PREFILL[{TAG},{name}]: ptok={ptok} ttft_s={tt:.2f} "
          f"tok_s={ptok / max(tt, 1e-6):.0f}", flush=True)

    mt = 900 if name == "65k" else 512
    print(f"=== decode {name} (mt={mt}) ===", flush=True)
    r = completions_stream(p, mt)
    print(f"DECODE[{TAG},{name}]: {json.dumps(r)}", flush=True)

print(f"DT_BENCH[{TAG}] DONE", flush=True)
