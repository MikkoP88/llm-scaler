#!/usr/bin/env python3
"""novel_probe — §24 S follow-up: discriminate cache-path vs lane-state.

A) novel65k: a brand-new ~65k filler (hash diverges at block 1 -> guaranteed
   full cold prefill) submitted NOW, i.e. in the same lane age/state the
   warm cells ran in. If decode ~359 -> the dip is cache-path-intrinsic
   (warm-resume scheduling). If ~302 -> it's lane-state/time, not the cache.
B) bench65k3: third submission of the dt_bench3x ctx65k prompt (warm again,
   arena is free) -> reproducibility of the 302 dip.
"""
import json
import time
import urllib.request

M = "http://localhost:8000"
MODEL = "qwen3.8-27b-fp8"
NOVEL = ("Harbor cranes swung containers from the freighter onto the rain-slicked "
         "quay while the night shift checked lashings and logged the manifest by "
         "hand under sodium lamps. ")


def stream_once(tag, prompt, max_tokens=256):
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
    reps = 65000 // 26
    stream_once("novel65k", NOVEL * reps + "\n\nSummarize the passage above briefly.")
    import dt_bench3x
    stream_once("bench65k3", dt_bench3x.long_prompt(65000))
