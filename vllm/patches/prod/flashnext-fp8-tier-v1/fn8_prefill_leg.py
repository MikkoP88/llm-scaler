#!/usr/bin/env python3
"""fn8 prefill-perf leg — 4K-prompt TTFT / prompt tok/s (P3 gate probe).

Gate (PHASES P3): prompt tok/s >= 600 @4K; no TTFT regression vs the P2
reference bank.  Streamed completion, TTFT = time to first SSE chunk;
prompt_tokens from the usage chunk (stream_options include_usage),
fallback chars/4.  4 legs with distinct ~60-token lead-ins so
--enable-prefix-caching cannot hand a later leg an earlier leg's prefix.

Run against the LIVE serve:
  docker exec -e FN8_PF_BANK=/stage/bank/prefill_p2ref.json fn8-p2 \
      python3 /stage/fn8_prefill_leg.py
"""
import json
import os
import time
import urllib.request

URL = "http://127.0.0.1:8321/v1/completions"
BANK = os.environ.get("FN8_PF_BANK", "/stage/bank/prefill_leg.json")

BASE = (
    "The tiered expert store keeps a bounded pool of expert tensors "
    "resident on the device while the remainder stay in host memory. "
    "Routing histograms collected during the profiling corpus decide "
    "which experts are pre-warmed.  A miss loads the expert through the "
    "side channel before the fused kernel gathers it, and an eviction "
    "writes the dummy slot id back into the mirror map.  ")

LEADS = [
    "First probe: summarize the mechanism.",
    "Second probe: summarize the mechanism.",
    "Third probe: summarize the mechanism.",
    "Fourth probe: summarize the mechanism.",
]


def leg(i):
    prompt = LEADS[i] + BASE * 68  # ~60 + ~4100 tokens
    body = json.dumps({
        "model": "flash-next-fp8",
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            try:
                j = json.loads(line[6:])
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            if ttft is None and j.get("choices") and (
                    j["choices"][0].get("text") or
                    j["choices"][0].get("finish_reason")):
                ttft = time.time() - t0
    ptok = (usage or {}).get("prompt_tokens") or len(prompt) // 4
    dt = time.time() - t0
    return {"leg": i, "ttft_s": round(ttft or dt, 3),
            "prompt_tokens": ptok,
            "prompt_tok_per_s": round(ptok / (ttft or dt), 1)}


legs = [leg(i) for i in range(len(LEADS))]
rates = [l["prompt_tok_per_s"] for l in legs]
med = sorted(rates)[len(rates) // 2]
out = {"legs": legs, "median_prompt_tok_per_s": med,
       "gate_600": med >= 600}
with open(BANK, "w") as f:
    json.dump(out, f, indent=1)
for l in legs:
    print("PREFILL leg%d ttft %.3fs ptok %d -> %.1f tok/s" % (
        l["leg"], l["ttft_s"], l["prompt_tokens"], l["prompt_tok_per_s"]))
print("PREFILL_MEDIAN %.1f tok/s gate_600=%s BANKED %s" % (
    med, out["gate_600"], BANK))
