#!/usr/bin/env python3
"""coh2_sanity — §24 S quality gate: 65k cold vs warm greedy text equality.

Submits the same ~65k prompt twice (temperature 0, fixed seed). Run 1 = cold
prefill; run 2 = prefix-cache hit -> arena copy-on-hit path (VLLM_COH2=1).
A bitwise-correct D2D copy must reproduce the cold text EXACTLY — any stale
or corrupted arena content shows up as a diff. Prints both text tails and an
EQUAL/DIFF verdict.
"""
import json
import time
import urllib.request

M = "http://localhost:8000"
MODEL = "qwen3.8-27b-fp8"
FILLER = ("The river carried silt and small branches past the mooring posts, and the "
          "water level changed with every lock opening upstream. ")


def long_prompt(approx_tokens):
    reps = approx_tokens // 22
    return (FILLER * reps
            + "\n\nSummarize the passage above in exactly two sentences.")


def run(tag, prompt, max_tokens=96):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1234}
    req = urllib.request.Request(
        M + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=900)
    d = json.loads(resp.read().decode("utf-8", "replace"))
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    u = d.get("usage", {})
    print(f"SANITY {tag}: ptok={u.get('prompt_tokens')} "
          f"ctok={u.get('completion_tokens')} total={dt:.1f}s", flush=True)
    print(f"SANITY {tag} text[-240:]: ...{content[-240:]!r}", flush=True)
    return content


if __name__ == "__main__":
    p = long_prompt(65000)
    a = run("cold65k", p)
    b = run("warm65k", p)
    print(f"SANITY VERDICT: {'EQUAL' if a == b else 'DIFF'}", flush=True)
