#!/usr/bin/env python3
"""dt_probe2 — chat-based correctness gate + acceptance probe.

v1 of this probe used raw /v1/completions, which sends the prompts without
the chat template; the chat-tuned thinking model then degenerates into
newline soup at temp-0 and acceptance numbers are meaningless (even the
stock MTP lane only reached 6%). v2 sends proper /v1/chat/completions so
the target behaves like the served product: coherent greedy text that a
drafter can actually track. Hashes reasoning+content so whole-output
equality across lanes is checkable.
"""
import hashlib
import json
import re
import time
import urllib.request

M = "http://localhost:8000"
MODEL = "qwen3.8-27b-fp8"

PROMPTS = [
    ("p1_short", "Explain in one concise sentence what a transformer neural network is."),
    ("p2_arith", "2+2=4, 4+4=8, 8+8=16. Continue this doubling pattern for exactly ten more steps, one per line."),
    ("p3_para", "The old lighthouse keeper counted ships by the sound of their bells rather than by sight, "
                "because the fog on this coast rarely lifted before noon. Write two paragraphs about a morning "
                "when the bells fell silent."),
    ("p4_code", "Write a Python function that returns the n-th Fibonacci number using memoization, "
                "with a short docstring and type hints."),
    ("p5_fact", "List the four largest planets in the solar system in order, with one distinguishing fact each."),
]

FILLER = ("The river carried silt and small branches past the mooring posts, and the "
          "water level changed with every lock opening upstream. ")

def long_prompt(approx_tokens: int) -> str:
    reps = max(1, approx_tokens // 22)
    return FILLER * reps + "\n\nSummarize the passage above in exactly two sentences."

def metrics():
    txt = urllib.request.urlopen(M + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^vllm:([a-z_]+)_total\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if m and any(m.group(1).endswith(k) for k in (
                "num_accepted_tokens", "num_draft_tokens", "num_emitted_tokens",
                "prefix_cache_queries", "prefix_cache_hits")):
            out[m.group(1)] = float(m.group(2))
    return out

def run(tag: str, prompt: str, max_tokens: int):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1234,
    }
    req = urllib.request.Request(
        M + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x1fCONTENT\x1f" + (msg.get("content") or "")
    u = r.get("usage", {})
    print(f"PROBE {tag}: sha={hashlib.sha256(text.encode()).hexdigest()[:16]} "
          f"prompt_tok={u.get('prompt_tokens', -1)} compl_tok={u.get('completion_tokens', -1)} "
          f"secs={dt:.1f} tps={u.get('completion_tokens', 0) / max(dt, 1e-9):.1f}")
    print(f"  head: {text[:100]!r}")

def main():
    m0 = metrics()
    for tag, p in PROMPTS:
        run(tag, p, 400)
    run("p6_ctx2k", long_prompt(2000), 256)
    time.sleep(1)
    m1 = metrics()
    print("METRICS before:", json.dumps(m0))
    print("METRICS after: ", json.dumps(m1))
    if "spec_decode_num_accepted_tokens" in m1 and "spec_decode_num_draft_tokens" in m1:
        d = m1["spec_decode_num_draft_tokens"] - m0.get("spec_decode_num_draft_tokens", 0)
        a = m1["spec_decode_num_accepted_tokens"] - m0.get("spec_decode_num_accepted_tokens", 0)
        if d > 0:
            print(f"ACCEPTANCE: {a:.0f}/{d:.0f} = {a/d:.3f}")

if __name__ == "__main__":
    main()
