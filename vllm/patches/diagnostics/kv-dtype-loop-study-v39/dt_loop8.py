#!/usr/bin/env python3
"""dt_loop8 — extended-budget think-trap differential (llm-scaler dtype matrix).

The 4 prompts that think-trapped at mt=4096 on BOTH fp16(auto) and
fp8_e4m3 (P2/P4/P5/P6 of dt_loop.py), re-run at mt=8192. Distinguishes:
  - budget-limited long thinking (escapes at 8192, finish=stop)
  - genuine dtype-amplified trapping (still trapped / tail loops)
Outputs to /root/build/loopout8-<tag>/.
"""
import json
import os
import sys
import time
import urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUTDIR = f"/root/build/loopout8-{TAG}"
os.makedirs(OUTDIR, exist_ok=True)
URL = "http://localhost:8000/v1/chat/completions"

PROMPTS = {
    2: "Write a Python function that finds the longest strictly increasing subsequence of an integer list in O(n log n), then prove the correctness and the complexity bound carefully.",
    4: "In how many ways can you tile a 3x12 rectangle with 2x1 dominoes? Derive the recurrence, compute the value, and verify it with a second independent method.",
    5: "Design a normalized database schema for a multi-tenant ticketing system with SLA tracking; walk through every table, key, and index, and explain how you would query the 95th-percentile first-response time per tenant.",
    6: "There are 3 doors; behind one is a car, behind the others goats. You pick door 1, the host (who knows) opens door 3 showing a goat, but this host opens a random goat door when he has a choice and prefers door 3 when both are goats. What is the probability the car is behind door 2 if he opened door 3? Reason carefully about the host policy.",
}


def detect_loop(text, min_p=4, max_p=160, min_reps=3, tail=1000):
    if not text:
        return None
    words = text.split()[-tail:]
    n = len(words)
    if n < min_p * min_reps:
        return None
    best = None
    for p in range(min_p, min(max_p, n // min_reps) + 1):
        unit = words[n - p:]
        reps = 1
        while reps < 64 and words[n - p * (reps + 1): n - p * reps] == unit:
            reps += 1
        if reps >= min_reps and (best is None or reps > best[1]):
            best = (p, reps)
    if best is None:
        return None
    p, reps = best
    return {"period_words": p, "reps": reps,
            "unit_sample": " ".join(words[n - p: n - p + min(p, 18)])}


def detect_loop_char(text, min_p=4, max_p=220, min_reps=3, tail=4000):
    if not text:
        return None
    t = text[-tail:]
    n = len(t)
    if n < min_p * min_reps:
        return None
    best = None
    for p in range(min_p, min(max_p, n // min_reps) + 1):
        unit = t[n - p:]
        reps = 1
        while reps < 128 and t[n - p * (reps + 1): n - p * reps] == unit:
            reps += 1
        if reps >= min_reps and (best is None or reps > best[1]):
            best = (p, reps)
    if best is None:
        return None
    p, reps = best
    return {"period_chars": p, "reps": reps,
            "unit_sample": t[n - p: n - p + min(p, 60)].replace("\n", "\\n")}


results = []
for i, prompt in PROMPTS.items():
    body = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    res = {"i": i}
    try:
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1200))
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
        content = msg.get("content") or ""
        finish = r["choices"][0].get("finish_reason")
        comp = r.get("usage", {}).get("completion_tokens", -1)
        with open(f"{OUTDIR}/p{i:02d}.json", "w") as f:
            json.dump({"prompt": prompt, "reasoning": reasoning,
                       "content": content, "finish": finish,
                       "completion_tokens": comp}, f, indent=1)
        lp_r = detect_loop(reasoning)
        lp_rc = detect_loop_char(reasoning)
        trapped = finish == "length" and len(content.strip()) == 0
        res = {"i": i, "finish": finish, "tokens": comp, "secs": round(dt, 1),
               "loop_reason": lp_r, "loop_reason_char": lp_rc,
               "think_trapped": trapped,
               "reason_len": len(reasoning), "content_len": len(content)}
    except Exception as e:  # noqa: BLE001
        res = {"i": i, "error": str(e)}
    results.append(res)
    flag = "  <<< LOOP" if (res.get("loop_reason") or res.get("loop_reason_char")
                            or res.get("think_trapped")) else ""
    print(f"P{i}: {json.dumps(res)}{flag}", flush=True)

ntrap = sum(1 for r in results if r.get("think_trapped"))
nloop = sum(1 for r in results
            if r.get("loop_reason") or r.get("loop_reason_char"))
print(f"DT_LOOP8[{TAG}]: {ntrap}/{len(results)} still trapped @8192, "
      f"{nloop} tail-loops", flush=True)
