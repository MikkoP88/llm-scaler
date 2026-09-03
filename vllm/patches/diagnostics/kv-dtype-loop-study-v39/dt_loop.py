#!/usr/bin/env python3
"""dt_loop — thinking-mode loop detector (llm-scaler dtype matrix).

8 chat prompts that force long <think> chains, greedy (temp 0), mt 4096.
Detects word-level tail loops (period 4..160, >=3 consecutive reps) in
reasoning_content and content separately; flags think-trapped generations
(finish=length, content empty). Full outputs saved to
/root/build/loopout-<tag>/ for cross-dtype diffing."""
import hashlib
import json
import os
import sys
import time
import urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUTDIR = f"/root/build/loopout-{TAG}"
os.makedirs(OUTDIR, exist_ok=True)
URL = "http://localhost:8000/v1/chat/completions"

PROMPTS = [
    "A train leaves Helsinki at 14:20 traveling at 132 km/h; a second train leaves the same station at 15:05 at 168 km/h in the same direction. At what time and distance from the station does the second train catch the first? Show every step.",
    "On an island every inhabitant is either a knight (always truthful) or a knave (always lying). You meet A, B, C. A says: 'exactly one of us is a knave.' B says: 'exactly two of us are knaves.' C says: 'B is a knight.' Determine who is what, with full case analysis.",
    "Write a Python function that finds the longest strictly increasing subsequence of an integer list in O(n log n), then prove the correctness and the complexity bound carefully.",
    "Prove that the square root of 2 is irrational, then explain step by step why the same argument fails for the square root of 4.",
    "In how many ways can you tile a 3x12 rectangle with 2x1 dominoes? Derive the recurrence, compute the value, and verify it with a second independent method.",
    "Design a normalized database schema for a multi-tenant ticketing system with SLA tracking; walk through every table, key, and index, and explain how you would query the 95th-percentile first-response time per tenant.",
    "There are 3 doors; behind one is a car, behind the others goats. You pick door 1, the host (who knows) opens door 3 showing a goat, but this host opens a random goat door when he has a choice and prefers door 3 when both are goats. What is the probability the car is behind door 2 if he opened door 3? Reason carefully about the host policy.",
    "A 2 kg block slides down a frictionless 30-degree incline of 4 m length and lands on a rough horizontal surface (mu=0.25). Compute the landing speed, the distance it slides, and the total time from release to rest. Take g=9.81. Show all steps.",
]


def detect_loop(text, min_p=4, max_p=160, min_reps=3, tail=1000):
    """Smallest-period, most-reps consecutive tail loop; word-level."""
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
        while (
            reps < 64
            and words[n - p * (reps + 1): n - p * reps] == unit
        ):
            reps += 1
        if reps >= min_reps and (best is None or reps > best[1]):
            best = (p, reps)
    if best is None:
        return None
    p, reps = best
    unit_txt = " ".join(words[n - p: n - p + min(p, 18)])
    return {"period_words": p, "reps": reps, "unit_sample": unit_txt}


def detect_loop_char(text, min_p=4, max_p=220, min_reps=3, tail=4000):
    """Character-level tail loop (catches whitespace-free repetition)."""
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
    unit_txt = t[n - p: n - p + min(p, 60)].replace("\n", "\\n")
    return {"period_chars": p, "reps": reps, "unit_sample": unit_txt}


def run(i, prompt):
    body = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
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
    lp_c = detect_loop(content)
    lp_rc = detect_loop_char(reasoning)
    lp_cc = detect_loop_char(content)
    trapped = finish == "length" and len(content.strip()) == 0
    h = hashlib.sha256((reasoning + "||" + content).encode()).hexdigest()[:12]
    return {
        "i": i, "finish": finish, "tokens": comp, "secs": round(dt, 1),
        "loop_reason": lp_r, "loop_content": lp_c,
        "loop_reason_char": lp_rc, "loop_content_char": lp_cc,
        "think_trapped": trapped, "hash": h,
    }


results = []
for i, p in enumerate(PROMPTS):
    try:
        res = run(i, p)
    except Exception as e:  # noqa: BLE001
        res = {"i": i, "error": str(e)}
    results.append(res)
    flag = ""
    if (res.get("loop_reason") or res.get("loop_content")
            or res.get("loop_reason_char") or res.get("loop_content_char")
            or res.get("think_trapped")):
        flag = "  <<< LOOP"
    print(f"P{i}: {json.dumps(res)}{flag}", flush=True)

nloop = sum(
    1 for r in results
    if (r.get("loop_reason") or r.get("loop_content")
            or r.get("loop_reason_char") or r.get("loop_content_char")
            or r.get("think_trapped"))
)
print(f"DT_LOOP[{TAG}]: {nloop}/{len(results)} prompts show looping", flush=True)
