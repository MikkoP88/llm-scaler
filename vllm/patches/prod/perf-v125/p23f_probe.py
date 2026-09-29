#!/usr/bin/env python3
"""p23f_probe.py — ns-path quality-rate measurement for v125 P23F.

P23E aborted at its first gate: e4m3ns (=1 ESIMD fp8) returned a WRONG
concurrent-math answer (x8d 7/8) on a FRESH boot — no prefix needed.
The old probe printed only the count; this one measures rates and dumps
every wrong answer verbatim.

mode A (fresh boot):
  1) 10 rounds x 8 concurrent distinct multiplications (round 0 = the
     exact P23E x8d question set), enable_thinking False, temp 0
  2) all 80 questions re-asked SERIALLY — separates concurrency-
     triggered crosstalk (concurrent-wrong, serial-right) from a
     numerics defect (wrong both ways)
  3) 30 alternating T1/T2 tool-call requests (p23d bodies, thinking ON)
mode B (after prefix): 3 more concurrent rounds + 30 tool-call requests.

ALWAYS exits 0 (measurement, not a gate).
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"

FLAT = {
    "name": "run_command",
    "description": "Run a shell command on the machine",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["command"],
    },
}

NESTED = {
    "name": "ask_user",
    "description": "Ask the user a multiple-choice question",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "header": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["question", "header", "options"],
                },
            }
        },
        "required": ["questions"],
    },
}

T1 = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Use the run_command tool to list files under /tmp. Keep the description short."}],
    "tools": [{"type": "function", "function": FLAT}],
    "max_tokens": 300,
    "temperature": 0.0,
}
T2 = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Use the ask_user tool to ask the user which color they prefer, red or blue. Header Color."}],
    "tools": [{"type": "function", "function": NESTED}],
    "max_tokens": 400,
    "temperature": 0.0,
}


def call(body: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def math_call(content: str, out: list, idx: int) -> None:
    body = {"model": MODEL, "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        out[idx] = (r["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        out[idx] = f"FAIL:{type(e).__name__}"


def qs(rnd: int) -> tuple[list, list, list]:
    a = [11 + ((rnd * 3 + i) % 19) for i in range(8)]
    b = [3 + ((rnd * 5 + i) % 7) for i in range(8)]
    return a, b, [f"What is {a[i]}*{b[i]}? Reply with just the number." for i in range(8)]


def math_rounds(rounds: int, tag: str) -> dict:
    """Concurrent rounds; returns {question_key: got} for all asked."""
    wrong = 0
    got: dict[str, str] = {}
    for rnd in range(rounds):
        a, b, prompts = qs(rnd)
        outs: list = [None] * 8
        ths = [threading.Thread(target=math_call, args=(prompts[i], outs, i)) for i in range(8)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        for i in range(8):
            want = str(a[i] * b[i])
            got[f"r{rnd}q{i}"] = outs[i]
            if outs[i] != want:
                wrong += 1
                print(f"MATH[{tag}] round{rnd:02d} req{i} q='{a[i]}*{b[i]}' want={want} got={outs[i][:80]!r}")
        time.sleep(1)
    print(f"MATH[{tag}] SUMMARY wrong={wrong}/{rounds * 8}")
    return got


def math_serial(limit: int, tag: str, conc_got: dict) -> None:
    """Re-ask the same questions SERIALLY; compare against concurrent."""
    wrong = 0
    flips = 0
    for rnd in range(limit):
        a, b, prompts = qs(rnd)
        for i in range(8):
            out: list = [None]
            math_call(prompts[i], out, 0)
            want = str(a[i] * b[i])
            key = f"r{rnd}q{i}"
            if out[0] != want:
                wrong += 1
                print(f"MATH[{tag}] serial r{rnd}q{i} q='{a[i]}*{b[i]}' want={want} got={out[0][:80]!r}")
            if key in conc_got and out[0] != conc_got[key]:
                flips += 1
                print(f"MATH[{tag}] FLIP r{rnd}q{i} concurrent={conc_got[key][:40]!r} serial={out[0][:40]!r}")
    print(f"MATH[{tag}] SERIAL SUMMARY wrong={wrong}/{limit * 8} concurrent-vs-serial flips={flips}")


def tool_probe(tag: str) -> int:
    counts = {"CLEAN": 0, "SALAD": 0, "OTHER": 0, "ERROR": 0}
    for i in range(30):
        body, want = (T1, "run_command") if i % 2 == 0 else (T2, "ask_user")
        try:
            j = call(body)
            ch = j["choices"][0]
            msg = ch.get("message", {})
            reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls") or []
            name = (tcs[0]["function"]["name"] if tcs else None)
            detail = (f"finish={ch.get('finish_reason')} comp={(j.get('usage') or {}).get('completion_tokens')} "
                      f"rlen={len(reason)} name={name!r}")
            if "!!!!!" in reason or "!!!!!" in content:
                head = reason[:60] if reason else content[:60]
                cls = "SALAD"
                detail += f" head={head!r}"
            elif name == want:
                cls = "CLEAN"
            elif not tcs and ch.get("finish_reason") == "length" and (len(reason) > 200 or len(content) > 200):
                cls = "SALAD"
                detail += f" head={(reason or content)[:60]!r}"
            else:
                cls = "OTHER"
                detail += f" head={(reason or content)[:60]!r}"
        except Exception as e:
            cls, detail = "ERROR", f"{type(e).__name__}: {e}"
        counts[cls] += 1
        if cls != "CLEAN":
            print(f"TOOL[{tag}] req{i:02d} {body['tools'][0]['function']['name'][:12]:12s} {cls:5s} {detail}")
    print(f"TOOL[{tag}] SUMMARY clean={counts['CLEAN']} salad={counts['SALAD']} "
          f"other={counts['OTHER']} error={counts['ERROR']} of 30")
    return counts["SALAD"] + counts["OTHER"] + counts["ERROR"]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    t0 = time.time()
    if mode == "A":
        conc = math_rounds(10, "fresh")
        math_serial(10, "fresh", conc)
        tool_probe("fresh")
    else:
        math_rounds(3, "prefix")
        tool_probe("prefix")
    print(f"OVERALL[{mode}] done in {time.time() - t0:.0f}s")
    print("P23F_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
