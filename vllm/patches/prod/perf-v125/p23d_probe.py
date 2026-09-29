#!/usr/bin/env python3
"""p23d_probe.py — salad-rate measurement for v125 P23D.

Sends N alternating T1/T2 tool-call requests (the exact battery bodies,
thinking default ON, temp 0) and classifies each response:
  CLEAN   — tool_calls present with the right name
  SALAD   — degenerate repetition in the reasoning channel (the
            'The!!!...!' attractor) or length-finish with no call and
            no meaningful reasoning
  OTHER   — any other miss (no call, wrong name, malformed args)
Prints per-request one-liners and a summary; ALWAYS exits 0 (this is a
measurement, not a gate). The caller greps the serve log for
grammar_matcher warnings before/after to get the matcher delta.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
N = 30

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


def classify(j: dict, want: str) -> tuple[str, str]:
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
        return "SALAD", detail + f" head={head!r}"
    if name == want:
        return "CLEAN", detail
    if not tcs and ch.get("finish_reason") == "length":
        head = (reason or content)[:60]
        return "SALAD" if len(reason) > 200 or len(content) > 200 else "OTHER", detail + f" head={head!r}"
    return "OTHER", detail


def main() -> int:
    counts = {"CLEAN": 0, "SALAD": 0, "OTHER": 0, "ERROR": 0}
    t0 = time.time()
    for i in range(N):
        body, want = (T1, "run_command") if i % 2 == 0 else (T2, "ask_user")
        try:
            j = call(body)
            cls, detail = classify(j, want)
        except Exception as e:
            cls, detail = "ERROR", f"{type(e).__name__}: {e}"
        counts[cls] += 1
        print(f"req{i:02d} {body['tools'][0]['function']['name'][:12]:12s} {cls:5s} {detail}")
    dt = time.time() - t0
    print(f"SUMMARY clean={counts['CLEAN']} salad={counts['SALAD']} other={counts['OTHER']} "
          f"error={counts['ERROR']} of {N} in {dt:.0f}s")
    print("P23D_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
