#!/usr/bin/env python3
"""p23b_diag.py — discriminate the P23 parser-battery failure on the =2
spec-fp8 postures (n2e4m3s4 T1/T2 FAIL, T3 PASS; cert16s4 all PASS).

P23's cc_parser_battery_v89.py prints only name=None/args=None on failure.
This probe re-sends the EXACT T1/T2/T3 bodies and dumps the FULL raw
response shape: finish_reason, usage, reasoning_content length + head,
content head, tool_calls raw. Two extra variants per tool case:
  -long    : max_tokens 1500   (is the call merely truncated by a long
                                thinking phase before max_tokens 300/400?)
  -nothink : enable_thinking False (does the call come immediately?)
Read as: if -long/-nothink PASS while the stock body FAILs, the failure is
truncation-by-thinking, not corruption. If -nothink still produces prose
with no call (or garbage), it is output-quality degradation.
"""
from __future__ import annotations

import json
import sys
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


def call(body: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def dump(tag: str, body: dict) -> None:
    try:
        j = call(body)
    except Exception as e:
        print(f"---- {tag} ERROR {type(e).__name__}: {e}")
        return
    ch = j["choices"][0]
    msg = ch.get("message", {})
    fr = ch.get("finish_reason")
    comp = (j.get("usage") or {}).get("completion_tokens")
    reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    tc0 = tcs[0] if tcs else None
    fn = (tc0 or {}).get("function", {})
    args_head = (fn.get("arguments") or "")[:200]
    print(f"---- {tag} finish={fr} comp_tok={comp} "
          f"reason_len={len(reason)} content_len={len(content)} "
          f"tc_n={len(tcs)} name={fn.get('name')!r} args={args_head!r}")
    if reason:
        print(f"     reason[:{180}]={reason[:180]!r}")
    if content:
        print(f"     content[:{260}]={content[:260]!r}")
    if not tcs and not content and not reason:
        print(f"     EMPTY-MESSAGE keys={sorted(msg.keys())} raw_choice={json.dumps(ch)[:400]}")


def main() -> int:
    t1 = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "Use the run_command tool to list files under /tmp. Keep the description short."}],
        "tools": [{"type": "function", "function": FLAT}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    t2 = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "Use the ask_user tool to ask the user which color they prefer, red or blue. Header Color."}],
        "tools": [{"type": "function", "function": NESTED}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    t3 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 200,
        "temperature": 0.0,
    }

    dump("T1 stock (mt300)", t1)
    t1l = dict(t1); t1l["max_tokens"] = 1500
    dump("T1 long (mt1500)", t1l)
    t1n = dict(t1); t1n["chat_template_kwargs"] = {"enable_thinking": False}
    dump("T1 nothink (mt300)", t1n)

    dump("T2 stock (mt400)", t2)
    t2l = dict(t2); t2l["max_tokens"] = 1500
    dump("T2 long (mt1500)", t2l)
    t2n = dict(t2); t2n["chat_template_kwargs"] = {"enable_thinking": False}
    dump("T2 nothink (mt400)", t2n)

    dump("T3 stock (mt200)", t3)
    print("P23B_DIAG_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
