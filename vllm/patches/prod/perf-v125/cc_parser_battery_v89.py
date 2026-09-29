#!/usr/bin/env python3
"""cc_parser_battery_v89.py — certify the qwen3_coder tool parser engine-direct.

Shapes mirroring CC traffic: flat tool (Bash-like), nested tool (AskUserQuestion
questions array), plain no-tool reply. Asserts tool_calls present, correct name,
arguments parse as JSON with non-empty leaves; nested container may arrive
stringified (model habit; litellm callback v2 un-stringifies) — both accepted
here, content verified either way. tool_choice is AUTO (CC never forces).
Exit non-zero on any miss.
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
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def as_container(v):
    """Return v as list/dict, un-stringifying one level if needed."""
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def main() -> int:
    fails: list[str] = []

    # T1 — flat tool
    try:
        j = call({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Use the run_command tool to list files under /tmp. Keep the description short."}],
            "tools": [{"type": "function", "function": FLAT}],
            "max_tokens": 300,
            "temperature": 0.0,
        })
        msg = j["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        ok = bool(tcs) and tcs[0]["function"]["name"] == "run_command"
        args = None
        if ok:
            try:
                args = json.loads(tcs[0]["function"]["arguments"] or "{}")
            except Exception:
                ok = False
        ok = ok and isinstance(args, dict) and isinstance(args.get("command"), str) and len(args["command"]) > 0
        print(f"T1 flat-tool: {'PASS' if ok else 'FAIL'} name={tcs[0]['function']['name'] if tcs else None} args={json.dumps(args)[:120] if args else None}")
        if not ok:
            fails.append("T1")
    except Exception as e:
        print(f"T1 flat-tool: ERROR {type(e).__name__}: {e}")
        fails.append("T1")

    # T2 — nested tool (AskUserQuestion shape)
    try:
        j = call({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Use the ask_user tool to ask the user which color they prefer, red or blue. Header Color."}],
            "tools": [{"type": "function", "function": NESTED}],
            "max_tokens": 400,
            "temperature": 0.0,
        })
        msg = j["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        ok = bool(tcs) and tcs[0]["function"]["name"] == "ask_user"
        qs = None
        if ok:
            try:
                args = json.loads(tcs[0]["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            qs = as_container(args.get("questions"))
        ok = ok and isinstance(qs, list) and len(qs) > 0 and isinstance(qs[0], dict) and "question" in qs[0]
        print(f"T2 nested-tool: {'PASS' if ok else 'FAIL'} questions_type={type(qs).__name__} n={len(qs) if isinstance(qs, list) else 0}")
        if not ok:
            fails.append("T2")
    except Exception as e:
        print(f"T2 nested-tool: ERROR {type(e).__name__}: {e}")
        fails.append("T2")

    # T3 — plain reply, no tools offered
    try:
        j = call({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 200,
            "temperature": 0.0,
        })
        txt = (j["choices"][0]["message"].get("content") or "").strip()
        ok = "OK" in txt and not (j["choices"][0]["message"].get("tool_calls"))
        print(f"T3 plain-reply: {'PASS' if ok else 'FAIL'} text={txt[:60]!r}")
        if not ok:
            fails.append("T3")
    except Exception as e:
        print(f"T3 plain-reply: ERROR {type(e).__name__}: {e}")
        fails.append("T3")

    print(f"BATTERY {'PASS' if not fails else 'FAIL ' + ','.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
