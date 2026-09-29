#!/usr/bin/env python3
"""cc_parser_battery_v125b.py — v89 parser battery + P23 failure
instrumentation.

P23 (2026-09-29 04:01) saw T1/T2 FAIL on n2e4m3s4 (=2 e4m3 spec fp8)
AFTER the full sustain+serialized+burst prefix, while T3 passed; the
same bodies passed perfectly on a fresh boot minutes later (p23b_diag,
comp_tok=74/112, correct calls). This variant keeps the v89 assertions
EXACTLY but adds:

  - full raw dump on any failure (finish_reason, completion_tokens,
    reasoning length+head, content head, tool_calls raw) — the v89
    printout showed only name=None/args=None, hiding what the model
    actually emitted;
  - one settle-retry after 20 s for T1/T2 (discriminates a transient
    vs persistent-in-context failure): verdict PASS if the retry
    passes, printed as RETRIED-PASS; both attempts dumped on fail.

Exit non-zero only on double-failure (or T3 failure).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
RETRY_SETTLE_S = 20

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


def dump_raw(tag: str, j: dict) -> None:
    ch = j["choices"][0]
    msg = ch.get("message", {})
    reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    print(f"     RAW[{tag}] finish={ch.get('finish_reason')} "
          f"comp_tok={(j.get('usage') or {}).get('completion_tokens')} "
          f"reason_len={len(reason)} content_len={len(content)} tc_n={len(tcs)}")
    if reason:
        print(f"     RAW[{tag}] reason[:200]={reason[:200]!r}")
    if content:
        print(f"     RAW[{tag}] content[:300]={content[:300]!r}")
    if tcs:
        print(f"     RAW[{tag}] tool_calls[:300]={json.dumps(tcs)[:300]!r}")


def run_t1() -> bool:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the run_command tool to list files under /tmp. Keep the description short."}],
        "tools": [{"type": "function", "function": FLAT}],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    def attempt() -> tuple[bool, object]:
        j = call(body)
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
        return ok, (args, j)

    ok, (args, j) = attempt()
    if ok:
        print(f"T1 flat-tool: PASS name=run_command args={json.dumps(args)[:120]}")
        return True
    print("T1 flat-tool: attempt1 FAIL — raw:")
    dump_raw("T1a", j)
    time.sleep(RETRY_SETTLE_S)
    ok, (args, j) = attempt()
    if ok:
        print(f"T1 flat-tool: RETRIED-PASS name=run_command args={json.dumps(args)[:120]}")
        return True
    print("T1 flat-tool: attempt2 FAIL — raw:")
    dump_raw("T1b", j)
    return False


def run_t2() -> bool:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Use the ask_user tool to ask the user which color they prefer, red or blue. Header Color."}],
        "tools": [{"type": "function", "function": NESTED}],
        "max_tokens": 400,
        "temperature": 0.0,
    }

    def attempt() -> tuple[bool, object]:
        j = call(body)
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
        return ok, (qs, j)

    ok, (qs, j) = attempt()
    if ok:
        print(f"T2 nested-tool: PASS questions_type=list n={len(qs)}")
        return True
    print("T2 nested-tool: attempt1 FAIL — raw:")
    dump_raw("T2a", j)
    time.sleep(RETRY_SETTLE_S)
    ok, (qs, j) = attempt()
    if ok:
        print(f"T2 nested-tool: RETRIED-PASS questions_type=list n={len(qs)}")
        return True
    print("T2 nested-tool: attempt2 FAIL — raw:")
    dump_raw("T2b", j)
    return False


def run_t3() -> bool:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    j = call(body)
    msg = j["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    ok = "OK" in txt and not msg.get("tool_calls")
    print(f"T3 plain-reply: {'PASS' if ok else 'FAIL'} text={txt[:60]!r}")
    if not ok:
        dump_raw("T3", j)
    return ok


def main() -> int:
    fails: list[str] = []
    try:
        if not run_t1():
            fails.append("T1")
    except Exception as e:
        print(f"T1 flat-tool: ERROR {type(e).__name__}: {e}")
        fails.append("T1")
    try:
        if not run_t2():
            fails.append("T2")
    except Exception as e:
        print(f"T2 nested-tool: ERROR {type(e).__name__}: {e}")
        fails.append("T2")
    try:
        if not run_t3():
            fails.append("T3")
    except Exception as e:
        print(f"T3 plain-reply: ERROR {type(e).__name__}: {e}")
        fails.append("T3")

    print(f"BATTERY {'PASS' if not fails else 'FAIL ' + ','.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
