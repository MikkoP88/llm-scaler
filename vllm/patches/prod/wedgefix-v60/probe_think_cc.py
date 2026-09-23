#!/usr/bin/env python3
"""CC thinking round-trip probe — Anthropic /v1/messages through litellm.

Reproduces what Claude Code does: turn 1 fresh question; turn 2 replays the
assistant reply INCLUDING its thinking block (CC sends prior thinking back);
turn 3 replays text only (control). Diagnoses whether thinking blocks in
history break the request (400/500) or poison the next reply.

Usage:
  python3 probe_think_cc.py [base_url] [model]
  defaults: base_url=http://127.0.0.1:4000 model=qwen3.8-27b-fp8-opus
"""
import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8-opus"
KEY = sys.argv[3] if len(sys.argv) > 3 else "sk-dummy"
URL = BASE + "/v1/messages"


def call(messages, max_tokens=600, tag=""):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": KEY,
            "authorization": "Bearer " + KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        print("[%s] HTTP_ERR %s %s" % (tag, e.code, detail))
        return None
    except Exception as e:  # noqa: BLE001
        print("[%s] ERR %r" % (tag, e))
        return None
    blocks = payload.get("content", [])
    kinds = [b.get("type") for b in blocks]
    think = "".join(
        b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
    )
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    print(
        "[%s] stop=%s kinds=%s think_head=%r text_head=%r"
        % (tag, payload.get("stop_reason"), kinds, think[:80], text[:80])
    )
    return payload


t1 = call(
    [{"role": "user", "content": "Name three primary colors and nothing else."}],
    tag="t1",
)

hist2 = [{"role": "user", "content": "Name three primary colors and nothing else."}]
if t1:
    assistant_content = []
    for b in t1.get("content", []):
        if b.get("type") == "thinking":
            assistant_content.append(
                {"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")}
            )
        elif b.get("type") == "text":
            assistant_content.append({"type": "text", "text": b.get("text", "")})
    hist2.append({"role": "assistant", "content": assistant_content})
hist2.append({"role": "user", "content": "Now name three secondary colors."})
t2 = call(hist2, tag="t2")


def has_think(p):
    return bool(p) and any(b.get("type") == "thinking" for b in p.get("content", []))


hist3 = [
    {"role": "user", "content": "Name three primary colors and nothing else."},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": (t1 and next((b.get("text", "") for b in t1["content"] if b.get("type") == "text"), "")) or "Red, blue, yellow.",
            }
        ],
    },
    {"role": "user", "content": "Now name three secondary colors."},
]
t3 = call(hist3, tag="t3")

t2_text = ""
if t2:
    t2_text = next((b.get("text", "") for b in t2["content"] if b.get("type") == "text"), "")
echoed = bool(t2_text) and ("primary" in t2_text.lower() or "red" in t2_text.lower())
print(
    "THINK_CC_VERDICT t1_ok=%s t2_ok=%s t2_has_thinking=%s t2_echo_old=%s t3_ok=%s t3_has_thinking=%s"
    % (bool(t1), bool(t2), has_think(t2), echoed, bool(t3), has_think(t3))
)
