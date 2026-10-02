#!/usr/bin/env python3
"""probe_thk_chain.py — thinking-chain hop diagnosis.

Hop 1: engine :8000 OpenAI stream  -> which reasoning field does the fork emit?
Hop 2: litellm :4000 OpenAI route   -> does reasoning survive litellm forwarding?
Hop 3: litellm :4000 /v1/messages   -> does the anthropic adapter emit thinking_delta?
Hop 4: litellm :4000 /v1/messages non-stream -> thinking block in content array?

Prints THK_* summary lines. Run on the host (python3, stdlib only).
"""
import json
import sys
import urllib.request

VLLM = "http://127.0.0.1:8000"
LITE = "http://127.0.0.1:4000"
MODEL = "qwen3.8-27b-fp8"          # engine name
MODEL_T = "qwen3.8-27b-fp8-opus"   # litellm tier name
KEY = "sk-dummy"
PROMPT = "Think step by step briefly, then answer: what is 17*23?"


def post(url, body, headers, timeout=420):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def sse_iter(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data: "):
            yield line[6:]


def hop1_engine():
    body = {
        "model": MODEL, "stream": True, "max_tokens": 400,
        "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    try:
        r = post(VLLM + "/v1/chat/completions", body,
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + KEY})
    except Exception as e:
        print("THK hop1 ERROR", repr(e)); return
    f = {"reasoning": 0, "reasoning_content": 0, "content": 0}
    sample = ""
    for d in sse_iter(r):
        if d == "[DONE]":
            break
        try:
            j = json.loads(d)
        except Exception:
            continue
        delta = (j.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("reasoning"):
            f["reasoning"] += 1; sample = sample or delta["reasoning"][:60]
        if delta.get("reasoning_content"):
            f["reasoning_content"] += 1
        if delta.get("content"):
            f["content"] += 1
    print("THK hop1 engine-stream reasoning=%d reasoning_content=%d content=%d"
          % (f["reasoning"], f["reasoning_content"], f["content"]))
    print("THK hop1 sample:", json.dumps(sample))


def hop2_lite_openai():
    body = {
        "model": MODEL_T, "stream": True, "max_tokens": 400,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    try:
        r = post(LITE + "/v1/chat/completions", body,
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + KEY})
    except Exception as e:
        print("THK hop2 ERROR", repr(e)); return
    f = {"reasoning": 0, "reasoning_content": 0, "content": 0}
    for d in sse_iter(r):
        if d == "[DONE]":
            break
        try:
            j = json.loads(d)
        except Exception:
            continue
        delta = (j.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("reasoning"):
            f["reasoning"] += 1
        if delta.get("reasoning_content"):
            f["reasoning_content"] += 1
        if delta.get("content"):
            f["content"] += 1
    print("THK hop2 lite-openai reasoning=%d reasoning_content=%d content=%d"
          % (f["reasoning"], f["reasoning_content"], f["content"]))


def hop3_lite_anthropic_stream():
    body = {
        "model": MODEL_T, "stream": True, "max_tokens": 400,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    h = {"Content-Type": "application/json", "x-api-key": KEY,
         "Authorization": "Bearer " + KEY, "anthropic-version": "2023-06-01"}
    try:
        r = post(LITE + "/v1/messages", body, h)
    except Exception as e:
        body_txt = ""
        if hasattr(e, "read"):
            try: body_txt = e.read().decode()[:300]
            except Exception: pass
        print("THK hop3 ERROR", repr(e), body_txt); return
    ev_types = {}
    think_deltas = 0; text_deltas = 0; think_head = ""
    raw_first = []
    for d in sse_iter(r):
        try:
            j = json.loads(d)
        except Exception:
            continue
        if len(raw_first) < 30:
            raw_first.append(d[:160])
        t = j.get("type", "?")
        ev_types[t] = ev_types.get(t, 0) + 1
        if t == "content_block_delta":
            d2 = j.get("delta") or {}
            if "thinking_delta" in d2:
                think_deltas += 1
                think_head = think_head or (d2.get("thinking_delta") or "")[:60]
            if "text_delta" in d2:
                text_deltas += 1
        if t == "error":
            print("THK hop3 SSE-ERROR", d[:300])
    print("THK hop3 anthropic-stream events=%s" % json.dumps(ev_types))
    print("THK hop3 thinking_delta=%d text_delta=%d" % (think_deltas, text_deltas))
    print("THK hop3 think_head:", json.dumps(think_head))
    print("THK hop3 raw_first:")
    for ln in raw_first:
        print("   |", ln)


def hop4_lite_anthropic_nonstream():
    body = {
        "model": MODEL_T, "stream": False, "max_tokens": 400,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    h = {"Content-Type": "application/json", "x-api-key": KEY,
         "Authorization": "Bearer " + KEY, "anthropic-version": "2023-06-01"}
    try:
        r = post(LITE + "/v1/messages", body, h)
        j = json.loads(r.read().decode())
    except Exception as e:
        body_txt = ""
        if hasattr(e, "read"):
            try: body_txt = e.read().decode()[:300]
            except Exception: pass
        print("THK hop4 ERROR", repr(e), body_txt); return
    blocks = [{"type": b.get("type"), "head": (b.get("thinking") or b.get("text") or "")[:60]}
              for b in (j.get("content") or [])]
    print("THK hop4 anthropic-nonstream stop=%s blocks=%s"
          % (j.get("stop_reason"), json.dumps(blocks, ensure_ascii=False)))


if __name__ == "__main__":
    print("=== THK hop1: engine :8000 ===");   hop1_engine()
    print("=== THK hop2: litellm openai ==="); hop2_lite_openai()
    print("=== THK hop3: litellm anthropic stream ==="); hop3_lite_anthropic_stream()
    print("=== THK hop4: litellm anthropic non-stream ==="); hop4_lite_anthropic_nonstream()
    print("THK_CHAIN_DONE")
