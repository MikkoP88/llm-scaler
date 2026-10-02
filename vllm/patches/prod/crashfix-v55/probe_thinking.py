#!/usr/bin/env python3
"""Probe: why do non-Claude-Code clients see no thinking + no STATUS block?

Three requests, all from GPUNODE01:
  A) litellm :4000 qwen3.8-27b-fp8-sonnet, plain user msg (no client system)
  B) litellm :4000 same model, WITH client system prompt (non-CC app shape)
  C) vLLM :8000 direct, explicit chat_template_kwargs enable_thinking
For each: reasoning_content field present? <think> inline? STATUS block head?
"""
import json
import urllib.request

BASE_LITELLM = "http://127.0.0.1:4000/v1/chat/completions"
BASE_VLLM = "http://10.20.3.65:8000/v1/chat/completions"
KEY = "sk-dummy"


def call(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def show(tag, d):
    m = d["choices"][0]["message"]
    c = m.get("content") or ""
    rc = m.get("reasoning_content")
    print(f"[{tag}] finish={d['choices'][0].get('finish_reason')} "
          f"usage_ct={d.get('usage', {}).get('completion_tokens')}")
    print(f"[{tag}] HAS_REASONING_FIELD={rc is not None} "
          f"rc_head={ (rc or '')[:120]!r}")
    print(f"[{tag}] THINK_INLINE={'<think>' in c} "
          f"content_head={c[:220]!r}")
    print(f"[{tag}] STATUS_BLOCK="
          f"{any(c.lstrip().startswith(p) for p in ('DONE:', 'RUNNING:', 'STATUS', '**STATUS**'))}")
    print()


Q = "What is 17*23? Answer with the number only."

# A: plain non-CC request, no client system
show("A sonnet/plain", call(BASE_LITELLM, {
    "model": "qwen3.8-27b-fp8-sonnet",
    "messages": [{"role": "user", "content": Q}],
    "max_tokens": 500,
}))

# B: non-CC app shape with its own system prompt
show("B sonnet/own-system", call(BASE_LITELLM, {
    "model": "qwen3.8-27b-fp8-sonnet",
    "messages": [{"role": "system",
                  "content": "You are a concise assistant."},
                 {"role": "user", "content": Q}],
    "max_tokens": 500,
}))

# C: direct to vLLM, thinking explicitly on
show("C vllm-direct", call(BASE_VLLM, {
    "model": "qwen3.8-27b-fp8",
    "messages": [{"role": "user", "content": Q}],
    "max_tokens": 500,
    "chat_template_kwargs": {"enable_thinking": True},
}))
