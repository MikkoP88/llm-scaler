#!/usr/bin/env python3
"""llm-scaler T2 discriminator: does chat_template_kwargs.enable_thinking
(False) turn structured-output requests into HTTP 500?

repro3 pinned the op behavior: xpu_topk_topp_sampler returns 0 iff its
logits row is ALL -inf. Something between the grammar apply and the op
-inf's the row. Sampler.apply_logits_processors consults
thinking_budget_state_holder (fork thinking-budget machinery) — both
failing probe bodies set enable_thinking False. Variants:
  A) json_schema + enable_thinking False   (known 500)
  B) json_schema, NO chat_template_kwargs
  C) json_schema + enable_thinking True
  D) json_schema + reasoning_effort low (no enable_thinking)
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"


def post(body, tag):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            code, d = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code = e.code
        try:
            d = json.load(e)
        except Exception:
            d = {}
    except Exception as e:
        print(f"[{tag}] EXC {type(e).__name__}: {e}")
        return
    wall = time.time() - t0
    if "choices" in d:
        m = d["choices"][0]
        c = (m.get("message") or {}).get("content") or ""
        ok = "?"
        try:
            j = json.loads(c)
            ok = f"VALID_JSON keys={sorted(j.keys())}"
        except Exception:
            ok = "NO_JSON"
        print(f"[{tag}] HTTP={code} wall={wall:.1f}s finish="
              f"{m.get('finish_reason')} {ok} head={c[:60]!r}")
    else:
        print(f"[{tag}] HTTP={code} wall={wall:.1f}s NO_CHOICES")


SCHEMA = {"type": "object",
          "properties": {"city": {"type": "string"},
                         "temperature_c": {"type": "number"}},
          "required": ["city", "temperature_c"],
          "additionalProperties": False}


def body(ctk):
    b = {"model": "qwen3.8-27b-fp8", "max_tokens": 512,
         "response_format": {"type": "json_schema", "json_schema": {
             "name": "weather", "strict": True, "schema": SCHEMA}},
         "messages": [{"role": "user",
                       "content": "Tokyo weather (guess values). Fill the JSON now."}]}
    if ctk is not None:
        b["chat_template_kwargs"] = ctk
    return b


CASES = [
    ("A-thinkFalse", body({"enable_thinking": False})),
    ("B-noctk", body(None)),
    ("C-thinkTrue", body({"enable_thinking": True})),
    ("D-effort-low", body({"reasoning_effort": "low"})),
]
for i in range(2):
    for tag, b in CASES:
        post(b, f"{tag}-run{i+1}")
    print("---")
print("THINK_PROBE_DONE")
