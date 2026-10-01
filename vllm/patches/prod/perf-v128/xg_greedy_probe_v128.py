#!/usr/bin/env python3
"""xg_greedy_probe_v128.py — v128 WS-D leg-1b: greedy/temp-floor probe.

The sweep's degeneration entry rate is temperature-INSENSITIVE (t06 10/20,
t07 10/20, t10 11/20) — inconsistent with a peaked 27B sampling
distribution (temp 0.6 should sharply suppress any coin-flip class), and
suggestive of post-mask logit flattening among the surviving legal tokens
(e.g. rarely-trained merged digit tokens competing with the closers).

Decisive discriminator:
  greedy temp 0.0 CLEAN  -> stochastic sampling-level attractor; the mask
                           permits the exits (`,`/`}`/`.`), the model just
                           doesn't prefer them -> fix = engine flag +
                           client schema bounding + default-temp fold.
  greedy temp 0.0 RUNAWAY-> the grammar/mask state itself forces the path
                           (exit tokens masked illegal at some digit
                           prefixes) -> engine/xgrammar bug, fork-level
                           investigation required.
Arms: g00 (temp 0.0), g01 (temp 0.1), 5 runs each.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
OUT = "/root/build/lce1/xg_greedy_probe_v128.jsonl"

WEATHER_SCHEMA = {"type": "object",
                  "properties": {"city": {"type": "string"},
                                 "temperature_c": {"type": "number"}},
                  "required": ["city", "temperature_c"],
                  "additionalProperties": False}


def post(temp, run_idx):
    body = {
        "model": "qwen3.8-27b-fp8", "max_tokens": 2048,
        "temperature": temp,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "weather", "strict": True, "schema": WEATHER_SCHEMA}},
        "messages": [{"role": "user",
                      "content": "Tokyo weather (guess values). Fill the JSON now."}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    rec = {"ts": time.strftime("%H:%M:%S"), "temp": temp, "run": run_idx}
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        d = json.load(e)
    except Exception as e:
        rec.update(status="EXC", exc=f"{type(e).__name__}: {e}")
        return rec
    m = (d.get("choices") or [{}])[0]
    c = ((m.get("message") or {}).get("content") or "")
    runs_ = [len(x) for x in re.findall(r"[0-9]+", c)]
    try:
        json.loads(c); parsed = True
    except Exception:
        parsed = False
    rec.update(status="ok", finish=m.get("finish_reason"),
               content_head=repr(c[:60]), max_digit_run=max(runs_) if runs_ else 0,
               parsed=parsed, wall=round(time.time() - t0, 1))
    return rec


def main():
    n = 5
    with open(OUT, "a") as fh:
        fh.write(f"# greedy probe {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for temp in (0.0, 0.1):
            for i in range(n):
                rec = post(temp, i + 1)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[t{temp} run{i+1}] {rec.get('finish')} "
                      f"digits={rec.get('max_digit_run')} parsed={rec.get('parsed')} "
                      f"head={rec.get('content_head')}", flush=True)
    print("GREEDY_PROBE_DONE")


if __name__ == "__main__":
    main()
