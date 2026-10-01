#!/usr/bin/env python3
"""xg_sweep_v128.py — v128 WS-D leg-1: temperature sweep of the guided-JSON
degeneration (P34 ladder + P38 quiet sample: 4/11 runs degenerate at the
probe's DEFAULT temperature 1.0 — grammar-legal unbounded digit emission,
finish=length, zero FSM/backend errors; the grammar only ever RESTRICTS to
schema-legal tokens, so the choice among legal tokens is pure sampling).

Arms (all explicit temperature in the request body; identical schema/prompt
to the banked T2 probe so rates are comparable with the 4/11 evidence):
  t10  temperature 1.0  — control, replicates the degeneration class
  t07  temperature 0.7  — Qwen3 non-thinking recommendation
  t06  temperature 0.6  — Qwen3 thinking recommendation
  pb1  temperature 1.0 + pattern-BOUNDED numeric field (string pattern) —
       client-side escape-hatch control: if 0 degenerations here, the
       guidance "bound the numeric pattern" is grounded even at temp 1.0.

Interleaved order (arm-major inside run-major) so any fleet-return drift
contaminates all arms equally. Output: JSONL per run + summary JSON.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
N_PER_ARM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = "/root/build/lce1/xg_sweep_v128.jsonl"
SUMMARY = "/root/build/lce1/xg_sweep_v128.json"

WEATHER_SCHEMA = {"type": "object",
                  "properties": {"city": {"type": "string"},
                                 "temperature_c": {"type": "number"}},
                  "required": ["city", "temperature_c"],
                  "additionalProperties": False}
# same object, numeric field as a pattern-bounded string: at most 4 integer
# digits + at most 4 fraction digits — grammar-enforced, so a runaway is
# impossible BY CONSTRUCTION while staying JSON-parseable
BOUNDED_SCHEMA = {"type": "object",
                  "properties": {"city": {"type": "string"},
                                 "temperature_c": {"type": "string",
                                                   "pattern": "^-?[0-9]{1,4}(\\.[0-9]{1,4})?$"}},
                  "required": ["city", "temperature_c"],
                  "additionalProperties": False}

ARMS = [
    ("t10", 1.0, WEATHER_SCHEMA),
    ("t07", 0.7, WEATHER_SCHEMA),
    ("t06", 0.6, WEATHER_SCHEMA),
    ("pb1", 1.0, BOUNDED_SCHEMA),
]


def body_for(temp, schema):
    return {
        "model": "qwen3.8-27b-fp8", "max_tokens": 2048,
        "temperature": temp,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "weather", "strict": True, "schema": schema}},
        "messages": [{"role": "user",
                      "content": "Tokyo weather (guess values). Fill the JSON now."}],
    }


def post(arm, temp, schema, run_idx):
    body = body_for(temp, schema)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    rec = {"ts": time.strftime("%H:%M:%S"), "arm": arm, "run": run_idx,
           "temperature": temp}
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            code, d = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        code, d = e.code, json.load(e)
    except Exception as e:
        rec.update(status="EXC", exc=f"{type(e).__name__}: {e}",
                   wall=round(time.time() - t0, 1))
        return rec
    rec["wall"] = round(time.time() - t0, 1)
    if "choices" in d:
        m = d["choices"][0]
        c = (m.get("message") or {}).get("content") or ""
        rec.update(status="ok", finish=m.get("finish_reason"),
                   content_len=len(c), content_head=c[:100])
        # longest digit run anywhere in the emission (the degeneration
        # signature: 20+ digits on an unconstrained numeric field)
        runs = [len(x) for x in re.findall(r"[0-9]+", c)]
        rec["max_digit_run"] = max(runs) if runs else 0
        try:
            j = json.loads(c)
            rec["parsed"] = True
            v = j.get("temperature_c")
            rec["temp_field"] = (str(v)[:40] if v is not None else None)
        except Exception as e:
            rec["parsed"] = False
            rec["parse_exc"] = f"{type(e).__name__}"
        rec["degenerate"] = bool(
            rec.get("finish") == "length" or rec.get("max_digit_run", 0) > 15)
    else:
        rec.update(status="NO_CHOICES",
                   body_head=json.dumps(d)[:200])
    return rec


def main():
    rows = []
    with open(OUT, "a") as fh:  # append: legs may be re-run after contamination
        fh.write(f"# sweep start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"n_per_arm={N_PER_ARM}\n")
        for i in range(N_PER_ARM):
            for arm, temp, schema in ARMS:
                rec = post(arm, temp, schema, i + 1)
                rows.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{arm} run{i+1}] {rec.get('status')} "
                      f"finish={rec.get('finish')} digits={rec.get('max_digit_run')} "
                      f"parsed={rec.get('parsed')} wall={rec.get('wall')}",
                      flush=True)
            time.sleep(1)

    summ = {"tag": "xg_sweep_v128", "n_per_arm": N_PER_ARM, "arms": {}}
    for arm, _, _ in ARMS:
        rs = [r for r in rows if r["arm"] == arm and r.get("status") == "ok"]
        n = len(rs)
        summ["arms"][arm] = {
            "n_ok": n,
            "n_total": sum(1 for r in rows if r["arm"] == arm),
            "degenerate": sum(1 for r in rs if r.get("degenerate")),
            "finish_length": sum(1 for r in rs if r.get("finish") == "length"),
            "finish_stop": sum(1 for r in rs if r.get("finish") == "stop"),
            "parse_fail": sum(1 for r in rs if not r.get("parsed", False)),
            "max_digit_run_max": max((r.get("max_digit_run", 0) for r in rs),
                                     default=0),
            "wall_p50": sorted(r["wall"] for r in rs)[n // 2] if n else None,
        }
    with open(SUMMARY, "w") as fh:
        json.dump(summ, fh, indent=1)
    print("SWEEP_DONE")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
