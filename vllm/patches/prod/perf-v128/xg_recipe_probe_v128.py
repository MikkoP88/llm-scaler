#!/usr/bin/env python3
"""xg_recipe_probe_v128.py — v128 WS-D leg-3: the COMPLETE-RECIPE probe.

Runs the pattern-bounded schema (pb1 from xg_sweep_v128: numeric field as
a string with ^-?[0-9]{1,4}(\\.[0-9]{1,4})?$) at temperature 1.0 on the
V1227_XGCOMPACT=1 lane (any_whitespace=False grammar).

The recipe claim this probe certifies:
  bounded schema (kills both digit classes: integer-runaway AND
  fraction-runaway, grammar-enforced) + disable_any_whitespace (kills the
  whitespace class, grammar-enforced) = 0 degenerations BY CONSTRUCTION
  at ANY temperature — here demonstrated at the worst case, temp 1.0.

Sweep baselines for the same schema/prompt: pb1 alone (no flag) = 7/20
whitespace-class degenerations.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = "/root/build/lce1/xg_recipe_probe_v128.jsonl"
SUMMARY = "/root/build/lce1/xg_recipe_probe_v128.json"

BOUNDED_SCHEMA = {"type": "object",
                  "properties": {"city": {"type": "string"},
                                 "temperature_c": {"type": "string",
                                                   "pattern": "^-?[0-9]{1,4}(\\.[0-9]{1,4})?$"}},
                  "required": ["city", "temperature_c"],
                  "additionalProperties": False}


def post(run_idx):
    body = {
        "model": "qwen3.8-27b-fp8", "max_tokens": 2048,
        "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "weather", "strict": True, "schema": BOUNDED_SCHEMA}},
        "messages": [{"role": "user",
                      "content": "Tokyo weather (guess values). Fill the JSON now."}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    rec = {"ts": time.strftime("%H:%M:%S"), "run": run_idx, "temperature": 1.0}
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        d = json.load(e)
    except Exception as e:
        rec.update(status="EXC", exc=f"{type(e).__name__}: {e}",
                   wall=round(time.time() - t0, 1))
        return rec
    m = (d.get("choices") or [{}])[0]
    c = ((m.get("message") or {}).get("content") or "")
    runs_ = [len(x) for x in re.findall(r"[0-9]+", c)]
    ws = [len(x) for x in re.findall(r"[:,:]([ \t\r\n]+)", c)]
    try:
        j = json.loads(c)
        parsed = True
        v = j.get("temperature_c")
        val = str(v)[:40] if v is not None else None
    except Exception:
        parsed, val = False, None
    rec.update(status="ok", finish=m.get("finish_reason"),
               content_head=repr(c[:60]),
               max_digit_run=max(runs_) if runs_ else 0,
               max_ws_after_sep=max(ws) if ws else 0,
               parsed=parsed, temp_field=val,
               wall=round(time.time() - t0, 1))
    rec["degenerate"] = bool(
        rec.get("finish") == "length" or rec.get("max_digit_run", 0) > 15
        or rec.get("max_ws_after_sep", 0) > 15)
    return rec


def main():
    rows = []
    with open(OUT, "a") as fh:
        fh.write(f"# recipe probe {time.strftime('%Y-%m-%d %H:%M:%S')} n={N}\n")
        for i in range(N):
            rec = post(i + 1)
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[pb1+flag run{i+1}] {rec.get('status')} "
                  f"finish={rec.get('finish')} digits={rec.get('max_digit_run')} "
                  f"ws={rec.get('max_ws_after_sep')} parsed={rec.get('parsed')} "
                  f"wall={rec.get('wall')}", flush=True)
            time.sleep(1)
    ok = [r for r in rows if r.get("status") == "ok"]
    summ = {
        "tag": "xg_recipe_probe_v128", "n": N,
        "n_ok": len(ok),
        "degenerate": sum(1 for r in ok if r.get("degenerate")),
        "finish_length": sum(1 for r in ok if r.get("finish") == "length"),
        "parse_fail": sum(1 for r in ok if not r.get("parsed", False)),
        "max_digit_run_max": max((r.get("max_digit_run", 0) for r in ok), default=0),
        "max_ws_after_sep_max": max((r.get("max_ws_after_sep", 0) for r in ok), default=0),
        "recipe_zero_by_construction": (
            len(ok) == N and not any(r.get("degenerate") for r in ok)),
    }
    with open(SUMMARY, "w") as fh:
        json.dump(summ, fh, indent=1)
    print("RECIPE_PROBE_DONE")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
