#!/usr/bin/env python3
"""xg_flag_probe_v128.py — v128 WS-D leg-2: validation probe for the
--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}
serve flag (booted via boot_v1227_restore.sh V1227_XGCOMPACT=1).

Hypotheses under test (against the banked xg_sweep_v128 baseline):
  H1  the whitespace-attractor degeneration class is GONE at any
      temperature (inter-token whitespace is grammar-ILLEGAL under
      any_whitespace=False — runaway impossible by construction);
  H2  output remains valid JSON at ~100% parse rate (compact form:
      no newlines/indent between JSON tokens; whitespace inside
      string values is content and stays legal);
  H3  residual digit-attractor rate on the unbounded numeric field is
      MEASURED (not assumed): the flag does not bound JSON numbers, so
      digit runaway may survive at temp 1.0 — this number decides
      whether the v1.2.28 posture also needs the generation_config
      default-temperature fold (0.9 -> 0.6) and/or client guidance.

Arms (identical schema/prompt to the sweep for comparability):
  f10  temperature 1.0  — the stress arm
  f07  temperature 0.7  — the vendor-rec arm
Usage: xg_flag_probe_v128.py [n_per_arm]
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
N_PER_ARM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = "/root/build/lce1/xg_flag_probe_v128.jsonl"
SUMMARY = "/root/build/lce1/xg_flag_probe_v128.json"

WEATHER_SCHEMA = {"type": "object",
                  "properties": {"city": {"type": "string"},
                                 "temperature_c": {"type": "number"}},
                  "required": ["city", "temperature_c"],
                  "additionalProperties": False}

ARMS = [("f10", 1.0), ("f07", 0.7)]


def post(arm, temp, run_idx):
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
    rec["status_code"] = code
    rec["wall"] = round(time.time() - t0, 1)
    if "choices" in d:
        m = d["choices"][0]
        c = (m.get("message") or {}).get("content") or ""
        rec.update(status="ok", finish=m.get("finish_reason"),
                   content_len=len(c), content_head=c[:100])
        runs = [len(x) for x in re.findall(r"[0-9]+", c)]
        rec["max_digit_run"] = max(runs) if runs else 0
        # whitespace-run fingerprint: longest run of ws chars outside of
        # string values (approximated by longest ws run after ':' or ','
        # or before '"' — where pretty-printing would put it)
        ws = [len(x) for x in re.findall(r"[:,:]([ \t\r\n]+)", c)]
        rec["max_ws_after_sep"] = max(ws) if ws else 0
        try:
            j = json.loads(c)
            rec["parsed"] = True
            v = j.get("temperature_c")
            rec["temp_field"] = (str(v)[:40] if v is not None else None)
        except Exception:
            rec["parsed"] = False
        rec["degenerate"] = bool(
            rec.get("finish") == "length" or rec.get("max_digit_run", 0) > 15)
        # the class the flag must kill: whitespace-runaway signature =
        # length-finish with tiny digit content, or ws run > 15
        rec["ws_degenerate"] = bool(
            rec.get("max_ws_after_sep", 0) > 15
            or (rec.get("finish") == "length" and rec.get("max_digit_run", 0) <= 3))
    else:
        rec.update(status="NO_CHOICES", body_head=json.dumps(d)[:300])
    return rec


def main():
    rows = []
    with open(OUT, "a") as fh:
        fh.write(f"# flag probe start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"n_per_arm={N_PER_ARM}\n")
        for i in range(N_PER_ARM):
            for arm, temp in ARMS:
                rec = post(arm, temp, i + 1)
                rows.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{arm} run{i+1}] {rec.get('status')} "
                      f"finish={rec.get('finish')} digits={rec.get('max_digit_run')} "
                      f"ws={rec.get('max_ws_after_sep')} parsed={rec.get('parsed')} "
                      f"wall={rec.get('wall')}", flush=True)
            time.sleep(1)

    summ = {"tag": "xg_flag_probe_v128", "n_per_arm": N_PER_ARM, "arms": {}}
    for arm, _ in ARMS:
        rs = [r for r in rows if r["arm"] == arm and r.get("status") == "ok"]
        n = len(rs)
        summ["arms"][arm] = {
            "n_ok": n,
            "n_total": sum(1 for r in rows if r["arm"] == arm),
            "degenerate": sum(1 for r in rs if r.get("degenerate")),
            "ws_degenerate": sum(1 for r in rs if r.get("ws_degenerate")),
            "finish_length": sum(1 for r in rs if r.get("finish") == "length"),
            "finish_stop": sum(1 for r in rs if r.get("finish") == "stop"),
            "parse_fail": sum(1 for r in rs if not r.get("parsed", False)),
            "max_digit_run_max": max((r.get("max_digit_run", 0) for r in rs),
                                     default=0),
            "max_ws_after_sep_max": max((r.get("max_ws_after_sep", 0) for r in rs),
                                        default=0),
            "wall_p50": sorted(r["wall"] for r in rs)[n // 2] if n else None,
        }
    f10 = summ["arms"]["f10"]
    summ["verdict"] = {
        "H1_whitespace_class_dead": f10["ws_degenerate"] == 0,
        "H2_parse_clean": f10["parse_fail"] == 0 and f10["n_ok"] > 0,
        "H3_digit_residual_per_arm": {
            a: summ["arms"][a]["degenerate"] for a, _ in ARMS},
    }
    with open(SUMMARY, "w") as fh:
        json.dump(summ, fh, indent=1)
    print("FLAG_PROBE_DONE")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
