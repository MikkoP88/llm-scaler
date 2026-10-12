#!/usr/bin/env python3
"""DEC40 decode-leg runner v2 (against the fn8 serve :8321).

Solo decode measurement, warm-engine law, COMPREHENSIVE TELEMETRY (user
directive 2026-10-12 "capture all issues and performance issues"):

  1. wait health
  2. WARMUP completion (256 tok, greedy) — engine warm AND tier FLUSH law
  3. PARITY mini-suite (3 deterministic prompts, substring checks) — PLE-
     under-spec state-corruption watch (04d made PLE run under verify;
     semantic drift here = red flag)
  4. per leg: non-stream max_tokens=1 (prefill lands in prefix cache) ->
     STREAMING leg capturing EVERY chunk: arrival times + text sizes.
     Decode tok/s = (completion_tokens - 1) / (t_final - t_first).
     Spec telemetry from the stream itself: n_chunks ~= n_steps,
     tokens/step ~= ctok / n_chunks (acceptance proxy), inter-chunk ms =
     step latency percentiles.
  5. /metrics scrape before/after each shape: spec_decode_*_total deltas
     (drafted/accepted/emitted = true acceptance), preemptions, cache hit
  6. bank: legs + medians + spec deltas + serve-log tail (errors/PLE telem)
     + tier dump summary
  7. N legs (default 3), median + all legs banked; GATE_40 verdict

Acceptance proxy (A/B law): report ratio vs the nospec P3 reference
(4.975/5.243 tok/s, banks p3_run12.json) via env DEC40_REF.

Env: FN8_P2_PORT (8321), DEC40_BANK (/stage/bank/dec_d1.json),
DEC40_LEGS (3), DEC40_MAXTOK (256), DEC40_REF (optional, tok/s),
DEC40_SERVE_LOG (/stage/serve_d1.log), DEC40_TIER_DUMP
(/stage/bank/tier_d1.json).
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('FN8_P2_PORT', '8321')}"
MODEL = os.environ.get("FN8_MODEL", "flash-next-fp8")
BANK = os.environ.get("DEC40_BANK", "/stage/bank/dec_d1.json")
LEGS = int(os.environ.get("DEC40_LEGS", "3"))
MAXTOK = int(os.environ.get("DEC40_MAXTOK", "256"))
REF = float(os.environ["DEC40_REF"]) if "DEC40_REF" in os.environ else None
SERVE_LOG = os.environ.get("DEC40_SERVE_LOG", "/stage/serve_d1.log")
TIER_DUMP = os.environ.get("DEC40_TIER_DUMP", "/stage/bank/tier_d1.json")

# Prose + code shapes (Strata law: code accepts higher; measure both).
SHAPES = [
    ("prose", "Write a vivid paragraph about a lighthouse keeper who "
     "discovers that the sea has begun returning things she lost decades "
     "ago. Continue the scene."),
    ("code", "Implement a Python LRU cache with O(1) get/put using a "
     "doubly-linked list plus dict, with tests and commentary."),
]

# Deterministic parity (greedy): substring must appear in the completion.
PARITY = [
    ("fib20", "The 20th Fibonacci number is", "6765"),
    ("gcd", "The greatest common divisor of 48 and 18 is", "6"),
    ("add", "17 + 26 =", "43"),
]

_SPEC_RE = re.compile(r"^vllm:(\S*spec\S*_total)\s+([0-9.eE+-]+)")


def wait_health(deadline_s=14400):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    print("HEALTH_200 after", round(time.time() - t0, 1), "s",
                          flush=True)
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(10)
    raise SystemExit("health deadline exceeded")


def post(payload, timeout=14400):
    r = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def scrape_metrics():
    """Raw spec/preemption counters; None-safe (never crash the leg)."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            txt = r.read().decode()
    except Exception as e:
        return {"_error": str(e)}
    out = {}
    for line in txt.splitlines():
        m = _SPEC_RE.match(line.strip())
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.startswith("vllm:num_preemptions_total"):
            try:
                out["num_preemptions_total"] = float(line.split()[-1])
            except ValueError:
                pass
        elif line.startswith("vllm:prefix_cache_hit_rate"):
            try:
                out["prefix_cache_hit_rate"] = float(line.split()[-1])
            except ValueError:
                pass
    return out


def metrics_delta(a, b):
    d = {}
    for k in set(a) | set(b):
        if k.startswith("_"):
            continue
        if k in a and k in b:
            d[k] = round(b[k] - a[k], 3)
    return d


def stream_leg(model_prompt, maxtok):
    """Streaming completion; capture every chunk time+size (spec steps)."""
    payload = {
        "model": MODEL, "prompt": model_prompt,
        "max_tokens": maxtok, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    r = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t_first = None
    t_final = None
    ctok = None
    chunks = []  # (t, char_len) per text chunk = per emitted step
    with urllib.request.urlopen(r, timeout=14400) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            try:
                ev = json.loads(body)
            except Exception:
                continue
            if ev.get("usage") and ev["usage"].get("completion_tokens"):
                ctok = ev["usage"]["completion_tokens"]
                t_final = time.time()
                continue
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                if t_first is None:
                    t_first = time.time()
                chunks.append((time.time(), len(ch[0]["text"])))
    if t_final is None:
        t_final = time.time()
    if ctok is None:
        ctok = maxtok

    # step-level spec telemetry from the chunk stream
    step = {}
    if len(chunks) >= 3:
        its = [chunks[i][0] - chunks[i - 1][0] for i in range(1, len(chunks))]
        step = {
            "n_chunks": len(chunks),
            "tok_per_step_proxy": round(ctok / len(chunks), 3),
            "step_ms_p50": round(statistics.median(its) * 1e3, 1),
            "step_ms_p90": round(
                sorted(its)[max(0, int(len(its) * 0.9) - 1)] * 1e3, 1),
            "step_ms_max": round(max(its) * 1e3, 1),
        }
    return ctok, t_first, t_final, step


def serve_tail():
    """Errors + PLE telem tail from the serve log (evidence banking)."""
    out = {"exists": False}
    try:
        with open(SERVE_LOG, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        out["_error"] = str(e)
        return out
    out["exists"] = True
    out["n_lines"] = len(lines)
    out["n_error"] = sum(1 for l in lines if " ERROR " in l)
    out["ple_telem_n"] = sum(1 for l in lines if "FN8_PLE_TELEM" in l)
    out["ple_telem_last"] = next(
        (l.strip()[-200:] for l in reversed(lines) if "FN8_PLE_TELEM" in l),
        None)
    out["nan_lines"] = sum(1 for l in lines if "NaN" in l)
    out["last_20"] = [l.rstrip()[-240:] for l in lines[-20:]]
    return out


def tier_summary():
    try:
        with open(TIER_DUMP) as f:
            d = json.load(f)
    except Exception as e:
        return {"_error": str(e)}
    # dump schema varies; keep it small + lossless on the common keys
    if isinstance(d, dict):
        keep = {}
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                keep[k] = v
        return {"top_keys": sorted(d.keys())[:20], **keep}
    return {"type": str(type(d))}


def main():
    ready = wait_health()
    bank = {"ready_s": round(ready, 1), "ts": time.time(),
            "maxtok": MAXTOK, "legs": LEGS, "shapes": {}}

    # -- warmup (engine + FLUSH law) ---------------------------------------
    w = post({"model": MODEL,
              "prompt": "Write a Python function that reverses a linked "
                        "list iteratively, then explain its complexity.",
              "max_tokens": 256, "temperature": 0.0})
    bank["warmup_ct"] = w.get("usage", {}).get("completion_tokens")
    print(f"WARMUP 256 tok {bank['warmup_ct']} ct", flush=True)

    # -- parity mini-suite (PLE-under-spec corruption watch) ----------------
    bank["parity"] = {}
    for name, prompt, expect in PARITY:
        # 64-token budget: think-block preamble can defer the answer past
        # 32 (D1 false negative: add expected "43", got it after preamble)
        r = post({"model": MODEL, "prompt": prompt,
                  "max_tokens": 64, "temperature": 0.0})
        txt = (r.get("choices") or [{}])[0].get("text", "")
        ok = expect in txt
        bank["parity"][name] = {"ok": ok, "expect": expect,
                                "got_head": txt[:80]}
        print(f"PARITY {name}: {'OK' if ok else 'FAIL'} got={txt[:40]!r}",
              flush=True)

    for name, prompt in SHAPES:
        m_before = scrape_metrics()
        legs = []
        for i in range(LEGS):
            # prefill into prefix cache (ttft ~= 0 on the timed leg)
            post({"model": MODEL, "prompt": prompt,
                  "max_tokens": 1, "temperature": 0.0})
            ctok, t_first, t_final, step = stream_leg(prompt, MAXTOK)
            decode_s = t_final - t_first
            tok_s = (ctok - 1) / decode_s if decode_s > 0 else 0.0
            legs.append({"leg": i + 1, "completion_tokens": ctok,
                         "decode_s": round(decode_s, 3),
                         "tok_per_s": round(tok_s, 3), "step": step})
            print(f"DEC {name} leg{i + 1}: {ctok} ct in {decode_s:.2f}s "
                  f"decode = {tok_s:.3f} tok/s "
                  f"tok/step~{step.get('tok_per_step_proxy')} "
                  f"step_ms~{step.get('step_ms_p50')}", flush=True)
        m_after = scrape_metrics()
        med = round(statistics.median(l["tok_per_s"] for l in legs), 3)
        entry = {"legs": legs, "median_tok_per_s": med,
                 "spec_metrics_delta": metrics_delta(m_before, m_after),
                 "metrics_before": m_before, "metrics_after": m_after}
        if REF is not None:
            entry["ratio_vs_ref"] = round(med / REF, 3)
        bank["shapes"][name] = entry
        print(f"DEC {name} MEDIAN {med} tok/s"
              + (f"  ratio_vs_ref={entry['ratio_vs_ref']}" if REF else ""),
              flush=True)

    verdict = max(s["median_tok_per_s"] for s in bank["shapes"].values())
    bank["best_median_tok_per_s"] = verdict
    bank["gate_40"] = verdict >= 40.0
    bank["serve_tail"] = serve_tail()
    bank["tier_summary"] = tier_summary()
    with open(BANK, "w") as f:
        json.dump(bank, f, indent=1)
    print("BANKED", BANK, "best_median", verdict,
          "GATE_40", "MET" if bank["gate_40"] else "NOT_MET", flush=True)
    print("FN8_DEC_LEG_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
