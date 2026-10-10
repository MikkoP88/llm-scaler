#!/usr/bin/env python3
"""p84_decode_drive.py — P84 decode measurement driver (2026-10-06).

Drives N_CONVS concurrent deep requests (fixed ~12k-token fill + 512-token
generation, temp 0.5) against the live lane, records per-request walls and
usage, then snapshots /metrics (TPOT histogram + spec counters) to the leg
stage dir. Comparable to the 2026-10-06 live v1.2.30 baseline fingerprint
(step ~165-167 ms, TPOT mean 83.3 ms, 12.0 tok/s/req @ 3-4 deep reqs).

Usage: p84_decode_drive.py <leg-name>   (stage dir /root/build/p84_stage)
Output: P84DECODE_DONE line + <leg>_decode.jsonl + <leg>_metrics.txt
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
N_CONVS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
FILL_CHARS = 48000          # ~12k tokens at the fork's 4 chars/tok heuristic
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 512

WORDS = ("kernel cache lane prefix block token queue scheduler budget "
         "barrier wedge fence stall prefill decode chunk watermark hash "
         "evict resident cold hot lru slot pool capacity concurrency "
         "telemetry gauge counter bucket latency tail median p99 step "
         "graph capture shape pad batch aggregate stream window storm "
         "quiet regime cohort signature causal pressure oversubscribe").split()


def synth(seed, approx_chars):
    out = []
    n = 0
    while n < approx_chars:
        out.append(WORDS[(seed * 131 + n // 5) % len(WORDS)])
        n += len(out[-1]) + 1
    return " ".join(out)


def one_req(idx, results):
    prompt = ("Background context follows.\n\n" + synth(idx + 7, FILL_CHARS)
              + "\n\nEnd of context. Write a concise numbered summary of "
                "exactly 8 points about the text above.")
    body = json.dumps({
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.5,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
        wall = time.time() - t0
        usage = payload.get("usage") or {}
        ctok = usage.get("completion_tokens") or 0
        ptok = usage.get("prompt_tokens") or 0
        rec = {"idx": idx, "status": "ok", "wall_s": round(wall, 2),
               "prompt_tokens": ptok, "completion_tokens": ctok,
               "tok_s_per_req": round(ctok / wall, 2) if wall > 0 else 0}
    except Exception as exc:                             # noqa: BLE001
        rec = {"idx": idx, "status": "err",
               "error": str(exc)[:200], "wall_s": round(time.time() - t0, 2)}
    results[idx] = rec
    print("P84DECODE_REQ %s" % json.dumps(rec), flush=True)


def main():
    leg = sys.argv[1] if len(sys.argv) > 1 else "leg"
    results = {}
    ths = [threading.Thread(target=one_req, args=(i, results))
           for i in range(N_CONVS)]
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.time() - t0
    ok = [r for r in results.values() if r["status"] == "ok"]
    tot_c = sum(r.get("completion_tokens", 0) for r in ok)
    with open("/root/build/p84_stage/%s_decode.jsonl" % leg, "w") as fh:
        for i in sorted(results):
            fh.write(json.dumps(results[i]) + "\n")

    time.sleep(2)
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=30) as resp:
            metrics = resp.read().decode()
    except Exception:                                     # noqa: BLE001
        metrics = ""
    with open("/root/build/p84_stage/%s_metrics.txt" % leg, "w") as fh:
        fh.write(metrics)
    tpot = [ln for ln in metrics.splitlines()
            if "time_per_output" in ln and not ln.startswith("#")]
    spec = [ln for ln in metrics.splitlines()
            if "spec_decode" in ln and not ln.startswith("#")]
    print("P84DECODE_DONE leg=%s ok=%d/%d wall=%.1fs agg_ctok=%d "
          "agg_tok_s=%.1f" % (leg, len(ok), N_CONVS, wall, tot_c,
                              tot_c / wall if wall > 0 else 0), flush=True)
    for ln in tpot[:12]:
        print("TPOT_METRIC %s" % ln, flush=True)
    for ln in spec[:8]:
        print("SPEC_METRIC %s" % ln, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
