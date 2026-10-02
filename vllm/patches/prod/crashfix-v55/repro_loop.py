#!/usr/bin/env python3
"""repro_loop.py — Phase C wedge trigger loop (p12 geometry, solo).

Sequential /v1/chat/completions: "Write a html car game", 4096 out,
crash sampling params (temp 0.9 / top_k 20 / top_p 0.95). Each request
crosses the 2048 mamba boundary — today's wedge trigger geometry.
Stops on: fence line in serve log, engine death, or N clean requests.
"""
import json
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def serve_log_grep(pat: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "exec", "lsv-test", "sh", "-c",
             f"grep -c '{pat}' /root/serve_full.log"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "?"


def one_request(i: int) -> str:
    body = json.dumps({
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user",
                      "content": f"Write a html car game (variant {i})"}],
        "max_tokens": 4096,
        "temperature": 0.9,
        "top_k": 20,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=420) as r:
            out = json.loads(r.read())
        usage = out.get("usage", {})
        ntok = usage.get("completion_tokens", -1)
        return f"req{i}: OK {ntok} toks in {time.time()-t0:.0f}s"
    except Exception as e:
        return f"req{i}: FAIL after {time.time()-t0:.0f}s: {type(e).__name__}: {e}"


for i in range(1, N + 1):
    line = one_request(i)
    fence = serve_log_grep("ASYNC-EVENT-STALL")
    print(f"[{time.strftime('%H:%M:%S')}] {line} (fence-hits={fence})",
          flush=True)
    if fence not in ("0", "?", ""):
        print(f"[{time.strftime('%H:%M:%S')}] WEDGE DETECTED at req {i} "
              f"(fence-hits={fence}) — STOPPING LOOP", flush=True)
        sys.exit(2)
    if "FAIL" in line:
        print(f"[{time.strftime('%H:%M:%S')}] REQUEST FAILED — checking "
              "engine", flush=True)
        time.sleep(5)
        fence = serve_log_grep("ASYNC-EVENT-STALL")
        if fence not in ("0", "?", ""):
            print("WEDGE CONFIRMED post-fail", flush=True)
            sys.exit(2)
print("LOOP_COMPLETE_NO_WEDGE", flush=True)
