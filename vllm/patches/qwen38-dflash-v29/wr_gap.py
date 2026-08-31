#!/usr/bin/env python3
"""wedge_repro gap variant: SLEEP between iterations (arg2, default 180s).
Tests whether the iter-2 wedge hazard decays with idle time between runs."""
import json, socket, sys, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 3
GAP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
WATCHDOG_S = 90.0

def one_iteration(i):
    prompt = f"Write a html car game. Gap run {i}-{int(time.time())}: "
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=WATCHDOG_S)
    body = json.dumps({"model": "qwen3.8-27b-fp8", "prompt": prompt,
                       "max_tokens": 4096, "temperature": 0.3, "top_k": 20,
                       "top_p": 0.95, "min_p": 0, "presence_penalty": 0,
                       "repetition_penalty": 1.0, "stream": True})
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    chunks = 0
    try:
        for line in r:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunks += 1
        c.close()
        return ("clean", chunks, time.perf_counter() - t0)
    except (socket.timeout, TimeoutError):
        c.close()
        return ("wedge", chunks, 0.0)

print(f"wr_gap: {N_ITER} iters, gap {GAP_S}s between", flush=True)
for i in range(1, N_ITER + 1):
    if i > 1:
        print(f"sleeping {GAP_S:.0f}s...", flush=True)
        time.sleep(GAP_S)
    status, chunks, wall = one_iteration(i)
    if status == "clean":
        print(f"iter {i}: CLEAN chunks={chunks} wall={wall:.1f}s", flush=True)
    else:
        print(f"iter {i}: WEDGE after {chunks} chunks", flush=True)
        print("WEDGE DETECTED", flush=True)
        sys.exit(7)
print("ALL CLEAN", flush=True)
