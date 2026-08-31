#!/usr/bin/env python3
"""wedge_repro variant: UNIQUE prompt per iteration (defeats prefix-cache
reuse between iterations). Discriminates iter-2 wedge trigger:
  still wedges ~480 chunks -> position/length-dependent (KV boundary);
  clean                     -> cached-decode/prefix-cache path involved.
Same 90 s watchdog semantics as wedge_repro.py (exit 7 = WEDGE)."""
import json, socket, sys, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WATCHDOG_S = 90.0


def one_iteration(i):
    prompt = f"Write a html car game. Variant {i}-{int(time.time())}: "
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=WATCHDOG_S)
    body = json.dumps({
        "model": "qwen3.8-27b-fp8", "prompt": prompt, "max_tokens": 4096,
        "temperature": 0.3, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1.0, "stream": True})
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    chunks = 0
    last = t0
    try:
        for line in r:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunks += 1
                last = time.perf_counter()
        wall = time.perf_counter() - t0
        c.close()
        return ("clean", chunks, wall)
    except (socket.timeout, TimeoutError):
        c.close()
        return ("wedge", chunks, time.perf_counter() - last)


def main():
    print(f"wedge_repro_u: {N_ITER} UNIQUE-prompt canonical iterations",
          flush=True)
    for i in range(1, N_ITER + 1):
        status, chunks, wall = one_iteration(i)
        if status == "clean":
            print(f"iter {i}: CLEAN chunks={chunks} wall={wall:.1f}s", flush=True)
        else:
            print(f"iter {i}: WEDGE after {chunks} chunks", flush=True)
            print("WEDGE DETECTED", flush=True)
            sys.exit(7)
    print("ALL CLEAN", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
