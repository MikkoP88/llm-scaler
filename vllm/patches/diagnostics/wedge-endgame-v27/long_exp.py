"""Long-exposure MTP-wedge probe: ~65k fox-filler ctx + ignore_eos +
2048 tokens = ~2000 decode steps per iteration in the degenerate token
distribution (the regime where historical wedges live), far beyond the
64-token historical probes. exit 7 on WEDGE (watchdog 120 s).
"""
import json, socket, sys, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
CTX = sys.argv[3] if len(sys.argv) > 3 else "65k"
WATCHDOG_S = 120.0
UNIT = "The quick brown fox jumps over the lazy dog while the engine warms up. "
REPS = {"32k": 2048, "65k": 4066, "131k": 8132}[CTX]
PROMPT = UNIT * REPS + "\nQuestion: Write a html car game."


def one_iteration(i):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=WATCHDOG_S)
    body = json.dumps({
        "model": "qwen3.8-27b-fp8", "prompt": PROMPT, "max_tokens": MAXTOK,
        "temperature": 0.3, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1.0,
        "ignore_eos": True, "stream": True})
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", body=body,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    chunks, last = 0, t0
    try:
        for line in r:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunks += 1
                last = time.perf_counter()
        return ("clean", chunks, time.perf_counter() - t0)
    except (socket.timeout, TimeoutError):
        return ("WEDGE", chunks, time.perf_counter() - t0,
                time.perf_counter() - last)


def main():
    print(f"long_exp: {N_ITER} x {MAXTOK} tokens @~{CTX} fox ctx, "
          f"ignore_eos, watchdog {WATCHDOG_S}s", flush=True)
    for i in range(1, N_ITER + 1):
        res = one_iteration(i)
        if res[0] == "clean":
            _, chunks, wall = res
            print(f"iter {i}: CLEAN chunks={chunks} wall={wall:.1f}s "
                  f"tok/s={chunks / max(wall, 1e-3):.1f}", flush=True)
        else:
            _, chunks, wall, stalled = res
            print(f"iter {i}: WEDGE after {chunks} chunks "
                  f"(wall={wall:.1f}s, stalled {stalled:.0f}s)", flush=True)
            sys.exit(7)
    print("ALL CLEAN", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
