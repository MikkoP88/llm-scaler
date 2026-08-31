"""High-rate MTP-wedge repro (v27 session): spec k4 + VIA_ALLGATHER at ~65k
ctx — the arm that wedged 1/2 probes there. Varied numbered filler (KNOWN
ISSUES #13: identical-repetition fillers instant-EOS >=32k and would mask
the wedge). Each iteration = one ~65k prefill + 128-token decode window,
120 s streaming watchdog. exit 7 on WEDGE.
"""
import json, socket, sys, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128
WATCHDOG_S = 120.0
PROMPT = " ".join(
    f"Marker {i:05d}: the engine warms up while the fox watches the quiet harbor. "
    for i in range(3800)
) + "\nQuestion: Write a html car game."  # ~65k tokens


def one_iteration(i):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=WATCHDOG_S)
    body = json.dumps({
        "model": "qwen3.8-27b-fp8", "prompt": PROMPT, "max_tokens": MAXTOK,
        "temperature": 0.3, "top_k": 20, "top_p": 0.95, "min_p": 0,
        "presence_penalty": 0, "repetition_penalty": 1.0, "stream": True})
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
    print(f"deep_repro: {N_ITER} x {MAXTOK}-token iterations @~65k ctx, "
          f"watchdog {WATCHDOG_S}s", flush=True)
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
