"""Fox-filler 65k wedge repro (v27 session): the HISTORICAL probe filler
(2048x... here 4066x identical sentence = ~65k @16.0 tok/rep) that measured
1/2 wedge rates under k4+VIA. A/B against deep_repro.py (varied numbered
filler) to test whether the wedge requires the degenerate token
distribution (KNOWN_ISSUES #13 collapse -> extreme drafter accept/reject
patterns) rather than depth alone.
"""
import json, socket, sys, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128
WATCHDOG_S = 120.0
UNIT = "The quick brown fox jumps over the lazy dog while the engine warms up. "
PROMPT = UNIT * 4066 + "\nQuestion: Write a html car game."  # ~65k tokens


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
    print(f"fox_repro: {N_ITER} x {MAXTOK}-token iterations @~65k "
          f"(FOX filler), watchdog {WATCHDOG_S}s", flush=True)
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
