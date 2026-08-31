"""Fast MTP-wedge repro harness (v27 session, KNOWN_ISSUES #11 graphs-x-spec
residual).

Control config = spec k1 + VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1 at SHORT ctx:
the residual there is ~7e-3/step (a canonical wedged at token 141), so a
4096-token canonical wedges with p ~= 1 per iteration and the whole repro
completes in minutes. Loops canonical generations with a 90 s streaming
watchdog: no data chunk for 90 s while the request is open = WEDGE.

exit 0 = N iterations clean; exit 7 = WEDGE (details on stdout).
"""
import json, socket, sys, time, http.client

PROMPT = "Write a html car game."
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WATCHDOG_S = 90.0


def one_iteration(i):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=WATCHDOG_S)
    body = json.dumps({
        "model": "qwen3.8-27b-fp8", "prompt": PROMPT, "max_tokens": 4096,
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
        stalled = time.perf_counter() - last
        return ("WEDGE", chunks, time.perf_counter() - t0, stalled)


def main():
    print(f"wedge_repro: {N_ITER} canonical iterations, watchdog {WATCHDOG_S}s",
          flush=True)
    for i in range(1, N_ITER + 1):
        res = one_iteration(i)
        if res[0] == "clean":
            _, chunks, wall = res
            print(f"iter {i}: CLEAN chunks={chunks} wall={wall:.1f}s "
                  f"tok/s={chunks / wall:.1f}", flush=True)
        else:
            _, chunks, wall, stalled = res
            print(f"iter {i}: WEDGE after {chunks} chunks "
                  f"(wall={wall:.1f}s, stalled {stalled:.0f}s)", flush=True)
            sys.exit(7)
    print("ALL CLEAN", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
