"""Concurrent-stream wedge pressure (v27 session): 2 parallel clients
looping ~65k fox probes (the E2-bs2 historical rate-raiser: concurrency
activates tiny gathers, forces batch-size/capture-desc switches and
eager-collective interleave between replayed pieces). Watchdog 120 s.
exit 7 on first WEDGE from either stream.
"""
import json, socket, sys, threading, time, http.client

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
STREAMS = 2
WATCHDOG_S = 120.0
UNIT = "The quick brown fox jumps over the lazy dog while the engine warms up. "
PROMPTS = [UNIT * 4066 + "\nQuestion: Write a html car game.",
           UNIT * 2048 + "\nQuestion: Write a html car game."]  # 65k + 32k
result = {"wedge": None}
lock = threading.Lock()


def worker(tid):
    prompt = PROMPTS[tid % len(PROMPTS)]
    for i in range(1, N_ITER + 1):
        if result["wedge"]:
            return
        try:
            c = http.client.HTTPConnection("127.0.0.1", 8000,
                                           timeout=WATCHDOG_S)
            body = json.dumps({
                "model": "qwen3.8-27b-fp8", "prompt": prompt,
                "max_tokens": 128, "temperature": 0.3, "top_k": 20,
                "top_p": 0.95, "min_p": 0, "presence_penalty": 0,
                "repetition_penalty": 1.0, "ignore_eos": True,
                "stream": True})
            t0 = time.perf_counter()
            c.request("POST", "/v1/completions", body=body,
                      headers={"Content-Type": "application/json"})
            r = c.getresponse()
            chunks, last = 0, t0
            for line in r:
                if line.startswith(b"data: ") and \
                        line.strip() != b"data: [DONE]":
                    chunks += 1
                    last = time.perf_counter()
            print(f"[t{tid}] iter {i}: CLEAN chunks={chunks} "
                  f"wall={time.perf_counter()-t0:.1f}s", flush=True)
            c.close()
        except (socket.timeout, TimeoutError):
            with lock:
                if not result["wedge"]:
                    result["wedge"] = (tid, i, chunks,
                                       time.perf_counter() - last)
            return
        except Exception as e:
            print(f"[t{tid}] iter {i}: EXC {type(e).__name__}: {e}",
                  flush=True)
            time.sleep(2)


def main():
    print(f"conc_repro: {STREAMS} streams x {N_ITER} iters (65k+32k fox), "
          f"ignore_eos, watchdog {WATCHDOG_S}s", flush=True)
    ts = [threading.Thread(target=worker, args=(t,)) for t in range(STREAMS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if result["wedge"]:
        tid, it, chunks, stalled = result["wedge"]
        print(f"WEDGE stream={tid} iter={it} chunks={chunks} "
              f"stalled={stalled:.0f}s", flush=True)
        sys.exit(7)
    print("ALL CLEAN", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
