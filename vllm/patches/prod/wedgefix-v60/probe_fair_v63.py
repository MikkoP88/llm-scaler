#!/usr/bin/env python3
"""v63 fairness probe: decode starvation during a big cold prefill.

Thread A: background chat decode stream (max_tokens 512) with per-token
timestamps. Main: after the decode stream is in flight, send ONE big fresh
(~106k tok) chat prefill (max_tokens 8) and time it. Reports decode tok/s
before / during / after the prefill window plus gap stats.

Usage: python3 probe_fair_v63.py [base] [model] [seed] [n_words_big]
  base  default http://127.0.0.1:8000   model default qwen3.8-27b-fp8
"""
import json
import sys
import threading
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 101
NWORDS_BIG = int(sys.argv[4]) if len(sys.argv) > 4 else 34000  # ~106k tokens

WORDS = ("amber birch cedar dogwood elm fir ginkgo hawthorn ironwood juniper "
         "kauri larch maple nigra oak pine quince rowan spruce teak ulmus "
         "viburnum walnut yew zebrawood eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen twenty thirty forty fifty").split()


def mk_text(n_words, seed=0):
    out = []
    i = seed
    while len(out) < n_words:
        out.append(WORDS[i % len(WORDS)])
        out.append(str(i))
        i += 1
    return " ".join(out[:n_words])


def post_stream(path, body, on_token=None, max_wait=420):
    """SSE post; returns (code, ttft, total). on_token(t_rel) per content delta."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={
        "Content-Type": "application/json", "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    ttft = None
    code = -1
    try:
        r = urllib.request.urlopen(req, timeout=max_wait)
        code = r.status
        for raw in r:
            if not raw.strip() or raw.startswith(b":"):
                continue
            # content delta lines: {"delta":{"content":...}} — the first
            # chunk is {"delta":{"role":...}} and must not count.
            if b'"delta"' not in raw or b'"role"' in raw:
                continue
            t = time.time() - t0
            if ttft is None:
                ttft = t
            if on_token is not None:
                on_token(t)
        return code, ttft, time.time() - t0, None
    except Exception as e:
        return code, ttft, time.time() - t0, str(e)[:200]


def main():
    print(f"FAIR_V63 base={BASE} model={MODEL} seed={SEED} nwords={NWORDS_BIG}",
          flush=True)

    # Background decode stream.
    stamps = []
    lock = threading.Lock()
    dec_done = threading.Event()

    def decoder():
        body = {"model": MODEL, "stream": True, "max_tokens": 4096,
                "temperature": 0.2,
                "messages": [{"role": "user", "content":
                    "Count from one upward, one number per line, no other text. "
                    "Keep counting until stopped."}]}
        def on_tok(t):
            with lock:
                stamps.append(time.time())
        code, ttft, total, err = post_stream("/v1/chat/completions", body, on_tok)
        with lock:
            dec_done.code = code
            dec_done.err = err
        dec_done.set()

    th = threading.Thread(target=decoder, daemon=True)
    t_dec_start = time.time()
    th.start()
    time.sleep(6.0)  # let decode reach steady state

    # Big cold prefill (fresh text => no prefix reuse).
    big = mk_text(NWORDS_BIG, seed=SEED)
    big_body = {"model": MODEL, "stream": True, "max_tokens": 8,
                "messages": [{"role": "user", "content": big +
                    "\nFinal question: reply with the single word: ok"}]}
    t_pf0 = time.time()
    code, ttft, total, err = post_stream("/v1/chat/completions", big_body)
    t_pf1 = time.time()
    print(f"FAIR_V63 big_seed{SEED} code={code} ttft={round(ttft or -1, 2)} "
          f"total={round(total, 2)} err={err}", flush=True)

    time.sleep(8.0)  # capture post-prefill recovery
    with lock:
        snap = list(stamps)
    # Decoder may still stream; stop it by closing? urllib has no clean cancel;
    # just wait a bounded time for finish.
    dec_done.wait(timeout=600)

    with lock:
        allst = list(stamps)
    d = dec_done.err
    pre = [t for t in allst if t < t_pf0]
    dur = [t for t in allst if t_pf0 <= t <= t_pf1]
    post = [t for t in allst if t > t_pf1]
    w_dur = max(t_pf1 - t_pf0, 1e-9)
    gaps = [round(b - a, 2) for a, b in zip(dur, dur[1:])] or [0.0]
    def rate(ts, t0, t1):
        return round(len(ts) / max(t1 - t0, 1e-9), 3)
    print(f"FAIR_V63 decode n_tok={len(allst)} dec_err={d}", flush=True)
    print(f"FAIR_V63 rate pre={rate(pre, t_dec_start, t_pf0)}/s "
          f"during={rate(dur, t_pf0, t_pf1)}/s (window {round(w_dur,1)}s, "
          f"n={len(dur)}) post={rate(post, t_pf1, allst[-1] if allst else t_pf1)}/s",
          flush=True)
    print(f"FAIR_V63 gaps_during min={min(gaps)} max={max(gaps)} "
          f"last10={gaps[-10:]}", flush=True)
    print("FAIR_V63_DONE", flush=True)


if __name__ == "__main__":
    main()
