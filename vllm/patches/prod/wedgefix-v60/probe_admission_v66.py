#!/usr/bin/env python3
# probe_admission_v66.py — reproduce/verify waiting-admission starvation.
# Mirror of the CC production shape: 3 long-decode sessions (small prompts,
# big max_tokens) hold the decode phase; then NEW clients arrive carrying
# BIG prompts (CC-style context, default ~12k tokens) and we measure each
# one's time-to-first-token (TTFT, streaming, any delta counts — thinking
# tokens count). On a starving posture the engine log shows
# "Running: 2/3, Waiting: N, Avg prompt throughput: 0.0 tokens/s" and the
# client TTFT climbs to minutes. The caller greps the serve log for peak
# Waiting + V66_FAIRFIX_ACTIVE after the run.
# usage: python3 probe_admission_v66.py <base> <model> <seed> [client_words]
import json
import random
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 661
CWORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 4000

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega tensor "
         "matrix kernel compiler runtime driver device memory lane queue").split()


def mk_text(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


def decode_session(idx, results):
    payload = {
        "model": MODEL, "max_tokens": 4000, "temperature": 0.7,
        "messages": [{"role": "user",
                      "content": "Write an extremely long numbered list "
                                 "starting at 1, one number per line, and "
                                 "do not stop until you must. %s"
                                 % mk_text(30, SEED + idx)}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            d = json.loads(resp.read())
        results[idx] = (time.time() - t0, d["choices"][0].get("finish_reason"),
                        (d.get("usage") or {}).get("completion_tokens"))
    except Exception as e:  # noqa: BLE001
        results[idx] = (time.time() - t0, "ERR:%s" % e, 0)


def ttft_client(tag, words, max_tokens=600):
    payload = {
        "model": MODEL, "max_tokens": max_tokens, "temperature": 0.7,
        "stream": True,
        "messages": [{"role": "user",
                      "content": mk_text(words, SEED + 500 + hash(tag) % 97)
                      + "\nSummarize this text in one sentence."}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            if b"[DONE]" in line:
                break
            try:
                j = json.loads(line[6:])
            except Exception:  # noqa: BLE001
                continue
            ch = (j.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if (delta.get("content") or delta.get("reasoning_content")
                    or delta.get("reasoning")):
                ttft = time.time() - t0
                # drain the rest (keep server-side load realistic)
    print("ADMIT %s ttft=%s words=%d" %
          (tag, ("%.2fs" % ttft) if ttft is not None else "NONE", words),
          flush=True)
    return ttft


def main():
    results = {}
    ts = [threading.Thread(target=decode_session, args=(i, results))
          for i in range(3)]
    for t in ts:
        t.start()
    print("3 decode sessions started (max_tokens 4000); settling 15 s...",
          flush=True)
    time.sleep(15)
    ttfts = []
    for k in range(3):
        ttfts.append(ttft_client("c%d" % k, CWORDS))
        time.sleep(10)
    for t in ts:
        t.join(timeout=120)
    print("DECODE_SESSIONS %s" %
          {k: ("%.1fs %s ctok=%s" % v) for k, v in results.items()}, flush=True)
    ok = [t for t in ttfts if t is not None]
    print("ADMISSION ttfts=%s" %
          [("%.2f" % t) if t is not None else "NONE" for t in ttfts], flush=True)
    if len(ok) == len(ttfts) and ok:
        mx = max(ok)
        print("ADMIT_MAX_TTFT %.2f" % mx, flush=True)
        print("ADMISSION_DONE verdict=%s" % ("PASS" if mx < 30.0 else "STARVED"))
    else:
        print("ADMISSION_DONE verdict=FAIL (missing ttfts: %s)"
              % [i for i, t in enumerate(ttfts) if t is None], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
