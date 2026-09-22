#!/usr/bin/env python3
"""v64 solo cold-TTFT probe: ONE big fresh chat prefill, nothing else
running. Measures pure chunked-prefill wall time (solo budget path).

Usage: python3 probe_solo_cold.py [base] [model] [seed] [n_words_big]
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 110
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


def main():
    print(f"SOLO_COLD base={BASE} model={MODEL} seed={SEED} nwords={NWORDS_BIG}",
          flush=True)
    big = mk_text(NWORDS_BIG, seed=SEED)
    body = {"model": MODEL, "stream": True, "max_tokens": 8,
            "messages": [{"role": "user", "content": big +
                "\nFinal question: reply with the single word: ok"}]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    ttft = None
    code = -1
    usage = None
    try:
        r = urllib.request.urlopen(req, timeout=420)
        code = r.status
        for raw in r:
            if not raw.strip() or raw.startswith(b":"):
                continue
            if b'"delta"' not in raw or b'"role"' in raw:
                continue
            if ttft is None:
                ttft = time.time() - t0
            if b'"prompt_tokens"' in raw:
                try:
                    j = json.loads(raw[6:] if raw.startswith(b"data: ") else raw)
                    usage = j.get("usage")
                except Exception:
                    pass
        print(f"SOLO_COLD seed{SEED} code={code} ttft={round(ttft or -1, 2)} "
              f"total={round(time.time() - t0, 2)} prompt_tokens="
              f"{(usage or {}).get('prompt_tokens')}", flush=True)
    except Exception as e:
        print(f"SOLO_COLD seed{SEED} code={code} ttft={round(ttft or -1, 2)} "
              f"total={round(time.time() - t0, 2)} err={str(e)[:200]}", flush=True)
    print("SOLO_COLD_DONE", flush=True)


if __name__ == "__main__":
    main()
