#!/usr/bin/env python3
"""v63 JIT warmup probe: one big cold prefill + sampled decode.

Compiles the kernel_unified_attention / reduce_segments / xpu topk_topp
variants at real CC shapes (first-big-turn JIT spikes: 3-5 s at the
prefill transition). Run ONCE per fresh container/image; stamp
.v63_jit_warmed afterwards. Torch JIT cache persists in container FS.

Usage: python3 probe_jitwarm_v63.py [base] [model] [seed]
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 411

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


def post(body, max_wait=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=max_wait)
        n = 0
        for _ in r:
            n += 1
        return r.status, round(time.time() - t0, 1), n, None
    except Exception as e:
        return -1, round(time.time() - t0, 1), 0, str(e)[:200]


def main():
    # 1) big cold prefill ~130k tokens, small sampled output (temperature on
    #    -> topk_topp path; stream off keeps it simple, max_tokens 4).
    big = mk_text(42000, seed=SEED)
    code, wall, _, err = post({"model": MODEL, "max_tokens": 4, "temperature": 0.7,
                               "messages": [{"role": "user", "content": big +
                                   "\nReply with one word: ok"}]})
    print(f"JITWARM big_seed{SEED} code={code} wall={wall}s err={err}", flush=True)
    ok1 = code == 200

    # 2) short sampled decode burst (decode-shape sampler variants).
    code2, wall2, _, err2 = post({"model": MODEL, "max_tokens": 32,
                                  "temperature": 0.7,
                                  "messages": [{"role": "user", "content":
                                      "Write a two-sentence note."}]})
    print(f"JITWARM decode code={code2} wall={wall2}s err={err2}", flush=True)

    print(f"JITWARM_{'OK' if ok1 and code2 == 200 else 'FAIL'}", flush=True)
    sys.exit(0 if ok1 and code2 == 200 else 1)


if __name__ == "__main__":
    main()
