import hashlib
import json
import time
import urllib.request

BASE = "http://localhost:8000/v1/completions"


def call(tag, prompt, max_tokens, quiet=False):
    body = {
        "model": "qwen3.8-27b-fp8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    ch = out["choices"][0]
    text = ch.get("text") or ""
    wall = round(time.time() - t0, 2)
    md5 = hashlib.md5(text.encode()).hexdigest()[:8]
    if not quiet or (quiet and "!" in text):
        print(
            tag, "wall", wall, "fin", ch.get("finish_reason"),
            "ctok", out.get("usage", {}).get("completion_tokens"),
            "md5", md5, repr(text[:40]), flush=True,
        )
    return text, wall


WORD = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
NONCE = "Nonce %d. " % int(time.time() * 1000 % 100000)
HEAD = NONCE + "Context block follows. "
TAIL = "End of context. The last Greek letter mentioned above is"
P3320 = HEAD + WORD * 300 + TAIL

# A: bank attn chunks + mamba boundary into LMCache CPU (and native GPU).
a_text, a_wall = call("A-bank", P3320, 24)
time.sleep(4)

# Fillers: 96 distinct ~3.8k-token prompts (~365k tokens total) to cycle the
# GPU KV pool and LRU-evict A's native blocks. CPU pool (60GB) keeps A warm.
for i in range(96):
    p = (
        "Filler %d-%d. Context block follows. " % (i, time.time() % 997)
        + WORD * 340
        + "End of context. The last Greek letter mentioned above is"
    )
    _, _w = call("filler%d" % i, p, 4, quiet=True)
    if i % 16 == 15:
        print("FILLERS_DONE", i + 1, flush=True)
    time.sleep(0.3)

time.sleep(3)
# A-again: native should be evicted; LMCache CPU hit must drive the full
# external load (attn KV chunks + mamba boundary seed).
a2_text, a2_wall = call("A2-reuse-after-evict", P3320, 24)
print("A==A2", a_text == a2_text, "walls", a_wall, a2_wall, flush=True)
print("EVICT_DONE", flush=True)
