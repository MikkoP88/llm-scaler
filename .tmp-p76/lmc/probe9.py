import hashlib
import json
import time
import urllib.request


def call(tag, prompt, max_tokens):
    body = {
        "model": "qwen3.8-27b-fp8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    ch = out["choices"][0]
    text = ch.get("text") or ""
    print(
        tag,
        "wall",
        round(time.time() - t0, 2),
        "fin",
        ch.get("finish_reason"),
        "ctok",
        out.get("usage", {}).get("completion_tokens"),
        "md5",
        hashlib.md5(text.encode()).hexdigest()[:8],
        repr(text[:60]),
        flush=True,
    )
    return text


WORD = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
P2215 = (
    "Context block follows. " + WORD * 200
    + "End of context. The last Greek letter mentioned above is"
)
P3320 = (
    "Context block follows. " + WORD * 300
    + "End of context. The last Greek letter mentioned above is"
)

# THE GROUND-TRUTH TEST. First request on a fresh engine: nonce-less P3320
# computed fully fresh (no GPU registry, no CPU cache). If the true temp-0
# answer is EOS@1 (EMPTY), then T2a's EMPTY was CORRECT and the whole
# "corruption" reading collapses.
gt = call("GT-P3320-FRESH", P3320, 24)
time.sleep(3)
a1 = call("P2215-after", P2215, 24)
time.sleep(3)
warm = call("T2-P3320-warm", P3320, 24)
print("GT==WARM", gt == warm, flush=True)
print("PROBE9_DONE", flush=True)
