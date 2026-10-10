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
        repr(text[:50]),
        flush=True,
    )
    return text


WORD = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
NONCE = "Nonce 4471. "
P2215N = (
    NONCE + "Context block follows. " + WORD * 200
    + "End of context. The last Greek letter mentioned above is"
)
P3320N = (
    NONCE + "Context block follows. " + WORD * 300
    + "End of context. The last Greek letter mentioned above is"
)

# Pure-local nonce pair (lmc_off): A1N banks chunks 0,1 fresh; T2N resumes
# at 2048 and straddles the 3072 boundary. No lmcache, no inherited poison.
a1n = call("A1N-2215-fresh", P2215N, 24)
time.sleep(3)
t2n = call("T2N-3320-straddle", P3320N, 24)
time.sleep(3)
t3n = call("T3N-3320-straddle2", P3320N, 24)
print("A1N==T2N", a1n == t2n, "T2N==T3N", t2n == t3n, flush=True)
print("PROBE6_DONE", flush=True)
