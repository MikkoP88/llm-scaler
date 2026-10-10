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
HEAD = "Context block follows. "
TAIL = "End of context. The last Greek letter mentioned above is"


def body(n):
    return HEAD + WORD * n + TAIL


# Length-scaled PURE-LOCAL nonce-less legs (lmc_off). P2215/P3320 chunks are
# GPU-registered already; each longer prompt resumes locally at 3072 and
# extends the SAME periodic body further. If GDN state degenerates with
# periodic length, these break with zero LMCache involvement.
l4437 = call("L4437-periodic-purelocal", body(400), 24)
time.sleep(3)
l5542 = call("L5542-periodic-purelocal", body(500), 24)
time.sleep(3)
l6647 = call("L6647-periodic-purelocal", body(600), 24)
print("PROBE8_DONE", flush=True)
