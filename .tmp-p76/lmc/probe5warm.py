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
P2215 = (
    "Context block follows. " + WORD * 200 + "End of context. The last Greek letter mentioned above is"
)
P3320 = (
    "Context block follows. " + WORD * 300 + "End of context. The last Greek letter mentioned above is"
)

# Warm: banks chunks 0,1 (GPU + lmcache) and a real chunk-2 compute.
a1 = call("A1-2215-warm", P2215, 24)
time.sleep(3)
t2a = call("T2a-3320-compute", P3320, 24)
time.sleep(4)
# Load leg: gated lmcache load of chunk 2 on the same warm engine.
t2b = call("T2b-3320-load", P3320, 24)
print("T2a==T2b", t2a == t2b, flush=True)
print("PROBE5WARM_DONE", flush=True)
