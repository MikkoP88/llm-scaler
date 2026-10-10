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

# Pure-local leg (lmc_off sentinel present): fork's own prefix cache +
# PRE-COPY mamba resume, connector reports zero.
t2c = call("T2c-3320-purelocal", P3320, 24)
time.sleep(3)
# A2 anomaly recheck on the same boot.
a2 = call("A2-2215-purelocal", P2215, 24)
print("PROBE5LOCAL_DONE", flush=True)
