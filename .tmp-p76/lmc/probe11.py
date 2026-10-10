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
P2215 = HEAD + WORD * 200 + TAIL
P3320 = HEAD + WORD * 300 + TAIL

# probe11: engine already carries the probe10 lineage (A1 fresh writer +
# T2w/T2W2/L280/L290/L400/P2215c resumed consumers, each with decodes).
# Consume it FURTHER: if accumulated rewrites of the shared seed page
# poison full-page resumes, these legs degrade now.
t1 = call("T2w-again-lineage5", P3320, 24)
time.sleep(3)
p1 = call("P2215-ctrl-lineage6", P2215, 24)
time.sleep(3)
t2 = call("T2w-again2-lineage7", P3320, 24)
print("P==", p1 == t1 == t2, flush=True)
print("PROBE11_DONE", flush=True)
