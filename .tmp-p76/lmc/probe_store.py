import hashlib
import json
import time
import urllib.request

PROMPT = (
    "Context block follows. "
    + ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200)
    + "End of context. The last Greek letter mentioned above is"
)
BODY = {
    "model": "qwen3.8-27b-fp8",
    "prompt": PROMPT,
    "max_tokens": 24,
    "temperature": 0,
    "stream": False,
}


def one(tag: str) -> str:
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:8000/v1/completions",
        data=json.dumps(BODY).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    usage = out.get("usage", {})
    content = out["choices"][0].get("text") or ""
    print(
        tag,
        "wall",
        round(time.time() - t0, 2),
        "ptok",
        usage.get("prompt_tokens"),
        "md5",
        hashlib.md5(content.encode()).hexdigest()[:8],
        repr(content[:40]),
        flush=True,
    )
    return content


a = one("first")
time.sleep(3)
b = one("second")
time.sleep(3)
print("MATCH" if a == b else "MISMATCH", flush=True)
print("PROBE_DONE", flush=True)
