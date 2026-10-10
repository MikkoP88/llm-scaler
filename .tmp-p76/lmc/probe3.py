import hashlib
import json
import time
import urllib.request


def call(tag, prompt, max_tokens, logprobs=None):
    body = {
        "model": "qwen3.8-27b-fp8",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = logprobs
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
    lps = ""
    lp = ch.get("logprobs")
    if lp and lp.get("top_logprobs") and lp["top_logprobs"][0]:
        items = list(lp["top_logprobs"][0].items())[:6]
        lps = " top=" + ",".join(f"{k}:{v:.2f}" for k, v in items)
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
        lps,
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

# T1: pure-prefix check on the warm engine (full local GPU hit expected)
call("T1-2215-prefix-lp", P2215, 1, 5)
time.sleep(2)

# T2: fresh 3-chunk prompt pair -> L3b should take the split-hit load path
a = call("T2a-3320-fresh", P3320, 24)
time.sleep(3)
b = call("T2b-3320-load-lp", P3320, 24, 5)
print("T2", "MATCH" if a == b else "MISMATCH", flush=True)
print("PROBE3_DONE", flush=True)
