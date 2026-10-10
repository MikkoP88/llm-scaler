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
NONCE = "Nonce 55923. "
P2215Z = (
    NONCE + "Context block follows. " + WORD * 200
    + "End of context. The last Greek letter mentioned above is"
)
P3320Z = (
    NONCE + "Context block follows. " + WORD * 300
    + "End of context. The last Greek letter mentioned above is"
)
# Style control: same 3320 shape, different nonce -> full-miss fresh answer.
NONCE_W = "Nonce 8837. "
P3320W = (
    NONCE_W + "Context block follows. " + WORD * 300
    + "End of context. The last Greek letter mentioned above is"
)

# Connector ON (no lmc_off sentinel). Fresh nonce -> empty lmcache + empty
# GPU block-hash registry on this boot.
# A1Z banks chunks 0,1 (GPU + lmcache). T2AZ: local 1024 + external 1024 ->
# gated async load of chunk 1, promotion, straddle pass [2048,3329).
# THE REPRO: does T2AZ emit EMPTY (EOS@1) like T2a did twice before?
a1z = call("A1Z-2215-warm", P2215Z, 24)
time.sleep(3)
t2az = call("T2AZ-3320-gated-straddle", P3320Z, 24)
time.sleep(4)
t2bz = call("T2BZ-3320-load3072", P3320Z, 24)
time.sleep(3)
t2a2z = call("T2A2Z-3320-repeat", P3320Z, 24)
time.sleep(3)
pw = call("PW-3320-fresh-style", P3320W, 24)
print("PROBE7_DONE", flush=True)
