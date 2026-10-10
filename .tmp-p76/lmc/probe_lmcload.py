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
# Fresh nonce every run: guarantees a cold LMCache + cold native start.
NONCE = "Nonce %d. " % int(time.time() * 1000 % 100000)
HEAD = NONCE + "Context block follows. "
TAIL = "End of context. The last Greek letter mentioned above is"
P3320 = HEAD + WORD * 300 + TAIL
# Partial-prefix safety leg: strict prefix of P3320's body (~1870 tok) with a
# different question. Attn chunks hit ~1536 but only the deepest (~3072)
# mamba boundary is banked -> mamba_hit=0 -> joint 0 -> must fully compute.
PSHORT = HEAD + WORD * 180 + (
    "End of context. How many Greek letters were listed in total? "
    "Answer with a number."
)

# Prefix caching is OFF for this boot: every warm resend must exercise the
# FULL external load (attn KV chunks + mamba boundary seed).
a = call("A-cold-compute", P3320, 24)
time.sleep(5)
b = call("B-warm-load", P3320, 24)
time.sleep(3)
c = call("C-warm-load2", P3320, 24)
time.sleep(3)
d = call("D-partial-safety", PSHORT, 24)
print("A==B", a == b, "B==C", b == c, flush=True)
print("LMCLOAD_DONE", flush=True)
