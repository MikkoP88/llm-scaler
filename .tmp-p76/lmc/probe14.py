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
NONCE = "Nonce 77. "


def body(n):
    return NONCE + HEAD + WORD * n + TAIL


P2215V = body(200)
P3320V = body(300)

# probe14: L1 (seed-provenance) vs L2 (engine-wide accumulation).
# This engine already ran seeded full-page passes (T2w [2048,3072)
# healthy, L400 [3072,4096) NaN) — so L400V below is the Nth seeded
# full-page, not the first.
#  T2V    fresh P3320V: unseeded mega [0,~3200) writes states@1024/
#        2048/3072 UNSEEDED-written; short seeded tail. Expect healthy.
#  P2215V resume@2048 short from unseeded-written seed. Healthy control.
#  L400V  resume@3072: seed = UNSEEDED-written state@3072 page; pass
#        [3072,4096) seeded full-page. Healthy => L1 (only seeded-
#        written seeds are toxic). '!!!!' => L2 (engine-wide workspace
#        accumulation from any prior seeded full-page).
#  T2V2  P3320V again: resume@3072 short from T2V's mega-written page.
#  L500V  resume@4096: seed = L400V's pass output (seeded-written);
#        [4096,~5333) seeded full-page — data point both laws.
t2v = call("T2V-3320V-fresh-mega", P3320V, 24)
time.sleep(3)
p2v = call("P2215V-short-ctrl", P2215V, 24)
time.sleep(3)
l400v = call("L400V-DISCRIMINATOR", body(400), 24)
time.sleep(3)
t2v2 = call("T2V2-3320V-again", P3320V, 24)
time.sleep(3)
l500v = call("L500V-seeded-written-seed", body(500), 24)
print("PROBE14_DONE", flush=True)
