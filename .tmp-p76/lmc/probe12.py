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

# probe12: EXACT probe9 order on a virgin boot (fresh registry), with the
# probe10 instrumentation (full-page hashes + finite scans) and v134 clamp.
# probe9: GT fresh (pass-2 = [2048,3315) fresh STRADDLE, unclamped) ->
# P2215-after -> T2-warm = '!!!!'. probe10 (no fresh P3320 straddle ever):
# healthy. If this reproduces '!!!!' with identical instrumentation, the
# fresh straddle pass (or its registration) poisons the lineage for later
# resumed full-page passes: H2. If healthy: timing/boot-state (H3).
gt = call("GT-P3320-FRESH", P3320, 24)
time.sleep(3)
a1 = call("P2215-after", P2215, 24)
time.sleep(3)
warm = call("T2-P3320-warm", P3320, 24)
time.sleep(3)
w2 = call("T2-P3320-warm2", P3320, 24)
print("GT==WARM", gt == warm, "GT==WARM2", gt == w2, flush=True)
print("PROBE12_DONE", flush=True)
