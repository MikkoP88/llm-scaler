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


P2215 = body(200)
P3320 = body(300)
NONCE = "Nonce 55923. "
P2215Z = NONCE + P2215
P3320Z = NONCE + P3320

# probe10 (lmc_off, v134 clamp ACTIVE, finite-scan instrumentation):
#  A1        banks chunks 0,1 nonce-less fresh.
#  T2w       the known-broken shape: resume@2048 -> clamp ->
#            [2048,3072) full-page resumed + [3072,3315) tail. Expect '!!!!'.
#  T2W2      P3320 again: chunks 0,1,2 now registered -> resume@3072,
#            243-tok single pass seeded by T2w's full-page output state.
#            Healthy => pass-1 wrote a GOOD state@3072 (corruption is in
#            pass-2/decode consumption); '!!!!' => the state itself is
#            poisoned by the resumed full-page pass.
#  L280/L290 share [0,3072) -> resume@3072 short continuations (sched ~23
#            and ~133) from the same lineage state.
#  L400      resume@3072 -> clamp -> [3072,4096) ANOTHER full-page resumed
#            pass + tail: does a second full-page resumed pass break too?
#  P2215c    bit-exact control (expect be6f5f02).
#  A1Z/T2AZ  nonce-55923 pair: same mechanics, shifted phase — the known
#            healthy control (T2AZ resume@2048 clamped like T2w).
a1 = call("A1-2215-fresh", P2215, 24)
time.sleep(3)
t2w = call("T2w-3320-broken-shape", P3320, 24)
time.sleep(4)
t2w2 = call("T2W2-3320-resume3072", P3320, 24)
time.sleep(3)
l280 = call("L280-short-cont", body(280), 24)
time.sleep(3)
l290 = call("L290-mid-cont", body(290), 24)
time.sleep(3)
l400 = call("L400-second-fullpage", body(400), 24)
time.sleep(3)
p2c = call("P2215c-control", P2215, 24)
time.sleep(3)
a1z = call("A1Z-2215Z-fresh", P2215Z, 24)
time.sleep(3)
t2az = call("T2AZ-3320Z-same-mech", P3320Z, 24)
print("P2215c==A1", p2c == a1, flush=True)
print("PROBE10_DONE", flush=True)
