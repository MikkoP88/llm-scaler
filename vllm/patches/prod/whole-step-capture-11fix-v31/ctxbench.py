#!/usr/bin/env python3
# ctxbench.py <tag> [ctx_csv] [max_tokens] [conc]
# Decode-perf sweep vs context length against localhost:8000.
# Greedy, stream=True; TTFT (prefill+queue) and steady decode tok/s reported.
# conc=N runs N parallel requests per ctx point; reports aggregate and
# per-stream decode tok/s (last arg, default 1).
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

TAG = sys.argv[1] if len(sys.argv) > 1 else "bench"
CTXS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2048,8192,32768,65536").split(",")]
MAXTOK = int(sys.argv[3] if len(sys.argv) > 3 else 192)
CONC = int(sys.argv[4] if len(sys.argv) > 4 else 1)

PARA = ("The quick brown fox jumps over the lazy dog near the river bank while "
        "the morning mist rises slowly over the sleeping village and the "
        "fishermen prepare their nets for the long day ahead. ")

def build_prompt(target):
    # ~1 token per ~3.7 chars for this text; calibrate via /tokenize if up
    unit = PARA * 10
    try:
        req = urllib.request.Request("http://localhost:8000/tokenize",
            data=json.dumps({"prompt": unit}).encode(),
            headers={"Content-Type": "application/json"})
        n = len(json.load(urllib.request.urlopen(req, timeout=30))["tokens"])
        reps = max(1, round(target / n * 10))
        est = "calibrated"
    except Exception:
        reps = max(1, round(target * 3.7 / len(PARA)))
        est = "estimated"
    # bare repeated text makes greedy decode emit EOS instantly; force continuation
    return PARA * reps + "\n\nDetailed summary of the above text:\n", est

def model_id():
    j = json.load(urllib.request.urlopen(
        urllib.request.Request("http://localhost:8000/v1/models"), timeout=10))
    return j["data"][0]["id"]

MODEL = model_id()

def run_once(prompt):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": MAXTOK,
        "temperature": 0.0, "stream": True, "min_tokens": 32,
    }).encode()
    req = urllib.request.Request("http://localhost:8000/v1/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            try:
                j = json.loads(line[6:])
            except Exception:
                continue
            if j.get("choices") and j["choices"][0].get("text"):
                n += 1
                if ttft is None:
                    ttft = time.time() - t0
    t1 = time.time()
    return ttft, n, t1 - t0

print(f"== ctxbench {TAG} ctx={CTXS} max_tokens={MAXTOK} conc={CONC} ==")
for ctx in CTXS:
    prompt, est = build_prompt(ctx)
    t0 = time.time()
    def one(i):
        # unique header per request defeats block-level prefix cache sharing
        return run_once(f"Request {i} unique seed {i*7919}: " + prompt)
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        res = list(ex.map(one, range(CONC)))
    wall = time.time() - t0
    ok = [r for r in res if r[0] is not None]
    if not ok:
        print(f"ctx~{ctx:<6} ({est:10s}) NO-OUTPUT n={[r[1] for r in res]}")
        continue
    tot_tok = sum(r[1] for r in ok)
    agg = tot_tok / wall                      # aggregate tok/s incl. prefill
    per = [ (r[1] - 1) / (r[2] - r[0]) for r in ok if r[1] > 1 and r[2] > r[0] ]
    per_avg = sum(per) / len(per) if per else 0.0
    print(f"ctx~{ctx:<6} conc={CONC:<3} wall={wall:6.1f}s tokens={tot_tok:5d} "
          f"agg={agg:6.2f} tok/s per_stream={per_avg:6.2f} tok/s "
          f"ttft_max={max(r[0] for r in ok):6.1f}s")
print("CTXBENCH-DONE")
