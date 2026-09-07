#!/bin/bash
# master127.sh — v127-audit full matrix on llm-scaler-exp:v1.2.5 (2026-09-07).
# 4 user spec configs, exact user flags (0.9/262144), + post phase on the
# final live lane (mtp4_tq4): 16k N=3 distinct-prompt reruns (v53 16k-dip
# classification) + D7 cold-cold deep pair (two DIFFERENT 128k prompts
# back-to-back) + memory snapshot.
set -u
mkdir -p /root/build/bench127
M=/root/build/bench127/master.log
echo "MASTER127_START $(date)" | tee -a "$M"

# Lane 1: dflash k7 + fp8_e4m3  (user baseline #1)
bash /root/build/run_lane127.sh df7_fp8 fp8_e4m3 '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' >> "$M" 2>&1
# Lane 2: dflash k7 + turboquant_4bit_nc (user baseline #2)
bash /root/build/run_lane127.sh df7_tq4 turboquant_4bit_nc '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7}' >> "$M" 2>&1
# Lane 3: mtp k4 + fp8_e4m3 (user baseline #3)
bash /root/build/run_lane127.sh mtp4_fp8 fp8_e4m3 '{"method":"mtp","num_speculative_tokens":4}' >> "$M" 2>&1
# Lane 4 (LAST — post phase runs on its live container): mtp k4 + tq4nc
bash /root/build/run_lane127.sh mtp4_tq4 turboquant_4bit_nc '{"method":"mtp","num_speculative_tokens":4}' >> "$M" 2>&1

P=/root/build/bench127/post.log
echo "POST_START $(date)" | tee -a "$P"

# P1: 16k N=3 with DISTINCT prompts (v53 observed 46.6->38.5 single-draw;
# classify noise vs systematic). ~16.1k tok each, varied seeds.
python3 - <<'PYEOF' >> "$P" 2>&1
import json, random, time, urllib.request
WORDS = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta",
         "iota","kappa","lambda","mu","nu","xi","omicron","pi"]
for seed in (1, 2, 3):
    random.seed(seed)
    filler = " ".join(random.choice(WORDS) for _ in range(13600))
    body = {"model": "qwen3.8-27b-fp8", "prompt": "Summarize the following document.\n" + filler,
            "max_tokens": 256, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://localhost:8000/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ctoks = 0; ptok = 0
    r = urllib.request.urlopen(req, timeout=600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "): continue
        d = line[6:]
        if d == "[DONE]": break
        try: j = json.loads(d)
        except Exception: continue
        ch = (j.get("choices") or [{}])[0]
        if ttft is None and (ch.get("text") or ""): ttft = time.time() - t0
        ctoks += len(ch.get("text") or "")
        u = j.get("usage")
        if u: ptok = u.get("prompt_tokens", 0)
    wall = time.time() - t0
    print(f"CTXSCAN2 p16k_n3 seed={seed}: ptok={ptok} ctok={ctoks} ttft={ttft:.1f}s wall={wall:.1f}s decode_tps={256/max(wall-ttft,1e-9):.1f}")
PYEOF

# P2: D7 cold-cold — two DIFFERENT 128k-class prompts back-to-back (no
# prefix-cache overlap): first-vs-second decode_tps tells whether the
# documented 4.2->7.3 / cold-vs-warm asymmetry is a first-request property
# (JIT/pool state) or a per-prompt property. Sizes 114000/116000 words
# (~134.9k/137.3k tok), distinct from lane-deep's 115000 (seed differs).
python3 - <<'PYEOF' >> "$P" 2>&1
import json, random, time, urllib.request
WORDS = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta",
         "iota","kappa","lambda","mu","nu","xi","omicron","pi"]
for seed, nwords in ((71, 114000), (72, 116000)):
    random.seed(seed)
    filler = " ".join(random.choice(WORDS) for _ in range(nwords))
    body = {"model": "qwen3.8-27b-fp8", "prompt": "Summarize the following document.\n" + filler,
            "max_tokens": 256, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://localhost:8000/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ctoks = 0; ptok = 0
    r = urllib.request.urlopen(req, timeout=2400)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: "): continue
        d = line[6:]
        if d == "[DONE]": break
        try: j = json.loads(d)
        except Exception: continue
        ch = (j.get("choices") or [{}])[0]
        if ttft is None and (ch.get("text") or ""): ttft = time.time() - t0
        ctoks += len(ch.get("text") or "")
        u = j.get("usage")
        if u: ptok = u.get("prompt_tokens", 0)
    wall = time.time() - t0
    print(f"CTXSCAN2 d7coldcold seed={seed} nwords={nwords}: ptok={ptok} ctok={ctoks} ttft={ttft:.1f}s prefill_tps={ptok/max(ttft,1e-9):.0f} wall={wall:.1f}s decode_tps={256/max(wall-ttft,1e-9):.1f}")
PYEOF

# P3: memory snapshot on the live lane
{ echo '--- xpu-smi ---'; xpu-smi dump -m 2>/dev/null | head -40; } >> "$P" 2>&1 || true

echo "MASTER127_DONE $(date)" | tee -a "$M" "$P"
