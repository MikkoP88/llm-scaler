#!/usr/bin/env python3
"""Raw SSE dump for one streaming chat completion — diagnose empty streams.

Usage: probe_raw_stream.py [base] [model] [words] [max_tokens] [contend]
  contend=1 -> run 2 background decode sessions first (admission conditions)
Dumps first RAW_STREAM_HEAD non-empty data lines, counts all data lines,
prints finish_reason + usage, and whether any delta carried content or
reasoning_content.
"""
import json
import random
import sys
import threading
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-fp8"
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 200
CONTEND = (sys.argv[5] if len(sys.argv) > 5 else "0") == "1"
HEAD = 14


def mk_text(n, seed=661):
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
             "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
             "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
             "omega", "quartz", "flint", "cedar", "birch", "maple", "rowan"]
    out = []
    for i in range(n):
        out.append("%d %s %s" % (i + 1, rng.choice(vocab), rng.choice(vocab)))
    return " ".join(out)


def decode_session():
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "stream": False,
        "messages": [{"role": "user", "content":
                      "Write an extremely long numbered list starting at 1 "
                      "about any topic you like. Keep going."}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print("[decode] ERR %r" % e, flush=True)


if CONTEND:
    ts = [threading.Thread(target=decode_session) for _ in range(2)]
    for t in ts:
        t.start()
    print("[contend] 2 decode sessions up; settling 15 s", flush=True)
    time.sleep(15)

body = {
    "model": MODEL,
    "max_tokens": MAXTOK,
    "stream": True,
    "messages": [
        {"role": "user", "content": mk_text(WORDS) + " Summarize this text in one sentence."}
    ],
}
req = urllib.request.Request(
    BASE + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
t0 = time.time()
n_data = 0
n_content = 0
n_reason = 0
head_lines = []
finish = None
usage = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        n_data += 1
        payload = line[5:].strip()
        if payload == "[DONE]":
            head_lines.append("[%5.1fs] data: [DONE]" % (time.time() - t0))
            continue
        try:
            obj = json.loads(payload)
        except Exception:  # noqa: BLE001
            head_lines.append("[%5.1fs] BADJSON %s" % (time.time() - t0, payload[:120]))
            continue
        if "error" in obj:
            head_lines.append("[%5.1fs] ERROR %s" % (time.time() - t0, payload[:200]))
            continue
        ch = obj.get("choices") or []
        d = ch[0].get("delta", {}) if ch else {}
        if d.get("content"):
            n_content += 1
        if d.get("reasoning_content"):
            n_reason += 1
        if ch and ch[0].get("finish_reason"):
            finish = ch[0]["finish_reason"]
        if obj.get("usage"):
            usage = obj["usage"]
        if len(head_lines) < HEAD:
            head_lines.append("[%5.1fs] %s" % (time.time() - t0, payload[:160]))

print("RAW_STREAM_RESULT data_lines=%d content_deltas=%d reasoning_deltas=%d "
      "finish=%s wall=%.1fs" % (n_data, n_content, n_reason, finish, time.time() - t0), flush=True)
if usage:
    print("RAW_STREAM_USAGE %s" % json.dumps(usage), flush=True)
for h in head_lines:
    print("RAW_HEAD %s" % h, flush=True)
print("RAW_STREAM_DONE contend=%d" % CONTEND, flush=True)
