# dt_cargame.py — real-life chat client for the crash repro + car-game bench.
# Uses /v1/chat/completions with sampling params intentionally OMITTED so the
# server applies the model's generation_config defaults (the user's real-life
# path: temp 0.7 / top_p 0.95 / top_k 20). Counts reasoning_content deltas too.
# Usage: dt_cargame.py [single|concurrent|bench|mixed] [max_tokens]
import hashlib
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = "Write a html car game."
PROMPT2 = "Explain how a refrigerator works step by step."


def chat(tag, max_tokens=512, stream=True, prompt=None):
    payload = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": prompt or PROMPT}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if stream:
        # llm-scaler v47b bench fix: request the usage chunk. Without it the
        # token count fell back to n = number of SSE delta events, which under
        # spec decode is one event per ENGINE STEP (~1/yield of the true token
        # count) — every streaming tok/s was understated ~2.5x.
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    parts = []
    n = 0
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            if stream:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    if j.get("usage"):
                        usage = j["usage"]
                    _ch = j.get("choices") or [{}]
                    d = _ch[0].get("delta", {})
                    c = (
                        (d.get("content") or "")
                        + (d.get("reasoning_content") or "")
                        + (d.get("reasoning") or "")
                    )
                    if c:
                        if ttft is None:
                            ttft = time.time() - t0
                        parts.append(c)
                        n += 1
            else:
                j = json.loads(r.read())
                usage = j.get("usage")
                msg = j["choices"][0]["message"]
                full = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
                parts.append(full)
                ttft = -1.0
    except Exception as e:
        print(f"[{tag}] ERROR {type(e).__name__}: {e}", flush=True)
        return
    full = "".join(parts)
    wall = time.time() - t0
    # v47b: exact token count from the server usage chunk; n (delta events
    # = engine steps) kept only as a last-resort fallback.
    ctoks = (usage or {}).get("completion_tokens") or n
    ttft_s = ttft if ttft is not None else -1.0
    dec = wall - (ttft_s if ttft_s > 0 else 0)
    dps = ctoks / max(dec, 1e-9)
    print(
        f"[{tag}] OK wall={wall:.2f}s ttft={ttft_s:.2f}s toks={ctoks} "
        f"steps={n} decode={dps:.1f} tok/s sha={hashlib.sha256(full.encode()).hexdigest()[:12]} "
        f"head={full[:70]!r}",
        flush=True,
    )


mode = sys.argv[1] if len(sys.argv) > 1 else "single"
mt = int(sys.argv[2]) if len(sys.argv) > 2 else 512
if mode == "single":
    chat("single", mt)
elif mode == "concurrent":
    ts = [threading.Thread(target=chat, args=(f"conc{i}", mt)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
elif mode == "concurrent2":
    # different prompts -> NO prefix-cache block sharing between the two reqs
    ts = [
        threading.Thread(target=chat, args=("cargame", mt, True, PROMPT)),
        threading.Thread(target=chat, args=("fridge", mt, True, PROMPT2)),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
elif mode == "bench":
    chat("bench", mt, stream=False)
elif mode == "mixed":
    # single stream first, second request fired mid-decode (crash-2 repro)
    t = threading.Thread(target=chat, args=("main", mt))
    t.start()
    time.sleep(6)
    chat("late2nd", mt)
    t.join()
else:
    print("modes: single | concurrent | concurrent2 | bench | mixed", file=sys.stderr)
