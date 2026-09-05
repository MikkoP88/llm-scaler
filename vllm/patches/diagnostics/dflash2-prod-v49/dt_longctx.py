# dt_longctx.py — long-context A/B bench (8k/16k/32k) for dflash2 vs mtp lanes.
# Same measurement path as dt_cargame.py (streaming, usage chunk for exact
# token counts, reasoning_content counted) but with a sized filler document
# before the car-game task. Reports prompt_tokens from the server usage chunk
# so the actual context length is measured, not estimated.
# Usage: dt_longctx.py single|concurrent 8k|16k|32k [max_tokens]
import hashlib
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
FILLER_WORDS = (
    "The industrial history of refrigeration begins with the harvesting of "
    "natural ice and the early experiments on the liquefaction of gases. "
    "Engineers studied the absorption of heat by expanding fluids and built "
    "successive generations of compression machines that transported "
    "perishable goods across continents and changed the diet of cities. "
    "Every chapter of this story couples a physical discovery with a "
    "practical machine, and every machine with a market that rewarded "
    "reliability over elegance, which is why the winning designs are rarely "
    "the beautiful ones. "
)


def build_prompt(target_tokens):
    # ~1.35 tokens per word for this text; calibrate by usage feedback.
    words = int(target_tokens / 1.35)
    part = FILLER_WORDS
    reps = max(1, words // 40)
    filler = part * reps
    return (
        f"Below is a long reference document.\n\n{filler}\n"
        "End of document. Now, ignoring the document above, write a html "
        "car game."
    )


def chat(tag, prompt, max_tokens=512):
    payload = {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
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
        with urllib.request.urlopen(req, timeout=3600) as r:
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
    except Exception as e:
        print(f"[{tag}] ERROR {type(e).__name__}: {e}", flush=True)
        return
    full = "".join(parts)
    wall = time.time() - t0
    ctoks = (usage or {}).get("completion_tokens") or n
    ptoks = (usage or {}).get("prompt_tokens") or -1
    ttft_s = ttft if ttft is not None else -1.0
    dec = wall - (ttft_s if ttft_s > 0 else 0)
    dps = ctoks / max(dec, 1e-9)
    print(
        f"[{tag}] OK wall={wall:.2f}s ttft={ttft_s:.2f}s prompt_toks={ptoks} "
        f"toks={ctoks} steps={n} decode={dps:.1f} tok/s "
        f"sha={hashlib.sha256(full.encode()).hexdigest()[:12]} "
        f"head={full[:60]!r}",
        flush=True,
    )


mode = sys.argv[1] if len(sys.argv) > 1 else "single"
size = (sys.argv[2] if len(sys.argv) > 2 else "8k").lower()
mt = int(sys.argv[3]) if len(sys.argv) > 3 else 512
target = int(size.rstrip("k")) * 1024
if mode == "single":
    chat(f"lc{size}", build_prompt(target), mt)
elif mode == "concurrent":
    # different filler seeds -> no prefix-cache sharing between the two
    p1 = build_prompt(target)
    p2 = build_prompt(target) + " "  # distinct content ends different
    ts = [
        threading.Thread(target=chat, args=(f"lc{size}-a", p1, mt)),
        threading.Thread(target=chat, args=(f"lc{size}-b", p2, mt)),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
else:
    print("modes: single | concurrent", file=sys.stderr)
