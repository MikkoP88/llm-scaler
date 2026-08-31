#!/usr/bin/env python3
"""Long-ctx canonical: 32k filler + 'Write a html car game.', streaming,
4096 max tokens, canonical sampling params. Same acceptance bar as
canonical.py (has_html + has_canvas + completes without client timeout)."""
import json, time, http.client

# Filler must be VARIED. 2048 identical repetitions of one sentence make the
# model emit EOS as its first token at ~32k ctx (verified greedy AND sampled
# on v27+eager: finish_reason=stop, 0 chars; a varied filler generates fine)
# - a prompt artifact, not a serve fault. Numbered sentences: ~17 tok each.
PROMPT = " ".join(
    f"Marker {i:05d}: the engine warms up while the fox watches the quiet harbor. "
    for i in range(1900)
) + "\nQuestion: Write a html car game."  # ~32k tokens

c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=900)
body = json.dumps({"model": "qwen3.8-27b-fp8", "prompt": PROMPT,
                   "max_tokens": 4096, "temperature": 0.3, "top_k": 20, "top_p": 0.95,
                   "min_p": 0, "presence_penalty": 0, "repetition_penalty": 1.0,
                   "stream": True})
t0 = time.perf_counter()
c.request("POST", "/v1/completions", body=body,
          headers={"Content-Type": "application/json"})
r = c.getresponse()
n = toks = 0
first = None
text = b""
while True:
    line = r.readline()
    if not line:
        break
    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
        n += 1
        try:
            j = json.loads(line[6:])
            d = j.get("choices", [{}])[0]
            piece = d.get("text", "")
            if piece:
                if first is None:
                    first = time.perf_counter()
                text += piece.encode("utf-8", "ignore")
                toks += 1
        except Exception:
            pass
tot = time.perf_counter() - t0
ttft = (first - t0) if first else -1
s = text.decode("utf-8", "ignore")
print(f"chunks={n} text_chunks={toks} ttft={ttft:.2f}s total={tot:.1f}s "
      f"tok/s={toks / max(tot - ttft, 1e-3):.2f}")
print(f"has_html={'<html' in s or '<!DOCTYPE' in s} "
      f"has_canvas={'<canvas' in s or 'canvas' in s}")
print("tail_120:", repr(s[-120:]))
