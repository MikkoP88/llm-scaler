#!/usr/bin/env python3
"""Canonical car-game bench via /v1/completions streaming (no reasoning parser):
the completions endpoint always fills choices[0].text, so event counting works
even when the model never closes <think>. Same canonical sampling."""
import argparse, json, sys, time, http.client

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--tag", default="cell")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    payload = {
        "model": "qwen3.8-27b-fp8",
        "prompt": "Write a html car game.",
        "max_tokens": args.max_tokens,
        "temperature": 0.3, "top_k": 20, "top_p": 0.95,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
    t_req = time.perf_counter()
    conn.request("POST", "/v1/completions", body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"FATAL: HTTP {resp.status}: {resp.read()[:300]!r}"); sys.exit(2)
    stamps, usage, buf, text_tail = [], None, b"", ""
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        now = time.perf_counter()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if ch.get("text"):
                    stamps.append(now)
                    text_tail = (text_tail + ch["text"])[-600:]
    conn.close()
    if len(stamps) < 2:
        print(f"FATAL: only {len(stamps)} events"); sys.exit(2)
    ttft = stamps[0] - t_req
    gen_time = stamps[-1] - stamps[0]
    n_events = len(stamps)
    n_true = usage.get("completion_tokens") if usage else None
    tok_per_event = (n_true / n_events) if n_true else 1.0
    t0 = stamps[0]
    half = t0 + gen_time / 2
    late = [t for t in stamps if t >= half]
    steady = ((len(late) - 1) / (late[-1] - late[0])) if len(late) > 2 else None
    steady_true = steady * tok_per_event if steady else float("nan")
    print(f"TAG={args.tag} events={n_events} tokens_true={n_true} "
          f"tok/event={tok_per_event:.2f} TTFT={ttft:.2f}s gen={gen_time:.1f}s "
          f"steady_true(last50%)={steady_true:.2f} tok/s")
    closed = "</think>" in text_tail or "<html" in text_tail.lower() or "<!doctype" in text_tail.lower()
    print(f"tail_has_close_or_html={closed}")
    print("tail:", text_tail[-200:].replace("\n", " ")[:200])
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": args.tag, "events": n_events,
                                "tokens_true": n_true, "steady_tok_s": round(steady_true, 2),
                                "gen_time_s": round(gen_time, 1)}) + "\n")

if __name__ == "__main__":
    main()
