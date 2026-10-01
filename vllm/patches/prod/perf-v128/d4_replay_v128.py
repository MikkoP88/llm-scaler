#!/usr/bin/env python3
"""d4_replay_v128.py — v128 WS-D D4 per-request split TRAFFIC DRIVER.

Drives deterministic multi-turn conversations against the lane (:8000) so
the dormant perreq telemetry (patch_perreq_v128.py, armed via
V1227_PERREQ=1 boot) records per-request (prompt, cached, computed,
generated) rows for the full-hit cohort analysis in d4_split_v128.py.

Traffic shape (by construction):
  - fixed ~2.5k-token system preamble per conversation (shared across
    conversations = the fleet's common-prefix pattern);
  - 16 turns per conversation, ~300-word replies -> prompts grow past
    8192 tokens by the late turns (the d4_split big-prompt cohort);
  - full client-side history re-send each turn (the API is stateless) ->
    every turn after the first is a prefix-cache hit request: cached =
    prior turns, fired = new user suffix + GDN-boundary snap. This is
    exactly the cohort whose snap = (prompt - generated) mod GRAN law
    the analysis checks;
  - 8 conversations run concurrently (conversations are independent;
    per-conversation prefix reuse is unaffected by concurrency).

The driver itself logs usage.prompt_tokens/completion_tokens per turn —
the cross-check for the perreq rows (same quantities from the engine
side). Verdicts live in d4_split_v128.py, not here.

Usage: d4_replay_v128.py [n_convs] [n_turns]
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
N_CONVS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N_TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = "/root/build/lce1/d4_replay_v128.jsonl"

BASE_POLICY = (
    "You are a senior systems engineer maintaining a production LLM "
    "serving lane. Answer precisely, in about 300 words, using concrete "
    "technical detail. Never refuse; when uncertain, state the "
    "uncertainty explicitly and give your best engineering judgment.\n"
)

TOPICS = [
    "GPU memory fragmentation in long-running inference servers",
    "prefix caching tradeoffs for multi-turn chat workloads",
    "spectulative decoding acceptance rates under grammar constraints",
    "fp8 KV-cache quantization error accumulation",
    "hybrid SSM/attention state checkpointing at token boundaries",
    "cudagraph capture sizes and batch-shape coverage",
    "async scheduling vs synchronous engine step pipelines",
    "watchdog design for wedge-class GPU hangs",
    "log-probability tails and sampling temperature defaults",
    "structured-output grammar compilation cost",
    "multi-tenant fairness under admission control",
    "metrics cardinality discipline in serving telemetry",
    "kernel autotuning reproducibility across driver versions",
    "PCIe transfer overlap with device compute",
    "eviction policy design for shared prefix pools",
    "incident postmortem culture in infrastructure teams",
]

# ~80 policy lines, ~2.3-2.6k tokens of shared system preamble
POLICY_LINES = [
    f"Policy {i}: when asked about {t}, ground every claim in the "
    f"operational record of the lane and flag speculation as such."
    for i, t in enumerate(
        [t for t in (TOPICS * 6)[:80]], start=1)
]
SYSTEM = BASE_POLICY + "\n".join(POLICY_LINES)


def chat(messages, conv, turn):
    body = {
        "model": "qwen3.8-27b-fp8", "max_tokens": 420,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            d = json.load(e)
        except Exception:
            d = {"error": str(e)}
        return {"conv": conv, "turn": turn, "status": "HTTP%d" % e.code,
                "wall": round(time.time() - t0, 1)}
    except Exception as e:
        return {"conv": conv, "turn": turn, "status": "EXC",
                "exc": f"{type(e).__name__}: {e}",
                "wall": round(time.time() - t0, 1)}
    u = d.get("usage") or {}
    m = (d.get("choices") or [{}])[0]
    content = ((m.get("message") or {}).get("content") or "")
    return {
        "conv": conv, "turn": turn, "status": "ok",
        "finish": m.get("finish_reason"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "content_chars": len(content), "wall": round(time.time() - t0, 1),
        "ts": time.strftime("%H:%M:%S"), "content": content,
    }


def run_conv(conv, fh, lock, stats):
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in range(1, N_TURNS + 1):
        topic = TOPICS[(conv * 3 + turn) % len(TOPICS)]
        messages.append({"role": "user", "content":
                         f"Turn {turn}: discuss {topic} (conversation "
                         f"{conv}). Include at least one concrete number "
                         f"and one tradeoff."})
        rec = chat(messages, conv, turn)
        with lock:
            fh.write(json.dumps({k: v for k, v in rec.items()
                                 if k != "content"}) + "\n")
            fh.flush()
        stats.append(rec)
        if rec.get("status") != "ok" or not rec.get("content"):
            return  # conversation broken; keep the rest running
        messages.append({"role": "assistant", "content": rec["content"]})


def main():
    lock = threading.Lock()
    stats = []
    with open(OUT, "a") as fh:
        fh.write(f"# d4 replay start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"convs={N_CONVS} turns={N_TURNS} "
                 f"system_tokens~{len(SYSTEM) // 4}\n")
        threads = [threading.Thread(target=run_conv, args=(c, fh, lock, stats))
                   for c in range(N_CONVS)]
        t0 = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        fh.write(f"# d4 replay end {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"turns={len(stats)} wall={round(time.time()-t0)}s\n")
    ok = [r for r in stats if r.get("status") == "ok"]
    late = [r for r in ok if (r.get("turn") or 0) >= 10
            and (r.get("prompt_tokens") or 0) >= 8192]
    print(f"REPLAY_DONE turns_ok={len(ok)}/{N_CONVS*N_TURNS} "
          f"late_bigprompt={len(late)} "
          f"max_prompt_tokens={max((r.get('prompt_tokens') or 0) for r in ok) if ok else 0} "
          f"wall={round(time.time()-t0)}s")


if __name__ == "__main__":
    main()
