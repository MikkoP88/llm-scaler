#!/usr/bin/env python3
"""wsc_pressure_v129.py — perf-v129 P51b WS-C controlled-pressure replay.

Goal: force KV prefix-cache eviction CAUSALLY at fleet-representative
shapes on the perreq-armed certified lane (:8000), so the per-request
telemetry (prompt/cached/computed per finished request) shows the
eviction signature per cohort. The live CC window (P51a) may be
storm-free; this leg guarantees pressure.

Cohorts (working set ~= 684k tokens vs 476,451-token KV pool ~= 1.44x):
  SHARED  7 convs, prompt ~42k = IDENTICAL ~8k shared head + unique
          ~34k body   -> tests whether LRU keeps hot shared-prefix
          blocks resident without any pinning code (the pinning-
          lever falsification cohort); size band ~38-45k
  PRIV    7 convs, prompt ~50k fully unique -> the eviction victims
          expected under LRU (fleet's big CC private histories);
          size band ~46k+ (bands disjoint so engine-side perreq rows
          are cohort-assignable by prompt size alone)
  SMALL  10 convs, prompt  ~4k unique       -> fleet-mean class,
          expected to stay resident (recently touched, small)

6 turns/conv, full client-side history re-send each turn (stateless
API), light turn barrier so cohort histories age together;
BrokenBarrier -> free-run fallback (a dead conv must not wedge the
rest). Analysis lives in wsc_split_v129.py, not here.

Usage: wsc_pressure_v129.py
"""
import json
import random
import threading
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
OUT = "/root/build/lce1/wsc_pressure_v129.jsonl"
N_TURNS = 6
MAXTOK = 300

TOPICS = [
    "GPU memory fragmentation in inference servers", "prefix caching",
    "speculative decoding acceptance", "fp8 KV quantization error",
    "hybrid SSM/attention checkpointing", "cudagraph capture sizes",
    "async scheduling pipelines", "watchdog design for GPU hangs",
    "sampling temperature defaults", "grammar compilation cost",
    "admission control fairness", "metrics cardinality discipline",
    "kernel autotuning reproducibility", "PCIe compute overlap",
    "shared prefix pool eviction", "incident postmortem culture",
    "block hashing granularity", "TTFT tail decomposition",
    "KV watermark policy", "recompute vs swap preemption",
    "long-context rope scaling", "quantized state recurrence",
]
WORDS = ("kernel cache lane prefix block token queue scheduler budget "
         "barrier wedge fence stall prefill decode chunk watermark hash "
         "evict resident cold hot lru slot pool capacity concurrency "
         "telemetry gauge counter bucket latency tail median p99 step "
         "graph capture shape pad batch aggregate stream window storm "
         "quiet regime cohort signature causal pressure oversubscribe").split()


def synth(seed, approx_tokens, label):
    """Deterministic unique-ish text of ~approx_tokens (4 chars/tok)."""
    rng = random.Random(seed)
    target_chars = approx_tokens * 4
    out = [f"[{label} sect {seed}] "]
    n = 0
    while n < target_chars:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))
        out.append(sent + ".")
        n += len(sent) + 2
    return " ".join(out)


# ~8k-token identical shared head (the fleet's common system prompt)
HEAD = synth(9001, 8000, "shared-head")
# topic string woven into each body for topical variety
CONVS = []
for c in range(7):
    body = synth(100 + c, 34000, f"shared-conv-{c:02d}")
    CONVS.append({"id": f"s{c:02d}", "cohort": "SHARED",
                  "system": HEAD + "\n\n" + body})
for c in range(7):
    body = synth(200 + c, 50000, f"priv-conv-{c:02d}")
    CONVS.append({"id": f"p{c:02d}", "cohort": "PRIV",
                  "system": body})
for c in range(10):
    body = synth(300 + c, 4000, f"small-conv-{c:02d}")
    CONVS.append({"id": f"m{c:02d}", "cohort": "SMALL",
                  "system": body})

BARRIER = threading.Barrier(len(CONVS), timeout=900)


def chat(messages, cid, turn):
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": MAXTOK,
            "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer sk-dummy"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        try:
            d = json.load(e)
        except Exception:
            d = {}
        return {"id": cid, "turn": turn, "status": "HTTP%d" % e.code,
                "wall": round(time.time() - t0, 1)}
    except Exception as e:
        return {"id": cid, "turn": turn, "status": "EXC",
                "exc": f"{type(e).__name__}: {e}",
                "wall": round(time.time() - t0, 1)}
    u = d.get("usage") or {}
    m = (d.get("choices") or [{}])[0]
    return {"id": cid, "turn": turn, "status": "ok",
            "finish": m.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "wall": round(time.time() - t0, 1),
            "ts": time.strftime("%H:%M:%S")}


def run_conv(conv, fh, lock, stats):
    messages = [{"role": "system", "content": conv["system"]}]
    for turn in range(1, N_TURNS + 1):
        topic = TOPICS[(int(conv["id"][1:]) * 7 + turn) % len(TOPICS)]
        messages.append({"role": "user", "content":
                         f"Turn {turn} for {conv['id']}: discuss {topic}. "
                         f"One concrete number, one tradeoff."})
        rec = chat(messages, conv["id"], turn)
        rec["cohort"] = conv["cohort"]
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats.append(rec)
        if rec.get("status") != "ok":
            BARRIER.abort()
            return
        # keep replies: read content via a second field we did not keep;
        # for history we only need a stable turn marker, so re-ask the
        # same shape -> use a fixed-size synthetic assistant turn
        messages.append({"role": "assistant", "content":
                         synth(int(conv["id"][1:]) * 50 + turn, 120,
                               f"reply-{conv['id']}-t{turn}")})
        try:
            BARRIER.wait()
        except threading.BrokenBarrierError:
            pass  # free-run fallback: a dead conv must not wedge the rest


def main():
    lock = threading.Lock()
    stats = []
    with open(OUT, "a") as fh:
        fh.write("# wsc pressure start %s convs=%d turns=%d "
                 "pool_tokens=476451 ws~684k\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), len(CONVS),
                    N_TURNS))
        fh.flush()
        ths = [threading.Thread(target=run_conv,
                                args=(c, fh, lock, stats))
               for c in CONVS]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        fh.write("# wsc pressure end %s turns=%d wall=%ds\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), len(stats),
                    round(time.time() - t0)))
    ok = [r for r in stats if r.get("status") == "ok"]
    print("PRESSURE_DONE ok=%d/%d wall=%ds max_prompt=%s"
          % (len(ok), len(CONVS) * N_TURNS, round(time.time() - t0),
             max((r.get("prompt_tokens") or 0) for r in ok) if ok else 0),
          flush=True)


if __name__ == "__main__":
    main()
