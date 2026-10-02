#!/usr/bin/env python3
"""wsc_pressure_v132a.py — perf-v129 P51c WS-C pressure re-run (moderated).

Re-run of wsc_pressure_v129.py after the P51b wave ended in
UR_RESULT_ERROR_DEVICE_LOST (15:03:47; see captures/
crash_WD_150545_excerpt.txt): the engine-side perreq jsonl lived
in-container and was destroyed by the crash teardown, so this leg
re-generates the eviction signal under MODERATED churn and harvests
the perreq stream CONTINUOUSLY to the host — a repeat crash can no
longer eat the data.

Moderation vs v1 (churn -25%, exposure -33%):
  convs  24 -> 18 (5 SHARED / 5 PRIV / 8 SMALL; same synth targets)
  turns   6 -> 4
Everything else identical (cohorts, barrier pacing, temp 0.6,
max_tokens 300, full client-side history re-send).

Harvester thread: every 120 s `docker cp` the in-container
/root/v128_perreq.jsonl to a snapshot, merge NEW rows (dedupe by
rid+ts) into /root/build/v132_stage/wsc_perreq_v132a.jsonl. Analysis:
wsc_split_v129.py <driver_jsonl> <harvested_perreq_jsonl>.

Caps: PRESSUREB_DONE ok=N/M plus HARVEST_ROWS=<n>.
"""
import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
OUT = "/root/build/v132_stage/wsc_pressure_v132a.jsonl"
PRQ_ACC = "/root/build/v132_stage/wsc_perreq_v132a.jsonl"
PRQ_SNAP = "/root/build/v132_stage/.wsc_perreq_v132a_snap.jsonl"
N_TURNS = 4
MAXTOK = 300
HARVEST_EVERY = 120

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


HEAD = synth(9001, 8000, "shared-head")
CONVS = []
for c in range(5):
    body = synth(100 + c, 34000, f"shared-conv-{c:02d}")
    CONVS.append({"id": f"s{c:02d}", "cohort": "SHARED",
                  "system": HEAD + "\n\n" + body})
for c in range(5):
    body = synth(200 + c, 50000, f"priv-conv-{c:02d}")
    CONVS.append({"id": f"p{c:02d}", "cohort": "PRIV", "system": body})
for c in range(8):
    body = synth(300 + c, 4000, f"small-conv-{c:02d}")
    CONVS.append({"id": f"m{c:02d}", "cohort": "SMALL", "system": body})

BARRIER = threading.Barrier(len(CONVS), timeout=900)
TOPICS = ("GPU memory fragmentation in inference servers", "prefix caching",
          "speculative decoding acceptance", "fp8 KV quantization error",
          "hybrid SSM/attention checkpointing", "cudagraph capture sizes",
          "async scheduling pipelines", "watchdog design for GPU hangs",
          "sampling temperature defaults", "grammar compilation cost",
          "admission control fairness", "metrics cardinality discipline")


def harvest_once(seen):
    n = 0
    subprocess.run(["docker", "cp", "lsv-test:/root/v128_perreq.jsonl",
                    PRQ_SNAP], check=True, capture_output=True, timeout=60)
    with open(PRQ_SNAP) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r.get("rid"), r.get("ts"))
            if key in seen:
                continue
            seen.add(key)
            with open(PRQ_ACC, "a") as g:
                g.write(line + "\n")
            n += 1
    return n


def harvester(stop_evt, stat):
    seen = set()
    while not stop_evt.is_set():
        try:
            stat["rows"] += harvest_once(seen)
        except Exception:
            pass                      # lane rebooting / file absent
        stop_evt.wait(HARVEST_EVERY)
    try:                              # final pass
        stat["rows"] += harvest_once(seen)
    except Exception:
        pass


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
        messages.append({"role": "assistant", "content":
                         synth(int(conv["id"][1:]) * 50 + turn, 120,
                               f"reply-{conv['id']}-t{turn}")})
        try:
            BARRIER.wait()
        except threading.BrokenBarrierError:
            pass


def main():
    lock = threading.Lock()
    stats = []
    stat = {"rows": 0}
    stop_evt = threading.Event()
    th = threading.Thread(target=harvester, args=(stop_evt, stat),
                          daemon=True)
    th.start()
    with open(OUT, "a") as fh:
        fh.write("# wsc pressure-B start %s convs=%d turns=%d "
                 "moderated-after-DEVICE_LOST\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), len(CONVS),
                    N_TURNS))
        fh.flush()
        ths = [threading.Thread(target=run_conv,
                                args=(c, fh, lock, stats)) for c in CONVS]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        fh.write("# wsc pressure-B end %s turns=%d wall=%ds\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), len(stats),
                    round(time.time() - t0)))
    stop_evt.set()
    th.join()
    ok = [r for r in stats if r.get("status") == "ok"]
    print("PRESSUREB_DONE ok=%d/%d wall=%ds HARVEST_ROWS=%d"
          % (len(ok), len(CONVS) * N_TURNS, round(time.time() - t0),
             stat["rows"]), flush=True)


if __name__ == "__main__":
    main()
