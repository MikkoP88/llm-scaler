#!/usr/bin/env python3
"""A5000 CUDA reference probe — stock vLLM MTP behavior vs the XPU fork lane.

Runs against http://127.0.0.1:8100 (vllm-mtp container, host network).
Measures: health/boot wait, short-request latency, spec-decode acceptance
counters (via /metrics deltas), deep-context hazard shape (multi-turn grow to
~16k+ then long-gen at depth — the shape that kills the XPU lane), and an
8-way concurrency burst. Writes /root/a5000_mtp/probe_report.txt.
"""
import json
import re
import threading
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8100"
MODEL = "qwen3.8-27b-int4"
OUT = "/root/a5000_mtp/probe_report.txt"

_filler_para = (
    "The pipeline compiles a staged execution graph where each operator "
    "declares its memory contract before scheduling. Between turns the "
    "recurrent state must remain resident while the paged cache rotates "
    "blocks under an LRU policy, so the admission controller watches the "
    "computed-token watermark rather than raw occupancy. "
)


def _lines(*msgs):
    with open(OUT, "a") as f:
        for m in msgs:
            f.write(str(m) + "\n")


def http(path, payload=None, timeout=900):
    if payload is None:
        req = urllib.request.Request(BASE + path)
    else:
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return time.time() - t0, body


def chat(messages, max_tokens, temperature=0.5):
    wall, body = http(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    data = json.loads(body)
    usage = data.get("usage", {})
    text = ""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return wall, usage, text


def spec_snapshot():
    _, body = http("/metrics")
    out = {}
    for ln in body.decode(errors="replace").splitlines():
        if ln.startswith("#") or "spec" not in ln.lower():
            continue
        # label-aware: name{labels} value  (nightly adds {engine=,model_name=})
        m = re.match(r"([\w:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)", ln)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def tok_count(s):
    """Token count via the server's /tokenize endpoint."""
    req = urllib.request.Request(
        BASE + "/tokenize",
        data=json.dumps({"model": MODEL, "prompt": s}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def derive(a, b):
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b if k in a}


def main():
    open(OUT, "w").close()
    _lines("# A5000 stock-vLLM MTP probe", time.ctime(), "")

    # 1. Wait for health (model load of ~16GB int4 can take minutes)
    t0 = time.time()
    code = None
    while time.time() - t0 < 2400:
        try:
            _, body = http("/health", timeout=10)
            code = 200
            break
        except Exception as e:  # noqa: BLE001
            code = repr(e)[:80]
            time.sleep(10)
    _lines(f"HEALTH: {code} after {time.time()-t0:.0f}s")
    if code != 200:
        _lines("BOOT_FAILED — aborting probes")
        return
    _, body = http("/v1/models")
    _lines("MODELS:", body.decode()[:200], "")

    base = spec_snapshot()

    # 2. Ten sequential short requests
    ok = 0
    walls = []
    toks = []
    for i in range(10):
        try:
            wall, usage, _ = chat(
                [{"role": "user", "content": f"Explain briefly what speculative decoding is. Variant {i}."}],
                256,
            )
            ct = usage.get("completion_tokens", 0)
            ok += 1
            walls.append(wall)
            toks.append(ct)
            _lines(f"SHORT {i}: wall={wall:.2f}s completion_tokens={ct}")
        except Exception as e:  # noqa: BLE001
            _lines(f"SHORT {i}: FAIL {e!r}")
    if walls:
        _lines(
            f"SHORT SUMMARY: ok={ok}/10 wall_med={sorted(walls)[len(walls)//2]:.2f}s "
            f"tok_med={sorted(toks)[len(toks)//2]}"
        )
    _lines("")

    after_short = derive(base, spec_snapshot())
    _lines("SPEC METRIC DELTAS (short):", json.dumps(after_short, indent=1), "")

    # 3. Deep-context hazard shape: grow one conversation toward the window edge,
    #    then long-gen at depth. Sized adaptively via /tokenize so prompt+gen
    #    stays inside max_model_len 16384 regardless of tokenizer.
    MAXLEN = 16384
    try:
        para_t = tok_count(_filler_para)
    except Exception as e:  # noqa: BLE001
        _lines(f"DEEP SHAPE: FAIL tokenize {e!r}")
        para_t = 0
    if para_t:
        deep_base = spec_snapshot()
        reserve = 6 * 430 + 2048 + 250  # 6 turns of growth + long-gen + slack
        count = max(1, int((MAXLEN - reserve) // para_t))
        big_input = " ".join([_filler_para] * count)
        conv = [{"role": "user", "content": big_input + "\n\nSummarize the pipeline in two sentences."}]
        _lines(f"DEEP fill: para={para_t} tok x{count} = ~{count*para_t} tok prompt")
        last_prompt = 0
        try:
            for turn in range(6):
                wall, usage, text = chat(conv, 400)
                last_prompt = usage.get("prompt_tokens", 0)
                conv.append({"role": "assistant", "content": text[-2000:]})
                conv.append({"role": "user", "content": f"Turn {turn+1}: expand on the admission controller and the watermark tradeoff."})
                _lines(
                    f"DEEP turn {turn+1}: wall={wall:.2f}s prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')}"
                )
            est_final = last_prompt + 500
            max_gen = min(2048, MAXLEN - est_final - 16)
            if max_gen < 256:
                _lines(f"DEEP long-gen: SKIPPED (est prompt {est_final} leaves <256 tok)")
            else:
                wall, usage, _ = chat(conv, max_gen)
                _lines(
                    f"DEEP long-gen: wall={wall:.2f}s prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')} tok/s={usage.get('completion_tokens',0)/max(wall,0.001):.1f}"
                )
            _lines("DEEP SHAPE: SURVIVED")
        except urllib.error.HTTPError as e:
            _lines(f"DEEP SHAPE: FAIL {e} body={e.read()[:300]!r}")
        except Exception as e:  # noqa: BLE001
            _lines(f"DEEP SHAPE: FAIL {e!r}")
        _lines("")

    after_deep = derive(deep_base, spec_snapshot())
    _lines("SPEC METRIC DELTAS (deep):", json.dumps(after_deep, indent=1), "")

    # 4. Concurrency burst: 8 parallel short requests
    results = [None] * 8

    def worker(i):
        try:
            wall, usage, _ = chat(
                [{"role": "user", "content": f"Write three sentences about cache eviction policies. Worker {i}."}],
                200,
            )
            results[i] = f"w{i}: wall={wall:.2f}s ct={usage.get('completion_tokens')}"
        except Exception as e:  # noqa: BLE001
            results[i] = f"w{i}: FAIL {e!r}"

    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    _lines(f"BURST 8-way total wall={time.time()-t0:.2f}s")
    _lines(*[r or "w?: no result" for r in results])
    _lines("")

    final = spec_snapshot()
    _lines("FINAL SPEC METRICS:", json.dumps(final, indent=1))
    _lines("PROBE DONE", time.ctime())


if __name__ == "__main__":
    main()
