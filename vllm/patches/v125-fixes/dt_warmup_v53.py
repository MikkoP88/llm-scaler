# dt_warmup.py — llm-scaler v53: boot-time request warmup.
#
# Lineage: v43b (crash-2 fix: drafter first-use JIT -> GuC engine reset),
# v50 F4 (phases 4+5: concurrent long decode + chunked prefill),
# v51 Phase L (p6 multi-chunk continuation-dequant JIT; p7/p8 36k/70k
# chunk buckets; p9 greedy sampler bs1+2).
#
# v53 (v125-audit L7, 2026-09-07): warmup stops at 70k, but the first
# 128k/227k-class request still JIT-compiles new shapes UNDER TRAFFIC
# (v125 REPORT D7: df7_tq4 cold 128k decode 4.2 vs 7.3 warm, first-deep
# TTFT 60-133 s; mtp4_fp8 JIT'd _fp8_mq_stage1/2 during ctxscan). Phases
# 10-11 pre-hit the ~131k (16-chunk) and ~227k (28-chunk) buckets with
# VARIED filler text (KNOWN_ISSUES #13: repetitive fillers trigger
# instant-EOS >=32k — v51's repeated-sentence trick stops working at
# these depths) and force greedy (temp=0) + ignore_eos decode steps AT
# DEPTH so the deep-context decode/graph-tier specializations and the
# greedy rejection-sampler row shapes compile at boot.
#
# Boot-cost note: p10+p11 add ~4-8 min (two deep prefills at production
# prefill rates). Opt out with V53_DEEP_WARMUP=0 (p1-p9 unchanged).
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
LOG = open("/root/dt_warmup.log", "a", buffering=1)


def log(msg: str) -> None:
    print(f"[warmup {time.strftime('%H:%M:%S')}] {msg}", file=LOG, flush=True)


def wait_health(timeout_s: int = 1800) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def fire(
    tag: str,
    max_tokens: int,
    prompt: str,
    temp: float | None = None,
    extra: dict | None = None,
) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temp is not None:
        payload["temperature"] = temp
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    ct = (j.get("usage") or {}).get("completion_tokens", -1)
    dt = time.time() - t0
    log(f"{tag}: {ct} toks in {dt:.1f}s ({ct / max(dt, 1e-9):.1f} tok/s)")
    return dt


def varied_prompt(target_tokens: int, seed: int) -> str:
    """Deterministic VARIED long document (~8-10 tokens per record).

    Repetitive fillers trigger instant-EOS at >=32k (KNOWN_ISSUES #13);
    the audit harness (dt_ctxscan) uses varied 16-word fillers for the
    same reason. Records are unique per index so no n-gram repetition
    pattern can dominate.
    """
    rng = random.Random(seed)
    words = [
        "service", "request", "cache", "latency", "document", "result",
        "memory", "worker", "client", "record", "queue", "response",
        "throughput", "schedule", "engine", "kernel", "buffer", "stream",
        "token", "batch", "prefix", "decode", "verify", "quant",
    ]
    parts = ["Summarize the following operational records.\n"]
    # ~9 tokens per record line; overshoot ~5% then let the server
    # tokenize-truncate naturally (no exact length needed for warmup).
    n = int(target_tokens / 8.5 * 1.05) + 100
    for i in range(n):
        parts.append(
            f"Record {i}: {rng.choice(words)} {rng.randrange(1000000)} "
            f"{rng.choice(words)} {rng.choice(words)} "
            f"{rng.choice(words)} {rng.randrange(1000)}.\n"
        )
    return "".join(parts)


def main() -> None:
    log("waiting for /health ...")
    if not wait_health():
        log("FATAL: health never came up; no warmup fired")
        sys.exit(2)
    log("health OK; phase 1 single short (absorbs first-use JIT)")
    fire("p1-single", 48, "Write a html car game.")
    log("phase 2 concurrent pair (mixed-batch specializations)")
    ts = [
        threading.Thread(target=fire, args=(f"p2-conc{i}", 48, "Write a html car game."))
        for i in range(2)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log("phase 3 single longer decode")
    fire("p3-long", 512, "Write a html car game.")
    log("phase 4 concurrent pair, longer decode (deeper-context JIT)")
    ts4 = [
        threading.Thread(
            target=fire, args=(f"p4-conclong{i}", 768, "Write a html car game.")
        )
        for i in range(2)
    ]
    for t in ts4:
        t.start()
    for t in ts4:
        t.join()
    log("phase 5 long-prompt chunked prefill (prefill-side JIT)")
    long_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while the engine decodes tokens in blocks; kernels "
        "compile once at boot so user traffic never pays the first-use cost. ") * 90
    )
    fire("p5-prefill", 32, long_prompt)
    # llm-scaler v51 (Phase L): phase 5's ~2.7k prompt fits ONE 8192-token
    # chunk, so the TQ continuation-dequant path never warmed (2026-09-06
    # 05:01:46 _tq_full_dequant_kv JIT under traffic + DFLASH_STALL peer-late
    # windows). Phase 6: ~18k tokens = 3 chunks -> continuations at cached_len
    # 8192 AND 16384 -> _tq_full_dequant_kv + concat-FA JIT at boot.
    log("phase 6 multi-chunk prefill (continuation-dequant JIT, cached_len 8k+16k)")
    multi_chunk_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while chunked prefill splits long prompts; continuation "
        "chunks dequant the cached prefix once at boot so traffic never pays "
        "the first-use compile cost. ") * 600
    )
    fire("p6-multichunk", 32, multi_chunk_prompt)
    # llm-scaler v51 (Phase L, 2026-09-06 ctx-scan evidence): even after
    # phase 6, the FIRST 32k/65k prompt JIT-compiles prefill-side kernels
    # DURING traffic. Phases 7-8 pre-hit the ~36k and ~70k chunk buckets
    # (5 and 9 chunks of 8192) so those specializations exist before any
    # user request lands on them.
    log("phase 7 ~36k prompt (5-chunk bucket JIT)")
    p36k_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while long documents split into five chunks; each new "
        "chunk bucket compiles its prefill kernels here at boot instead of "
        "under user traffic. ") * 1150
    )
    fire("p7-36k", 16, p36k_prompt)
    log("phase 8 ~70k prompt (9-chunk bucket JIT)")
    p70k_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while very long documents split into nine chunks; the "
        "last bucket compiles here at boot so sixty-five-thousand-token "
        "requests never pay a first-use compile. ") * 2200
    )
    fire("p8-70k", 16, p70k_prompt)
    # llm-scaler v51 (Phase 4b Boot B evidence, 2026-09-06): phases 1-8 fire
    # temp-default requests, so the GREEDY specialization of the spec-decode
    # rejection sampler (rejection_greedy_sample_kernel, temp=0 path) still
    # JIT-compiled on the first greedy request under traffic. Phase 9 pre-hits
    # the greedy sampler variants at bs=1 and bs=2.
    log("phase 9 greedy sampler variants (bs=1 + bs=2, temp=0)")
    fire("p9-greedy1", 32, "Write a html car game.", temp=0.0)
    ts9 = [
        threading.Thread(
            target=fire, args=(f"p9-greedy{i}", 32, "Write a html car game.", 0.0)
        )
        for i in range(2)
    ]
    for t in ts9:
        t.start()
    for t in ts9:
        t.join()
    # llm-scaler v53 (v125-audit L7 / REPORT D7, 2026-09-07): warmup
    # previously stopped at the 70k bucket. The first 128k/227k-class
    # request compiles chunk-count, KV-splits-tier, fp8-MQ-shape and
    # graph specializations UNDER TRAFFIC (measured: first-deep TTFT
    # 60-133 s; df7 cold-128k decode 4.2 vs 7.3 warm; mtp4_fp8 JIT'd
    # _fp8_mq_stage1/2 during ctxscan). Phases 10-11 pre-hit the two
    # deep buckets. VARIED fillers are mandatory here: v51's repeated-
    # sentence prompts trigger instant-EOS >=32k (KNOWN_ISSUES #13) and
    # would skip the decode-side warm entirely. temp=0 + ignore_eos
    # forces greedy deep decode steps (rejection-sampler row shapes at
    # depth + TQ KV-splits tier + fp8 MQ verify shapes).
    if os.environ.get("V53_DEEP_WARMUP", "1") != "0":
        # Tokenizer-calibrated 2026-09-07 (container tokenizer, seeds as
        # fired): varied_prompt token yield is ~2.97 x its target-unit
        # argument at these sizes (the ~8.5 tok/record design estimate
        # was ~3x low; records average ~25 tokens). Targets below are
        # measured EXACT: 41000 -> 121,701 tok (15 chunks @8192);
        # 67000 -> 198,004 tok (25 chunks). Both + 24 gen + template
        # overhead stay far under max_model_len 262,144. The original
        # uncalibrated 131072/227000 targets produced ~389k/~674k-token
        # prompts and the p10 request was 400-rejected at boot
        # (VLLMValidationError "at least 262121" = boundary watermark).
        log("phase 10 ~122k-tok varied prompt (15-chunk bucket + deep greedy decode JIT)")
        p122k = varied_prompt(41000, seed=131072)
        fire("p10-122k", 24, p122k, temp=0.0, extra={"ignore_eos": True})
        log("phase 11 ~198k-tok varied prompt (25-chunk bucket + deep greedy decode JIT)")
        p198k = varied_prompt(67000, seed=227000)
        fire("p11-198k", 24, p198k, temp=0.0, extra={"ignore_eos": True})
    else:
        log("V53_DEEP_WARMUP=0 -> skipping p10/p11 deep buckets")
    log("WARMUP_DONE")


if __name__ == "__main__":
    main()
