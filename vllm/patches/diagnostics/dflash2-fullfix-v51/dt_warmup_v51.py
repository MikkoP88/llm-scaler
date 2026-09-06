# dt_warmup.py — llm-scaler v51: boot-time request warmup.
#
# Lineage: v43b (crash-2 fix: drafter first-use JIT -> GuC engine reset),
# v50 F4 (phases 4+5: concurrent long decode + chunked prefill).
#
# v51 (Phase L fix, 2026-09-06): production runs --max-num-batched-tokens
# 8192. v50's phase-5 prompt was ~2.7k tokens = SINGLE chunk, so the TQ
# chunked-prefill CONTINUATION path (_continuation_prefill ->
# _tq_full_dequant_kv + concat flash-attn, turboquant_attn.py:822+) never
# ran at boot. First >8k prompt under traffic (user cell 2026-09-06 05:01:46)
# JIT-compiled _tq_full_dequant_kv DURING inference; the multi-second
# host-side compile made one TP rank late to the eager collectives ->
# DFLASH_STALL propose() 0.180/0.201 s peer-late windows on both workers
# (windows >0.64 s trip the xe GuC preempt watchdog -> engine reset, the
# crash-2 mechanism). Phase 6 fires a ~18k-token prompt = 3 chunks
# (8192 + 8192 + tail): two continuations (cached_len 8192 and 16384) so
# the dequant kernel and the concat-FA path JIT at boot, before traffic.
import json
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


def fire(tag: str, max_tokens: int, prompt: str, temp: float | None = None) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temp is not None:
        payload["temperature"] = temp
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.loads(r.read())
    ct = (j.get("usage") or {}).get("completion_tokens", -1)
    dt = time.time() - t0
    log(f"{tag}: {ct} toks in {dt:.1f}s ({ct / max(dt, 1e-9):.1f} tok/s)")
    return dt


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
    # DURING traffic (ttft 22-40 s spikes, 9 jit_monitor warnings, decode
    # tps also depressed on that first request). Phases 7-8 pre-hit the
    # ~36k and ~70k chunk buckets (5 and 9 chunks of 8192) so those
    # specializations exist before any user request lands on them.
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
    # JIT-compiled on the first greedy request under traffic (jit_monitor
    # warning 10:59:44 during ctxscan 2k cell). Phase 9 pre-hits the greedy
    # sampler variants at bs=1 and bs=2 — the two batch-divisibility buckets
    # the earlier phases already cover for the sampled path.
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
    log("WARMUP_DONE")


if __name__ == "__main__":
    main()
