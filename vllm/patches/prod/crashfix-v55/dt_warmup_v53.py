# dt_warmup_v53.py — llm-scaler v55 (crashfix-v55): boot-time warmup.
#
# Lineage: v43b (crash-2 fix: drafter first-use JIT -> GuC engine reset),
# v50 F4 (phases 4+5: concurrent long decode + chunked prefill), v51
# Phase L (p6/p7/p8 chunk-bucket continuations; p9 greedy sampler),
# deployed 2026-09-06; certified "ZERO jit_monitor post-WARMUP_DONE" on
# the tq4nc lane.
#
# v53 (crashfix-v55, 2026-09-10): the 2026-09-10 crash pair (14:34
# e4m3+mtp4 DEVICE_LOST; 15:44 e5m2+mtp3 async-event wedge) both ran on
# boots with NO warmup executed, and the first-traffic JIT storm
# (_topk_topp at 14:31:43 / 15:39:32, kernel_unified_attention,
# reduce_segments, eagle padding kernels, batch_memcpy, PRE-COPY flood)
# coincided with the failure window in BOTH logs. The crash traffic
# profile — staggered multi-stream ramp with DISTINCT prompts (no prefix
# sharing), chunked prefills co-admitted with decodes, explicit sampled
# params, then a long repetitive html-game decode — maps onto three
# coverage gaps in v51:
#
#   p10: EXPLICIT sampled params (temp 0.9 / top_k 20 / top_p 0.95 and
#        the top_p-only and top_k-only kernel branches) — v51 p1-p8 send
#        no explicit sampling fields, so the _topk_topp specializations
#        for client-shaped requests JIT on first traffic.
#   p11: STAGGERED CONCURRENT DISTINCT-PROMPT streams (4 streams,
#        +0/+3/+6/+9 s, ~2k/6k/12k/18k prompts) — v51's concurrent
#        phases reuse ONE prompt (prefix cache coalesces them), so the
#        co-admission PRE-COPY/DEF-PP machinery and chunked-prefill
#        continuations never ran with four independent sequences at
#        boot.
#   p12: LONG REPETITIVE DECODE on the exact crash prompt
#        ("Write a html car game", 4096 out, sampled) — the crash-2
#        wedge lane: high-acceptance repetitive code output driving
#        many 2048-token mamba boundary carries and spec-acceptance
#        decay, at boot instead of under traffic.
#   p13 (v53.1): VALIDATION-GEOMETRY 5-STREAM STAGGER (+0/+4/+8/+12/+16s:
#        short sampled long-decode + ~17k/~8k doc prefills + greedy and
#        sampled shorts) — validation run 5 convicted that p11's 4 streams
#        and p12's solo decode left the bs=5 mixed-context step shapes
#        cold (kernel_unified_attention + reduce_segments JIT-compiled at
#        19:18:06 on the first cycle).
#   p14 (v53.2): DRESS-REHEARSAL CYCLE — run 6 proved p13's 512-out docs
#        insufficient (the fired shapes need docs decoding 1024-2048 deep
#        while the htmlgame 4096 decode runs); warmup's final phase runs
#        one full v55_client cycle (seed base 90) so the validated
#        traffic class is compiled at boot. JIT inside warmup is allowed
#        by definition; the acceptance gate is ZERO jit_monitor lines in
#        the POST-WARMUP log section.
#
# All other phases and the WARMUP_DONE marker protocol are unchanged
# from v51 (prod_restore*.sh greps /root/dt_warmup.log on the host).
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


def fire(
    tag: str,
    max_tokens: int,
    prompt: str,
    temp: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temp is not None:
        payload["temperature"] = temp
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
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


def _filler(stream_no: int, unit: str) -> str:
    # Distinct per stream: no prefix-cache coalescing between them.
    return f"Summarize the following document {stream_no}. " \
           f"Document {stream_no} section: {unit} "


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
    for t4 in ts4:
        t4.start()
    for t4 in ts4:
        t4.join()
    log("phase 5 long-prompt chunked prefill (prefill-side JIT)")
    long_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while the engine decodes tokens in blocks; kernels "
        "compile once at boot so user traffic never pays the first-use cost. ") * 90
    )
    fire("p5-prefill", 32, long_prompt)
    log("phase 6 multi-chunk prefill (continuation-dequant JIT, cached_len 8k+16k)")
    multi_chunk_prompt = (
        "Summarize the following text. " + ("The quick brown fox jumps over "
        "the lazy dog while chunked prefill splits long prompts; continuation "
        "chunks dequant the cached prefix once at boot so traffic never pays "
        "the first-use compile cost. ") * 600
    )
    fire("p6-multichunk", 32, multi_chunk_prompt)
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
    log("phase 9 greedy sampler variants (bs=1 + bs=2, temp=0)")
    fire("p9-greedy1", 32, "Write a html car game.", temp=0.0)
    ts9 = [
        threading.Thread(
            target=fire, args=(f"p9-greedy{i}", 32, "Write a html car game.", 0.0)
        )
        for i in range(2)
    ]
    for t9 in ts9:
        t9.start()
    for t9 in ts9:
        t9.join()

    # ---- v53 (crashfix-v55): crash-coverage gap phases p10-p12 ----
    log("phase 10 explicit sampled params (topk+topp / topp-only / topk-only)")
    fire("p10-tk-tp", 48, "Write a html car game.",
         temp=0.9, top_k=20, top_p=0.95)
    fire("p10-tp", 48, "Write a html car game.", temp=0.9, top_p=0.95)
    fire("p10-tk", 48, "Write a html car game.", temp=0.9, top_k=50)

    log("phase 11 staggered concurrent distinct-prompt streams "
        "(co-admission prefill+decode, PRE-COPY/DEF-PP machinery)")
    units = {
        1: "alpha kernels compile at boot and never stall the decoder; ",
        2: "beta streams arrive staggered so chunked prefill co-admits "
           "with running decodes; ",
        3: "gamma documents differ per stream so prefix caches never "
           "coalesce the warmup; ",
        4: "delta requests mix sampled and greedy parameters to warm "
           "every sampler branch; ",
    }
    reps = {1: 70, 2: 200, 3: 400, 4: 600}  # ~2k / ~6k / ~12k / ~18k tok
    delays = {1: 0.0, 2: 3.0, 3: 6.0, 4: 9.0}
    def _stream(no: int) -> None:
        time.sleep(delays[no])
        if no == 4:
            fire(f"p11-s{no}", 256, _filler(no, units[no]) * reps[no],
                 temp=0.0)
        elif no == 2:
            fire(f"p11-s{no}", 256, _filler(no, units[no]) * reps[no],
                 temp=0.9, top_k=20, top_p=0.95)
        else:
            fire(f"p11-s{no}", 256, _filler(no, units[no]) * reps[no])
    ts11 = [threading.Thread(target=_stream, args=(n,)) for n in (1, 2, 3, 4)]
    for t11 in ts11:
        t11.start()
    for t11 in ts11:
        t11.join()

    log("phase 12 crash-lane long repetitive decode "
        "(html game, 4096 out, sampled; mamba boundaries + spec decay)")
    fire("p12-htmlgame", 4096, "Write a html car game",
         temp=0.9, top_k=20, top_p=0.95)

    # v53.1: validation-run-5 conviction — first traffic under the 5-way
    # concurrent cycle geometry (short sampled long-decode + 17k/8k doc
    # prefills + greedy/sampled shorts co-batched) JIT-compiled
    # kernel_unified_attention + reduce_segments (2 jit_monitor WARNINGs
    # at 19:18:06): p11 tops out at 4 streams and p12 is solo, so the
    # bs=5 mixed-context step shapes were cold. p13 replicates that
    # geometry with distinct prompts so the validated section is JIT-free.
    log("phase 13 validation-geometry 5-stream stagger "
        "(bs=5 mixed-ctx decode + long-decode/prefill co-residency)")
    def _p13(no: int) -> None:
        if no == 1:
            time.sleep(0.0)
            fire("p13-longdecode", 4096, "Write a html snake game",
                 temp=0.9, top_k=20, top_p=0.95)
        elif no == 2:
            time.sleep(4.0)
            fire("p13-doc17k", 512, _filler(5, "epsilon documents span "
                 "seventeen thousand tokens so chunk budgets split across "
                 "five-chunk buckets at five-way admission; ") * 530,
                 temp=0.9, top_k=20, top_p=0.95)
        elif no == 3:
            time.sleep(8.0)
            fire("p13-doc8k", 512, _filler(6, "zeta records keep the "
                 "eight-thousand-token lane distinct from every other "
                 "warmup stream; ") * 260)
        elif no == 4:
            time.sleep(12.0)
            fire("p13-greedy", 256,
                 "Explain in one paragraph why five-way concurrent "
                 "admission must be warmed at boot.", temp=0.0)
        else:
            time.sleep(16.0)
            fire("p13-sampled", 512,
                 "Write a short python function that merges five sorted "
                 "iterators and yields the smallest head each step.",
                 temp=0.9, top_k=20, top_p=0.95)
    ts13 = [threading.Thread(target=_p13, args=(n,)) for n in (1, 2, 3, 4, 5)]
    for t13 in ts13:
        t13.start()
    for t13 in ts13:
        t13.join()

    # v53.2: run-6 conviction — p13 (512-out docs) still left the cycle
    # geometry cold (kernel_unified_attention + reduce_segments JIT'd at
    # 19:47:38 on cycle 1): the fired shapes need the docs decoding
    # 1024-2048 deep WHILE the htmlgame-class 4096 decode runs, the exact
    # evolving-batch mix only the real cycle produces. Runs 5/6 both
    # showed cycles 2-3 clean after one exposure. p14 = dress rehearsal:
    # one full v55_client cycle with seed base 90 (validation cycles use
    # 101-103; same prompt LENGTHS -> same shape classes, distinct text).
    # JIT inside warmup is allowed by definition — the acceptance gate is
    # ZERO jit_monitor lines in the POST-WARMUP section.
    log("phase 14 dress-rehearsal validation cycle (v55_client seed 90)")
    try:
        import subprocess
        r14 = subprocess.run(
            [sys.executable, "/root/build/v55_client.py", "90"],
            capture_output=True, text=True, timeout=1800)
        _last14 = (r14.stdout or "").strip().splitlines()
        log("p14: rc=" + str(r14.returncode) + " "
            + (_last14[-1] if _last14 else "(no output)"))
        if r14.returncode != 0 and _last14:
            log("p14 stderr tail: " + " | ".join(
                (r14.stderr or "").strip().splitlines()[-3:]))
    except Exception as e14:  # noqa: BLE001
        log(f"p14: dress rehearsal failed ({type(e14).__name__}: {e14}) "
            f"— warmup coverage degraded, continuing")

    log("WARMUP_DONE")


if __name__ == "__main__":
    main()
