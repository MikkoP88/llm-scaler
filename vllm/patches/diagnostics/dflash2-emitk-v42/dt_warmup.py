# dt_warmup.py — llm-scaler v43b (crash-2 fix): boot-time request warmup.
#
# Defect: the draft/spec Triton kernels (TQ MQ stage1/2, eagle_prepare_*,
# copy_and_expand_dflash_inputs, batch_memcpy, _zero_kv_blocks,
# _compute_slot_mapping, ...) and the mixed-batch specializations JIT on the
# FIRST real requests. A multi-second host-side JIT window makes one TP rank
# late to the eager oneCCL collectives; peer-late windows > ~640 ms trip the
# xe GuC preempt watchdog -> engine reset on BOTH ranks (dmesg "Engine reset:
# engine_class=ccs" at 04:54:22 == user crash 2) -> UR_RESULT_ERROR_DEVICE_LOST.
# The engine's own warmup only synthesizes sampler inputs (gpu_worker.py
# #05b/#03); the drafter was never warmed.
#
# Fix: fire a fixed warmup sequence through the LIVE engine at boot (single,
# concurrent pair, one longer decode) so every first-use JIT happens in a
# controlled, ~symmetric boot window before any user traffic. Runs INSIDE the
# serving container as a background subshell (survives the exec of vllm
# serve); writes WARMUP_DONE to the serve log.
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


def fire(tag: str, max_tokens: int, prompt: str) -> float:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
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
    log("WARMUP_DONE")


if __name__ == "__main__":
    main()
