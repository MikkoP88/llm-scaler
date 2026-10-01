# STALL_ROOT_CAUSE.md — multi-agent "hard stalling" incident (2026-09-30, P32)

Task trigger: "do deep analyze there is now running multiple agents and has
major issue multiple sessions has hard stalling, this has to be solved, using
any possible solutions" + "start the weeks-scale scaled-space fp8 SSM kernel
project" + "Start implementing improvements but currently do not stop running
vllm instance."

## Verdict (two independent causes, both convicted by live evidence)

**Cause A — LANE CONFIG DRIFT (the trigger).** At 08:56 today, ~5 min after a
host reboot, the lane was relaunched NOT by the certified
`repro_bootV1226_prod.sh` chain but by a manually pasted `docker run` +
interactive serve (root `.bash_history` lines 1072-1080, pts/1 launch). The
running posture deviates from the certified v1.2.26 posture in four ways:

| Deviation | Certified (v1.2.26) | Running (08:56) | Effect |
|---|---|---|---|
| Model mount | `/models/qwen3.8-27b-fp8` (native fp8 ckpt) | `/models/swift-qwen3.8-27b` — **52 GB bfloat16, `quantization_config` ABSENT** | 2× weight memory load + on-the-fly quant path |
| Quantization flag | (none — inferred from ckpt) | `--quantization fp8` forced | Unmeasured W8A8 dynamic path for a bf16 ckpt |
| Chat template | NONE (tokenizer default; "no-template serve" is a ship gate) | `--chat-template /root/chat_template_qwen38_high.jinja` (vision-style template) | Unmeasured prompt formatting; quality posture unknown |
| Serve launch | `docker exec -d bash /root/serve_user.sh` → `/root/serve_full.log` | interactive pts/1, **`/root/serve_full.log` untouched since 04:44** | Zero engine telemetry (violates the crash-4 lesson baked into serve_user.sh) |

**Cause B — LONG-CONTEXT LOAD REGIME beyond the certified envelope.** The
multiple agent sessions are Claude Code-class clients resending whole
conversations every turn. Live metrics at 12:00 (engine cumulative since boot):

- 247 requests, **avg prompt 48,214 tokens** (53% of requests 5k–50k; 47%
  50k–100k; bucket evidence `request_prompt_tokens_bucket`).
- Cumulative prefix-cache hit 91.4% (10.67M cached / 11.67M total) — good —
  but the 1.003M `local_compute` tokens concentrate in ~10 full-prefix
  RE-PREFILLS (first turns + evictions) of ~100k tokens each.
- **TTFT: p50 ≈ 4 s, p90 ≈ 15 s, 8 requests 40–160 s, 2 requests 160–640 s**
  (`time_to_first_token_seconds_bucket`). Those tail requests ARE the
  "hard stall" the agents experience — a 100k-token re-prefill at the
  measured effective prefill throughput (~1.1–2.5k tok/s under contention)
  costs 40–90+ s, during which chunked prefill also starves every other
  session's decode cadence.
- Per-request TPOT: **no request averaged > 0.75 s/token** (buckets cap at
  0.75 = 243/243) — decode is FLOWING, avg 62.8 ms/tok at 5 concurrent
  long-context streams. So this is NOT a GPU wedge / engine stall.
- KV pool ~493k tokens (fp8_e4m3 KV, +3.5% v1.2.25) fits only ~4–5
  concurrent 100k-context sessions → at higher concurrency the LRU evicts a
  live session's prefix → its next turn re-prefills 100k → stall cascade.
  (At the 12:00 sample: 5 running, KV 16.4%, waiting 0, waiting_by_reason
  capacity/deferred 0 — the storm had passed; the counters keep the scars.)

**Exonerated by evidence:** GPU wedge (current-boot `dmesg` Engine-reset
count = 0; the f15b_dmesg.log reset lines are historical baked content — the
image was a warm lane-commit); the 09:10 f15b capture is a BOOT-PHASE
false-fire (py-spy dump #1 stack = EngineCore in `_init_executor`, dump #2 =
idle `run_busy_loop`); litellm errors 0 in 3 h; preemption/retraction lines 0;
spec MTP acceptance healthy (3.19 mean).

## Why certified batteries missed this

Every certified battery (drills/serial/bursts/SHIPMAT) uses ≤1024-token
contexts with fresh small prompts. The 48k-avg-prompt, 5–8-stream, turn-taking
CC-fleet regime was never in any gate. v63/v64 scheduler knobs were tuned and
frozen on that small-context regime (V63=512 REJECTED, floor guard ≥1024).

## Fix layers (see FIX_AND_TEST_PLAN.md)

1. **Restore certified posture** (model, quantization, template, serve chain,
   log capture) — needs one lane restart (maintenance window; the user
   directed the running instance NOT be stopped now).
2. **Long-context regime tuning round** — mnbt / V63 contended budget / gmu
   legs measured with a CC-fleet replay harness (new gate).
3. **Scaled-space fp8 SSM kernel project (weeks-scale, started)** — halves
   the GDN pool and removes the gather/scatter roundtrips → more KV headroom
   at the same gmu → fewer evictions → kills the re-prefill storm class.
4. **Lane-config drift guard** — watchdog v2 checks the SERVE CONFIG (model
   path, no --quantization, no --chat-template, fresh serve log), not just
   health 200, so another agent's manual relaunch cannot silently detune the
   lane again.

---

## AMENDMENT (2026-09-30, later user directive — supersedes parts above)

User directive: **`swift-qwen3.8-27b` + `--quantization fp8` +
`--chat-template chat_template_qwen38_high.jinja` are the DESIGNATED
certified configuration.** The 08:56 model/quant/template choice was
intentional, not drift. Re-reading the table above under that directive:

- The **config deviations** column becomes the designated posture
  (plus `--served-model-name qwen3.8-27b-fp8`, which the 08:56 launch also
  carried — that is what kept litellm's `openai/qwen3.8-27b-fp8` fleet
  mapping working).
- Cause A shrinks to its true residue — both still real, both still fixed:
  1. **UNCERTIFIED ROLLOUT**: the posture ran outside the boot chain (no
     patcher verification, no marker discipline, no battery ever run on
     swift+on-the-fly-fp8+template) — designated ≠ certified until W1's
     battery passes (FIX_AND_TEST_PLAN Layer 1).
  2. **TELEMETRY BLINDNESS**: stdout to a dead pts, `serve_full.log` stale —
     the reason the stall had to be forensically reconstructed instead of
     watched. Fixed by the boot chain + watchdog v2.
- Cause B (long-context re-prefill storms) is posture-independent and
  stands unchanged — it is the actual stall mechanism and Layers 2/3
  address it.
- Watchdog v2 semantics invert accordingly: drift now = serving anything
  OTHER than the designated posture (or losing telemetry); the template's
  presence inside the container is itself a drift tripwire because the file
  is NOT in the v1.2.26 image — only the boot chain stages it.

