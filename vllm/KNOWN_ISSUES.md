
# 01. System Hang During Ubuntu 25.04 Installation with B60 Card Plugged In
The issue is caused by an outdated GPU GuC firmware bundled in the official Ubuntu 25.04 Desktop ISO image.

Workaround: Remove the B60 card before starting the Ubuntu installation, and plug it back in once the installation is complete.
We are also working with the Ubuntu team to address this issue upstream.

# 02. Limited 33 GB/s Bi-Directional P2P Bandwidth with 1x GPU Card
When using a single GPU card over a x16 PCIe connection without a PCIe switch, the observed bi-directional P2P bandwidth is limited to 33 GB/s.

Workaround: Change the PCIe slot configuration in BIOS from Auto/x16 to x8/x8.
With this change, over 40 GB/s bi-directional P2P bandwidth can be achieved.
Root cause analysis is still in progress.

# 03. Un-clean vLLM Shutdown (kill -9 / crash / docker rm -f) Wedges the xe Driver Until Host Reboot
Killing a vLLM XPU worker while a kernel is in flight (SIGKILL, OOM kill,
`docker rm -f`, engine crash) leaves a no-timeout job on the engine. The xe
driver can only reclaim it with an engine reset (`dmesg`: "Engine reset:
engine_class=ccs" + Xe devcoredump "Reason: LR job cleanup"), and that reset
silently breaks whatever process is live on the engine at that moment. The
next vLLM instance then hangs forever at its first device operation and
spams `shm_broadcast.py: No available shared memory broadcast block found in
60 seconds` (observed both at first request and mid-generation on long
contexts).

There is no in-host recovery: a sysfs FLR (`echo 1 > /sys/class/drm/cardN/
device/reset`) disables the mei_gsc firmware and makes both cards
unusable (`No XPU devices are available`). Only a host reboot clears it.

REBOOT NOTE (2026-08-28): the bench host sits on a fast short-lease DHCP
network — after every reboot it comes back on a NEW address, typically the
old IP + 1 (10.20.3.59 -> 10.20.3.60 observed). After issuing the protocol
reboot, scan for the host at <old IP>+1 (or the /24) instead of assuming the
address is stable; a new IP also means a new SSH host key
(`-o StrictHostKeyChecking=accept-new`).

ESCALATION (2026-08-27, v17 battery tail): WARM reboots are sometimes not
enough. After a long fault-heavy day (3 mid-decode xe faults + 5 protocol
reboots), the host entered a state where three CONSECUTIVE serve boots
faulted during warmup (ccs resets, GPU1 da:00.0 guc_id=24 twice then GPU0
b1:00.0 guc_id=38 once), each on a FRESH warm boot — plus `rmmod xe`
blocked ("in use"), `delayed_fput hogged CPU` kernel warnings, and an
xpu-smi query hang against the wedged container. The benchmark host is a
Dell PowerEdge R740 with iDRAC (`/dev/ipmi0`, `ipmitool` preinstalled):
`ipmitool chassis power cycle` = true COLD power cycle (full GPU power
drain; Power Restore Policy restores power automatically). Cold-cycle
whenever a first-serve-after-fresh-warm-reboot faults — do NOT keep
warm-rebooting into the same wall.

ROOT CAUSE + RESOLUTION (2026-08-27 late, proven by elimination): the
"3 consecutive boot faults" were NOT host degradation. Discriminators on
the same host, same GPUs, same drafter weights: rmacy v16 (eager + spec
active, all gates clean) and adv:v14 (graphs-off + spec active, log
prints "XPU Graph is disabled by environment variable") both boot
HEALTHY with zero resets; only adv:v17 + drafter faults (4/4). The v17
image ENV bakes `VLLM_XPU_ENABLE_XPU_GRAPH=1` — first enabled by the
~20:55 v17 rebuild (every pre-rebuild arm, incl. healthy spec boots
a1r/a3, ran graphs-OFF; "graphs-ON first-ever" was a5, post-rebuild).
With XPU graphs ON + the dflash drafter, the eager post-capture
`_dummy_run` (max_num_reqs x (k+1) spec tokens — a never-captured
shape) drives TP collectives that fault the xe ccs engine ~60 s after
capture → shm_broadcast wedge → boot death. Survives warm AND cold
reboots because it is in-image, not host state (the cold cycle DID
clear all host state — a9r3 then faulted identically on the pristine
host, which is what pinned the cause to the image). graphs-on + nospec
(a4-a7) is healthy — the spec shapes are required. FIXED in
llm-scaler-vllm-adv:v18 (gpu_worker.py only): the #05(b) post-capture
skip is extended from turboquant-KV-only to `spec_active AND
VLLM_XPU_ENABLE_XPU_GRAPH=1` (knob `VLLM_XPU_SPEC_SAFE_WARMUP`, default
1) — synthesized hidden states warm the sampler; the captured graphs
already exercised the kernels. The iDRAC cold-cycle protocol above
remains valid for genuine host-state wedges (rmmod blocked, xpu-smi
hang) — but boot faults that persist across a cold cycle are in-image:
bisect the image before touching the host again.

FIX VALIDATED LIVE (2026-08-28, arm `v17a9_default_spec` on adv:v18):
the skip fired verbatim ("skipping eager post-capture _dummy_run -
XPU graphs + speculative drafter - synthesized hidden states (320,
5120)"), boot healthy first-try in ~6 min, KV pool 142,317, and the
FULL suite ran zero-reset: gates 21.6-22.1 tok/s, 60k longctx PASSED
(138.3 s), deep10k @ temp 0.6/1.0 = 69.2/67.1 tok/s, conc4 66.5 tok/s,
graceful teardown with zero dmesg events. Same host, same env, same
drafter — the exact recipe that faulted 4/4 on v17.

Workarounds:
- Stop instances gracefully (`docker stop` / Ctrl-C, ~10 s grace) so the
  worker drain hook (`torch.xpu.synchronize()` + `empty_cache()` at exit,
  timeout-bounded) can finish in-flight kernels. Do not use `kill -9` /
  `docker rm -f` on a running server. Verified live: after a 248k-token
  generation, a SIGTERM teardown produced zero dmesg engine events and a
  restart on the same boot served requests normally.
- After any crash or hard kill, reboot the host before starting a new
  vLLM instance. vLLM probes the device at startup
  (`VLLM_XPU_STARTUP_PROBE_TIMEOUT_S`, default 60 s) and fails fast with
  this guidance instead of hanging; the worker step watchdog
  (`VLLM_WORKER_STEP_TIMEOUT_S`, default 600 s on XPU) bounds mid-run
  hangs the same way. IMPORTANT LIMITATION (verified live): the startup
  probe catches context-level device loss (hard errors, basic-op hangs)
  but NOT the post-SIGKILL silent wedge, which only manifests once both
  TP workers submit real collectives during warmup - on that class the
  probe passes, startup then hangs with the `shm_broadcast` spam, and a
  reboot is still the only fix. Graceful shutdown is the primary
  defense.
- Do not run multiple concurrent `xpu-smi dump --metrics ALL` loops
  against the GPUs while serving. Stacked monitoring pollers (observed:
  8 concurrent dump loops from a broken `pgrep` guard in a launch
  script) coincide with warmup-time dual engine resets
  (`UR_RESULT_ERROR_DEVICE_LOST` on the first request). One pair of
  pollers was harmless; several were not. Guard patterns must match the
  real process name (`pgrep -f 'xpu-smi dump'`).
- Note: the AI host gets a NEW DHCP IP on every reboot (observed roaming
  10.20.3.44 -> .45 -> .46 -> .47, and .55 -> .56 -> .57 on 2026-08-27).
  If the last known IP stops answering after a reboot, probe the next
  IPs (+1, +2) before assuming the host is down.

# 04. Silent TP Corruption When the all_reduce Custom Op Runs Under Inductor-Lowered Pieces with XPU Graphs Disabled (adv images v4-v9)
Serving qwen3.8-27b-fp8 TP=2 with `VLLM_XPU_ENABLE_XPU_GRAPH` unset/0 (but
without `--enforce-eager`) returned deterministic garbage from the very first
token while `/health` stayed green. The corruption is in PREFILL: every config
produced byte-identical garbage, i.e. the hidden state is wrong before decode
even starts.

## True root cause (proven by instrumented run + inductor codegen, 2026-08-26)

Fork commit 07827c0 added a `torch.compiler.is_compiling()` bounce in
`XpuCommunicator.all_reduce` onto the registered `torch.ops.vllm.all_reduce`
custom op, and removed the `output = input_.clone()` so the underlying oneCCL
all-reduce runs in place and the function returns the INPUT tensor itself.
That breaks the custom-op contract the op was registered under: the schema
declares `mutates_args=[]` (pure) and the fake impl returns
`torch.empty_like(tensor)` — i.e. "returns a fresh tensor, does not touch the
input". The XPU runtime path returns an alias and mutates the input.

Under `VLLM_XPU_ENABLE_XPU_GRAPH=0`, the AR-containing pieces are lowered by
inductor into generated wrappers with static memory planning
(`TORCHINDUCTOR_CACHE_DIR/**/c*.py`). The planner TRUSTS the op contract: it
marks the op's input storage dead after the call and hands it to the next
buffer (`buf22 = buf17; del buf17  # reuse`). At runtime the op returns that
very storage, so the next fused kernel gets the SAME buffer as both input and
output, e.g. (measured, v9 image):

    buf18 = torch.ops.vllm.all_reduce.default(buf17, 'tp:0')   # returns buf17!
    buf20 = buf18; del buf18  # reuse
    buf22 = buf17; del buf17  # reuse        # == buf20's storage
    triton_red_fused_..._rms_norm_t_5.run(buf20, ..., buf22, ...)  # READS and WRITES same memory
    extern_kernels.mm(buf22, ...)            # then reads the clobbered workspace

A fused kernel reading and writing identical memory corrupts the hidden state
deterministically for every token, prefill included → byte-identical garbage
from token 1.

Instrumented proof (4-arm run, sitecustomize wrapping
`GroupCoordinator._all_reduce_out_place`, trace-transparent):
- All 698 op-path AR calls show `alias=True` (output data_ptr == input
  data_ptr) — yet partial sums are numerically SANE and identical on both TP
  ranks. The collective itself is CORRECT; the corruption happens when the
  consumer kernel reads the aliased buffer that inductor already reassigned.
- Graphs ON (coherent): same op, same alias — but the stack shows the op is
  invoked from dynamo-split MODEL frames (`eval_frame.py<qwen3_next.py<...`),
  NOT from an inductor-generated wrapper: eager tensor semantics, no static
  memory plan, so the aliased return is harmless. Immunity is a property of
  the execution path, not of the op.
- v10 (28ff055 gate): 0 op-path calls with graphs off (all ARs eager) —
  coherent, matching the codegen-free path.

## Trigger cell (corrected)

torch.compile active (no `--enforce-eager`) x `VLLM_XPU_ENABLE_XPU_GRAPH=0`.
BOTH dtypes corrupt: bf16 yields digit-salad ("6?/900922992119999/..."),
fp16 yields multilingual word-salad — the earlier "fp16+compile coherent"
exoneration does NOT hold under the dflash serve config (the classify()
letter-ratio heuristic also false-positived on CJK/Cyrillic garbage).
`--enforce-eager` is immune (no compile). rmacy v14/v15 ship vllm
0.21.1.dev0+gad7125a43 WITHOUT the fork patch and never reproduce it.

Fixed by 28ff055 (image `llm-scaler-vllm-adv:v10`): the bounce now
additionally requires `VLLM_XPU_ENABLE_XPU_GRAPH` enabled, so graphs-off
compile mode falls through to the plain oneCCL all_reduce like pre-07827c0
builds. Validated on v10 (coherent, and instrumented: zero op dispatches).

A follow-up attempt to harden the op itself (adv:v11, 108cfdd:
`ALLREDUCE_BOUNCE_FIX_v2`, cloning the input inside the op so the
custom-op out-of-place contract would hold on any future inductor-lowered
path) was a REGRESSION and is REVERTED. v11 battery evidence: PIECEWISE
graphs-ON serving (the only regime where the op is dispatched, thanks to
the v1 gate) corrupted in both nospec and spec k=4 arms (multilingual
word salad, spec acceptance collapsed to 0.0%), while graphs-off (gate,
no op dispatch) and the fp16 FULL_DECODE_ONLY champion stayed coherent.
Instrumented v11: op-path AR outputs read as 0.0 in 672/690 calls (vs
sane values on v9/v10), direct-path ARs sane, corruption deterministic
(byte-identical garbage across arms). Mechanism: under PIECEWISE XPU
graphs the op executes EAGERLY in the gap between captured pieces, once
per step; a fresh clone is freed every step (address cycling) and its
contents do not reliably reach the next captured piece, whereas the
alias return writes the stable activation buffer that the piece graph
was captured against. Under FULL capture the op body runs only at
capture time and the clone's address is frozen for every replay, which
is why the champion arm survived. Conclusion: the alias return is
LOAD-BEARING under PIECEWISE; the op must stay in-place/aliasing and
rely on the v1 gate for compile-safety.

Workarounds on v4-v9 images: serve with `VLLM_XPU_ENABLE_XPU_GRAPH=1`
(recommended, also fastest), or `--enforce-eager` plus
`-e DISABLE_ESIMD_GDN_OUTPROJ=1` (needed because true-eager bf16 otherwise
trips the `esimd_norm_gemv_fp8_blockscale: norm inputs must be fp16`
TORCH_CHECK in the M==1 GDN out-proj fusion — a separate, loud,
correctly-guarded fp16-only fusion). Do NOT use `--dtype float16` as a
workaround: fp16 is equally corrupted under compile+graphs-off, just with
a different garbage flavor.

## 05 — xe DEVICE_LOST family: spec-decode long streams, host-load cascades,
and TurboQuant graph-capture warmup (adv:v13/v14, b19d92f)

Three failure modes, one root signature — `UR_RESULT_ERROR_DEVICE_LOST`
(error 20) surfacing at `num_accepted_tokens_event.synchronize()` in the
dflash verify loop — plus one dtype-gate crash. Observed on 2× Arc Pro
B70, TP=2, adv:v14 image, xe driver, 2026-08-26 batteries.

### (a) Single-stream long generation dies mid-flight — v15/v15b clean-boot batteries passed; RANDOM recurrence confirmed 2026-08-27 (see addendum)

RECURRENCE 2026-08-28 (v19 car-game matrix, cell c2 = fp8_e4m3 KV + dflash
k4, bar-shape fp16/block128/FULL_DECODE_ONLY/maxlen 98304): 4 engine resets
(ccs guc 23/24 + bcs guc 31/32, BOTH GPUs) mid-generation ~2 min into the
4096-token sampled run; tail bucket collapsed to 3 tok/s; the same shape on
bf16 (c1) and turboquant_4bit_nc (c3) was reset-free in the same battery.
The protocol `shutdown -r now` afterwards did NOT complete — host hung in
POST (>20 min unreachable, ping sweep found it gone from the subnet) and
needed a physical power cycle. Add to the #03 playbook: after multi-reset
events, schedule the reboot via a mind that may have to power-cycle.

The v14 result table (kept below for the record) was a compound artifact
of host-state accumulation, a random boot-time wedge, and a request-layer
early-finish bug — NOT an intrinsic graphs-mode depth limit:

| arm | mode | died at | signature |
|---|---|---|---|
| k14 | graphs + dflash, bf16 KV | 32,395 | DEVICE_LOST at acceptance-event sync |
| kv3b | graphs + dflash, 3bit_nc | 25,279 | same |
| kvf13 | graphs + dflash, fp8 (build load) | ~20,275 | same |
| ns (control) | graphs, NO spec, bf16 KV | 2,889 | `RPC call to sample_tokens timed out`, worker wedged |
| tq4e | eager + dflash, 4bit_nc | ~4,336 | engine death mid-10k |
| k8e / k3e | eager + dflash, k8v4/k3v4_nc | completed 10k | coherent head+tail |

v14 forensics: every graphs-mode death showed the worker blocking ~5 s
after a xe "Engine reset" in dmesg, then `UR_RESULT_ERROR_DEVICE_LOST` at
`num_accepted_tokens_event.synchronize()` (the first blocking sync — an
observation point, not the cause), then exactly 300 s of dead air
(shm_broadcast RPC read timeout) → `TimeoutError: RPC call to
sample_tokens timed out` → HTTP 500. Critically, the v14 battery ran all
six arms back-to-back on one boot with engine resets in between and NO
host reboots — violating #03's recovery protocol.

v15/v15b (2026-08-27, fresh reboot per reset, `--max-model-len 80000`,
PIECEWISE graphs ON, dflash k=4, no `--enforce-eager`):

- **ctl arm (v14 recipe verbatim): ZERO deaths.** 18 min continuous
  spec-decode, 0 resets, 0 DEVICE_LOST, clean SIGTERM teardown. The
  request returned early at ~23.5k/40k tokens with HTTP 200 — see (d);
  the engine never faulted.
- **v15b forced validation (min_tokens pinned, same boot): 40,000-token
  gen finish=length WALL=700 s (57 tok/s avg) AND a 76,000-token
  full-window gen finish=length WALL=1880 s (40 tok/s avg), both
  COHERENT head+tail, then a 512-token gen at 8 s — zero resets across
  all 116k generated tokens.** Spec acceptance held at depth (mean
  4.7-5.0, per-position 90-100%). KV peaked at 53% (143,808-token pool).
- Random **boot-time wedge (~3/9 serves)**: hangs right after "Graph
  capturing finished", worker spins at ~80% CPU, health never greens;
  killing it leaves 4 engine resets. Hits the base config on fresh
  reboots too (2 consecutive), and is the same signature as v14's TQ
  warmup hangs (b). Recovery: reboot + relaunch (~2/3 of boots healthy).
  Mechanism suspect: oneCCL race at the first eager collective after
  graph capture.
- A concurrent docker build (57 compile jobs, load 42) still reliably
  triggers DEVICE_LOST across ALL serving arms within minutes — do not
  build images while serving; the GPUs recover by themselves once host
  load ends (verified by post-build matmul probe).
- USER PRODUCTION CRASH 2026-08-27 09:51 (same env/flags/clean boot as
  v15b, `docker inspect`-verified env identity): a 20-min
  `/v1/chat/completions` session using the model's generation_config
  sampling defaults (temp 1.0/top_k 20/top_p 0.95, acceptance 1.55-2.9)
  died with the classic signature — single ccs engine reset on one GPU
  then a ccs+bcs cascade, DEVICE_LOST at the acceptance-event sync. The
  v16 battery (same day, fresh reboot) REPLAYED that workload exactly
  (chat endpoint, default sampling, 15k gen -> 210 s idle -> 40k gen)
  plus greedy/temp-sweep controls: 6 healthy serves, ~265k generated
  tokens, ZERO resets, all COHERENT. Verdict: the crash is NOT
  workload-deterministic — it is the random instance-level xe fault
  family (same distribution as the boot wedge, which itself hit 2/8
  serves that day). Sampling defaults are a THROUGHPUT hazard (see
  PERF_TUNING temp curve), not a proven stability hazard. Operational
  protocol above (reboot after any reset; retry bad boots/serves)
  remains the mitigation; for long-lived production serves, a supervisor
  that relaunches after DEVICE_LOST + enforced host reboot is the
  practical answer until the driver-level race is pinned.

**Operational fix (validated):** (1) reboot the host after ANY xe engine
reset before serving again — never chain arms on a reset host; (2) if a
serve wedges post-capture, count it as a bad boot: kill, reboot, retry;
(3) for deep single streams set `min_tokens == max_tokens` (see (d));
(4) keep the v14 base env unchanged — the "fix levers" all failed
empirically: `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1` → GARBAGE output with AND
without spec (NaN hazard confirmed, acceptance 0.0); L0 legacy adapter
(`SYCL_UR_USE_LEVEL_ZERO_V2=0` + `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`)
→ coherent but 6 tok/s (4× slower); `CCL_ZE_IPC_EXCHANGE=pidfd` →
post-capture wedge on every attempt (unbootable; keep drmfd).

ADDENDUM 2026-08-27 16:52 (recurrence, driver-decoded): a healthy 0.75
serve (fresh boot, 14 min into a single-stream prompt test, 28.9k
tokens, spec acceptance mean 3.81 / 70.2% — normal operation) died
exactly per this signature: ccs engine reset on GPU0 →
`UR_RESULT_ERROR_DEVICE_LOST` (error 20) at
`num_accepted_tokens_event.synchronize()` → request hung (tokens
frozen, health still 200, workers spinning 90/83%). First-ever Xe
devcoredump captured for this family
(`/root/telemetry/captures/2026-08-27_165243_reset_1/`, 516 kB):
`Reason: LR job cleanup, guc_id=28`, hung job seqno=3194 finished=0,
context ccs28 timeslice=1 ms / preempt-timeout=640 ms, schedule state
0x250 = DESTROYED|RESET|BANNED (bits decoded against xe_guc_submit.c).
Mechanism: oneCCL's spin-wait collective kernels run on LR-flagged
(no-job-timeout) ze queues; a spin kernel that cannot be preempted
while the same rank's main compute queue has runnable work trips the
640 ms GuC preempt timeout → context ban → engine reset → DEVICE_LOST
for both ranks. GPU wait kernels cannot sleep/yield (no oneCCL env
changes this); xe 6.17 exposes NO preempt/timeslice module params
(only `force_execlist`, untested and needs reboot). The trigger is any
>640 ms host-side peer-late window; GC is already frozen by vLLM, so
the leading remaining source is XPU caching-allocator slow paths under
fragmentation/memory pressure (matches xpu.py's capture-size-cap
rationale and the dflash scratch-churn comments). Fixes shipped
2026-08-27 (patches/qwen38-dflash/dflash.py, hot-applied to adv:v14):
(1) DFLASH_STALL guard logging any propose() >150 ms so the next
occurrence pinpoints the phase; (2) grow-only scratch for the markov
latent gather (per-step allocator churn removed); (3) serve env
`PYTORCH_XPU_ALLOC_CONF=expandable_segments:True` (torch 2.11 XPU
allocator) to eliminate fragmentation-driven alloc stalls. Workload-
random: v15b's clean 116k tokens were protocol luck, not a fix.

ADDENDUM 2 (2026-08-27 19:49, adv:v17 PRISTINE-BOOT recurrence —
host-state hypothesis retired): a fresh reboot (0 prior resets, clean
dmesg), fresh v17 serve (arena pre-alloc #05e live, expandable_segments
live, dflash grow-only scratch + stall guard live, gmu 0.8, dflash k=4,
greedy) passed all gates then died 2 min 49 s into a deep40k at
15,341 tokens with the 16:52 signature verbatim: single ccs reset on
GPU1 (gpu da:00.0, guc_id=34), health STILL 200, workers spinning
99.5%/75.3%, BOTH GPUs pegged ~100% util at ~80 W in xpu-smi (the
oneCCL spin kernels burning both engines), client stream frozen. NO
DFLASH_STALL line preceded the reset (the only 4 were cold-JIT at
19:45) — the drafter is exonerated for this instance; the trigger is
again the collective spin path. Teardown of the wedged serve added the
documented 3 cleanup resets (1->4). Discriminator verdict: (a) the
fault is instance-random, NOT accumulated host state — a pristine boot
does not immunize; (b) all four shipped mitigations (stall guard,
grow-only scratch, expandable_segments, arena pre-alloc) do NOT
prevent this family — they fixed the OTHER failure modes (#05b/#05e
boot wedges), which is why v17 boots 3/3 first-try at gmu 0.8 where
v14 wedged ~1/4-1/3 of launches; (c) memory pressure shifts the odds
(v15b 2/2 deep survivals at 0.75 vs v17 0/2 at 0.8, but the 16:52
death was AT 0.75) — probabilistic, not deterministic. Operational
posture stands: shallow/interactive serving is rock-solid in every
healthy mode; deep single streams carry irreducible xe-fault risk
until the driver race is fixed — bound it with the supervisor
(serve_supervised.sh) + reboot protocol, and prefer gmu 0.75 for
deep-stream-only deployments (+26% KV pool at 0.8 is not worth the
shorter mean-time-to-fault for that workload class).

### (b) TurboQuant padded presets wedge in post-capture warmup (graphs ON)

With b19d92f (stride-safe flat view in `triton_turboquant_store`) the
historical `view(-1)` crash on padded KV pages is gone and graph capture
COMPLETES — but `turboquant_k8v4` and `turboquant_k3v4_nc` then hang in
`compile_or_warm_up_model` → `_dummy_sampler_run` → `make_dummy`
(metadata.py:41, a 64-int32 H2D copy): the queue is already poisoned by a
faulted kernel from the preceding eager `_dummy_run` (320 tokens = 64 reqs
× 5, a shape never captured). py-spy shows both workers spinning at 87%
CPU on that line; host dmesg shows GuC engine resets (bcs+ccs, both GPUs).
`turboquant_4bit_nc` (grow branch, code path untouched by b19d92f) hangs
earlier, DURING capture at 63% (12/19 sizes) — so this hang family
pre-dates the stride fix, which merely unmasked it for the padded presets.

`--enforce-eager` sidesteps the warmup hang: k8v4 comes healthy in ~3 min
and generates COHERENT greedy text (adv:v14e battery). The triton TQ
store/read kernels are stride-correct on padded pages — the
incompatibility is specifically the PIECEWISE-capture-context +
eager-collective warmup, the same interaction `xpu.py:443` warns about
for large capture lists.

RESOLVED 2026-08-27 (adv:v17, two-part fix in gpu_worker.py / vllm.py):

1. **TQ-safe warmup** — when `cache_dtype` starts with `turboquant` and
   `VLLM_XPU_TQ_SAFE_WARMUP=1` (default), the post-capture eager
   `_dummy_run` (the 64-reqs×5 shape that poisons the queue) is skipped
   entirely; sampler warmup runs on synthesized zero hidden states
   instead. `hidden_size` is resolved via `hf_config.get_text_config()`
   (hybrid configs like `Qwen3_5Config` nest it inside `text_config` —
   the first cut read `hf_config.hidden_size` directly and crashed
   cleanly; fixed the same day). If the size cannot be resolved the code
   falls through to the upstream eager path.
2. **Spec auto-disable** — requesting a drafter together with a TQ KV
   dtype on XPU now logs a WARNING and serves target-only
   (`VLLM_ALLOW_TQ_SPEC=1` overrides), instead of wedging in the drafter
   warmup.

   REVISED 2026-08-28 (adv:v19): the auto-disable is now OFF by default —
   `VLLM_ALLOW_TQ_SPEC` defaults to **1** (`=0` restores target-only). Both
   original causes are gone: the drafter's own KV is forced unquantized
   `auto` (dflash.py), and the #05b/#03 warmup skips (items 1/3) protect
   boot. The remaining TQ+spec perf blocker (5x KV rescan in verify — the
   "synthetic decode" per-token loop) is fixed by the v19 multi-query
   verify kernel (`turboquant_attn.py` dispatch + MQ kernels in
   `triton_turboquant_decode.py`; rollback `VLLM_TQ_MQ_VERIFY=0`). NEW
   bound discovered with the drafter actually loading under TQ: the
   drafter's UNCOMPRESSED KV pool is sized by the same max_model_len —
   ~11.3 GiB/GPU @262144 vs 6.67 GiB free after its fp8 weights — so
   TQ+spec needs max_model_len ≲118k at gmu 0.8 (measured: 13.48 GiB
   needed vs 6.67 available; nospec TQ4nc fits 262144 with 4.62x
   concurrency). Long-context + spec requires quantizing the drafter KV
   (follow-up).

3. **Generalized in adv:v18** — the post-capture skip is no longer
   TQ-only: `spec_active AND VLLM_XPU_ENABLE_XPU_GRAPH=1` also skips
   the eager post-capture `_dummy_run` (knob
   `VLLM_XPU_SPEC_SAFE_WARMUP`, default 1). The same never-captured
   spec shape (64 reqs × (k+1)) faults the xe ccs engine even on
   bf16/fp8 KV once graphs are on — this was the #03 boot-fault
   cluster (a8/a9/a9r/a9r3, 4/4). See #03 ROOT CAUSE.

Live validation (2026-08-27 battery, arm `v17a5_k8v4_spec`): k8v4 booted
healthy in 330 s **with graphs ON** — first time ever (every prior boot
wedged per this section). Gates: warm512 17.8-19.3 s @ 26.5-28.8 tok/s,
1536 in 59.9 s @ 25.7 tok/s — ~3.7× the historical `--enforce-eager`
fallback (7.3 tok/s). Auto-disable fired on both launch attempts.

NEW BOUND (same battery): the 60k-token longctx probe (60k prompt +
1536 gen) on k8v4 tripped the (a) xe-fault family — single ccs engine
reset (GPU0 b1:00.0, guc_id=29) ~79 s into the probe, right at
end-of-prefill / start-of-decode at 60k depth, with spec already
auto-disabled (pure TQ target-only, no drafter in play). Health stayed
200 (zombie serve), standard teardown + reboot protocol applied.
Shallow-context serving (≤~2k depth gates) was clean before and after,
and a continuous 5k-token deep gen on k8v4 in the same battery
(`v17a6_k8v4_deep5k`) completed 5000/5000 @ 26.65 tok/s with ZERO
resets. Counter-data point: the SAME 60k longctx probe on
`turboquant_4bit_nc` (`v17a7_tq4nc_longctx`) PASSED cleanly (wall
100.7 s, 15.25 tok/s, ttft 32 s, zero resets) — so the extreme-depth
hazard is k8v4-specific or a random-family hit (n=1 each; the (a)
family is instance-random), NOT TQ-universal. Conclusion: TQ presets
are now *bootable and fast* with graphs (both flavors), k8v4 should be
kept out of ≈60k-deep contexts until more data exists (fp8/bf16 KV are
validated clean there), and the standard supervisor/reboot protocol
covers the residual risk.

### (c) `--kv-cache-dtype float16` is rejected outright

`reshape_and_cache_flash` (flash-attn KV write op, upstream csrc compiled
in-image) raises `RuntimeError: Unsupported data type of kv cache: float16`
during first capture on both v13 and v14. Supported: bf16 (default with
dtype none), fp8. fp16 offers no memory advantage over bf16 anyway — use
the default, or fp8 for 2× KV capacity (validated coherent to 15k+).

### (d) `ignore_eos` long gens can return early with HTTP 200 (EOS leaks through the dflash path)

Discovered in the v15 battery: `ignore_eos: true` 40k requests returned
`200 OK` after ~19-24k tokens (three independent configs: base,
in-graph-collectives, L0-legacy), text ending in a run of hundreds of
newlines (greedy whitespace collapse at depth), decode still healthy. The
engine never faulted — the sequence was simply FINISHED early; the stop
decision slips past `ignore_eos` somewhere in the dflash/async-scheduling
finish path. Not fixed in code yet.

FIXED 2026-08-27 (adv:v17, v1/core/sched/utils.py `check_stop`):
`ignore_eos` is now a hard guard on every stop branch — the eos check,
stop_token_ids (default-sampling trap: model generation_config stop
ids honored only when ignore_eos is false), and the length cap (which
previously let an early-window finish through). Early/ambiguous
finishes log `FINISH_DIAG` lines (stop path + token counts) so any
residual leak is visible in serve logs instead of silent. Client-side
`min_tokens == max_tokens` pinning remains good hygiene for deep gens
(it also protects against pre-v17 servers).

Workaround (validated in v15b): set `min_tokens` equal to `max_tokens` in
the request. Both 40k and 76k gens then return `finish=length` with
exactly the requested `completion_tokens`, coherent end-to-end. Any
client doing deep `ignore_eos` generation should pin `min_tokens`.

### (e) Startup wedge at the FIRST TP collective — UR error 39 (OUT_OF_DEVICE_MEMORY) — RESOLVED 2026-08-27 (`--gpu-memory-utilization 0.75`)

Symptom ("vllm shut down but docker running"): a fresh serve gets
through graph capture, then health never greens; one VLLM::Worker spins
at 65-101% CPU; every 60 s exactly one
`shm_broadcast.py:681 No available shared memory broadcast block found
in 60 seconds` line repeats forever; the container stays alive because
PID 1 is bash (serve is a docker-exec child). Hit on ~1/4-1/3 of serve
launches at `--gpu-memory-utilization 0.8` (4 occurrences on
2026-08-27, zero workload required — fires before any prompt arrives).

Golden evidence (live capture, instrumented serve at 0.8,
`/root/telemetry/captures/` on the benchmark host):

```
RuntimeError: level_zero backend failed with error: 39
(UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
  ... tensor_model_parallel_all_reduce
  File ".../vocab_parallel_embedding.py", line 489, in forward
  ... drafter.load_model -> embed_input_ids(dummy)   [dflash drafter warmup]
```

i.e. the FIRST oneCCL TP all-reduce on a fresh L0 context collides
with the device-memory ceiling. One rank dies at that first
collective; the peer rank spin-blocks inside oneCCL forever (~80% CPU)
— that IS the wedge. The 60-second shm_broadcast line is the
reader-side poll heartbeat of that one-sided deadlock, NOT a stream of
new errors. The driver later reaps the orphaned GPU work
(`Kernel-submitted job timed out ... in no process [-1]` → xe engine
resets) — the "4 resets after killing a wedged serve" signature,
linking this family to #03's reset-recovery protocol.

One mechanism, three costumes: (1) error-39 crash + wedge; (2) silent
wedge (same spin, no surfaced error); (3) post-kill engine-reset
cascade.

Fix (validated 2026-08-27): `--gpu-memory-utilization 0.75` leaves
oneCCL/L0 scratch headroom for the first collective. The
first-collective window is slow (~3 min, up to 3 transient
shm_broadcast 60 s timeouts) but COMPLETES → healthy serve (99.99%
GPU util, 183 W under load). Scope: 0.75 fixes ONLY the startup wedge
— a mid-flight GuC long-runner reset (family (a), see addendum below)
still struck 14 min into the first 0.75 test, at 28.9k tokens. Cost: KV
pool 143,808 → 112,479 tokens (~22% smaller, still 1.4× the 80k
max-model-len window).

Deeper candidate fix (fork): run one dummy TP
all-reduce BEFORE loading weights (while device memory is still free)
to pre-allocate the oneCCL scratch arena, restoring 0.8 utilization.

RESOLVED 2026-08-27 (adv:v17, `VLLM_XPU_PREALLOC_CCL_ARENA=1` default
on, gpu_model_runner.py load_model start): exactly that fix — a
min(max_num_batched_tokens, 8192)-token TP all-reduce scratch arena is
allocated and freed before KV pool sizing. Validated live: 3/3
first-try healthy boots at `--gpu-memory-utilization 0.8` (previously
~1/4-1/3 wedge rate) with zero UR39, and KV pool restored to 142,317
tokens (+26% vs the 0.75 workaround's 112,479 — matches the pre-fix
0.8 pool exactly, i.e. no hidden memory cost). Scope unchanged: this
fixes the STARTUP wedge only; the mid-flight (a) family is unaffected
(see addenda). `--gpu-memory-utilization 0.75` remains available as a
deep-stream risk-reduction lever.

CEILING NOTE (v17 battery, arm `v17a8_gmu09_spec`): at
`--gpu-memory-utilization 0.9` graph capture completes (19/19 sizes,
2.15 GiB, KV pool 204,231 tokens = +43% vs 0.8) but a ccs engine reset
fires ~60 s AFTER capture in the post-capture warmup collective phase
(GPU1 da:00.0, guc_id=24), followed by the shm_broadcast 60 s heartbeat
wedge.

REVISED VERDICT (2026-08-27 late — supersedes the confound note): the
a8@0.9 fault and the "confounding" a9@0.8 fault are the SAME bug, and
it is NOT headroom: it is the #03 root cause (XPU graphs + dflash
drafter; eager post-capture `_dummy_run` on never-captured spec shapes
→ xe ccs reset). Both arms loaded the drafter on the post-rebuild v17
image; every healthy 0.8 boot that day either predated the rebuild
(graphs off) or ran without a drafter. The "post-capture collectives do
not fit at 0.9" headroom theory is retired; the #05e arena fix itself
stands (3/3 healthy pre-rebuild boots at 0.8 with the fix, KV pool
+26%). 0.9 is UNTESTED since the v18 fix — the +43% KV pool datapoint
(capture completed) makes it a worthwhile future cell, now that the
graphs+spec boot fault is fixed in adv:v18.

Workarounds (section 05 footer): interactive requests ≤1536 tokens are
rock-solid in every healthy mode; deep single streams are safe on a
clean host with graphs ON (76k full-window validated) — pin
`min_tokens`, and follow the (a) operational protocol (reboot after any
reset; retry bad boots). A serve that wedges in the first ~5 min at 0.8 utilization is (e) —
kill, reboot the host, relaunch at 0.75; never retry a wedge on an
un-rebooted host (pending resets poison it, #03). fp8 KV + dflash is the best validated
capacity/speed point; TQ presets boot WITH GRAPHS on adv:v17 (26-29
tok/s single stream, (b) resolved; tq4nc even cleared the 60k longctx)
— keep k8v4 out of ≈60k-deep contexts (see (b) NEW BOUND); fp8 or bf16
KV for long-context work.

## 06 — TurboQuant x spec-decode: XPU verify graphs baked context-blind
(FOUND + FIXED 2026-08-28, adv:v19b; present but unreachable ≤v18 because
the #05b guard dropped the drafter under TQ)

**Symptom.** With `--kv-cache-dtype turboquant_*` + dflash k=4 +
`cudagraph_mode FULL_DECODE_ONLY`, generations hallucinate a nonexistent
context (greedy: "The user just sent a blank message…"; sampled: "user
just said hello") and spec acceptance collapses (sampled per-position
0.000; greedy ~8% = row-0 markov coincidences only). Byte-identical
across boots (seed 0). Nospec TQ serving is perfectly healthy, and the
drafter is healthy (DFLASH_DEBUG dumps: sane hidden states/logits).

**Root cause (proven by env-gated instrumentation, probes 11-12).**
Under spec, verify steps run as XPU graph REPLAYS (decode-graph machinery
at bs×(k+1) tokens). At capture, `build_for_cudagraph_capture` builds
dummy metadata with `seq_lens == q_len == k+1`, so in
`_prefill_attention` the raw-KV flash fast path
(`max_query_len == max_seq_len`) fires and the graph records flash-varlen
over ONLY the in-batch verify tokens — the KV cache is never read. At
replay, dynamic buffers (seq_lens/block_table) are refreshed, but Python
never runs again: every real verify step attends just its 5 tokens, so
the target is context-blind. This also explains why the v18→v19 verify
kernel changes never altered the bug (probe8: `VLLM_TQ_MQ_VERIFY=0`
bit-identical) — the continuation path was never reached. The earlier
"greedy accepts 2.25 tok/step" read was the blind target's row-0
agreement with the markov-1 drafter (both condition on the single last
real token), not health.

**Fix (v19b, `turboquant_attn.py`).** (1) In
`build_for_cudagraph_capture`, multi-token captures force the
continuation path (`max_seq_len = max_query_len + 1`, seq_lens_cpu/seq_lens
bumped) so the graph records the KV-reading kernels instead of the flash
fast path; decode (q_len==1) graphs are unchanged. (2) The continuation
call sites derive per-row causal limits from the DYNAMIC `seq_lens`
buffer (`seq_lens[i:i+1] - (q_len-1)` / `+ arange(q_len)`) instead of
slicing a static arange with capture-time CPU scalars — captured GPU ops
re-read the refreshed buffer at replay, so attention extent is always
the real context length. Rollback: `VLLM_TQ_VERIFY_GRAPH_FIX=0` (not
recommended — restores the blind graphs); `cudagraph_mode NONE` also
sidesteps (eager verify is correct but slow).

**Validation.** Deterministic arms that failed identically on 6 boots
(probes 3-12) all produce correct HTML-car-game text on v19b; acceptance
8% → 25.5% greedy (49/192, ~2.0 tok/step gross), sampled 21-46%
windowed, mean acceptance length 1.87-2.85; zero engine resets.

## 07 — fp8 KV x nospec x FULL_DECODE_ONLY: ESIMD decode fast path D2H-syncs
inside XPU graph capture (boot crash)
(FOUND + FIXED 2026-08-28, adv:v19c; pre-existing — v19 changed nothing in
flash_attn.py)

**Symptom.** `--kv-cache-dtype fp8_e4m3` + nospec + `FULL_DECODE_ONLY`
crashes at boot 4/4 deterministic (262k and 98k maxlen, so not a memory
issue), while every other nospec cell (bf16, tq4nc, k8v4) boots and fp8+spec
(c2) boots. Worker dies with `RuntimeError: wait method cannot be used for
an event associated with a command graph` at `_warmup_and_capture` →
`_dummy_run` → `flash_attn.py:1136`.

**Root cause.** The ESIMD page-attention decode fast path (fp16 query +
XPU-graph env) reads the fp8 KV descale factors with
`_k_scale = float(layer._k_scale)` — `float()` on a device tensor is a D2H
sync, and any sync inside XPU graph capture raises the command-graph event
error. bf16 KV takes the else-branch (python `1.0`, no sync) and boots; TQ
dtypes never enter flash_attn; fp8+spec never captures single-token target
decode graphs (dflash replaces decode with k+1 verify), so only the
fp8 x nospec combination hits the sync — deterministically.

**Fix (v19c, `flash_attn.py`).** The scale tensors are static after model
load, so cache the python floats per layer (`self._esimd_kv_scales`) at the
first eager call; if capture ever precedes every eager call, skip the ESIMD
fast path for that capture (falls through to `flash_attn_varlen_func`)
rather than bake wrong scales into the graph. The kernel-dispatch lines are
guarded on `eagle_ops is not None` accordingly. Rollback:
`VLLM_ESIMD_F8_SCALE_FIX=0` restores the original read (A/B only — it
re-crashes capture).

**Validation.** nfp8 boots first try: 4096 tokens, steady 33.34 tok/s (the
fastest nospec cell), zero engine resets; nbf16 re-run on the same image is
unchanged (33.11 vs 33.10 tok/s) — the bf16 path is untouched.

## 08 — v19 bench client counted SSE delta EVENTS as tokens: every spec-decode
"steady tok/s" underreported ~1.9x (the "spec loses to nospec" result was an
artifact)
(FOUND + FIXED 2026-08-28, adv:v20; measurement bug only — the engine was
never at fault)

**Symptom.** The v19 car-game matrix read "healthy TQ spec at 17.56 tok/s
loses to nospec 32.79" and a derived "95 ms/step, 3.15x overhead" — driving
the whole v19→v20 step-cost reduction plan.

**Root cause.** The client tallied one token per streamed `delta` event. With
spec decode the detokenizer flushes the whole accepted block per event
(~E[len] ≈ 1.9 tokens/event), so spec cells underreported by exactly
tok/event; nospec emits 1 token/event and was correct. Proof chain (all on
the v19 image): (1) engine `SpecDecoding metrics` windows satisfy
`emitted = mean_accept_length x drafted/k` and give c3 ≈ 35.6 true tok/s;
(2) `VLLM_SPEC_TIMING` step instrumentation measured 56-63 ms/step, i.e.
32-36 tok/s at ~2.0 tok/step — the 95 ms figure was derived FROM the bad
client number; (3) re-runs with the fixed client reproduce the engine rates
(bar 32.79 / c3 34.09 vs 35.6 engine / k2 39.90 vs 39.8 engine) while the
legacy event metric reproduces v19's numbers (17.60 ≈ 17.56) on the same
runs.

**Fix (v20, `cargame_client.py`).** Request
`stream_options: {"include_usage": true}` and report `usage.completion_tokens`
(`tokens_true`), `tok/event`, and true steady = event rate x tok/event, with
the legacy event metrics kept alongside and a loud fallback warning if the
server omits usage.

**Lesson.** Never compare throughput across spec/nospec arms using streamed
event counts; use server-side token counters (`include_usage`, or the engine
SpecDecoding windows, which remain the authoritative steady source). The
authoritative steady source for spec cells = serve-log SpecDecoding windows
(tail-6).

