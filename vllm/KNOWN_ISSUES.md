
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

ADDENDUM 3 (2026-08-29, v21b bar cell — instance-random, reboot-clean): a
healthy tq4nc nospec @262144 serve (fresh boot, 2 min into the canonical
car-game run) died at ~2300/4096 tokens with 4 ccs/bcs engine resets (both
GPUs) and a frozen stream — the same family, nospec this time. dmesg
showed GuC 70.44.1 loaded vs the 70.49.4 recommended by the driver
package (a post-reboot firmware-version drift is suspected but unproven).
Protocol reboot resolved it cleanly (host DHCP-moved .61 -> .62); the
subsequent v21c/v21d/v21e chains (12+ serves across all KV dtypes, spec
and nospec, including 262144 spec cells) ran with ZERO resets.

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

## 09 — MTP + cudagraphs: oneCCL allgather segfault inside the eagle_head
torch.compile warmup (MTP dead with any graphs mode on XPU TP=2)
(FOUND + FIXED 2026-08-29, adv:v22; user-reported "MTP do not work" on all
images incl. intel's)

**Symptom.** `--speculative-config {"method":"mtp",...}` with
`--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}` (or PIECEWISE)
dies during boot on both TP ranks:

```
sycl queue_impl::submit_impl -> invoke_barrier ->
ccl allgatherv_large_su_ring<half> -> ... ->
c10d ProcessGroupXCCL::allgather_into_tensor_coalesced ->
all_gather_into_tensor -> pythonFallback (dynamo_eval_custom_code)
!!!!!!! Segfault encountered !!!!!!!  -> VllmWorker died unexpectedly
```

**Isolation matrix (v21 image, one boot per cell).** graphs crashed at k=4
AND k=1; `--enforce-eager` booted at both (E[len] 3.38 @ k4). k, KV dtype,
TQ, our overlays, parsers: all irrelevant. Raising
`CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD` did not help (user-tested). Same crash
on the stock intel image.

**Root cause.** The MTP head classes (`Qwen3_5MTP` + inner
`Qwen3_5MultiTokenPredictor`) are `@support_torch_compile`-decorated; with
graphs mode `llm_base_proposer.initialize_cudagraph_keys` forces the drafter
to PIECEWISE, so the head enters torch.compile (tag `eagle_head`). The
head's sampling path issues oneCCL allgathers — the full-vocab fp16 logits
gather (`LogitsProcessor._get_logits/_gather_logits` behind
`compute_logits().argmax()`) or the padded [batch, 64] gather in
`get_top_tokens` — from inside dynamo-evaluated code, and that combination
segfaults. The identical collectives run fine eager (the matrix's eager
cells). #05 family: eager collectives x compiled regions on oneCCL/xe. The
TARGET backbone is unaffected (its collectives run inside full XPU graph
capture, which is validated).

**Fix (v22, `VLLM_XPU_MTP_EAGER_HEAD`, default 1 on XPU, =0 = stock).**
(1) `qwen3_5_mtp.py`: module-tail `ignore_torch_compile()` on
`Qwen3_5MultiTokenPredictor`, `Qwen3_5MTP`, `Qwen3_5MoeMTP` — the head never
compiles. Both decorated classes must carry the key (the decorators
machinery only skips the exact class holding it; the inner `self.model` is
separately decorated). (2) `llm_base_proposer.py`:
`initialize_cudagraph_keys` forces the drafter to NONE for `method=="mtp"`
(no compiled subgraphs exist to capture piecewise; also selects the
`direct_eager_inputs` propose fast path). Method-gated: dflash/eagle
drafters keep PIECEWISE. Effect: target keeps FULL_DECODE_ONLY graphs, the
single-layer head runs eager — MTP k4 boots and reaches steady 72-76 tok/s
on the canonical test (2.2x the dflash k2 user config), E[len] 4.2, 0
resets.

**2026-09-02 close-out (v35dp1 re-test).** `VLLM_XPU_MTP_EAGER_HEAD=0`
on the current tree (v31.1 lineage) boots CLEAN — zero segfaults, head
compile warmup passes, decode graphs captured 18/18. The crash site is
gone by restructure: the head's `forward` now returns hidden states
only (`compute_logits`/`get_top_tokens` are invoked by the proposer's
`_greedy_sample`, eager by construction), so the logits allgather no
longer sits inside any dynamo-evaluated region. Historically closed;
moot in practice because the v31.1 guard (#11) sets
`TORCH_COMPILE_DISABLE=1` for spec+TP2 before the head could compile —
and the PIECEWISE-registered-without-compile drafter state fails the
hash gate anyway (see #17 v35dp1 addendum).

**Observed alongside (NOT a v22 issue, documented for awareness).** On
2026-08-29 the chat endpoint (any image, incl. v21 replicas, any spec method)
stopped producing `delta.content` within 4096 tokens: the model's `<think>`
phase runs past the budget (coherent planning text, no corruption, no
marker-loss; the fork classifies it under `message.reasoning`). The
completions endpoint (no chat template) finishes naturally in ~3200-3500
tokens. Discriminator: v21-image dflash replica same-day reproduced the
identical profile (0 content events, engine 30-33 tok/s) as v22 —
environmental/model-behavior shift after the host reboot, not code. Chat
content-event benches are therefore not comparable across days; use
`bench_completions.py` or engine SpecDecoding windows.

**Lesson.** Speculative-draft heads that sample through TP-gathered logits
must not be torch.compile'd on this oneCCL/xe stack; keep drafters eager
(their device cost is negligible next to the graphed target verify) — and
gut-check "spec changes output quality" claims with a same-day eager/stock
replica before blaming the new code.


## 10 — "Prefix cache hit rate: 0.0%" under MTP is architectural, not a bug:
hybrid GDN page unification forces 4096-token prefix granularity on XPU
(ANALYZED 2026-08-29, adv:v22 live serve; no code defect found — do not
"fix" the scheduler)

**Symptom.** With `--enable-prefix-caching` + MTP the logged hit rate stays
0.0% for chat-sized prompts; repeated identical prompts of 2-8k tokens reuse
nothing, while the same server without speculative decoding shows hits. Old
logs eventually show non-zero rates only from large-prompt traffic.

**Root cause (proven offline-sim + live-measured).** Not the eagle/MTP
cache-drop machinery and not an insertion bug: replaying the scheduler's
`allocate_slots` sequences through the container's real `KVCacheManager` /
`HybridKVCacheCoordinator` / `MambaManager` (`.tmp-tq/sim_prefix.py`,
`sim2_prefix.py`, `sim3_prefix.py`) shows blocks insert and look up
correctly. The engine simply cannot reuse at fine granularity:

1. qwen3.8 is a GDN hybrid; one GDN cache page = one fixed-shape state
   snapshot (~0.8 MB regardless of token count). The hybrid pool requires
   equal page bytes per group, so the attention block is raised until its
   page matches the GDN page: `interface.py:645` "Setting attention block
   size to 1664 tokens...". The fork's `platforms/xpu.py` pow-2 rewrite
   (ESIMD paged-attention kernel asserts `isPowerOf2(pageSize)`, #51 family)
   then lands the live serve at **attn block = GDN block = 4096 tokens**.
2. `resolve_kv_cache_block_sizes` gives `hash_block_size = 4096` (equal
   blocks; the mamba back-off and `--hash-block-size` override are moot).
3. Prefix reuse needs WHOLE physical blocks, so reuse is quantized to 4096
   tokens. Live ladder probes confirm exactly: 9139-tok pair hits 4096,
   16340-tok pair hits 8192, 8497/11217 hit 4096, 30413 hits 24576; every
   size fits `(floor(n/4096) - 1) * 4096` inserted blocks.
4. MTP (eagle cache-drop) additionally sacrifices ~1 block per lookup
   (`_mamba_block_aligned_split` cut + match-one-more-then-pop), so prompts
   < ~8192 tokens net ZERO reuse with MTP while nospec keeps the first
   4096-token block — the exact "works without MTP" delta the user saw.

**Measured benefit that DOES exist (live, MTP k4 on).** TTFT for repeated
prefixes: 9.1k prompt 4.20s -> 2.47s (4096 hit), 16.3k prompt 7.19s ->
3.86s (8192 hit), 30.4k 17.1s -> 4.7s (24576 hit). Prefill ~2k tok/s.

**Levers assessed.** (a) Scheduler/KV-manager patch: none sane — hash ==
block == 4096 is already optimal given the page size; the eagle pop/cut is
upstream correctness machinery. (b) Finer `--hash-block-size`: proven no-op
(sim2 Run B: hash 128 with 4096 blocks still 0 for small prompts). (c) Real
lever = shrink the negotiated page: ESIMD kernel accepting mult-of-64
instead of pow-2 pages (1664-token blocks ≈ 2.5x finer reuse) — kernel
surgery, high risk, not undertaken. (d) Halving per-token attn page bytes
(e.g. different KV dtype) halves granularity but costs accuracy. Practical
guidance: contexts > ~8k get real MTP-compatible reuse today; chat-sized
multi-turn traffic structurally cannot on this model+XPU stack.

**Lesson.** On hybrid-SSM models the prefix-cache granularity is set by the
SSM state-page size, not by `--block-size`; before chasing scheduler bugs,
ladder-probe the live hit quantization (unique blob pairs per size) and
check the boot log's page-negotiation lines (`Setting attention block size
to N tokens`, `Padding mamba page size`).

**Comprehensive validation (2026-08-29 evening, live serve, MTP k4).**
- Ladder floors 0-16: `(floor(n/4096)-1)*4096` held EXACTLY at every size
  (9/9 rows: 1938/4818 -> 0; 9618/10258 -> 4096; 13618 -> 8192; 17298 ->
  12288; 24818 -> 20480; 32978 -> 28672).
- Cross-request share (11.7k common prefix, different suffixes): 2nd request
  hit 4096 and answered its own question - sharing works, outputs correct.
- Multi-turn chain (5 turns, +4.2k each): each turn reused exactly what the
  previous turn cached (4096/8192/12288/16384); warm TTFT stayed FLAT at
  ~3.7-4.4s while cold grew 3.7 -> 11.3s (61% cut at 25k context).
- Correctness: greedy (temp 0) 64-tok outputs IDENTICAL cold vs warm vs
  control; 4 concurrent same-prompt requests produced 4 identical outputs
  (zero errors). Decode speed unaffected; only prefill shortens.
Conclusion: reuse is CORRECT and beneficial at the architectural granularity;
the 0% chat-traffic rate stands as granular, not broken. (Stalls seen during
this campaign are #11, unrelated to caching.)


## 11 — MTP + TP=2 + large prompts: ~7.5% of requests WEDGE INSIDE THE
ENGINE forever (oneCCL spin-clog, #05 family, prefill/verify variant);
NOT scheduler loss, NOT GuC-reset residue, NOT host state
(FOUND 2026-08-29, live serve adv:v22 MTP k4; CORRECTED VERDICT
2026-08-30 after instrumented reproduction on v22/v23/v24 boots of the
rebooted host; ROOT-CAUSED 2026-09-01 v31 GPU window: INDUCTOR-COMPILED
pieces x spec decode — capture and every custom kernel exonerated;
FIXED-BY-CONFIG in v31.1, see v31 update at the end of this section)

**v26 update (2026-08-30 evening, image adv:v26, host 10.20.3.65).** Two
results sharpen this entry decisively:

1. **The GDN spec-kernel OOB is real, fixed, and NOT this wedge.** The
   two XPU GDN spec kernels walked a rectangular `batch_id*k + t_local`
   index over raggedly-truncated spec buffers (verify-tail 1-of-5,
   drafter-replay, cudagraph padding rows) — OOB reads/writes that
   device-fault the GPU deterministically on the exact tail shape
   (`repro_gdn_spec_oob.py`, `UR_RESULT_ERROR_DEVICE_LOST` + xe ccs
   engine reset on iteration 0, twice). v26 fixes the walks
   (`vllm/patches/qwen38-dflash-v26/`, wheel-level regression CLEAN
   0/50, full-batch outputs bit-identical). The ≥32k serve stall,
   however, persists unchanged on the fixed wheel.

2. **The wedge requires spec x (any) cudagraphs x ≥32k context.** Arm
   matrix on v26, probe `wedge_probe.py 32k:3` (MTP k4 tq4nc unless
   noted): stock FULL_DECODE_ONLY **2/3 wedge**; PIECEWISE **1/3**;
   k=1 (verify q_len=2) **1/3**; KV `auto`/fp16 **1/3**;
   `CCL_ENABLE_SYCL_KERNELS=0` **2/3**; `DISABLE_ESIMD_GDN_SPEC=1`
   **1/3** — everything that keeps graphs wedges. `--enforce-eager`
   (no compile, no graphs) **0/11 CLEAN**; no-spec + FULL graphs
   **0/3 CLEAN**. The GDN spec kernels, the ESIMD fused kernel, the TQ
   kernels and the oneCCL kernel transport are all exonerated as the
   primary cause; the oneCCL spin kernels seen in the native stacks are
   the *symptom* (peer-rank collectives that can never complete), not
   the root. The root lives in the graphed spec pipeline meeting the
   eager MTP drafter under TP=2 at long context — most plausibly a
   replay/interleave hazard (captured wait or stale dynamic state on
   the victim rank's queue) that desynchronizes the ranks; the victim's
   host then cannot even SUBMIT the post-sample D2H (align-mode
   `.cpu()`, gpu_model_runner.py:1489 in v26) while the peer runs a
   step ahead into the eager head (py-spy 3-round capture, 2026-08-30).
   Wedged requests are lost; the ENGINE itself recovers after the
   client disconnect and serves subsequent traffic.

**Workarounds (validated 2026-08-30).**
- Long-context serving (≥32k): run **without spec** (nospec +
  FULL_DECODE_ONLY graphs) — 0 wedges across the 32k-262k battery AND
  the fastest arm at long ctx (28.2 tok/s @32k vs 19-25 healthy MTP
  graphs / 15-20 MTP eager). MTP is a net loss at ≥32k on this setup
  even when healthy.
- If MTP must stay on at any length: `--enforce-eager` — now validated
  **clean across the entire envelope** (0/11 @32k, 0/1 @133k, 0/1 @262k),
  at a heavy decode cost (15-20 tok/s @32k, 2.4-3.8 tok/s @133-262k).
- `VLLM_XPU_ALLOW_COMM_IN_GRAPH=0` (keep oneCCL out of the graphs) is a
  PARTIAL fix: 0/5 clean at 32-67k (stock wedges 2/3) but still wedged
  1/1 at 133k, and decode drops to 9-17 tok/s @32k. Not recommended as
  a workaround; kept as mechanism evidence (below).
- MTP + graphs remains fine for short contexts (the historical ~4.8k
  population wedged ~7.5%; 12-token prompts 0/200).

**Mechanism (CCL trace, 2026-08-30 late — adv:v26 + CCL_LOG_LEVEL=debug).**
oneCCL debug tracing during a live wedge (2/3 at 33k) shows the exact
interleave. Per spec decode step the EAGER drafter enqueues ~6 small
collectives — 4x `allreduce 5120 fp16` (the k=4 sequential head
forwards), `allgather 2560 fp16`, and `allgather 64 fp32` (the
`get_top_tokens` greedy-sample gather) — while the target's large
collectives are CAPTURED inside the verify decode graph (never logged at
step time). At the wedge, both ranks' last logged entry is the *same*
sample-gather, host-side "done", then total silence: the subsequent
graph replay's captured collectives never complete, the in-flight
window fills, and the victim host blocks at the align-mode `.cpu()` D2H
(gmr:1489) while the peer's queue stalls one op later. Reading: the
interleaved eager collectives advance oneCCL's device-side
matching/flag state between replays, and the captured collectives —
frozen at capture time — eventually spin forever on stale flags.
This explains every arm: eager-only clean (0/11; no captured colls),
graphs-only clean (nospec 0/10; no eager colls interleaved), mixed
wedges with rate scaling by the eager-coll count before each replay
(0% @12-tok prompts, ~7.5% @4.8k, ~deterministic @≥33k where 5-chunk
prefills pump eager colls), host barriers (v23-v25) useless, and
`draft_tensor_parallel_size=1` NOT a fix — the trace under DTP1 is
coll-identical (the head forwards keep issuing the same ARs/AGs).
Corroboration: the compiled-drafter boot crash
(`VLLM_XPU_MTP_EAGER_HEAD=0`, log `v26head0_crash.log`) is the same
oneCCL ring path segfaulting (`allgatherv_large_su_ring →
invoke_barrier`) when collectives run inside dynamo-evaluated code.
`VLLM_XPU_ALLOW_COMM_IN_GRAPH=0` (W3) removes captured colls from the
equation and indeed clears ≤67k — but wedges once at 133k, so a second
long-context trigger (independent of comm capture) remains. A true fix
needs either graph-replay-safe collectives in oneCCL, or a fully
coll-free drafter step (replicated head / per-domain comm), both
kernel/stack-level work.

**Corrected verdict.** The original entry below suspected stale GuC-reset
state (17:36 CCS reset never followed by a host reboot). That is disproven:
identical behavior on the freshly rebooted host, and the wedges are
reproducible on demand with sequential ~4.8k-token prompts. The earlier
belief that wedged requests "eventually complete server-side" is also
wrong — that was a misread of the success counter (neighboring requests
ticking). A wedged request NEVER finishes: it sits `Running` with KV
allocated until the client disconnects, and even then is freed without
being counted as aborted.

**Symptom (precise).** With the user's flags (TP=2, `--enable-prefix-caching`,
MTP k=4, `--kv-cache-dtype turboquant_4bit_nc`, `--async-scheduling`,
FULL_DECODE_ONLY graphs): ~7.5% of ~4.8k-token-prompt requests never return
(streaming and non-streaming equally). Normal latency is a flat 2.2 s — the
loss is all-or-nothing, not tail latency. 12-token prompts never wedge
(0/200); ~740-token single-block prompts almost never wedge. Serve log and
dmesg completely clean; no xe reset fires.

**In-engine signature (metrics).** `vllm:num_requests_running=1` frozen for
90-600+ s; `time_to_first_token` count already incremented (prefill done,
decode started); `request_success_total` frozen in ALL reasons (no stop, no
length, no abort, no error); freed only when the client disconnects — and
then still without abort accounting.

**py-spy during the wedge (native, multiple catches).** Both TP workers
"active":

- v22: TP0 spinning in `ur_command_list_manager::appendUSMMemcpy` — the
  align-mode `.cpu()` D2H in `_update_states_after_model_execute`
  (gpu_model_runner.py:1476/1483/1502 across builds) cannot be SUBMITTED
  (in-flight queue never drains); TP1 either a step ahead polling
  `urEventGetInfo` in `_calc_spec_decode_metadata` or inside the MTP head's
  vocab all-gather (`qwen3_5_mtp.py:476 -> _gather_logits ->
  ProcessGroupXCCL._allgather_base`).
- v23/v24: BOTH ranks stuck at the same `.cpu()` submission — the clog is
  symmetric and forms upstream of the drafter seam.
- xpu-smi during the wedge: both GPUs Compute+Copy engines 100%, EU ~11%
  active / ~56% stall — oneCCL spin-wait kernels resident. No competing
  queue work at this seam => the 640 ms GuC preempt timeout documented in
  #05 never fires: no reset, no DEVICE_LOST, eternal spin.

**Isolation matrix (v22 image, ~4.8k prompts, 80 reqs/cell).** BASE 6/80;
no `--async-scheduling` 6/80; `--max-num-batched-tokens 4096` 1/80;
`--mamba-cache-mode all` 2/80 — **but see the correction below**;
`--speculative-config` OFF **0/80**. MTP is the trigger. The
`mamba-cache-mode all` cell is INVALID as a delta:
`model_executor/models/config.py:438-446` coerces `all` -> `align` for
hybrid models without `supports_mamba_prefix_caching`, so BASE and that
cell ran the same mode (2/80 vs 6/80 is noise at this n).

**Mechanism.** The ranks' XPU streams mutually clog on oneCCL
non-preemptible spin-wait kernels (kernel transport, `CCL_ENABLE_SYCL_KERNELS=1`,
TP=2). Every wedge stack shows hosts blocked submitting a tiny post-sample
D2H (`appendUSMMemcpy`) — the victim, not the culprit: the in-flight window
is full of collective kernels that never retire. This is the documented #05
peer-late race family; the MTP head's eager collectives interleaved with the
target's (graphed verify / eager prefill) collectives create the crossing.

**Fix attempts — both disproven live (kept as negative results).**

- v23 `qwen38-dflash-v23`: one-sided `torch.xpu.synchronize()` before the
  drafter on steps >= 512 tokens. The fast rank drains and proceeds alone
  into the head's all-gather; the slow rank never reaches the barrier (it
  is stuck upstream in the `.cpu()`). No effect on the wedge rate.
- v24 `qwen38-dflash-v24`: true rendezvous — drain + gloo `cpu_group`
  barrier before the drafter. No effect either: on v24 wedges BOTH ranks
  are stuck upstream of the rendezvous (which is never reached), inside
  the verify/decode step's post-sample path. The clog forms in the
  target/verify/drafter collectives themselves; no host-side barrier
  placed at the drafter seam can reach it.
- First-pass "0 lost" validation batches were INVALID: `loss_rate2.py` arg
  2 is prompt REPS, not timeout — `80 45` measured ~740-token prompts (a
  population that does not wedge). Valid probes: REPS=300 (~4.8k).

**What does work (validated on every boot).** Prefix caching + MTP itself:
warm hits land exactly at `(floor(n/4096) - 1) * 4096` (block size
negotiates to 4096 under tq4nc on this hybrid), warm TTFT 2.3-3.2x faster
than cold; canonical car-game output clean; zero resets. The wedge is
independent of prefix caching (it needs MTP + large prompts + TP>1).

**Operational guidance (current).**

1. Clients: timeouts >= 120 s + one retry. Every observed retry recovered
   immediately; the orphaned duplicate is harmless and pre-warms the cache.
2. Large-prompt-heavy, loss-intolerant workloads: drop
   `--speculative-config` (only measured-clean arm, 0/80; costs ~1.65x
   decode speed).
3. `VLLM_WORKER_STEP_TIMEOUT_S` (StepWatchdog, 600 s default on XPU) will
   eventually kill a wedged worker; most wedges are "resolved" earlier by
   the client disconnect.
4. A real fix likely requires upstream work: preemption/timeout for oneCCL
   spin-wait kernels, or collective-order hardening between graphed verify
   and eager MTP-head collectives (the #05 thread). Removing the align-mode
   `.cpu()` sync (upstream TODO) would only move the victim line, not the
   clog.

**Original entry (superseded, kept for the timeline).** Sporadic
multi-minute stalls on the 17:54 boot (~12 of ~120 requests, in windows);
suspected the 17:36 GuC CCS engine reset was never cleared by a host
reboot. dmesg showed `xe 0000:da:00.0: [drm] Tile0: GT0: Engine reset:
engine_class=ccs ... guc_id=32` pre-boot. Disproven by reproduction on the
rebooted host; the reset was incidental.

**Update 2026-08-30 (long-context sweep): the wedge rate climbs with prompt
length until it is effectively deterministic.** Sequential single-request
probes with the same flags: ~2k prompt completes (15.0 tok/s); **~32k
wedged 2/2; ~131k wedged 2/2** — each at the prefill→decode handoff
(prefill bursts at up to ~13k tok/s inside a 10-s window, then `run=1`,
`generation_tokens_total` frozen forever). Prefix-cache retries of the same
prompt ALSO wedge (the retry prefills cheaply via the cached blocks, then
freezes identically) — client retry alone does not rescue a long prompt; it
must be served without MTP. Net: with MTP k=4 + TP=2, contexts >= ~32k are
UNUSABLE (near-100% loss), which upgrades guidance item 2 from
"loss-intolerable workloads" to "any workload with >= ~16-32k prompts".
Same prompts on the identical boot minus `--speculative-config`: all
lengths through 262k complete cleanly (see PERF_TUNING
LONG_CONTEXT_ANALYSIS for the perf curves). Also measured: MTP's draft KV
pool cuts request KV capacity 23% (869,550 -> 1,130,964 tokens without
spec; max 262k-request concurrency 3.32x -> 4.31x).

**UPDATE 2026-08-30 (evening, v25 image): ten-arm isolation matrix + two
live py-spy/xpu-smi captures. Verdict at the time — "the wedge is a
DEVICE-SIDE kernel spin in the GDN spec-state path; no host-side or
config knob reaches it" — is SUPERSEDED by the v26 matrix above: fixing
the GDN spec kernels (v26 wheel) did NOT stop the serve wedge, and
`--enforce-eager` is 0/11 clean, so a config knob does reach it. The GDN
spec kernels are exonerated as the primary cause; the trigger surface is
the graphed spec pipeline x the eager drafter under TP=2 at >=32k. The
arm table and live captures below remain valid data. Ship = drop
`--speculative-config` (clean at every length, and fastest at >=32k);
`k=1` is a validated opt-in for <=32k envelopes only.**

*Calibration correction first:* the wedge-probe filler measures 15.7443
tokens/repetition (server-reported: 16,646 reps = 262,081 tokens), 2x the
7.87 the probe assumed until 2026-08-30. Every "32k" row in the arm table
below and in the v25 README was actually ~65k of prompt. Conclusions are
unchanged (k=4 wedges at true 32k as well; see DTP1/MALL rows), but length
labels before that fix read one octave low. (Late correction, 2026-08-30:
`/tokenize` ground truth on the v26 boot is **16.0 tok/rep** — 16,570 reps
= 265,138 tokens. The 15.7443 constant was back-computed from an HTTP-400
"prompt contains at least N" message, which is derived from the context
limit (`limit+1-max_tokens`), not the true count; lengths after the
15.7443 fix therefore read ~2% low. Immaterial to every conclusion; the
262k battery probe was recalibrated to 16.0.)

*Arm matrix (k=4 unless noted; wedge = stream never returns; sequential
single-stream unless noted):*

| arm | delta vs user baseline | result |
|---|---|---|
| E0 | none | 3/3 WEDGE (~65k real) |
| E1 | `CCL_ENABLE_SYCL_KERNELS=0` | 2/3+ WEDGE |
| E2 | `use_local_argmax_reduction:true` | 2/3 WEDGE |
| E2-bs2 | E2 + 2 concurrent streams (tiny gather ACTIVE) | 5/8 WEDGE |
| v25 | every-step pre-drafter `torch.xpu.synchronize()` + tiny-gather default | 3/3 WEDGE |
| NG | `cudagraph_mode NONE` (no graphs at all) | 2/3 WEDGE |
| DTP1 | `draft_tensor_parallel_size=1` (zero draft collectives) | true-32k 1/3, 64k 0/2, 131k 1/2 WEDGE |
| MALL | `--mamba-cache-mode all` (async pinned-copy path; no align `.cpu()` D2H, no postprocess_mamba copies) | 2/3 WEDGE |
| E4 | `--max-num-batched-tokens 4096` | 1/3 WEDGE |
| K1 | `num_speculative_tokens=1` | **32k 7/7 OK**, 64k 2/3, 131k 1/3 WEDGE |
| v27 | dedicated drafter oneCCL communicator (`VLLM_XPU_DRAFTER_PG`, adv:v27) | true-32k 1/3 WEDGE (neutral — same as DTP1) |
| v27+MALLred | + `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` k4 | **32k 7/7 clean**, 65k 1/2, 131k 1/2 WEDGE |
| v27+nukesimd | + MALLred + every `DISABLE_ESIMD_*=1` | 65k 2/2 clean, 131k 1/2 WEDGE |
| v27+k1+MALLred | k=1 + MALLred | probes clean 32k-262k (27.7/23.3/14.6/8.48 tok/s) but **canonical WEDGE @ token 141** |

Refuted causes: oneCCL SYCL-kernel transport (E1), collective size /
argmax path (E2, E2-bs2), host-side interleave barriers (v23/v24/v25),
XPU graphs as a whole for the <=32k class (NG), draft-model collectives
(DTP1, v27 dedicated communicator — the drafter-comm interleave story is
dead), the align-mode blocking
D2H + mamba state copies (MALL), prefill chunk size (E4), the ESIMD
kernel family (v27+nukesimd: every `DISABLE_ESIMD_*=1` still wedges).
CONVICTED for the <=32k class: the eager oneCCL `all_reduce` kernel —
`VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1` (allgather + local add, already in
`xpu_communicator.py`, default off) takes k4 @32k from 1-2/3 wedge to
7/7 clean. What survives every lever is a graphs-x-spec residual
(~1e-2/step; see the v27 update below). What remains of the original
GDN reading is only the graphs-x-spec residual's location: it needs
graph-replay-level debugging (allocator/replay interaction or a captured
kernel with data-dependent termination), still beyond vLLM-side config.

*Live captures (the mechanism):*

- v25 boot, k=4: py-spy shows BOTH TP ranks blocked at
  `_update_states_after_model_execute` -> align-mode `.cpu()`
  (gpu_model_runner.py:1489) — the submit cannot enter an in-flight queue
  that never drains. Identical stacks at +20 s and +40 s (hard stuck).
- DTP1 boot, k=4: BOTH ranks blocked while ENQUEUEING
  `torch.ops._xpu_C.gdn_attention` (`_xpu_ops.py:183`, called from the
  compiled `qwen3_next.py` forward) — and the wedge formed during the
  PREFILL chunk of the next probe, not decode.
- xpu-smi during wedges: Compute Engines 100% AND Copy Engines 100% on
  both GPUs, but GPU Utilization only 11-22% at ~155 W; worker threads in
  R state with wchan 0 (user-space spin-wait in a blocked submit).

Reading: a resident kernel spins forever (engines busy, EUs mostly idle);
the host threads merely block wherever the full in-flight queue happens
to catch them — which is why the py-spy "stuck line" moves between the
align `.cpu()` and the GDN enqueue. The spin lives in device code, so
every host-side fix is unreachable by construction. The engine only
un-wedges when the client disconnects (running 1->0; the next request
serves normally), i.e. the request is silently discarded server-side.

*k=1 mitigation (validated):* dropping `num_speculative_tokens` 4 -> 1
cuts the per-step spec-state surface ~4x. At 32k: 7/7 clean sequential at
27.9-30.3 tok/s (nospec parity is 27.9); short-ctx canonical car-game
45.4 tok/s vs 33.2 nospec (+37%), correct `<think>` + clean HTML, healthy
acceptance (E[len] 1.67-1.95, position-0 rate 0.80-0.95). At >=64k it
still wedges ~40-50% (64k 2/3, 131k 1/3 across arm + battery), so k=1 is
NOT a general fix — it is an opt-in for serves bounded to <=32k contexts.

*v27 update (2026-08-30/31, host 10.20.3.65, adv:v27 = v26 + dedicated
drafter communicator overlay): the canonical-grade standard changes the
verdicts.*

1. **Probes under-expose ~60x.** The historical probe = a 64-token decode
   window; the canonical = a 4096-token generation (~60x the per-step
   exposure). Probe-clean therefore does NOT mean per-step-zero. The
   v22-era k=1 "7/7 @32k + canonical PASS" was a lucky draw by this
   arithmetic; last session's W3 (comm-out-of-graph <=67k) and eager
   0/11 verdicts read the same way.
2. **k1+MALLred is NOT canonical-safe.** It passed every probe in the
   full envelope (32k/65k/131k/262k, 27.7/23.3/14.6/8.48 tok/s) and then
   wedged a canonical at token 141 at SHORT ctx — the graphs-x-spec
   residual is ~1e-2/step and context-independent; only exposure drives
   it. Every probe-clean graphs+spec arm above carries the same caveat.
3. **`--enforce-eager` + MTP k4 is the only spec config validated at
   canonical exposure:** 3x canonical PASS back-to-back (short-ctx
   48.4 tok/s, +18% over the nospec canonical 40.8; 32k-ctx PASS at
   4.85 tok/s, html+canvas OK) + the 0/13 historical probe record incl.
   131k/262k. Deep-context decode is slow (>=131k 2.4-3.8 tok/s) —
   it is a short/medium-ctx latency config, not a deep-ctx one.
4. **Methodology trap (see #13):** a long filler of identical repetitions
   instant-EOSes at ~32k ctx (greedy too) and returns empty text in
   ~1-2 s — trivially misread as a serve failure during canonical
   testing. A wedge HANGS and never returns. Long-ctx canonicals must
   use varied fillers (`patches/qwen38-dflash-v27/canonical32k.py`).

*v28dbg update (2026-08-31, flight-recorder session): instrumentation for
the graph-replay-level debugging exists now, and the residual's surface is
narrower than "graphs".*

1. **The spec path never runs full-graph replay.** With FULL_DECODE_ONLY +
   MTP, `CudaGraphManager.run_fullgraph` executed **0 times** across two
   full boots (flight-recorder markers) while each rank logged 43,552 eager
   `all_reduce` calls. The graphs surface under spec = torch.compile
   PIECEWISE captured pieces + EAGER oneCCL collectives between them —
   platforms/xpu.py (~370-400) adds `vllm::all_reduce` to `splitting_ops`
   unless `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1`, so collectives are excluded
   from pieces BY DEFAULT (and the historical "comm-out-of-graph" arm was a
   no-op). The graphs-x-spec residual therefore lives at the
   piece-replay x eager-collective x allocator boundary — captured-
   collective replay was already convicted and is not in the path.
2. **Flight recorder** (`patches/qwen38-dflash-v28dbg`, image
   `llm-scaler-vllm-adv:v28dbg`, zero-config, `VLLM_XPU_FR=0` kills it):
   `fr.py` at site-packages root writes timestamped phase markers (replay
   begin/end, AR begin/end, AG begin/end) to `/tmp/fr_<pid>.log`. EVERY
   call site is `torch.compiler.is_compiling()`-guarded — dynamo traces
   through `all_reduce` during profile_run and faults on file I/O /
   f-string tensor methods otherwise (verified boot crash, fixed). Healthy
   baseline (2 ranks x 2 boots): exact AR begin/end pairing (43,552/43,552,
   zero orphans), ~200 us per eager AR with ~3 ms piece-replay gaps.
3. **Decision rule at the next live wedge** (tail of `/tmp/fr_*.log`):
   last line `AR begin` with no `AR end` => the eager collective itself
   hangs (on the VIA path that is `all_gather`); `AR end` then silence =>
   a replayed/compiled piece spins on device. Host-side watcher
   (`patches/qwen38-dflash-v27/frw3.sh` -> host /root/build) auto-captures
   xpu-smi + py-spy + fr tails on a 50 s /metrics stall into
   /root/build/wedge_cap/ — run it next to any wedge-susceptible boot.
4. **Boot-lottery behavior (rate evidence):** two fresh k4+VIA boots
   (incl. one deliberate re-roll) ran the whole battery clean — short-ctx
   10/10, 65k probes 14/14, 2048-token ignore_eos 4/4 (twice), concurrent
   2-stream 20/20 (twice) — vs ~1/2 wedge rate on the same shape the night
   before. Susceptibility looks decided per-boot at capture time (piece
   layout / allocator state); a clean battery bounds only that boot.
5. **Untried levers for the next wedge session** (deferred — no
   discriminating power against tonight's clean control): ALLOC
   (`PYTORCH_XPU_ALLOC_CONF=expandable_segments:False`; boot pins True and
   the VIA path clones ~1 buffer per AR against the load-bearing
   stable-activation aliasing), CAPCOMM (`VLLM_XPU_ALLOW_COMM_IN_GRAPH=1`),
   NOASYNC (`--async-scheduling` off). Also from a crash log: large default
   capture lists "intermittently fault the xe engine
   (UR_RESULT_ERROR_DEVICE_LOST) during PIECEWISE capture with eager
   collectives" — this tree caps default capture sizes to <=128.

*v29 update (2026-08-31, live-capture session): the wedge mechanism is now
INSTRUMENTED — a rank-desynced device-side livelock — and five candidate
causes are convicted-innocent by direct arms.*

1. **Mechanism (from 7 auto-captured live wedges, `/root/build/wedge_cap/`
   w080544-w123120):** both GPUs sit at
   Compute Engines 100% + Copy Engines 100% with GPU util only 22% (EU
   ~10% active / ~48% stall in the detailed dump) — a tiny-kernel/copy
   storm that never retires. The two TP workers' HOSTS freeze at DIFFERENT
   points of the same engine step, one region apart: one rank parks in
   `_update_states_after_model_execute` at the mamba-align D2H sync
   (`gpu_model_runner.py:1489`, `num_accepted_tokens.gpu[:n].cpu()`
   `.numpy()` — GIL released, classic blocked-sync), while the other is a
   region ahead inside the drafter `_propose_impl` (or once deeper, in an
   fp8 `apply_block_scaled_mm` op call that never returned). Which rank is
   ahead flips between wedges — a symmetric race, not a rank-0 pathology.
   Eager collectives are fully exonerated: perfect AR begin/end pairing
   (76,414/76,414 per rank in wedge #1); the fr log's mid-propose tail can
   under-report (buffered writer never flushes at freeze) — trust py-spy.
2. **Exonerated by direct arms (all on v29 = v28dbg + stablebuf +
   capture-stab):** NOASYNC (`--async-scheduling` off: battery 8/8+4/4+20/20
   clean at ZERO wall cost, then wedged), ALLOC
   (`expandable_segments:False`: battery clean, then wedged), prefix-cache
   reuse (unique-prompt canonicals still wedged), mid-serving recompiles
   (`TORCH_LOGS=recompiles` boots: all 20 dynamo recompiles fire during
   warmup, ZERO during serving), and **boot-lottery itself** (see 3). The
   remaining surface is the device-side piece-replay storm, not host
   configuration.
3. **Hazard model — ACQUIRED per-boot state, not boot-lottery (A5
   within-boot proof):** the SAME boot ran ~3.2k short-ctx canonical
   chunks clean, then after the fox battery (128k) + 2x long_exp (65k)
   wedged at the NEXT decode start (chunk 1, 65k ctx). Boots that only
   ever saw short-ctx traffic stayed clean through 6.6-10.7k chunks;
   boots with large-ctx history wedged the subsequent canonical barrage
   at ~1,250-1,410 cumulative chunks (5 runs, mean ~1,300). 180 s idle
   gaps stretched survival ~2.6x but did not immunize — the state
   persists for the boot's lifetime and large-context traffic is the
   accelerant. This reinterprets v28dbg's boot-lottery: fresh boots
   start LOW-hazard; its clean fresh-boot batteries had simply not
   crossed threshold. The user's production pattern (>=32k ctx) hits it
   fastest, matching the historical "user arm >=32k effectively
   deterministic" record.
4. **ARs/step partition (v29 propose markers):** exactly 12.00 eager ARs
   per propose call on the drafter side (3 x 4 draft passes,
   deterministic) + ~23.4 target-side per step; most layer ARs live
   inside compiled pieces and never hit the eager wrapper.
5. **Auto-recovery exists:** `patches/qwen38-dflash-v29/frw4.sh [once|
   restart] <logname>` polls /metrics, on a 50 s stall captures xpu-smi +
   TP-worker py-spy (grep `Worker_TP` — EngineCore just parks in shm
   `get_response` and is the WRONG target) + fr/engine tails, and in
   `restart` mode re-rolls the boot (`docker restart` = fresh capture
   layout) — converts a permanent hang into one lost request + ~7 min
   reboot with no human intervention.

*v29b update (2026-08-31, device/comm-level session): the livelock is
IDENTIFIED as a oneCCL SYCL-kernel collective spin, and its PERMANENCE is
now attributable to the IPC handle-exchange path.*

6. **B-phase arms (standard provocation = fox 6x128 + long_exp 3x2048 @65k
   + canonical 10x4096):** B1 `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1` returns
   DETERMINISTIC GARBAGE at temp 0 (captured collectives corrupt numerics —
   disqualified without provocation). B2' `VLLM_XPU_ENABLE_XPU_GRAPH=0`
   (compile-without-capture) survived 19/19 with coherent output but decodes
   at 8.7 vs 14.7 tok/s (-40%) — proves piece capture is a NECESSARY
   ingredient of the wedge. B3a `CCL_ZE_IPC_EXCHANGE=sockets` (2 boots):
   full 14.5-15.0 tok/s; boot 1 fully clean; boot 2 froze >=120 s at fox
   iter1 AND long_exp iter1 (in-flight requests died) but the engine
   SELF-RECOVERED both times and then served 10/10 clean canonicals —
   **drmfd = permanent wedge, sockets = self-healing livelock**. B3b docker
   cpuset NUMA pin: unbootable (oneCCL worker-affinity pthread_create EINVAL;
   `CCL_WORKER_AFFINITY` parser rejects any in-set list) — the NUMA-locality
   lever is untestable without a oneCCL fix.
7. **w144515 (frw5 full-telemetry capture, first at a live freeze):** both
   workers parked TOGETHER (not desynced) at gmr:1489; device in the same
   storm as permanent wedges — Compute 100% + Copy 100% + util 22%, EU 66%
   stall, HBM reads 80 GB/s, but **PCIe only 33 MB/s** (rising to 68 MB/s as
   recovery began). The storm is a LOCAL spin polling peer state, not data
   transfer: the fabric is healthy while both engines peg.
8. **Transport characterization (fence-safe; copies bypass Event/sync
   fencing on this build):** same-GPU D2D 240 GB/s; cross-GPU "P2P" copy
   9.5 GB/s < via-host staging 12.2/12.8 GB/s (P2P is host-routed);
   oneCCL (torch 2.11+xpu backend name is **`xccl`**) AR 56 us @ 8x5120,
   ceiling 9.7-9.8 GB/s == the raw copy ceiling; `CCL_ENABLE_SYCL_KERNELS=0`
   = 2.5x BW / 8x latency regression (never ship); `CCL_TOPO_P2P_ACCESS=0`
   HANGS the first collective; mamba-align 10 KB D2H sync = 19.7 us
   (negligible); governor powersave->performance lifts launch-bound work
   49->86 TFLOPs but leaves the fabric unchanged. Serve env uses
   `CCL_ZE_IPC_EXCHANGE=drmfd`; oneCCL's DEFAULT is pidfd (untested cell).
10. **Posture (unchanged conclusion, sharper boundary):** prod stays v27
   nospec. graphs+spec remains uncertifiable: the best comm-env arm (sockets)
   converts permanent death into repeated multi-minute freezes that kill
   in-flight requests — operable only with auto-restart + client retry.
   Next untried cells: pidfd exchange, `VLLM_XPU_MAX_CAPTURE_SIZE=0`,
   verify-region-only capture disable (needs image change).

*v29c update (2026-08-31 evening): all three "next untried cells" are CLOSED,
and every wedge datapoint above is now known to be a k=4 datapoint.*

11. **pidfd**: unsupported by this oneCCL build —
   `|CCL_WARN| pidfd is not supported, fallbacks to drmfd exchange mode` on
   both TP workers at init (silent fallback). Only drmfd and sockets are real
   exchange paths here. **MAX_CAPTURE_SIZE=0**: semantics are UNCAPPED (0
   disables the size cap => full 51-size list), not "no capture" — more
   capture cannot help, and that list carries the documented xe-engine
   DEVICE_LOST boot-fault risk. **Verify-region capture disable**: moot —
   see #12's v29c update: piece capture + k4 corrupts numerics at bs=1 in the
   smallest pieces (any list serving bs=1 is affected). **k3 verdict is in**
   (`/root/build/prov_k3.out`): k3 ALSO wedges under the standard
   provocation — fox iter 1 WEDGE (wall 158.1 s, stalled 120 s, ~65k
   prefill->decode handoff), long_exp RC=7 (iter 1 wedge at 122.3 s), frw5
   capture `w181259` shows BOTH workers parked at `gmr:1489` — the k4
   wedge signature. The hypothesized collapse is REFUTED: **the #11 wedge
   is k-AGNOSTIC (k3 and k4 both wedge at >=32k); the #12 corruption is
   k4-ONLY** — two distinct defects that upstream must ticket separately.

*Operational guidance (final, supersedes the k=1 paragraph above):* for
the full 262k envelope run WITHOUT `--speculative-config` (nospec +
FULL_DECODE_ONLY graphs: canonical PASS, 0/10 probes, fastest >=65k
decode — 23.6/18.5/12.7 tok/s @ 67/133/262k). For spec latency at
short/medium ctx, `--enforce-eager` + MTP k4 is the only
canonical-validated option (48.4 tok/s short-ctx). k=1+MALLred
(`VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1`) removes the convicted reduce
kernel and is probe-clean across the whole envelope, but retains the
~1e-2/step graphs-x-spec residual — acceptable only for exposures well
below canonical (tens of requests x short generations) with the client
timeout+retry discipline below. A true graphs+spec fix needs
graph-replay-level debugging (allocator/replay interaction or a captured
kernel with data-dependent termination), not vLLM-side configuration.
Keep client timeouts >=120 s + one retry regardless.

*v29c guidance addendum:* graphs + MTP **k4 is convicted on NUMERICS** (see
#12 v29c) — never serve it in any transport, even where it doesn't wedge.
graphs + **k<=3** is deterministic on every short-ctx probe (k1/k2/k3
bit-stable, logprobs == eager reference) BUT k3 wedges under the standard
provocation at 65k exactly like k4 (prov_k3: FOX_RC=7, LONGEXP_RC=7,
w181259 both workers at `gmr:1489`) — the #11 wedge is k-agnostic, so the
spec opt-in guidance does NOT widen: **k=1 only, short ctx, unchanged**.
The k<=3 numerics result matters for root-cause separation, not for ops.

*v29d update (2026-08-31 night, Tier-3 upstream arms): both oneCCL upstream
workarounds are DEAD ENDS on 2021.15; onset can be delayed ~4-6x but the
wedge stands. Upstream recon found near-matching public reports —
uxlfoundation/oneCCL #212 (2x B70: stale cached Level-Zero IPC handle after
peer buffer realloc -> page fault + infinite `is_event_completed` spin;
workaround `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`), #215 (multi-GPU Battlemage
default-allreduce hang in vLLM TP; 2021.17.2+SYCL RT 2025.3.2 = PASS row;
workarounds `CCL_SYCL_ALLREDUCE_TMP_BUF=1` / `ALLGATHERV_TMP_BUF=1` /
`CCL_ALLREDUCE=ring`), #213 (pidfd unsupported -> silent drmfd override —
exactly our v29c T2a closure). Live arm results (v27 image, k4, graphs,
drmfd, standard provocation):*

- **`CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` is UNBOOTABLE on 2021.15** — both TP
  workers die at the first all_reduce in `init_device`:
  `CCL_ERROR ze_handle_manager.cpp:42 mem_to_ipc_handle: device_fd !=
  invalid_fd failed`, under BOTH drmfd and sockets exchange. The #212
  workaround is version-gated (their reporter is on 2021.17).
- **`CCL_SYCL_ALLGATHERV_TMP_BUF=1` is hard-refused on BMG in 2021.15** —
  `allgatherv_sycl.cpp:112: To run on BMG, CCL_SYCL_ALLGATHERV_TMP_BUF must
  be set to 0`. Our env's `=0` pins are a hardware requirement, not tuning.
- **`CCL_SYCL_ALLREDUCE_TMP_BUF=1` (alone)** boots and serves at full graphs
  speed (14.7-15.0 tok/s canonical). Provocation pass 1: **19/19 CLEAN**
  (fox 6/6 + long_exp 3/3 + canonical 10/10, zero watcher captures) — the
  first clean graphs+k4 battery in project history. Pass 2 on the SAME boot:
  fox 6/6 + long_exp 3/3 clean, then **WEDGE at canonical iter 2, 488 chunks
  (~8k cumulative chunks; w213816: same Compute 100% + Copy 100% storm,
  worker parked in the drafter propose path)**. Baseline wedges at ~1.3k
  chunks — so the tmp-buf path delays onset ~4-6x but does NOT fix the
  livelock. Mechanistic read: in-place SYCL-kernel allreduce exposes the
  caller's graph-replayed buffer to the IPC handle cache more often than a
  oneCCL-owned tmp buffer — consistent with #212's staleness mechanism
  surviving.
- Corollaries confirmed live: oneCCL 2021.15's **default exchange is pidfd**
  (our env pins drmfd), and pidfd is unsupported on this build/kernel ->
  silent drmfd fallback (#213), i.e. `CCL_ZE_IPC_EXCHANGE` effectively
  offers only drmfd/sockets here.

*v29e update (2026-09-01, oneCCL 2021.17.2 upgrade arm): the version lever is
CLOSED — the #11 wedge is oneCCL-version-INDEPENDENT on this stack.*

- Image `llm-scaler-vllm-adv:v27-ccl1717`
  (`patches/qwen38-dflash-v27-ccl1717/`): v27 + the
  `intel-oneapi-ccl-2021.17-2021.17.2-5_amd64.deb` component package
  (apt.repos.intel.com, SHA-pinned in-build). Layout-compatible overlay: new
  `ccl/2021.17` tree (2021.15 kept), `latest` + `.bashrc` repointed, venv
  bindings repointed + NEW `libccl.so.2` link — 2021.17's `libccl.so.1`
  NEEDEDs `libccl.so.2` (no RPATH on any ccl lib) — and
  `/etc/ld.so.conf.d/ccl-2021.17.conf` + ldconfig: serving runs with NO
  LD_LIBRARY_PATH (non-interactive `bash -c`; the `.bashrc` vars.sh line only
  fires interactively) and torch preloads the svml/imf/sycl deps from
  `/opt/venv/lib`, but NOTHING preloads `libccl.so.2` — without the cache
  entry the whole dlopen chain fails at serve time. SYCL RT untouched
  (2025.3.2, torch-bundled libsycl.so.8).
- **2021.17.2 removes `drmfd`**: our pinned `CCL_ZE_IPC_EXCHANGE=drmfd` is a
  hard boot error (`env_parser.hpp:100 ... expected values: sockets, pidfd`).
- **pidfd now WORKS** — 2021.17 carries the #213 fix (2021.15's "pidfd is not
  supported, fallbacks to drmfd" WARN is gone; kernel 6.17, both TP workers
  in one container pidns). Healthy boot ~340 s, all 6 vllm procs map
  `2021.17/lib/libccl.so.1.0` + `libccl.so.2.0`.
- **WEDGE PERSISTS at 2021.17.2 + pidfd** (standard provocation): fox 6/6
  clean, then long_exp iter 1 WEDGE after 39 chunks, canonical RC=7. w073435:
  BOTH workers at `gmr:1489`, Compute 100% + Copy 100% both GPUs — the #11
  signature exactly. New behavior: the wedge **self-heals on pidfd**
  (~minutes; like 2021.15+sockets, unlike 2021.15+drmfd permanent), but the
  first post-recovery request returns degenerate text ("!!!!!!!!") —
  self-healing is NOT production-usable. #12 k4 corruption also persists
  (coh P1 distinct=2 on this boot).
- **#212's cache-off workaround BOOTS on 2021.17 but wedges IMMEDIATELY**:
  `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` + pidfd passes `init_device` (2021.15
  died right there: `mem_to_ipc_handle` fd assertion) — then provocation
  wedges at fox iter 1 after ONE chunk, long_exp iter 1 one chunk, canonical
  RC=7 (w075336, same signature/site). Onset ~40x FASTER than baseline —
  the IPC handle cache is load-bearing for survival, not the bug.
- Verdict: the oneCCL axis is exhausted — 2021.15 wedges (drmfd permanent /
  sockets self-healing), 2021.17.2 wedges (pidfd self-healing; cache-off
  accelerates), 2022.x = HANG per #215's own matrix. #215's
  2021.17.2+SYCL-RT-2025.3.2 PASS row does not cover our defect class
  (theirs is a TMP_BUF-adjacent hang; ours is the graphs x spec x ctx
  livelock). The remaining fix surfaces are graph-replay-level debugging
  (v28dbg flight recorder is in place) or an upstream oneCCL
  timeout/preemption primitive in `ccl_executor::wait()`.

*Posture (final): UNCHANGED — prod stays v27 nospec + FULL_DECODE_ONLY graphs
(restored after the arm; coh_probe Paris −0.451 x3 bit-stable). Ticket drafts
at `patches/qwen38-dflash-v29/upstream_{oneccl,vllm}_ticket.md` now carry the
2021.17.2 datapoint — the strongest single comment we can post on #212/#215.*

*v31/v31.1 update (2026-09-01 GPU window) — ROOT CAUSE CONVICTED AND
FIXED-BY-CONFIG: the wedge is the INDUCTOR-COMPILED piecewise path x
speculative decode. NOT XPU graph capture, NOT any custom kernel, NOT oneCCL.*

- Discriminator matrix (v31 image = v30 gate + v31 k-clamp + split knobs;
  MTP k4 auto-clamped to 3; oneCCL 2021.15; standard battery):
  - **compile + capture** (v30 bypass path = the historic wedging config):
    fox iter1 / long_exp iter1 WEDGE after ONE chunk each (w122359 canonical
    signature: both workers `gmr:1489`, Compute 100% + Copy 100%), while
    canonical 10/10 CLEAN @17.4 tok/s (~10k chunks) — short/medium ctx is
    clean, ≥32k wedges; engine self-recovers on client watchdog abort.
  - **+ GDN splits** (v31 patch B): identical wedge — post-hoc REDUNDANT,
    `vllm::gdn_attention_core(_xpu)` were already in the default
    `_attention_ops` split list (`vllm/config/compilation.py`), so the arm
    changed nothing (duplicate-suppressed).
  - **+ `moe_ops::moe_forward_full_fp8_block` split** (the ESIMD fp8-block
    MoE decode variant for ≤12-token steps — the ONE variant missing from
    `_esimd_moe_splits`, i.e. exactly spec steps; it also keeps a host-side
    dict-of-output-buffers): identical WEDGE @1 chunk. NEGATIVE.
  - **+ ALL custom ops split** (every `custom_esimd_kernels_vllm::*`
    gemm/gemv/norm/qkv_rope op + `_xpu_C::fp8_gemm*` + the moe variants;
    over-splitting requires `VLLM_DISABLE_COMPILE_CACHE=1` or the compile
    cache artifact fails serialization): identical WEDGE @1 chunk. Every
    vllm/esimd/`_xpu_C` kernel EXONERATED — captured pieces contained only
    inductor-generated code + aten elementwise and still spun.
  - **compile, no capture** (other editor's v30_safe_p2): WEDGE at 563
    chunks — same defect, ~500x slower onset.
  - **capture, NO compile** (arm D: `TORCH_COMPILE_DISABLE=1` +
    `VLLM_XPU_ENABLE_XPU_GRAPH=1`; the XPU `CUDAGraphWrapper` captures the
    whole decode step with eager kernels, dynamo never engages): **fox 6/6 +
    long_exp 3/3 @65k + canonical 10/10 ALL CLEAN @17.7 tok/s** (+79% over
    eager 9.9; also faster than compile+capture 17.4), coh bit-stable
    (Paris −0.451 == eager reference), boot ~100 s faster (~230 s).
- **Fix shipped: `llm-scaler-vllm-adv:v31.1`**
  (`patches/qwen38-dflash-v31/Dockerfile.v31_1` + `gate_v311.patch`): the
  spec+TP>1 gate now sets ONLY `TORCH_COMPILE_DISABLE=1` — graphs/capture
  KEPT, dynamo/inductor off, splitting machinery inert.
  `VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1` still restores the wedging
  compiled config for on-demand repro. The v31 k-clamp is retained (#12 is
  capture-level, see #12 v31 addendum).
- **v31.1 certified on the DEFAULT posture (no bypass env)**: markers
  (inductor-disabled ×4, k 4→3 ×1, zero fully-eager fallback), coh
  bit-stable, **full battery ALL CLEAN: fox 6/6 + long_exp 3/3 @65k +
  canonical 10/10 @16.4 tok/s** (+66%; the arm D twin boot measured 17.7),
  post-battery coh P1 distinct=1.
- Mechanism (working model): an inductor-generated decode-piece kernel
  (combo_kernels-class fusion) livelocks under spec-step shapes at ≥32k
  cumulative context; capture replay accelerates onset (chunk 1 vs chunk
  563). Flight-recorder fit: every stall gap ends with `AR end` then
  silence — the eager collective retires, the following compiled/replayed
  region spins. Remaining upstream-debug surface if a true fix (rather than
  config avoidance) is wanted: torch-inductor codegen for the decode pieces
  under spec (bypass env reproduces on demand, w122359-class signature).
  The kernels lane (`/root/build/vxk`) is exonerated for #11.
- Posture: prod restored to v27 nospec at window end (coh-verified);
  **v31.1 is the validated promotion candidate** for spec+graphs serving
  (16.4-17.7 tok/s canonical, 65k-clean, k=3).

*v31.2 update (2026-09-01, second window — PROMOTION EXECUTED, PERF
VERDICT, NATIVE-STACK EVIDENCE)*:

- **Perf verdict — spec k3 is a NET LOSS; promotion posture = v31.1 image
  with NOSPEC config.** Controlled same-client sweep (ctxbench.py, greedy,
  unique-prefix prompts, warm): spec k3 vs v27-nospec = 16.5 vs 33.6 tok/s
  (conc=1 @2k), 15.5 vs 29.5 (agg, conc=4), 35.3 vs 137 (agg, conc=16 @2k),
  13.6 vs 53.9 (agg, conc=16 @32k), 7.5 vs 23.5 (conc=1 @65k). The "17.7 vs
  9.9" promotion premise was an artifact — 9.9 was the v30 gate's FULLY
  EAGER reference, not v27-nospec-graphs. Spec decode here pays ~7
  row-forwards/step (draft k3 + verify k+1) for ~1-2 accepted tokens at
  small bs (decode is row-serial: conc=4 agg < conc=1 agg); it never wins
  at any measured point.
- **Promotion executed**: prod = `llm-scaler-vllm-adv:v31.1` with NO
  speculative config. Warm parity vs v27 confirmed: conc=1 @2k 33.56 vs
  33.57, @32k 27.81 vs 27.76, @65k 23.52 vs 23.53, conc=16 @2k steady ~353
  tok/s both; coh bit-stable Paris -0.451; gate markers inert (0 warnings,
  0 clamps — posture identical to v27). All v31/v31.1 safety machinery is
  carried and auto-fires if spec is ever requested. First-request-after-boot
  at a new prefill shape is compile-polluted (ttft +7-25s) — measure warm.
- **Native-stack evidence at the live wedge** (gdb mid-stall, both TP
  workers identical, `gdb_wedge_evidence.txt` +
  `/root/build/keeper_gdb_wedge_evidence/`): main thread parked in
  `torch.ops...gdn_attention -> chunk_gated_delta_rule_impl_xe2 -> sycl
  q.wait() -> urQueueFinish -> ur_queue_immediate_in_order_t::queueFinish
  -> libze_intel_gpu` — the EAGER GDN kernel's wait on an IN-ORDER L0
  queue: work enqueued ahead of it never retires (device storm Compute 100%
  + Copy 100%); per the discriminator matrix that work is the
  inductor-compiled region on the same stream. Confirms + localizes the
  fr-record reading (AR end -> silence). Repro nuances: needs COLD-prefill
  6-concurrent unique-prefix 65k requests early in a bypass boot;
  prefix-warm and single-stream requests did not re-trigger on an
  already-exercised boot; the 2021.15 boot self-heals ~2 min (first
  post-recovery request can be degenerate).
- Upstream: k4 corruption filed as vllm-project/vllm#54785; #11 livelock
  filed as a separate vLLM issue with the gdb chain (URL in the drafts
  post-log); oneCCL #212/#215 cross-posts reframed as triage datapoints
  (oneCCL exonerated) — all posted 2026-09-01.


## 12 — temperature=0 outputs on LARGE CHUNKED prompts are not bit-stable
run-to-run under MTP (fp near-tie flips); bare prompts ARE stable —
prefix-cache correctness checks must not compare full greedy text
(OBSERVED 2026-08-30, adv:v22-v24 MTP k4, TP=2)

**Observation.** Same 8.3k-token prompt, `temperature=0`, sent 2-4x
sequentially: texts diverge after 55-215 chars (e.g. "reasonably polished
**but** not too long" vs "reasonably polished**:** player car, oncoming
cars, score") and can differ in length — including warm-vs-warm (identical
prompt, identical 4096-token prefix-cache reuse, back-to-back). A 7-token
bare prompt is 4/4 identical under the same serve. One 8.3k cold/warm pair
did match exactly, so the divergence is probabilistic per run.

**Reading.** Greedy argmax flips on fp16 near-ties when kernel reduction
order varies between runs; large prompts chunk through differently-shaped
batches (8335 tokens = 8192 + 143 chunk split; mamba align postprocessing;
spec verify shapes), and MTP's draft/verify path amplifies any single flip
through the autoregressive chain. NOT a prefix-cache corruption: the cache
returns bit-identical quantized KV, and bare-prompt greedy is stable. The
canonical correctness gate remains: hits land exactly at
`(floor(n/4096) - 1) * 4096`, warm TTFT improves, and the output is sane,
coherent text — not cold==warm string equality on large prompts.

*v29c update (2026-08-31 evening) — UPGRADED to a k4-specific replay defect;
the "fp near-tie" reading above is superseded, and "bare prompts ARE stable"
no longer holds on current images:*

- **graphs + MTP k4 corrupts temp-0 output** on a 6-token BARE prompt:
  identical requests cycle between 2-3 execution paths whose logits differ at
  NAT level (top-1 logprob for the same position: -0.099 / -0.294 / -1.598;
  the top token itself can flip ' Paris' -> 'The'), producing
  fluent-but-unrelated continuations and — after the boot accumulates ~40
  probes — degenerate repetition (' France France France...'). Found by
  accident during the pidfd arm's sanity check; characterized with
  `coh_probe.sh` (8x temp-0 first-token + 6x 40-tok + 3x logprobs=5).
- **The boundary is exactly k4.** v27 image, default graphs, k=1/2/3: P1
  distinct=1, top-1 -0.451 bit-stable, equal to the compile-no-capture
  reference (-0.453). k4 corrupt. Padding refuted as the trigger (k1 bs=3
  6-rows->padded-8-piece clean; k2 bs=1 3-rows->padded-4-piece clean) — the
  boundary tracks draft-token count, implicating multi-draft-position spec
  machinery (GDN state / verify mask) under captured replay.
- **Byte-reproducible**: same responses in the same order across boots,
  across v27 AND v29 images, and across capture lists [1..128] vs [1,2,4,8] —
  a deterministic function of request ordinal, i.e. the scheduler alternates
  among 2-3 paths and at least one computes wrong logits. temp 0 IS honored
  (high-margin 40-token prompt: 6/6 identical even on corrupt boots).
- **Capture is necessary** (`VLLM_XPU_ENABLE_XPU_GRAPH=0` + k4 = deterministic
  E1 reference) and **bs>=2 above the capture list falls back to eager and is
  clean** (2 concurrent k4 requests on a [1,2,4,8] boot: both return the
  reference continuation ' Paris.\nThe capital of Germany...').
- **Retrospective**: #12's original large-chunked-prompt divergences were
  observed at k4 on adv:v22-v24 — in today's light they were probably this
  same corruption, not benign near-ties. Every graphs+k4 arm in v25-v29
  measured wedges/throughput only; output text was never gated. **Dating:
  adv:v24 reproduces the corruption byte-for-byte** (same 8 responses, same
  logprobs) — not a v25-v27 regression; the original "bare prompt stable
  4/4" was a small-sample artifact of per-request-ordinal path cycling. The
  defect is at least as old as v24 / this stack generation.
- **Prod impact**: NONE as running (prod = nospec). Guidance: never serve
  graphs+k4; k<=3 + graphs passes every short-ctx determinism probe but k3
  WEDGES at 65k under the standard provocation exactly like k4 (prov_k3:
  FOX_RC=7, LONGEXP_RC=7, w181259 both workers at `gmr:1489`) — the wedge
  (#11) is k-agnostic, this corruption is k4-only; two distinct defects.
  The eager+MTP-k4 option in the #11 guidance remains valid and is
  deterministic (compile-free). Repro: 3 curls, see README v29c section.

*v35 fix campaign (2026-09-02, current tree v31.1+v32 patches) — fix
FAILED locally; corruption re-confirmed with sharper forensics; k4 left
user-selectable for diagnosis:*

- **Re-test (v35k4, clamp bypassed via `VLLM_XPU_ALLOW_K4_CAPTURE=1`)**:
  corruption UNCHANGED and FASTER-ONSET — fresh boot P1 distinct=2 with
  top-1 logprob drift across reps (-1.642 / -0.258 / -0.061); after ~14
  requests P1 distinct=8/8 with mojibake (' MS"$//}{}{', 'bs域8⨨',
  ' theTheTheTheTheThe'). High-margin output intact as before (P2 6/6
  identical, hash f167d905a10b invariant). **No wedge at 65k** (5x65k
  clean completions, warm 24.1 tok/s — the v29c-era k4@65k wedge was the
  compile path, consistent with #11's conviction).
- **Falsified fix arms (all three byte-identical corruption — same f8ref
  hashes e899790d3635/f167d905a10b/d84100508821, same logprob drift,
  same mojibake strings in the same order)**: (1) async scheduling OFF
  (`--no-async-scheduling`; needed serve_boot_var_noasync.sh — EXTRAFLAGS
  precedes the hardcoded `--async-scheduling`, and the fork default is
  True); (2) exact 5-multiple cudagraph capture sizes [5..320], verified
  in the serve config — zero padding anywhere, corruption byte-identical;
  (3) verify input staging code read clean (all buffers num_spec_tokens-
  sized, no hardcodes). Conclusion: the corruption is a deterministic
  function of the REQUEST SEQUENCE alone — independent of scheduling
  mode, capture bucket geometry, and (per v29c) padding, capture lists,
  oneCCL version, image. Remaining surface = GDN state rollback inside
  the captured verify path / closed kernels — upstream-scale.
- **k4+ user-selectable (per user request, 2026-09-02; clamp REMOVED
  same day)**: the xpu.py k>3 clamp and its `VLLM_XPU_ALLOW_K4_CAPTURE`
  bypass are deleted by the v35 boot patch (`v35_k4_unclamp.py`, applied
  to every lane by bootp.sh; baked into llm-scaler-vllm-adv:v36/v37) — k=4+
  is selected directly in the boot
  JSON, no env var, no code protection; the warning lives in this file
  and the README only. Verified bit-identical to the former bypass lane
  (f8ref e899790d3635/f167d905a10b/d84100508821, arm v35k4u:
  SpeculativeConfig num_spec_tokens=4 kept, zero clamp warnings) — all
  forensics above carry over unchanged. BLUNT CAVEATS, measured on the
  corrupt lane: k4 is DOMINATED by k3 everywhere (65k warm 24.1 vs k3
  29.25; conc16 9.6-12.9 vs 13.99; 2k early-stops into garbage), and
  output degrades to mojibake within ~tens of requests on low-margin
  prompts. k>4 passes through untested. Never prefer k4 over k3 on
  this stack.

*v31 update (2026-09-01): #12 is CAPTURE-LEVEL and COMPILE-INDEPENDENT —
inductor is NOT involved.* Under capture-no-compile (arm D posture:
whole-step XPU graph capture, `TORCH_COMPILE_DISABLE=1`) k=4 STILL corrupts:
coh P1 distinct=3 with drifting top-1 logprobs across identical requests
(−1.668 / −0.187 / −0.050, token flips ' The'/'The'), while high-margin
P2 40-tok stays 6/6 identical; no wedge at 65k in that posture even at k4
(fox 5/5 clean). The v31/v31.1 k-clamp (`num_speculative_tokens>3` → 3
whenever `cudagraph_mode != NONE`; `VLLM_XPU_ALLOW_K4_CAPTURE=1` for
diagnosis) is therefore MANDATORY on any capture posture. Suspect surface
for a true fix remains the multi-draft-position spec machinery under
captured replay (kernels lane, `/root/build/vxk`); #11's inductor
conviction does not transfer to #12.

*v31.3 update (2026-09-01, third window — ROOT-CAUSE NARROWED to the
5-token spec step; SHORT-PROMPT law found; four suspects exonerated):*

- **New empirical law: corruption requires k=4 AND a short prompt.** Same
  k4 capture boot, 8 identical greedy reqs: ~10-token prompt → **8/8
  distinct** outputs (logprob swings >1 nat); same prompt with ~25k tokens
  of filler → **1/8, text-stable** (residual logprob wobble ≤0.02). k=3
  bit-stable at both lengths. Interpretation: when the GDN recurrence
  carries the whole context (short prompts), stale spec-state content is
  catastrophic; long contexts are dominated by full-attention KV which
  masks it. This also retro-explains the "prompt-sensitivity" of the
  logprob swings.
- **Exonerated (live telemetry + env arms):** (1) the triton multi-query
  verify kernel — `VLLM_TQ_MQ_VERIFY=0` (per-row synthetic path for both
  verify and drafter) still corrupts; (2) spec mamba slot allocation —
  instrumented `GDNAttentionMetadataBuilder.build`: rows allocate k+1
  fresh blocks/step at BOTH k3 (`[1..4]→[5..8]→[9..12]`, num_accepted
  1→2, output correct) and k4 (`[1..5]→[6..10]→…`), and the block table
  is exactly `[batch, k+1]` — churn is by design (state migrates via the
  manager's last_state_block_idx/copy path), not the bug; (3) capture
  padding — `adjust_cudagraph_sizes_for_spec_decode` rounds capture sizes
  to multiples of k+1, so k4/bs=1 runs the EXACT-fit size-5 piece, no pad
  rows, and still corrupts; (4) `VLLM_XPU_MTP_EAGER_HEAD` is default-on
  already (not a discriminator).
- **Convicted region (what remains):** the 5-token spec step's numerics —
  the ESIMD `gated_delta_rule_spec_kernel` at num_spec_tokens=5 (k≤3 → 4)
  and/or the 5-row verify/rejection mapping. Next tool: per-layer device
  state dumps comparing captured k4 vs eager k4 for the same step.
- **Decision: no speculative patch this window** — the shipped k-clamp is
  the correct mitigation, prod (nospec) is unaffected, and an unverified
  numerics patch is worse than the clamp. No v32 image; posture stays
  v31.1. Upstream addendum with the short/long discriminator + telemetry:
  vllm#54785 issuecomment-5497862084.


## 13 — ~65k-token highly-repetitive prompts yield DEGENERATE completions
(instant-EOS empty text, or a `"!"`-loop) on the no-spec arm; lengths
131k/262k on the same prompt shape are clean (OBSERVED 2026-08-30,
adv:v22 NOSPEC boot, TP=2, tq4nc, fp16)

**Observation.** Streaming completions of ~65,069-token prompts (one text
unit repeated 4066x + a final question, temperature 0.3, top_k 20, top_p
0.95) returned HTTP 200 with server-side "success": 3 of 4 tries produced
`finish_reason=stop` after ONE token with EMPTY text
(`completion_tokens: 1`), and 1 of 4 produced a literal `"!"` repeated for
all 48 tokens. One ~32k prompt of the same shape hit the instant-EOS form
once (its retry was normal). ~131k and ~262k prompts of the same shape
streamed normal, sane text. No engine log errors, no aborts, no
preemptions; metrics count these as successful stops.

**Reading.** Not a transport or wedge issue (#11-family wedges never return
anything); the engine genuinely emitted EOS-on-first-token or a degenerate
loop. Most likely a sampling/logit pathology on an extremely repetitive
prompt distribution (the model's next-token distribution after thousands of
identical repetitions is degenerate), not an infra bug — but the length
asymmetry (65k bad, 131k/262k good) is unexplained. Open isolation steps
before escalating: repeat with realistic (non-repeating) 65k text; greedy
temperature 0; bf16 vs fp16; and with mamba align mode off (no prefix
caching). If real-text 65k prompts also degenerate, suspect the hybrid
mamba/align state path at the 2^16-token neighborhood instead.

**RESOLVED 2026-08-30 (evening): prompt-distribution pathology, not infra,
not MTP.** The isolation steps ran on fresh no-spec and k=1 boots:

- Instant-EOS reproduces on **no-spec** at the same rate as on k=1:
  sequential 131k 1/1, concurrent 32k x2 1/2 (distinct fillers) and 2/2
  (shared cached filler). k=1 same-shape battery: 4/6 concurrent. It is
  uncorrelated with MTP, concurrency, and prefix-cache state.
- A deliberately varied synthetic filler (seeded non-repeating "operator
  log" sentences) was clean concurrently on one boot (32k x2 no-spec: 2/2
  healthy `<think>` completions) — then the SAME two seeds on the next
  boot degenerated (`"!"`-loop + instant-EOS). Synthetic "log-style" text
  is still pattern-dominated enough to collapse; only genuinely natural
  text is trustworthy for content gates. Every degenerate case shares the
  same factor: low-entropy filler tens of thousands of tokens long, where
  the model's next-token distribution collapses and tiny numeric
  differences flip it to EOS or a `"!"` fixation.
- Practical consequence: any probe/benchmark built from a repeated unit
  (including this repo's `wedge_probe.py` filler) measures throughput and
  wedge behavior fine — its completion CONTENT is meaningless at >= ~32k.
  Correctness gates must use realistic text; `realistic_probe.py` (host
  /root/build) generates it with a seeded varied-sentence generator.

The 65k-bad / 131k-clean asymmetry of the original observation is simply
where the distribution collapse crossed the sampling threshold on those
runs — it is probabilistic per request, not length-deterministic.


## 14 — TurboQuant MQ verify kernel: [Q_BLOCK, BLOCK_KV, BLOCK_D] register
temps made spec-decode verify pay up to 1.7x the single-token KV scan at
long context (FIXED LOCALLY, v32, 2026-09-01)

Spec verify (q_len = k+1) routes through the multi-query kernel
`triton_turboquant_mq_decode_attention` (routing confirmed via triton JIT
cache: `_tq_mq_decode_stage1` present, single-token `_tq_decode_stage1`
absent on a k1 boot). The kernel shares KV tile LOADS across query rows,
but three sites built `[Q_BLOCK, BLOCK_KV, BLOCK_D]` fp32 intermediates
(FP8-key scores, MSE-path term1, and the value accumulation
`p[:, :, None] * values[None, :, :]`). At Q_BLOCK=2 / BLOCK_KV=4 /
BLOCK_D=128 that is 1024-float temps — 2x the single-token kernel's
footprint — forcing register spills that only bite on long scans:

- k1 @65k: verify forward 72.0ms vs nospec single pass ~40ms (~1.7x),
  while @2k the two are at parity (25ms) — context-scaled regression,
  the signature of occupancy loss, not extra bytes.
- `VLLM_TQ_MQ_STAGE1_WARPS=4` REGRESSES both ctx (77.4ms @65k, 18.65
  tok/s @2k vs 19.42 baseline): more warps per CTA = fewer resident CTAs
  on BMG. The knob is a trap for this kernel.

**Fix (llm-scaler v32, `vllm/patches/qwen38-dflash-v32/v32_mq_regpatch.py`)**:
per-row `tl.static_range(Q_LEN)` loops with where-masked row assembly;
per-row temps stay `[BLOCK_KV, BLOCK_D]` (identical to the tuned
single-token kernel). Identical reduction axes per row — bit-stable
(coh gate: distinct=1, Paris -0.452). Results (k1): tforward @65k
72.0 -> 66.1ms, drafter paths sharing the kernel -15%/-18% (propose d,
dforward d), @2k 19.42 -> 19.65 tok/s. Remaining ~26ms over a single
pass is the per-row SCORE/VALUE ALU (MSE unpack + centroid gather +
2x MACs), addressable only by XMX `tl.dot` tiles (tf32 numerics
decision) — see README v32 P1b. Upstreamable as-is (pure kernel-body
rewrite, no API change).

## 15 — fp8 KV-cache family x MTP spec x long context: e4m3 verify is
pathological at 65k (250ms/step), e5m2 unbootable (2026-09-01)

`--kv-cache-dtype fp8` (e4m3) bypasses the TurboQuant backend entirely.
With MTP k1 + graphs + our v32 kernel patches (patches inert on this
path): @2k 17.88 tok/s (-10% vs 4bit_nc), coh PASS (Paris -0.446,
deterministic) — but @65k the verify forward runs **~249.5ms/step**
(3.8x the TQ-4bit path's 66.1ms; propose d 21ms, dforward 17.8ms — the
drafter's cache reads degrade too). The standard XPU extend/verify path
with a 65k fp8 paged KV at q_len=2 is catastrophically slow on BMG:
**fp8 KV is NOT viable WITH SPEC at long context on this stack.**
`fp8_e5m2` is rejected at boot (WorkerProc startup failure — same
class as the float16 rejection, #05c).

**NOSPEC counterpoint (same window) — fp8-e4m3 is the FASTEST
long-ctx lane we have measured:** a true nospec fp8 boot (EngineCore
`speculative_config=None`) measured **33.51 tok/s @2k** (parity with
4bit_nc's 33.56) and **25.06 tok/s steady @65k (+6% over 4bit_nc's
23.61**; 39.4s chunked-prefill TTFT), coh PASS (Paris -0.446,
distinct=1). This confirms the #14 B/C decomposition from the other
side: the TQ 4-bit decode lane is ALU-bound (MSE unpack + centroid
gather ≈ 26ms of the ~40ms 65k scan), while raw fp8 attention skips
the unpack and wins despite 2x KV bytes. fp8-e4m3 NOSPEC is a prod
CANDIDATE (+6% @65k single-stream) pending the full adoption battery:
KV capacity halves (2x bytes/token -> fewer resident long ctx at
conc16), and the #07-era ESIMD sync posture under sustained load.
`turboquant_4bit_nc` remains the shipped dtype until that battery
runs; spec stays off fp8 regardless.

**RESOLVED 2026-09-02 (v33)**: the e4m3+spec verify pathology is FIXED
— root cause was C++ `_vllm_fa2_C` branch1 (chunk_prefill, no KV
splits, 2 CTAs/GPU serially scanning the 65k fp8 KV). Routing small-q
paged fp8 verify to the in-tree Triton `unified_attention` 3D split
path (`patches/qwen38-dflash-v33/v33_mq3d_triton.py`) is bit-identical
and 2.56x faster in warm pure-decode @65k (7.08 -> 18.14 tok/s). The
e5m2 half of this issue is root-caused in #19. fp8-nospec keeps its
prod-candidate status (23.99 @65k warm, v33 methodology) with the
newly-found greedy bimodality caveat (#18).

## 17 — spec decode collapses under concurrency: eager MTP draft is
never graph-captured; our v2x per-step sync barrier removed (2026-09-02)

Two findings from the v33 window:

(a) **_SPEC_DRAFT_BARRIER removed.** py-spy on TP0 showed 40.6% of
host samples blocked in `torch.xpu.synchronize()` called from
`propose_draft_token_ids` — this was OUR v2x oneCCL wedge mitigation
(`VLLM_XPU_SPEC_DRAFT_BARRIER`, default ON = full device drain EVERY
spec step), added before the v31.1 "whole-step graph is clean"
conclusion. `VLLM_XPU_SPEC_DRAFT_BARRIER=0` (now passed at boot for
all spec lanes): 4bit+spec 2k 19.65 -> 25.25 (+28.5%), fp8+spec 2k
+13.7%, 65k +3-4%; hashes unchanged. No wedge in ~10 min of sustained
65k spec decode across batteries — but it is a WATCH ITEM for
multi-hour prod: if #11-class wedge symptoms reappear on a spec lane,
re-enable the barrier first.

(b) **conc16 collapse is the eager-draft ceiling.** 4bit+spec k1
conc16@8k = 21.92 tok/s aggregate vs nospec 55.67 (TTFT 80.9s vs
52.6s). py-spy during conc16: eager MTP draft = 52-62% of TP0 host
samples (python IR-op dispatch per layer op), plus TQ prefill python
and per-call Triton JIT cache-key computation for the MQ kernel —
none graph-captured, all serialized by the #16 acceptance dependency,
and the draft work scales with batch while overlapping nothing.
Fix = P3 (capture propose+sample in the decode graph) / P2
(worker-resident acceptance loop) — README v33.

v34 addendum (2026-09-02): post-v33 py-spy on k3@65k decode (TP0):
draft/propose = **71% of host samples** (qwen3_5_mtp 64% — spread thin
across rms_norm / rotary / GEMM IR-op eager dispatch per layer op;
qwen3_5 67% incl. vocab_parallel_embedding lm_head allreduce,
get_top_tokens sharded argmax, sample_tokens), GDN build 9.1%, triton
10.2%, jit dispatch 6.0%, all_reduce 4.9% — no single killer function;
the fix is structural (capture), not local. k3 conc16@8k = 13.99 agg
(WORSE than k1's 21.92 — draft work scales with k while #16 still
serializes). The capture blocker is now precisely located: the draft
loop (`llm_base_proposer.py`) calls
`build_per_group_and_layer_attn_metadata(common_attn_metadata,
draft_index=...)` PER POSITION — fresh tensor objects per position per
step defeat XPU-graph capture (staleness); static input buffers
(`self.input_ids`, `self.hidden_states`) already exist. The capture seam
is the dispatcher (`cudagraph_dispatcher.get_capture_descs` /
`_warmup_and_capture`) — upstream-scale surgery, and the v29c lesson
(piece-capture x MTP corrupts temp-0 numerics, k4-boundary) argues
against blind capture attempts. NOT attempted in v34; P3 remains the
fix. A GDN steady-state build memo was attempted and REJECTED — and
the A/B data falsified the APPROACH, not just the implementation:
v1's premise ("every derived tensor is step-invariant except
num_accepted_tokens") is wrong — the build's derived tensors (state
indices, query offsets, block-table slices) encode per-step-varying
acceptance/context state, i.e. the per-step build work IS the freshness
mechanism. Skipping it returns stale GDN state (all-3 greedy hashes
broke, mismatched within one invocation) — and the memo key
(tensor `.numpy().tobytes()` on staging tensors) cost more than the
build it skipped (65k warm 18.76 -> 10.24, -45%). A "correct" memo
would have to recompute the varying fields — which IS the build. No
v2; the ~4.4ms/step is only removable inside P3 (captured graph),
where the whole build is frozen and replayed.

P3 direct test (v35dp1, 2026-09-02): the nearest reachable
capture-adjacent config — stock PIECEWISE drafter via
`VLLM_XPU_MTP_EAGER_HEAD=0` — BOOT-OK but REJECTED by the hash gate:
f8ref c77d4c73ba1b / f167d905a10b / b78b0a33f97f (P1/P3 diverge, P2
invariant — #18-class batch-state numerics; Paris logprob -0.452 vs
ref -0.451), perf parity at best (2k 25.34 vs 25.25, 65k 23.82 vs
23.73, conc16 19.7/30.5 vs 21.92 — no capture jump). Mechanism: the
v31.1 guard (xpu.py TORCH_COMPILE_DISABLE for spec+TP2) keeps the head
eager, so PIECEWISE registration only reroutes draft inputs through
the static-buffer path (direct_eager_inputs off) — numerics change,
zero benefit. Re-enabling compile for the drafter is CONVICTED by the
v31 matrix (wedge @1 chunk with capture / @563 chunks without, every
splitting-op variant incl. all-custom-ops-split). Draft capture is
upstream-blocked on #11; no local lever remains.

## 18 — fp8-e4m3 nospec greedy "bimodality": ESIMD decode-kernel race —
ROOT-CAUSED & FIXED in v38 (filed 2026-09-02; solved 2026-09-03)

2 of 3 fixed greedy prompts alternated between stable completions (prompt
2 never flipped). ORIGINAL HYPOTHESIS (WRONG, see v38 below): C++ FA2
decode KV-split count varies with resident batch -> reduction-order
changes -> single argmax flips early -> text diverges. Not corruption;
both modes coherent; the 4bit TQ lanes fully rep-stable on the same
prompts.

v34 closure (2026-09-02) — closed as INTRINSIC on the split-count
mechanism, three arms (all measurements real; the MECHANISM ATTRIBUTION
was wrong — v38 proved FA2 was never on this path):
(1) `VLLM_BATCH_INVARIANT=1` (upstream's own pin: num_splits=1 +
batch-invariant backends) is UNBOOTABLE on this stack: it reroutes the
attention selector (`v1/attention/selector.py:153` — mamba/GDN
`supports_batch_invariance()` gate / backend switch) and the container
crash-loops during model init with a silent native death (no
traceback; exit swallowed by restart=on-failure).
(2) Our surgical pin `VLLM_XPU_FA2_PIN_SPLITS=N` (v34_f8bi_pin.py,
flash_attn.py max_num_splits site): N is a **CAP, not an exact count**
— the C++ op still heuristically picks actual splits <= N per step.
[v38 correction: the observed "bimodality PERSISTS under pin=64 and
flips WITHIN a single f8ref invocation (P1 modes {91d489262cb5,
08fae14be388}, P3 modes {c03d1a317495, e15216abe6b0}, P2
3705c4621a59 invariant)" was the ESIMD kernel race below — the FA2
kernel those pins target was never reached. The cap-not-exact finding
about the C++ op itself stands.]
(3) The only value that forces determinism (splits=1) recreates the
#15 serial-scan pathology class at long ctx (2 CTAs scanning the whole
context).

v38 ROOT CAUSE (2026-09-03, kernel-level isolation, all reproducible):
The patch-stack `PAGED_ATTN_ESIMD_INSERTED_v1` gate (flash_attn.py
~1085) routes fp16-Q + XPU-graph + head-256 + GQA>=2 decoder decode to
`custom_esimd_kernels_vllm.eagle_ops.page_attn_decode` (compiled ESIMD
.so, no source) — **fp8 decode BYPASSES the vxk FA2 kernel entirely.**
That ESIMD kernel is RUN-TO-RUN NONDETERMINISTIC on fp8 KV with FIXED
inputs (v38quant.py, 999 calls, bit-compare vs call-0): fp8@511 eager
969/998 differ (same 235-elem signature — MULTISTABLE discrete modes);
fp8@512 eager 132/998, graph 998/998 (toggles 2 modes/replay); fp8@1024
988/998; fp8@2048 150/998; fp8@117 ~1/50; fp16@512 5/998 single-element
(latent on the fp16 lane too). Magnitudes 1-2 fp16 ULP — invisible on
fat-margin tokens, flips knife-edge argmax tokens -> coherent alternate
generations. Engine proof: same request x6 batch=1 cache-hit gave 4
distinct outputs with the gate active; 10/10 identical + f8ref x5 +
700-token gens x4 through the 512/1024 boundary hot zone bit-identical
with fp8 excluded. The vxk FA2 kernel is EXONERATED: 50/50
bit-identical all split configs (v38krig.py), invalid-tail-poison
invariant + fp32-ref correct to 1.4e-5 (v38ref.py). Also why no split
pin could ever fix it: f8ref prompts ~12 tokens -> wkb=2 < 16 ->
heuristic num_splits=1, split machinery INACTIVE at those lengths.

v38 FIX (image adv:v38, `v38_esimd_reroute.py`, baked over v37): the
ESIMD gate additionally excludes fp8* kv_cache_dtype lanes
(A/B force-back `VLLM_XPU_ALLOW_ESIMD_F8=1`). fp8 decode routes to the
vxk FA2 kernel — deterministic, correct, AND FASTER: 2k warm 33.2-33.8
vs 23.99 (+40%, 4bit parity); 65k warm 28.24-28.27 vs 23.99 (+18%;
prod 4bit 22.28-22.36 -> +27%); conc16 coherent; 4bit prod lane
bit-exact untouched (`0ce080630035`). RECLASSIFIED: local patch-stack
kernel race, NOT intrinsic fp8, NOT split-count. fp8-e4m3-nospec is now
a bit-stable lane and the fastest at long ctx (prod-promotion
candidate; prod restored on the v38 image as 4bit pending decision).
fp16/auto keeps the ESIMD kernel — latent 5/998 single-element
nondeterminism documented; a bit-stability demand on that lane would
need the same reroute.

## 19 — fp8_e5m2 KV cache: upstream hard reject with fp8 checkpoints
(wontfix; was half of #15) (2026-09-02)

`vllm/model_executor/layers/attention/attention.py:168` raises
`ValueError("fp8_e5m2 kv-cache is not supported with fp8
checkpoints.")` at WorkerProc init — both ranks, deterministic, ~15s
into boot. Not an XPU/kernel gap. WONTFIX: e5m2 shares e4m3's 1B/elem
with worse mantissa for bounded KV values; e4m3 strictly dominates.

v34 closure (2026-09-02) — TRIPLE-BLOCKED, wontfix now at KERNEL level,
not policy level. Bypassing the guard (v34_e5m2_guard.py env-gates the
raise behind `VLLM_XPU_ALLOW_E5M2_FP8_CKPT=1`) exposes two further,
independent fatal blockers: (a) with graphs ON the boot passes warmup
then DIES HARD (no traceback) at "Capturing CUDA graphs (decode,
FULL): 0%" — the e5m2 KV path faults inside XPU graph capture;
(b) with `--enforce-eager` the boot is healthy (200) but the FIRST
real request kills the worker:
`RuntimeError: Worker failed with error 'Unrecognized FP8 dtype:
fp8_e5m2'` — the XPU FA varlen dispatch (`flash_attn.py:191` lineage)
implements the e4m3 descale path ONLY; there is NO e5m2 kernel. The
upstream guard exists precisely because the kernel does not.

## 16 — vLLM v1 spec decode: acceptance of step N gates scheduling of
step N+1 — the host round-trip chain (~54ms/step @2k) is EXPOSED for
spec but HIDDEN for nospec (architectural, 2026-09-01)

With async_scheduling, nospec overlaps scheduler/prep host work with
worker device execution, hiding the entire host chain (33.6 tok/s @2k =
pure device rate). Spec cannot: the scheduler needs step N's acceptance
counts before it can build step N+1's verify batch, so the chain
worker-finish -> outputs D2H -> shm_broadcast -> engine output processor
-> scheduler -> submit -> worker prep runs SERIALLY every step.
Measured decomposition (k1 @2k, step 98.6ms, device ~44ms): drafter
propose EAGER python ~7ms (k3: 20ms — the v31.1 gate disables inductor
for spec safety, so the MTP drafter runs eager: all_gather, rotary,
get_top_tokens GEMM frames dominate), GDN attention metadata build
~4.4ms (gdn_attn.py build per step), block-table commit H2D ~2ms, and
~30ms IPC/output/wakeup latency — the EngineCore py-spy shows it 100%
IDLE in shm_broadcast acquire_read/sched_yield (latency, not CPU).
The align-mode accepted-count drain (57% of worker SAMPLES) is a
device-wait that async execution already overlaps: removing it
(`v32_align_async_v2.py`, correctness-validated) changes nothing
stacked on the kernel fix. Fix requires a worker-resident acceptance
loop (compute acceptance + next propose on-worker for m steps before
returning) — an upstream v1 architectural change; documented as P2 in
README v32.

## 20 — "thinking loop" reports on fp8_e4m3 KV are budget-limited long
thinking, NOT dtype looping (closed 2026-09-03)

Reports of the model looping when thinking with
`--kv-cache-dtype fp8_e4m3`. Differential battery (8 prompts,
temperature 0, reasoning field inspected with word- AND char-periodicity
loop detectors; full study in
`vllm/patches/diagnostics/kv-dtype-loop-study-v39/`):

- @max_tokens=4096: exactly the same 4 prompts (LIS proof, domino
  tiling, DB schema, host-policy Monty Hall) hit `finish=length` with
  zero output on ALL FOUR dtypes — auto (fp16, unquantized KV
  baseline), fp8_e4m3, turboquant_4bit_nc, turboquant_k8v4 (fp8 keys).
  Zero periodic text loops detected anywhere; trapped tails are
  coherent mid-derivation reasoning.
- @max_tokens=8192: fp16 baseline 4/4 STILL trapped, 0 tail-loops;
  fp8_e4m3 3/4 — the Monty Hall prompt ESCAPES at 7883 tokens and
  produces a full correct answer. The unquantized baseline is not
  better than fp8; there is nothing dtype-specific to fix.
- fp8_e5m2 is not a servable KV dtype on this stack (#19).

Verdict: model behavior (long thinking exceeding the token budget on
certain problem types). Operational mitigation only: raise max_tokens
for thinking requests, or disable/limit thinking
(`chat_template_kwargs: {"enable_thinking": false}`). The mapped
Hadamard-wrap surface at the fp8 store (`flash_attn.py:1407` + FA call
sites) was NOT needed and is parked.

Perf context from the same study (prod:v1, per-dtype steady numbers):
TQ 4bit beats fp8_e4m3 on deep prefill (+18% @16k, +44% @65k tok/s) but
trails on 2k prefill (−37%, attribution open) and deep decode
(−18%/−36%) — the decode gap is architectural (TQ triton 1-warp tiled
kernel vs fp8's native ESIMD flash decode; `VLLM_TQ_STAGE1_STAGES=2`,
`VLLM_TQ_BLOCK_KV=8` and the v39a nibble-split patch all failed to move
it — see `failed/tq-nibble-unpack-v39/`).


