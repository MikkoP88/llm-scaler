
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
  10.20.3.44 -> .45 -> .46 -> .47). If the last known IP stops answering
  after a reboot, probe the next IPs (+1, +2) before assuming the host
  is down.

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

Deeper candidate fix (fork, not yet implemented): run one dummy TP
all-reduce BEFORE loading weights (while device memory is still free)
to pre-allocate the oneCCL scratch arena, restoring 0.8 utilization.

Workarounds (section 05 footer): interactive requests ≤1536 tokens are
rock-solid in every healthy mode; deep single streams are safe on a
clean host with graphs ON (76k full-window validated) — pin
`min_tokens`, and follow the (a) operational protocol (reboot after any
reset; retry bad boots). A serve that wedges in the first ~5 min at 0.8 utilization is (e) —
kill, reboot the host, relaunch at 0.75; never retry a wedge on an
un-rebooted host (pending resets poison it, #03). fp8 KV + dflash is the best validated
capacity/speed point; TQ presets only with `--enforce-eager` (7-7.5
tok/s single stream, spec acceptance ~1-2% — debugging only, graphs are
~9× faster).
