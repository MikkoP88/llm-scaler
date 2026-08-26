
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
Follow-up hardening (v11): the op's XPU path is made genuinely out-of-place
(clone) so the custom-op contract holds on any future inductor-lowered path
— see ALLREDUCE_BOUNCE_FIX_v2 in the patch.

Workarounds on v4-v9 images: serve with `VLLM_XPU_ENABLE_XPU_GRAPH=1`
(recommended, also fastest), or `--enforce-eager` plus
`-e DISABLE_ESIMD_GDN_OUTPROJ=1` (needed because true-eager bf16 otherwise
trips the `esimd_norm_gemv_fp8_blockscale: norm inputs must be fp16`
TORCH_CHECK in the M==1 GDN out-proj fusion — a separate, loud,
correctly-guarded fp16-only fusion). Do NOT use `--dtype float16` as a
workaround: fp16 is equally corrupted under compile+graphs-off, just with
a different garbage flavor.
