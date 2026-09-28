# COMPLETE ROUND WRITE-UP — decode-speed program T0–T3 + async round (2026-09-27 → 2026-09-28)

Consolidated record of everything tested, everything that passed/failed, every root cause,
every captured number, thoughts, and open items. Source docs: `DECODE_SPEED_ANALYSIS.md`,
`PHASES.md` (P1–P14), `T3_ROUND_REPORT.md`, `../wedgefix-v75/PATCH_STACK_ANALYSIS.md`,
`../wedgefix-v60/README.md`, `../../../KNOWN_ISSUES.md` #24–#29.

Task: *"Run deep analyze why there is drop of token generations speed, do comprehensive plan
to max out token generation speed… then run full implementation and testing suite, bake new
production image `llm-scaler-exp:v*` using best values. Important! Spec and XGrammar-2 has to
have supported on all images always."* Standing constraints: no subagents (all execution
direct); never BARRIER=1; barrier defaults disabled (v1.2.23); never bake from the running
lane; `KERNELS_MAX_JOBS=52` for all wheel builds; host reboot before test rounds; live
investigation record after every phase.

---

## 1. Image lineage (what shipped, when)

| Image | ID | Size | Delta | Verdict |
|---|---|---|---|---|
| v1.2.20 | — | — | v66 FAIRFIX starvation fix | certified, deployed |
| v1.2.21 | `09e114a46903` | 24.7 GB | v88 int64 conv-kernels wheel = tiny-step wedge ROOT FIX | certified, deployed |
| v1.2.22-raw | `e6735a4c72a2` | 24.7 GB | fresh bake: parser qwen3_coder + 85 capture sizes + V63 floor | certified |
| **v1.2.22** | `89b17e0b0f8d` | 24.9 GB | warm lane-commit, 74-entry triton cache | certified, deployed 09-27 |
| v1.2.23-raw | `480840080515` | 24.7 GB | + `--async-scheduling` + `VLLM_RPC_TIMEOUT=60000` + barrier default 0 (67 bake gates) | certified |
| **v1.2.23** | `23a07b1e5ff6` | 24.7 GB | clean warm lane-commit (triton floor 72 = raw 72) | **PRODUCTION since 09-28 11:03** |

Spec MTP ×4 + XGrammar-2 0.2.7 asserted present on **every** image in every gate battery (standing requirement).

---

## 2. Analysis findings (v89, read-only lane)

**The engine did NOT regress — the drop was demand arithmetic + a sync-serialization ceiling.**

- TP2 pair delivered a fixed ~82 tok/s AGGREGATE decode ceiling **under sync scheduling**; the CC
  fleet ran 7–8 concurrent streams (337 req since boot, avg prompt 38.1k, client 10.20.3.18) →
  10–15 tok/s per stream by simple division; 82 tok/s solo (~43 ms/step @ 23 steps/s).
- Step time scaled ~LINEARLY with decode batch: 43 ms solo → 349 ms @ batch 7.2 (2.86 steps/s,
  28.6 tok/iter) = near-zero cross-stream amortization.
- GPU telemetry (`xpu-smi --device N --metrics UTILIZATION,MEMORY --number N`; short flags
  rejected on this host): engines 99.6 % busy but **EU active 16 % / EU stall 57 %**, memory read
  287–305 GB/s = 50–53 % of ~573 GB/s peak → memory-bandwidth/latency-bound, NOT compute-bound.
  Incremental per-stream cost = mamba/GDN state ops + KV reads over 38k contexts + 5
  forwards/iter (4 draft + 1 verify) under TP2 sync.
- Healthy subsystems exonerated with numbers: MTP acceptance 75.3 % in-window (per-position
  84.0/68.7/56.4/47.1 → E=3.56/5); prefix cache 61.0 % hit (37888/62071); dmesg clean, resets 0;
  V66 fired once lifetime (working as designed); mean TTFT 40.6 s = 18.7 queue + 18.8 prefill.
- INCIDENT (operational): the Sep-27 serve was launched BY HAND (fd→/dev/pts/1, no serve log)
  with non-certified flags (mnbs 32, explicit fp8, obsolete serve-side chat-template). Container
  env verified intact via docker inspect. Fixed by T0 certified relaunch.
- **The "~82 tok/s ceiling" verdict was later falsified** by the async round (§4): it was a
  sync-serialization artifact. Aggregate 122–158 tok/s measured under async.

---

## 3. What was tested — full inventory with results

### T0 — certified relaunch ✅ PASS (v1.2.22 round)
Relaunch via certified boot script; hand-launched serve replaced; parser **qwen3_coder**
validated under live CC fleet all day, then BAKED (qwen3_xml lineage retained).

### T1 — scheduler reallocation A/B ❌ REJECTED BY MEASUREMENT (nothing ships)
| Leg | Result | Verdict |
|---|---|---|
| `V63_CONTENDED_BUDGET` 1024→512 | 77.0 / 76.8 vs 77.1 baseline | no gain → default stays 1024; baked guard `if 0 < V63 < 1024: = 1024` so the env knob can never regress below certified |
| `V64_DECODE_INTERLEAVE` 2→3 (budget 512) | 77.6 vs 77.1 | noise → K=2/B=512 confirmed optimal among tested |

Fairness gate re-ran at the certified point: 6.997 tok/s @ 0.05–0.06 s gaps.

### T2a — XPU-graph capture sizes ✅ SHIPPED
Verified padding was real on live shapes (draft batch ≈7, verify ≈35 tokens). Extended to
**85 capture sizes** (dense 1..16 ∪ 5k k=1..64 ∪ legacy) = exact-fit spec-verify graphs.
Solo 71.2–71.4 → **73.6/74.1 tok/s (+3–4 %)**; contention neutral; boot neutral (64 graphs /
31 s / 2.32 GiB; KV unchanged 476,451 tok). Baked in v1.2.22, carried into v1.2.23.

### T2b — barrier-off validation leg ✅ RUN → §3.6 CLOSED
Post-v88 A/B: barrier-OFF 73.3/73.0 vs BARRIER=2 73.6/74.1 = identical; contention same
distribution. PATCH_STACK_ANALYSIS §3.6 closed with live measurement. Per the later user
directive ("disable all default on BARRIERs"), v1.2.23 flipped the **default to 0** (single
site `gpu_model_runner.py:184`, marker + env `=0` + gate `False False 0`; v62 mode-2 machinery
retained). BARRIER=1 remains forbidden (strictly dominated).

### Spec depth 5 ❌ REJECTED ON MATH (never run — documented)
pos3 acceptance 47.1 % → pos4 marginal ~35–40 %; each extra draft = full forward on a
bandwidth-bound stack → net-negative. Question closed.

### T3 — py-spy profiling ✅ RUN (validation lane, never live lane)
EngineCore ~100 % healthy idle (NOT Python-bound). Worker_TP0 **74.6 %** of host samples in
`rejection_sampler.parse_output`.

### T3-1 — host-side parse/D2H lever ❌ FALSIFIED BY MICROBENCH
D2H microbench measured **−12 %** (a loss): the 74.6 % was the host SLEEPING inside `.cpu()`
waiting for verify GPU work — inherent to sync scheduling, not recoverable from Python.
**Lesson: py-spy percentages on a sync-scheduled worker measure where the host sleeps, not
what wastes time. Host-hotspot claims need a GPU-side counterfactual.**

### Async scheduling — attempt 1 ❌ FAILED → root cause found
`--async-scheduling` boot died in EngineDeadError. Root cause (code-level): under async,
EngineCore `sample_tokens` via `collective_rpc` (`multiproc_executor.py:403` →
`future.result()` → `:388` TimeoutError) waits behind ALL queued worker work; default
`VLLM_RPC_TIMEOUT=10000` ms (`envs.py:94`) < heavy-chunk queued latency. dmesg clean — NOT
the v88 wedge class. **Rule: the timeout must exceed worst QUEUED-work latency.**

### Async scheduling — attempt 2 (`+VLLM_RPC_TIMEOUT=60000`) ✅ FULL BATTERY ALL GREEN (07:52:15)
| Test | Result |
|---|---|
| 14-phase drill ×3 | 3/3 `SUSTAIN_COMPLETE_NO_WEDGE`, fence-hits 0 |
| serialized 24 ×3 | SURVIVED ok=24 fail=0 each (72/72) |
| burst_harsh ×3 | SURVIVED ok=36 each (108/108), resets_after=0 |
| genspeed battery-end | **153.40 / 137.58 / 157.98 tok/s aggregate** (4×1024) |
| ASYNC-EVENT-STALL | 0 |
| dmesg engine resets | 0 |
| serve log | 0 errors after restart line 2813 |
| py-spy re-check | `parse_output` 74.6 % → **30.9 %** |

**Headline: 4×1024 aggregate 122–158 tok/s vs 72.6–74.0 sync = +85–110 %; per-stream p50
25–26 vs 10–15 tok/s; solo parity 73–74 (73.2/74.1/74.2) — the win is pipelining, not
single-stream.**

### P12 — bake v1.2.23-raw ✅ 67 GATE-OK / 0 FAIL (08:37:11, ~40 min)
Fresh bake from v1.2.22-raw (never the lane); wheel d20260925 version-gated; barrier patch +
import print `False False 0`; async seds count-1 gates; full lineage set (v60/f15b/v55.3/
arstage/STALFIX/v60c/parser, v63 default+floor, v64 K=2/512, v66 ×4, probe absence v65/v84-87/
v89, xgrammar 0.2.7, spec MTP ×4, pedigree v1218→v1222). Image `480840080515`.

### P13 — fresh-host full validation ✅ ALL GATES PASS (10:44:08)
Host reboot 09:24 → boot 09:26 (standing directive). First-try boot green (async-ON gate
passed first live run), HEALTH_OK 09:31:55 (~180 s). Solo cold seed114 **100.99 s** (TTFT
100.84) = v1222 parity. Drills 3/3 no-wedge. Fair V66 PASS (6.5 tok/s during 106k prefill,
gaps 0.06 s). serial24 ×3 72/72. Bursts 108/108 resets 0. Tracebacks 0. Parser battery
T1/T2/T3 PASS. Solo genspeed 73.2/74.1/74.2. Async 4×1024 **135.45 agg** (floor ≥100).

**JIT gate abort — adjudicated, not an image defect:** the driver's inherited zero-JIT-LINES
gate aborted after the chain was fully green (got=8 lines). Root cause = KNOWN_ISSUES #28
semantics: monitor lines are FIRST-USE events (compile OR disk-cache load), once per process
per kernel name. Evidence: 11 lines, cache 72→73 = **+1 bounded compile** (rejection_greedy
drill shape) = v1222-raw precedent (+2). Gate root-fixed in the driver (host + repo):
**-raw/validation round = cache-delta ≤2; strict zero-writes = production-image gate.**
All 21 driver acceptances re-verified programmatically (`p13_jit_adjudicate_v1223.sh`).

### P14 — ship v1.2.23 ✅ SHIPPED + CC battery adjudicated
Clean-warm relaunch from -raw → lane-commit → `gates_v1223_lane.sh` **ALL PASS 10:53:28**
(incl. dynamic triton floor: shipped 72 ≥ raw 72 = zero compiles during shipwarm; inverted
async-REQUIRED gate). Fresh-boot prod sanity + admission + posture PASS (async engaged,
barrier `False False 0`, RPC_TIMEOUT env). Watchdog repointed (`.pre_v1223` backup) + armed
(`alive armed code=200` 11:08:20; verify window raised 125→220 s — 10-cycle cadence).

**CC battery failure — root-caused to the ship script, NOT the image:** initial t1/t2/t3
http=400, thinking NO_THINKING. Two defects: (1) shell-assembled `-d` JSON tail `")]}"` was
missing the message-object close brace — reproduced exactly (len=134,
`Expecting ',' delimiter: line 1 column 133 (char 132)`); (2) no `Authorization` header —
the proxy requires the master key (401 `No api key passed in.`). Fixed in `ship_v1223.sh`
(repo+host); `cc_battery_fix_v1223.sh` re-ran GREEN: t1/t2/t3 http=200 (tool_use/text),
thinking http=200 + thinking block. **Lesson: build JSON bodies with python json.dumps; always
send `Authorization: Bearer sk-dummy` through :4000.**

---

## 4. Failures encountered and resolved (all)

| # | Failure | Root cause | Resolution |
|---|---|---|---|
| 1 | Bake run-1 abort (v1222) | gate-script bug: `grep -c \|\| echo 0` double-prints | sibling-gate shape; lesson recorded |
| 2 | Async attempt-1 EngineDeadError | RPC timeout 10 s < queued-work latency | `VLLM_RPC_TIMEOUT=60000` baked |
| 3 | Validation watcher false-FAIL | `grep -c` across two files prefixes filenames → `paste\|bc` choke | single `cat f1 f2 \| grep -c` |
| 4 | Bake marker not in bake log | final echo outside internal `> $L` redirect | direct verification; noted quirk |
| 5 | ABORT_JIT_RECHECK got=8 | #28 miscalibration (lines ≠ compiles) | gate recalibrated to cache-delta ≤2 |
| 6 | CC battery 400/401 | ship-script JSON tail + missing auth | fixed + re-run green |
| 7 | plink nested-quoting failures ×3 | escaped quotes through docker exec sh -c | stage scripts / simple calls |
| 8 | Watchdog "no fresh armed-line" | 125 s window < 10-cycle cadence | raised to 220 s |
| 9 | Validation watcher window too short | serial24 ×3 @ async pacing ≈ 2–3 h | restarted 36×300 s |

---

## 5. Final production posture (v1.2.23) — the certified configuration

**Docker run** (from `repro_bootV1223_prod.sh`, verified live): privileged, `--device /dev/dri`,
host network/ipc, CCL_* fabric env, `ZE_AFFINITY_MASK=0,1`, offline flags,
`VLLM_QUANTIZE_Q40_LIB`, `PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`,
`VLLM_V55_EVENT_TIMEOUT_S=600` + `VLLM_V55_DECODE_TIMEOUT_S=30` (v55.3 watchdog),
`VLLM_XPU_SPEC_DRAFT_BARRIER=0` + `MIN_CTX=0` (default off),
**`VLLM_RPC_TIMEOUT=60000`**, `VLLM_F15B=1` + `STALL_S=45`, `VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1`,
**`VLLM_GUIDED_DECODING_BACKEND=xgrammar`**, mounts target/drafter/dflash2 read-only,
image `llm-scaler-exp:v1.2.23`.

**Serve** (baked `serve_user.sh`, log → `/root/serve_full.log`): TP2, gmu 0.8, max-model-len
262144, mnbt 8192, mnbs 64, block 64, fp16, `kv-cache-dtype fp8_e4m3`, mamba fp16,
prefix caching, reasoning-parser qwen3, **tool-call-parser qwen3_coder**, auto tool choice,
**spec MTP ×4**, `cudagraph_mode FULL_DECODE_ONLY` + **85 capture sizes**,
**`--async-scheduling`**, port 8000.

**Boot-script TRAP (standing):** V1212-lineage scripts FORBID `--async-scheduling` (exit 10);
only `repro_bootV1223.sh` / `repro_bootV1223_prod.sh` (inverted gate + marker
`.llm_scaler_exp_v1223_baked`) are valid for this posture.

---

## 6. Performance summary (measured)

| Metric | Sync (v1.2.22) | Async (v1.2.23) | Δ |
|---|---|---|---|
| Solo 1×1024 decode | 73.6–74.1 tok/s | 73.2–74.2 tok/s | parity (win is pipelining) |
| 4×1024 aggregate | 72.6–74.0 tok/s | **122–158 tok/s** | **+85–110 %** |
| Per-stream p50 (4 streams) | 10–15 tok/s | **25–26 tok/s** | ~2× |
| Battery-end agg (3 runs) | — | 153.40 / 137.58 / 157.98 | — |
| `parse_output` host share | 74.6 % | 30.9 % | −44 pts |
| Solo cold TTFT (seed114) | ~101 s | 100.99 s | parity |
| Crash posture (drills/serial24/bursts/resets) | clean | clean ×2 rounds | no regression |

---

## 7. Thoughts / ideas / lessons

1. **py-spy on a sync worker measures where the host sleeps.** Any host-hotspot claim needs a
   GPU-side counterfactual (step time vs GPU-busy) before it drives code changes.
2. **The "bandwidth ceiling" was a scheduling artifact** — always re-test ceiling claims after
   changing the execution model, before investing in kernel work justified by the old ceiling.
3. **Async's one real cost is RPC latency masking** — the timeout rule (exceed worst QUEUED
   latency) generalizes to any future engine-level collective under async.
4. **JIT-line gates must be cache-delta gates** — line counts are first-use events, not compiles.
5. **Validate leaf VALUES, not structure** (CC battery JSON lesson; same class as the v1 callback
   null-args defect): a malformed body fails even when the image is perfect.
6. **Device-side acceptance bookkeeping is now a small residual** (30.9 %) — only attractive
   fused into the verify epilogue; standalone host-sync work is dead post-async.

---

## 8. Open items (not implemented) + next-round phases (P15–P20)

| Phase | Lever | Type | Expected |
|---|---|---|---|
| P15 | Preflight + baseline on live lane | measurement | numbers recorded |
| P16 | **spec_timing composition** (spec-off vs spec-4 A/B + depth sweep 1/2/4) | measurement | draft-vs-verify share → picks kernel target |
| P17 | **xpu-smi EU re-baseline under async load** | measurement | refreshed bandwidth-bound verdict |
| P18 | Config lever A/Bs: L1 `--mamba-ssm-cache-dtype` fp8 probe (halves per-stream state bandwidth; fast numerics accept/reject) · L2 `mnbt 16384` cold-TTFT A/B (contended path capped by V63=1024 regardless) | implementation | ships only on measured win |
| P19 | Kernel spec decision (mamba/GDN state ops vs KV gather vs prefill FA2/GEMM; v65: cold = FA2 40.1 s + GEMM 57.6 s) · torch.compile stays blocked (v31.1, upstream #11) | scoping | build spec for weeks-scale wheel work |
| P20 | **Bake + ship v1.2.24** only if ≥1 lever won (no no-op images) | ship | full battery + gates |

Also open: demand-side policy (litellm per-tier concurrency caps, CC context hygiene) — improves
per-stream TTFT feel (18.7 s of the 40.6 s mean TTFT was queueing), does not raise tok/s.

**Host state at write-up time (12:2x):** user removed `lsv-test` at ~12:11 for manual runs
(shell history shows stale v1.2.21 commands with BARRIER=2 — corrected copy-paste set for the
v1.2.23 posture was provided); lane-watchdog PAUSED 12:14 (documented experiment mechanism;
re-arm `rm /root/build/lane_watchdog.paused`); host 10.20.3.65 unreachable from the admin
host at ~12:20 (likely reboot in progress per user's manual-run pattern). P15 blocked until
the host returns.
