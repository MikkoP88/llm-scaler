# v89 execution phases — live investigation record

Task (2026-09-27): full implementation + testing suite of T0–T3, bake new production image `llm-scaler-exp:v*` with best values. Constraints: spec MTP + XGrammar-2 supported on ALL images; no subagents; BARRIER=2 never 1; no async scheduling; never bake from the running lane; `KERNELS_MAX_JOBS=52`; host reboot before test rounds; full ship-chain validation.

Host state at start: user rebooted 10.20.3.65 at ~17:49Z (watchdog stopped 17:46:51, container killed 17:47:01 exit 137 — deliberate, not a crash), lane-watchdog auto-started 17:52. Fresh host = clean slate for the test round.

## Phase log

### P0 — Recon + tooling (done)
- Read full `repro_bootV1221.sh` (certified chain: docker run env block incl. BARRIER=2, 9 boot patchers, config gates, serve launch with logging, health loop 60×10 s).
- Baked `serve_user.sh` (v1.2.21) verified in image: mnbs 64, qwen3_xml, no chat-template, no explicit quantization.
- Tooling inventory: `bench_genspeed.py` (short-prompt decode sharing), `probe_fair_v63.py`, `probe_admission_v66.py`, `burst_harsh.sh`, `repro_sustain.sh` (14-phase × N), `probe_think_cc.py` + `cc_tool_probe.py` (CC battery), `probe_solo_cold.py`.
- New: `bench_v89_contention.py` (8×6k-word distinct-prompt streams, streaming TTFT + per-stream tps + aggregate; prefix-cache-defeating heads per tag/round/stream).

### P1 — T0 certified relaunch (done)
- `repro_bootV1222_T0.sh` = certified V1221 + 3-line insert before serve launch: sed `qwen3_xml`→`qwen3_coder`. Boot 1 HEALTH_OK ~160 s, resets 0.
- Engine-resolved graph sizes (T2a evidence): `[1,2,4,8,16,24,32,40,48,…,512]`, 37/52 graphs, `FULL_DECODE_ONLY`. Padding waste: verify 5·B tokens (batch 7 → 35→40, +14 %); draft batch space 5/6 → 8 (+60/+33 %).
- Parser certification (cc_parser_battery_v89.py engine-direct): qwen3_coder **3/3 PASS** (flat args real strings; nested questions native list; plain reply clean).
- **Baselines (Boot 1, V63=1024/V64=2 defaults):** solo 71.2–71.4 tok/s; contention 8×600×6k agg_median **71.1** (74.5/67.7 — slow mode ~1-in-3 rounds recurs all legs), TTFT p50 25.6–27.0 s; fairness during 11.686/s gaps 0.06; admission max 14.36 s PASS.
- Incident + fix: first env-change restart used wrong pkill pattern (`openai.api_server`; serve runs as `vllm serve`) → stale serve survived SIGTERM (vLLM defers signals during init) and bound :8000; duplicate pid killed via `pkill -9 -P` + `kill -9`. Hardened procedure: pkill → wait-loop (SIGKILL escalation at 15 s) → port/idle verify → relaunch → `/proc/PID/environ` verify. Used for every subsequent leg, zero repeats.

### P2 — T1 scheduler A/B (done — CORRECTED after floor discovery)
- Leg A `V63_CONTENDED_BUDGET=512` (V64 default): agg_median 77.1; leg B +`V64_DECODE_INTERLEAVE=3`: 77.6; all gates PASS (fairness 11.65/11.62 gaps 0.06, admission 14.40/14.39 PASS). Initially read as a +8.4 % win for 512.
- **CORRECTION (decisive):** the baked scheduler floors the env — `if 0 < _V63_CONTENDED_BUDGET < 1024: = 1024` — so **both legs actually ran at 1024** (ground truth = the announce line: `V63_TTFTFIX_ACTIVE contended_budget=1024`). The apparent +8.4 % over the T0-boot baseline (71.1) was boot-to-boot round variance, not a budget effect. Pooled same-config (1024) clean rounds across the day: 61.2–79.3, median ≈ 74.3 — the baseline sample was a low draw. **Lesson: verify the announce line; env presence via /proc is not proof a knob is active when a guard can floor it.**
- **TRUE 512 test** (test-only lane floor sed 1024→256, announce verified `contended_budget=512`): 4 rounds agg_median **70.3** (67.3/66.5/73.3/80.9) vs ~74.3 pool at 1024 = **regression**. The v63-era floor was wisdom: sub-1024 budgets starve block-aligned prefill chunks. Lane floor restored to 1024 (verified), env line dropped.
- Leg B V64=3 attribution was genuine (no floor on v64): 77.6 vs 77.1 = no measurable gain → keep K=2.
- **T1 FINAL: no scheduler change ships — V63=1024 + floor, V64 K=2/512 confirmed optimal among tested.** (Plan hypothesis rejected by clean measurement; "only improvements allowed".)

### P3 — T2a capture-size extension (done — KEEP)
- `patch_cc_sizes.py`: serve `--compilation-config` capture_sizes = dense 1..16 ∪ {5k}₁..₆₄ ∪ defaults-preserved = **85 sizes** (exact-fit: verify 5·B and draft batch B).
- Boot: HEALTH_OK ~140 s (no boot penalty), **64/64 graphs captured in 31 s, 2.32 GiB**; KV cache unchanged 476,451 tok; usage peak 36 %; 0 evictions; v31.1 gate still firing.
- **Solo: 73.6/74.1 tok/s vs 71.2–71.4 baseline = +3–4 %** (batch-1 verify 5→exact, was 5→8 pad +60 %). Cross-checked on a second boot (73.3/73.0 during the barrier-0 leg) — consistent.
- Contention: warm median 75.4 ≈ the 1024-pool median — neutral as predicted (batch-8 verify 40 already exact in the default list). Cold-serve pair (61.2/70.2, TTFT 46/32 s) diagnosed cold-restart jitter, not the graphs.
- Gates: fairness during 10.80 gaps 0.06 (one 6.7 s gap in window; historical max-gap 3.9); admission max **13.19 s PASS — best leg**; 0 resets.
- **T2a verdict: KEEP sizes (solo +3–4 %, contention neutral, zero penalties).**

### P4 — T2b barrier-off leg (done — keep BARRIER=2)
- `VLLM_XPU_SPEC_DRAFT_BARRIER=0` on the winner config, env verified, HEALTH_OK 140 s, single serve, resets 0.
- **Solo: 73.3/73.0 vs 73.6/74.1 with BARRIER=2** (sizes constant, barrier the only delta) — identical within ±0.5 %. Contention median 74.3 (78.1/67.0/74.3) — same distribution as the 1024 pool.
- Decision rule (adopt OFF only on measurable improvement): **no improvement → keep BARRIER=2.** This closes PATCH_STACK_ANALYSIS §3.6 with live measurement: post-v88, the barrier's cost is not measurable at decode granularity.
- Reverted (line removed; container env BARRIER=2 governs). Post-revert HEALTH_OK 150 s, single serve.

### P5 — T3 profiling + scoped kernel plan (done)
- py-spy raw captures under live 8-stream contention (`t3_ec.raw` 1442 samples EngineCore, `t3_w0.raw` 867 samples Worker_TP0, `/root/build/lce1/`).
- **EngineCore: ~100 % healthy idle** (sched_yield 39.6 % + shm_broadcast wait/acquire_read/fence ~45 %) — the scheduler loop is NOT Python-bound; v63/v64 knobs act purely via GPU work allocation.
- **Worker_TP0: 74.6 % of host-side samples in `parse_output (vllm/v1/sample/rejection_sampler.py:270)`** — the MTP acceptance parse/device-sync dominates the worker host critical path; everything else diffuse (<2 % each: layernorm, rope, gemm launches). No Python hotspot to shave; the step cost is GPU-side, as the v89 analysis concluded.
- Scoped kernel plan (weeks-scale, no blind rewrites): (1) rejection-sampler parse path — keep acceptance bookkeeping on device / batch the host sync (per-step win at all batch sizes); (2) mamba/GDN state ops + KV reads at 38k contexts — the EU-stall 57 % headroom measured via xpu-smi; all kernels builds `KERNELS_MAX_JOBS=52`.
- CC battery on the winner config: `probe_think_cc.py` via litellm :4000 tier `qwen3.8-27b-fp8-opus` — t1/t2/t3 OK, thinking streamed, no old-context echo; parser battery engine-direct 3/3 PASS. (Gotcha: :4000 exposes TIER ids — bare `qwen3.8-27b-fp8` 400s "Invalid model name"; probe with a tier id.)

### P6 — v1.2.22 bake (in flight)
- `stage5_bake_v1222.sh` = v1221 chain from **v1.2.21 base** (no engine code, no wheel change — v88 wheel version-gated as inherited) + TWO serve-config changes: parser **qwen3_coder** + extended capture sizes; 53 gates incl. new (parser coder/xml-gone, sizes prefix, v63 floor 1024 + no-256, v64-no-K3, spec-MTP×4 exact string, xgrammar 0.2.7, barrier "False True 0", v1218/v1220/v1221 pedigree markers, no-template serve, no test probes).
- Lane torn down (lane-watchdog stopped first), bake warming. Next: commit → host reboot (standing directive) → fresh-boot full validation suite on v1.2.22 → ship (watchdog repoint, docs, memory, commit).
- **Run 1 ABORT at the final content gate — gate-script bug, not an image defect:** `no_v89_pyspy_probe` carried a `|| echo 0` fallback, but `grep -c` already prints `0` on zero matches (exit 1 just triggers the fallback too) → got=`0\n0` ≠ `0`. All 61 other gates OK; abort preceded marker+commit, so nothing unsound shipped. Fixed to the sibling-gate shape (plain `grep -c`), restaged, re-run from a fresh bake container. Lesson: never add `|| echo 0` after `grep -c` in a count gate — the count is always printed.
- **Run 2 DONE 21:32Z — raw bake committed: `e6735a4c72a2` (24.7 GB).** 0 GATE-FAIL across the full chain (lineage marks, v63 floor 1024 intact + no-256, v64 K=2/B=512, v66 ×4, probe-absence v65/v84–v87/v89, barrier `False True 0`, xgrammar 0.2.7, spec-MTP×4 exact, pedigree v1218/v1220/v1221, parser coder + 85 sizes, no-template serve); warm chain incl. `warm_round2_zero_jit=0`; `repro_bootV1222.sh` derived — diff vs V1221 is ONLY the image line (51) + marker line (85).
- Host rebooted ~21:37Z (standing directive) before the validation round; lane brought up on the raw image via `bash repro_bootV1222.sh v1222raw` (MODE arg is REQUIRED).

### P6b — fresh-boot validation on the raw image (done; surfaced the JIT question)
- Full chain green: HEALTH_OK ~140 s, sanity+ADMISSION PASS, solo cold rc=0, genspeed 72.6/73.9/74.0 tok/s, 4 sampler shapes 200×4, warm_ext replay rc=0, final health 200, dmesg engine resets 0.
- BUT the JIT recheck showed **8 monitor lines** at the recheck point (11 distinct kernel names over the lane's life) and **+2 triton cache dirs written during validation** (rejection_greedy_sample 21:50, eagle_prepare_next_token_padded 22:05 — both autotune-group kernels). Working theory at the time (later superseded — see P7): "bake-window cache entries lost across docker commit".
- Interim mitigation (later adopted as the production ship): `docker commit` of the validated warm lane → **`llm-scaler-exp:v1.2.22` = `89b17e0b0f8d` (24.9 GB)** carrying the complete warm cache (74 entries); raw preserved untouched as **`v1.2.22-raw` = `e6735a4c72a2`**. Documented exception to never-commit-the-lane: full content-gate battery re-run on the committed image — `gates_v1222_lane.sh` **ALL PASS 23:19:41** (asserts the image id, re-runs all 60+ lineage/content gates in a throwaway container, cache ≥74 + raw contrast 72).
- Decisive fresh boot on the lane-commit image (`decisive_v1222_wb.sh`): whole chain green again (sanity+admission PASS, solo cold rc=0, samplers, warmext, health 200, resets 0, cache 74→74 through all traffic) — but `jit_monitor_lines_whole_log=11` → gate FAIL. Not accepted at face value → P7.

### P7 — JIT-mechanism root cause + final ship verdict (done — CLOSED at source level)
- Probes (all direct, no subagents, per user directive):
  - E1 replay (fresh seed): 11→11 lines, 74→74 dirs — once-per-process-per-kernel-name, no value-specialization.
  - E2 novel `top_k=37`: NO new line — sampler keys don't specialize on parameter values.
  - Cross-process disk cache PROVEN working in-container: trivial `@triton.jit` probe, first process 3.097 s compile+store (+2 dirs), second process 0.545 s = disk-cache hit.
  - `find / -xdev -newermt 23:24:15 -path "*triton*"` inside the lane: ONLY the probe's own artifacts — the serve workers' 11 first-use events wrote NOTHING to the root FS. `/dev/shm`: psm/loky IPC only, no triton cache. Worker env: `TRITON_CACHE_AUTOTUNING=1` (set by vllm `env_override.py:113` itself), no `TRITON_CACHE_DIR` override (the `compiler_interface.py:480` site only fires when `--compilation-config cache_dir` is configured — not this lane).
- **ROOT CAUSE (code-level, triton 3.7):** `compiler/compiler.py:274-289` returns the disk-cached `CompiledKernel` on a cache hit without compiling, and `runtime/jit.py:878/886` fires `jit_post_compile_hook` after `compile()` returns **regardless of hit or miss**. Therefore vllm's "JIT compilation during inference" warning = **first-use event per (process × kernel name), whether truly compiled OR served from the shipped disk cache**. The earlier reading "line = real compile" was wrong.
- Reconciliation (every observation explained):
  - raw image: 11 first-use lines over its life = 9 loads from the inherited 72-entry cache + **2 real compiles** (autotune kernels whose specializations no prior image had cached — novel 85-size traffic mix), which stored the 2 new dirs (72→74).
  - lane-commit image: 11 first-use lines, **0 real compiles, 0 dirs written (74→74)** — all 11 served as loads. The shipped cache payload is genuinely load-serving, not inert.
  - **"Zero monitor lines" is unattainable on ANY image by design** (the hook fires on loads; fresh processes always first-use). The correct, attainable zero-gate is **cache-dir delta == 0 during traffic** — v1.2.22 passes it; raw does not (+2).
  - No "cache loss across docker commit" ever occurred: the bake's warm round wrote nothing because every first-use event in the bake container was a load from v1.2.21's complete cache. The 72-vs-74 delta is the raw LANE's own post-commit validation compiles.
- Raw-image A/B re-boot: SKIPPED as superseded — source-level proof + raw's own life numbers (same 11-name set, 2 stores vs 0) already establish lane-commit ≥ raw on every measured axis.
- **FINAL SHIP: `llm-scaler-exp:v1.2.22` = `89b17e0b0f8d` (24.9 GB), v1.2.22-raw `e6735a4c72a2` preserved.** Lane-clean (probe pollution removed, cache back to 74, /root/t.py removed — the shipped image was committed BEFORE the pollution so was never affected), CC battery green through litellm :4000 (t1/t2/t3 OK, thinking present, no template echo), `lane-watchdog` ACTIVE with boot script repointed (V1222 content at the V1212 lineage name; `.pre_v1222` backup preserved). Expected steady-state per boot: ~11 once-per-boot first-use monitor lines, all cache loads, ~ms each — no action item.
