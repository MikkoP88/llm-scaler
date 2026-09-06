# dflash2-crashfix-v50 — image lineage + evidence

## Lineage

| image | base | contents |
|---|---|---|
| `llm-scaler-exp:v1.2` | — | v48 overlays (5 files, md5-verified) + `dflash_v49.py` |
| `llm-scaler-exp:v1.2.1` | v1.2 | swap `dflash.py` v49 → v49c (draft-KV dtype policy: match `--kv-cache-dtype`, `VLLM_DFLASH_DRAFT_KV_DTYPE` override) |
| `llm-scaler-exp:v1.2.2` | v1.2.1 | v50 crash fixes 1-5 (small-fix bump per user scheme) |
| `llm-scaler-exp:v1.2.3` | v1.2.2 | **this recipe + F8** (crash-3 runaway-request guard; small-fix bump). **V123-CERTIFIED 2026-09-06 (battery +24/+20/+2% vs r6c, coh -0.452 bit-identical, zombie 5/5 → guard 5× → engine survived all) + PROD-RESTORED on it** |

## v50 overlays (all md5-gated at bake; validated live via EXTRA_VOL before baking)

| repo copy | in-image target | pristine md5 (v1.2.1) | v50 md5 | change |
|---|---|---|---|---|
| `core_v50.py` | `vllm/v1/engine/core.py` | `74e100658870a2ab56d6b2aa4c973b87` | `3581a600a755aaa542a257f183320484` | F2a hasher resync: capture closure granularity at hasher build; after Scheduler construction, rebuild hasher at the pool's final `hash_block_size` when pool < captured (starving direction only). Loud warning on resync + INFO probe line (logs captured vs pool hbs at boot). **F6** (same file): EngineCore init raises RuntimeError when kv dtype (probes fork `cache_dtype` AND upstream `kv_cache_dtype`) ∈ {fp8_e4m3, fp8_e5m2, fp8} and a speculative config is present (convicted broken lane, v50 B5/B5b/B5d/B5e). |
| `block_pool_v50.py` | `vllm/v1/core/block_pool.py` | `e0f3628790c0718668d2d19063daba29` | `14bf0787301e8b15d78820c9e20d9b2b` | F2b starvation clamp: `cache_full_blocks` clamps `num_full_blocks` to available hashes (log loud) instead of bare `assert` — engine never dies from caching bookkeeping. |
| `dflash2_v50.py` | `vllm/v1/spec_decode/dflash2.py` | `507d09bc2802f00fce8114ab7bec58af` | `66d43621180d5169511a86543788aa13` | F1: `VLLM_XPU_DFLASH_TP1` default 1 → 0 at all 3 read sites; loud CONVICTED-CORRUPT warning on explicit opt-in. |
| `drafter_comm_v50.py` | `vllm/v1/spec_decode/drafter_comm.py` | `865949702f180e4a543843324d4ac747` | `7a78ba74f707feb8b335ac64d098132b` | F1: 4th read site (`dflash_tp1_enabled`) default flip. |
| `dt_warmup_v50.py` | `/root/dt_warmup.py` | `54896f74286301ece8083db781e3ec61` | `254d1de55fc220fe29a6572d9feacdae` | F4: warmup phases 4 (concurrent 2×768 decode) + 5 (~3.5k-token chunked-prefill) so first-use JIT of deeper-context propose/TQ + prefill kernels happens at boot, not under traffic. |
| `async_sched_v50.py` | `vllm/v1/core/sched/async_scheduler.py` | `f3f1fc2444388ca94ee817e7446982db` | `a096c5a846409e83f83445fe97620af3` | **F8** runaway-request guard (v1.2.3): in `AsyncScheduler._update_after_schedule`, force-finish (`FINISHED_ABORTED`, loud warning w/ full telemetry) any request whose `num_output_placeholders` > `VLLM_V50_PLACEHOLDER_LIMIT` (default 64; 0 disables) or whose real `num_output_tokens` > `max_tokens + num_spec_tokens + 8`. Closes the crash-3 death chain: usage-less silent stream end → no abort → outputs stop resolving while scheduling continues → placeholder inflation → worker gather OOB (fatal SYCL assert). Validated F8a 2026-09-06: class recurred deterministically in both rp rounds; guard fired 2× (72>64, ~9 unresolved steps each), requests reaped FINISHED_ABORTED, engine survived both (health 200, 0 asserts, fresh requests served live mid-hang). Residual: each zombie's client SSE stream never terminates (finish can't traverse the stalled output leg) — clients need timeouts; serving-layer abort = upstream ticket Layer 1. Guard is inert on healthy traffic (battery + coh×3 numerically identical, -0.452 reference). |

## Host-side serve script (`dt_dflash2_serve2.sh`, NOT baked — deploy copy)

Guards in deploy order: (v42) `DFLASH2_EMIT_K=4` + graphs → refuse;
**(F7)** `DFLASH2_SPEC=mtp4|mtp7` → refuse — MTP k≥4 temp-0 outputs CORRUPT
under graphs (byte-reproducible garbage) AND eager (nondeterministic garbage;
v50 cells C/I 2026-09-06 overturned the v29c "eager = reference" claim);
mtp1 validated clean, dflash lanes unaffected; **(F6)** fp8-class KV + any
spec → refuse (convicted crawl/death lane); F1 `DFLASH2_TP1` default 0;
BLOCKSIZE knob. Validated live: mtp4/mtp7 exit 1 with FATAL; mtp1/dflash7
pass through and boot.

## Crash 1 (block_pool.py:240 fatal assert) — root cause + evidence

- Cell: v1.2.1, manual lane, **TP1 default ON** (`VLLM_XPU_DFLASH_TP1=1`), tq4nc
  (draft=tq4nc v49c default), `--block-size 512`, memutil 0.92, MAXLEN 262144,
  prefix caching ON, async scheduling ON, dflash k=7, pool 270,087 tokens.
- R2 reproduced: tiny OK → long0 stream ends **without usage chunk** at 784 toks
  → long1 → fatal `assert len(request.block_hashes) >= num_full_blocks` ~4 s
  later. Acceptance 2.1–2.8% (v48 TP1 corruption signature).
- R3 instrumented dump (decisive): gid=14, group block_size=1024, pool
  hash_block_size=1024, **hasher closure captured 2048**, len(block_hashes)=0,
  num_full_blocks=1, all_token_ids=1024, status=RUNNING (zombie long0 decoded
  past its usage-less stream end; 8 stuck output placeholders).
- Mechanism: dflash2 v48 TP1 draft-spec remap (2048→1024) shrinks the pool gcd
  AFTER `EngineCore.__init__` built the request hasher from
  `resolve_kv_cache_block_sizes()` (lcm value 2048). Draft group then demands a
  full-block hash at 1024 tokens; the 2048-granularity hasher has produced 0
  hashes → assert.
- R1 control (TP1=0): pool 373,824; identical client ALL-CLEAN with usage
  chunks; acceptance healthy (0.589/0.339/0.161). TP1=1 is the necessary
  co-factor.

## Fix validation

- **R4** (exact R2 crash cell + v50 overlays via EXTRA_VOL): ALL-CLEAN —
  tiny + 3×8192 streams, every stream ended WITH usage chunk; boot log shows
  `llm-scaler v50: request block hasher re-synced to final pool hash
  granularity 2048 -> 1024`; `hash-starvation clamp` fired **0** times (the
  resync removes the starvation; clamp is defense-in-depth); zero errors;
  engine alive post-run. TP1-lane acceptance remains degenerate (3.4–9.5%,
  per-position 0.22/0.02/0.00) — expected; the lane is corrupt, F1 defaults
  it OFF; v50 only makes the engine survive it.
- **R5** (certified lane TP1=0 + v50 overlays): see PLAN.md status log.

## Posture

- **Prod (2026-09-06 04:01): `llm-scaler-exp:v1.2.3`**, dflash7 + tq4nc +
  k8v4 draft override, WARMUP=1, via stock `dt_dflash2_serve2.sh`
  (no overlays — all fixes baked). V123 certification = the final-image
  gate per the standing mandate (all crashes fixed before the final image).
- Certified fallbacks: `llm-scaler-exp:v1.2.2` (TP1=0 lanes, pre-F8 —
  crash-3 NOT contained, zombie recurrence is engine-fatal there);
  `llm-scaler-exp:v1.2.1` (TP1=0 lanes, pre-v50).
- Rollback knobs: `VLLM_XPU_DFLASH_TP1=1` re-enables the (corrupt) replication
  lane; the clamp/resync are pure hardening with no behavior change when
  hasher and pool agree (R5 silent no-op verified); `VLLM_V50_PLACEHOLDER_LIMIT=0`
  disables the F8 guard (default 64 — a healthy request never exceeds ~2 steps
  of unresolved placeholders; the zombie signature crosses 64 in ~9 steps).
- Residual known-risk classes NOT covered by any fix (both need upstream
  work, see upstream_vllm_ticket.md + the v29 oneCCL tickets):
  (a) the serving-layer trigger of the usage-less silent stream end — F8
  contains the blast radius engine-side (request lost, engine survives);
  (b) the #11 oneCCL collective-spin wedge — observed once on the mtp1 lane
  (v50 mtp1_loop, wedge_mtp1_0228.log); metrics freeze, workers+EngineCore
  spin, health stays 200 = silent service loss; dflash7 never wedged across
  17+ v50 cells; mtp1 stays F7-allowed experimental, not promoted.
