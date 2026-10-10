# LMCache Integration — Full Discovery & Phase Writeup
**Lane:** 10.20.3.65 / container `lsv-test` / vLLM fork `0.21.1.dev0+gad7125a43.d20260826`, XPU TP2, Qwen3.5-27B hybrid (48 GDN + 16 full-attn + 1 MTP), lmcache 0.5.5 (locally patched)
**Window:** 2026-10-08 → 2026-10-10 (continuing)
**Task:** deep-research LMCache → integrate into fork → max perf → crash-free validation → bake production image → baseline-vs-LMCache table. LMCache runs locally, no external server.

---

## Phase map

| Phase | Name | State |
|---|---|---|
| P1 | LMCache deep research (architecture, V1 connector protocol, storage backends, hybrid-state implications) | DONE |
| P2 | Fork integration v136 (LMCacheConnectorV1 + mamba recurrent-state store/load) | DONE |
| P3 | Corruption capture under eviction pressure (telemetry legs) | DONE |
| P4 | Root cause proof — TWO components | DONE |
| P5 | Fix layers v138 (inert) → **v138b** (per-group padded-slot zeroers) → **v138c** (carry sanitize) | DONE, deployed |
| P6 | Validation matrix (hammer + natural-rate) | DONE — natural-rate connector-OFF **PASSED** |
| P7 | Connector-ON: worker freeze class + A2-reuse divergence | **CORRECTNESS CLOSED (v139)** — freeze = watch-item (2 clean runs since; py-spy watchdog stays armed) |
| P8 | Perf parity (baseline v1.2.32 vs LMCache) + knob sweep | PENDING |
| P9a | Clean-CCL perf matrix: LMCache vs baseline v1.2.32 | CLOSED — gate MET |
| P9b | Bake production image + final table | PENDING |

---

## P1 — Deep research findings that shaped the design

- LMCache V1 connector (`LMCacheConnectorV1`) hooks `wait_for_lookup` / `start_load` / `wait_for_save` per scheduler step; attn KV store is chunk-based (default 256; lane uses 512 = mamba tokens_per_page).
- LMCache is KV-cache-only upstream: **zero support for recurrent (mamba/SSM) state**. A hybrid model's prefix cache without mamba state is unusable — resume needs the frozen recurrent state at the resume depth. → integration must add a parallel "mamba engine" keyed by token-prefix hash, with an ordering law: an attn hit without its mamba state must never happen (both-or-neither).
- Storage: local only per task constraint → CPU-memory/disk backends via `kv_connector_extra_config {"use_native":true}`; no external server.
- Upstream store path assumes pool pages are final at `wait_for_save` time — **false under this fork's async scheduling** (see P2 store-lag law).

## P2 — v136 integration (deployed)

- Connector line: `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true},"engine_id":"lsv65"}'` (serve script line 57; deleting the line = baseline boot).
- **Mamba engine:** second LMCache engine instance; one object per materialized 512-token boundary, keyed by stock chain hash of `tokens[0:B]`, filled from CPU seed bytes frozen at materialization time (`_LMC_BOUNDARY_REGISTRY` in `mamba_utils`), never from pool pages (align mode nulls boundary pages). Any later resume at any banked depth strict-prefix-hits.
- **Ordering law enforced:** failed mamba store ⇒ skip that request's attn store; lookup requires both engines to hit.
- **Store-lag law (LMC fix 2026-10-09):** store-side `from_gpu` gathers raced the KV-write kernels under async scheduling (span handed to `wait_for_save` before the chunk's kernels were submitted → gathered all-zeros → cached zeros → diverged output). Fix: snapshot each store, execute on the NEXT `wait_for_save` after a device sync (`LMC_STORE_LAG=1` default). Lookups racing the lag miss cleanly.
- MambaDType support: engine accepts `float16 / fp8_e4m3 / fp8_e5m2`; store+lookup path proven on fp8_e4m3; fp16/fp32/e5m2 legs pending.
- Debug/instrument surface (TEMP, strip before bake): `LMC_DBG_FWD/TOGPU/BYTES/POLL/CHUNKS`, `LMC_PJ` (EngineCore journal delay), LWSDbg* lines, `/tmp/lmc_off|lmc_snap_off|lmc_store_off` kill switches.

## P3 — Corruption capture (the eviction-pressure legs)

Probe: A-bank (long request) → 96 short fillers (eviction pressure) → A2 (A re-sent; must byte-match A). Fillers ending `md5 98abe3a2 '!!!!'` = v60c POISONED-ROW sampler guard replacing NaN-logit final tokens.

| Leg | time | connector | PJ | sleep | mamba dtype | fix | result |
|---|---|---|---|---|---|---|---|
| base_evict | 18:41 | OFF | – | 0.3 | fp8_e4m3 | none | filler26 `'!!!!'` then **RemoteDisconnected — engine DIED**; dmesg: ccs guc_id=24 + bcs guc_id=32 engine resets 18:41:09 (both cards) — the historical guc-ccs death chain |
| telem_evict | 19:01 | ON | 1 | 0.3 | fp8_e4m3 | none | 96/96 clean, NANDET 0, LMCBOUND 392/0 finite |
| telem_a | 19:13 | ON | 1 | 0.3 | float16 | none | 96/96 clean, NANDET 0 |
| telem_b | 19:0x | ON | 2 | **0.0** | fp8 | none | **8 NANDET** (A-side all-NaN conv pages), corrupt capture |
| C1 | | OFF | 2 | 0.0 | fp8 | v138 inert | 6 NANDET, 2 poison |
| C2 | | OFF | 2 | 0.0 | fp8 | inert+instr | 2 poison, A==A2 False |
| C3 | | OFF | 2 | 0.0 | fp8 | **v138b live** | 4 NANDET, 2 poison; **B-side hashes all-zeros** (zeroer works) |
| C4 | | OFF | 2 | 0.0 | fp8 | **v138b+c** | 2 NANDET, 2 poison; **A-side n393216 all-NaN conv page carried 5→6 → dest n0** (sanitize healed live) |
| **leg_nat** | 20:46 | OFF | – | 0.3 | fp8 | **v138b+c** | **95/96 clean, NANDET 0, anan 0, A==A2 True, TRACEBACKS 0, engine alive, no dmesg reset** — the base_evict death conditions, survived |
| leg_conn | 20:58 | **ON** | – | 0.3 | fp8 | v138b+c | **NEW CLASS: workers froze** at filler ~50 (see P7) |
| leg_conn2 | 21:26 | ON | – | 0.3 | fp8 | v138b+c | 96/96, no freeze, but **A2 reuse diverged** (md5 cc32abcf ≠ 23af1bed, parked B=2560) — see P7 |
| **leg_conn2 v139** | 21:52 | ON | – | 0.3 | fp8 | v138b+c+**v139** | **A==A2 True (7f0bfa4b), A2 wall 1.85s vs A 5.69s (3.1×), parked B=3072, store(final) banking confirmed, 0 poison, 0 tracebacks, engine alive, no freeze** |

Calibration laws:
- Clean legs (PJ=1): **0 poison, NANDET 0** — that is the acceptance reference, POISONED_ROWS ≈ 98 = normal guard noise.
- Zero-sleep+PJ2 hammer is an artificial worst case; its constant ~2 poison = short fillers ending mid-block (see P4).
- `A==A2` md5 varies ACROSS boots (23af1bed / 932b5c22 / 7f0bfa4b all legitimate for the same prompt — spec/async nondeterminism). Only WITHIN-run A==A2 is meaningful.
- LMCache exonerated as corruption source: all corrupt legs had connector OFF. Block-pool bookkeeping exonerated: ANOM 0 through 798 evictions.

## P4 — Root cause (proven, two components)

**Component 1 — ALLOCATION RESIDUE (killed by v138b).**
`KVBlockZeroer.init_meta` skipped MambaSpec layers; the single-type manager recorded only FullAttentionSpec block ids → recycled mamba pool pages kept previous occupants' content (saturated-448 / all-NaN fp8 conv + fp16 ssm residue). Async-overlap read-before-PRE-COPY-carry consumed the residue as live recurrent state → `'!!!!'` outputs / NaN-zombie spiral. Proof: telem_b A-side tables; C3 B-side hashes all-zeros with zeroers live.

**Component 2 — IN-COMPUTE NaN ORIGINATION + CARRY SPIRAL (propagation killed by v138c).**
A request's live conv state goes 100% NaN mid-block: fp16 accumulation overflow → inf → fp8_e4m3fn cast = NaN bytes 0x7F/0xFF (the 448 store-clamp only catches finite overflow; fn has no inf). The PRE-COPY carry then propagates the NaN page boundary-to-boundary forever (telem_c3: same md5 1f0e8545 n393216 anan at prev=5→6 AND 6→7 = carried NaN page). Origination is a kernel-numeric property of the compiled GDN .so — not patchable at this layer; a short request that ends inside its final block after origination can still emit one struck final token (the residual 1-2/96 under hammer / 1/96 at natural rate).

**Geometry & bookkeeping laws discovered (v138b design constraints):**
- PADDED-SLOT POOL GEOMETRY: mamba pools are NOT packed. ssm (1059,7,5120) fp16: slot stride 512 KiB, content 70 KiB; conv (1059,24,128,128) fp8: stride 512 KiB, content 384 KiB. In int32 words: stride 131072, ssm len 17920, conv len 98304. Pools are PER-LAYER (16 layers/group → 16 segments per class). Zeroing must be content-bounded (len), addressing by stride.
- PER-GROUP DISJOINT ID LAW: the 3 mamba groups have DISJOINT block-id spaces (telem_b: g0 [1-5,21,26], g1 [6-10,22,27], g2 [11-15,23,28]). A flat union applied across group pools would zero LIVE pages.
- Coordinator law: `single_type_managers` is a positional 1:1 tuple over `kv_cache_groups` → manager index == group index.

## P5 — Fix layers (deployed, host staging /root/build/fixv138/)

1. **utils.py** (md5 0743c44e…): `_zero_kv_blocks_kernel` gained `PAGE_STRIDE_EL`; `from_raw_segments(device, pin_memory, seg_addrs, page_size_el, page_stride_el)`; meta now 5-tuple (segs, len, stride, blk, n_segs).
2. **gpu_model_runner.py** (md5 e20dca2f…): `_init_kv_zero_meta` builds `_mamba_block_zeroers_by_group` — per mamba group, per (stride,len) class, per-layer guards (alignment/contiguity/content≤stride), data_ptr dedup; logs `llm-scaler v138 mamba zeroers: group=%d [(131072,17920,16),(131072,98304,16)]` ×3 groups ×2 ranks. `_zero_mamba_block_ids(ids_per_manager)` zeroes per-group ids only.
3. **kv_cache_manager.py** (md5 880646f2…): `take_new_mamba_block_ids()` → list over `single_type_managers` (per-group lists).
4. **output.py** (md5 5bd0d7f3…): `new_mamba_block_ids_to_zero: list[list[int]] | None` (separate from attn `new_block_ids_to_zero` — NEVER mix the id spaces).
5. **scheduler.py** (md5 fe2ce15b…): collects per-manager ids when `needs_kv_cache_zeroing`.
6. **single_type_kv_cache_manager.py** (md5 5771a8b4…): records mamba ids at the 3 alloc sites (from earlier round, unchanged).
7. **mamba_utils.py** (md5 d820e320…): v138c `_lmc_sanitize_state_page` — fp8_e4m3fn: byte-mask 0x7F/0xFF → 0; else `page[~isfinite] = 0`. `MambaCopyBuffers.sanitize_pages` appended at PRE-COPY dst fill; executed in `do_mamba_copy_block` right after `batch_memcpy` (heals the carried page at each 512-token boundary; stale-list sanitize is harmless).
8. **vllm_v1_adapter.py v139** (md5 12cabc38…, backup `.v138` alongside): token-id stash + finish/lag flush of `_LMC_BOUNDARY_REGISTRY` (see P7.2) — kills the one-step store lag and the overlap-dropped final wfs row; marker `LMC mamba store(final)`.

**Env kill-switches:** `VLLM_MAMBA_ZERO_NEW_BLOCKS=0` (zeroers off), `VLLM_MAMBA_SANITIZE_CARRY=0` (sanitize off).

## P6 — Validation verdict

- Natural rate (the production class — base_evict's death conditions): **v138b+c SURVIVES**. 95/96 clean + 1 struck final token (in-compute origination inside the filler's final block; no boundary reachable by design), NANDET 0, no persistent corruption, no cascade, no dmesg reset, A2-reuse byte-identical.
- Under the artificial hammer: fixes monotonic (NANDET 8→6→4→2; C4 shows live healing), poison constant ≈2 = mid-block-ending class, fix-independent.
- dmesg proof: last ccs/bcs resets 18:46:30; ZERO resets during any v138b/c leg (19:13–20:55).
- Dtype coverage: fp8_e4m3-mamba natural-rate PASSED (leg_nat); float16-mamba clean under PJ=1 (telem_a); e5m2/fp16/f32 natural-rate legs pending. User directive: standing combo = `--kv-cache-dtype fp8_e4m3 --mamba-ssm-cache-dtype float16` (both flags safe — attn fp8 was never implicated; mamba fp16 covered by both fix layers and is the more forgiving dtype: overflow saturates to ±inf, caught by isfinite).

## P7 — Connector-ON: freeze class + A2-reuse divergence

### 7.1 Worker freeze (WATCH-ITEM, not reproduced since)

leg_conn (connector ON, no PJ, natural rate, v138b+c): both Worker_TP ranks froze simultaneously at 21:06:56,94x — last log `LWSDbg mamba-save: req=…afeffbd1 m_skip=0 last=False ntok=3072 bnd=None` → 5 min silence → `RPC call to sample_tokens timed out` (EngineDead, HTTP 500). 

- Freeze point narrowed to the attn-store loop between the mamba-skip and the `Storing KV cache` log — only ops: `slot_mapping.to(device)` H2D, `torch.ones`, logger.info.
- **No dmesg/GPU reset during 5 min of hang** → not a hung kernel (guc would have reset) → CPU-side block: H2D on a wedged transfer context, allocator/logging lock, or GIL held by a deadlocked LMCache daemon thread.
- Both ranks within 1 ms = deterministic shared-state block, not an incidental thread race.
- Discriminating fact: telem_evict ran the SAME connector path 96/96 clean **under PJ=1** (EngineCore journal delay closes the async-overlap window). Connector-ON + no-PJ re-opens it → same overlap-race family as P85 M1 (parked side-copy context), now on the connector store path.
- leg_conn2 + v139 leg: **2 consecutive clean runs, no freeze**. All future connector legs keep the py-spy stall watchdog (serve-log staleness ≥90 s ⇒ dump both Worker_TP stacks, ≤6 dumps).

### 7.2 A2-reuse divergence — ROOT-CAUSED and FIXED (v139, 2026-10-10)

leg_conn2 (pre-fix): A (prompt 3323 tok = 2560+512+251) stored attn 2560/2560 then 512/3072, mamba ONLY deepest=2560; A2 (same prompt) hit the cache (2.32 s vs 5.26 s) but resumed at B=2560 → 763-token recompute → md5 cc32abcf ≠ 23af1bed. **Cached bytes were proven bit-faithful** (LMCDbgBytes store-time page hashes 0c75f716/29c63138 == load-time) — completeness bug, not corruption. Chain:

1. **One-step store lag**: boundary snapshots materialize in the copy path one step AFTER the chunk that crosses them; each wfs can therefore be one boundary behind (A's chunk-1 boundary [2560] was consumed at chunk-2's wfs — the mamba-store log timestamps prove it).
2. **Final-wfs row dropped**: A's chunk-3 (3072→3323) never produced ANY wfs line (no attn "251 out of 3323", no mamba store) — the async-overlap empty-row class (P84/v60c POISONED-ROW) ate the metadata row, so the [3072] boundary sat unconsumed in the registry until FIFO eviction.
3. **251-token tail is uncacheable by design**: 3323 is not 512-aligned; the lookup-side joint depth (attn∩mamba) capped at 2560.

Fix **v139** (vllm_v1_adapter.py, md5 12cabc38, backup .v138): (a) `_lmc_token_stash` — last cumulative token span per request (FIFO 64) filled at every wfs; (b) finish/lag flush after the mamba store loop — for registry req_ids ABSENT from the current step's requests, pop and store all boundaries from stashed tokens, marker `LMC mamba store(final)`. Early flush of mid-flight requests is harmless (bytes frozen at materialization, prefix-hash keys, idempotent overwrites); flush failure is non-fatal (lookup-side intersection already protects the ordering law). `save_unfull_chunk=True` (LMCache default) means `process_tokens(tokens[:B])` natively yields the unaligned-tail hash — no custom key scheme needed.

**Proof (leg_conn2 v139, fp8 mamba):** `store(final)` lines fired (deepest=3072/3584), A2 parked **B=3072** (was 2560), 251-token tail recomputed, **A==A2 True (7f0bfa4b)**, A2 wall **1.85 s** vs A 5.69 s (**3.1×**, and faster than the broken 2.32 s — deeper resume wins twice). 96/96 fillers, 0 poison md5, 0 tracebacks, engine alive, no freeze. NANDETECT ×1 = v138c carry-sanitize catching an in-compute origination at boundary 6→7 and healing it (output stayed clean — the same 1/96 natural class as leg_nat's struck filler, but caught because the request lived past a boundary).

Corollary law: **short-tail recompute preserves byte-parity** (59-token engine-native tail: telem_evict; 251-token connector tail: v139 leg) — exact-depth finish-time park (F-B) is NOT needed for correctness; it remains an optional perf micro-lever only.

### 7.3 fp16 mamba lookup=0 — ROOT-CAUSED: mamba CPU-pool eviction (fixed by `mamba_pool_ratio`)

Standing combo leg at fp16 (`telem_conn2_fp16_serve.log`, md5 d378ace4): acceptance all green (A==A2 True 932b5c22, 96/96, 0 NANDETECT, no freeze) but **via engine-native fallback** — 0 `seed parked`, 0 `async load complete`, A2 3.0 s ≈ native 2.83 s, not the 1.85 s LMCache class. The one joint-lookup line: `attn_hit=3072 mamba_hit=0 -> 0` — mamba storage answered MISS for content provably stored minutes earlier (A's regular deepest=2048 + `store(final)` deepest=3072 both committed at fp16).

Forensics that killed the false leads: `kv_dtype=torch.uint8` on BOTH the scheduler-side lookup client and worker engine (key dtype identical — dtype NOT in the failure path); `kv_shape`/`page_size_bytes` post-init override (adapter 1812-1838) touches only the worker engine and `kv_shape` is not part of `CacheEngineKey`; zero eviction/no-space log lines. **The distribution was the tell**: BOTH legs' `LMCache hit tokens` lines are 97× zero — the probe's churn phase evicts everything at both dtypes — and exactly ONE rehit (doc1's A2, sent before churn) consults LMCache with a live store: fp8 → attn 3072 + mamba 3072 → park; fp16 → attn 3072 + mamba 0 → clamp → full recompute. The entire fp8-vs-fp16 contrast is that single lookup.

Root cause (arithmetic): mamba boot line `CPU pool scaled 60 GB -> 7 GB (ratio=8, page_bytes=1048576, tokens_per_page=1024)` — fp16 `page_bytes` is exactly 2× fp8's 524288, so per-chunk objects double (~89.3 MB vs 44.6 MB per 1024-tok chunk; derived from the fp8 seed 22,315,008 B / 512 tok = 43,584 B/tok). The probe's A-phase (32 docs × 3 chunk-objects) = **4.3 GB at fp8 (fits 7 GB) vs 8.6 GB at fp16 (exceeds it)** → LMCache CPU backend silently LRU-evicts; doc1's mamba chunks (stored first = oldest) are gone before its A2 arrives 5 min later. Eviction is silent at INFO level. Attn side never affected (60 GB pool vs ~2 GB stored).

Fix: **`mamba_pool_ratio` extra-config knob** (already read at `_init_mamba_engine`, adapter ~line 925) → 15 GB mamba CPU pool (host RAM: 212 GB available, zero GPU cost), covers fp16's doubled footprint with headroom; identical for fp8 (harmless). LAW: **mamba pool sizing must be derived from page_bytes × workload, not token count** — any dtype change re-scales the pool requirement 1:1 with page bytes. LAW (plumbing, cost two failed boots): `kv_connector_extra_config` keys reach LMCacheConfig ONLY with the **`lmcache.` prefix** and only for existing config fields (`_validate_and_set_config_value`, adapter line 1273-1281); `get_extra_config_value` reads ONLY the `extra_config` sub-dict — so the working form is `{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":4}}` (top-level `mamba_pool_ratio` and nested-but-unprefixed `extra_config` are both silently ignored). Boot proof: `Updated config extra_config from vLLM extra config: {'mamba_pool_ratio': 4}` + `Mamba engine CPU pool scaled 60 GB -> 15 GB (ratio=4, page_bytes=1048576)`.

**Verification leg (telem_conn2_fp16b_serve.log, md5 12c197ee):** mechanism fully restored at fp16 — doc1-A2 joint lookup returned full depth (no clamp line), **`seed parked B=3072 bytes=41189376`** (exactly 2× the fp8 seed's 22,315,008 — dtype scaling end-to-end), hit distribution 97×0 + 1×3072 (identical to fp8), 2× async-load-complete, 194 store(final). **A2 wall 1.3 s** (vs 1.85 s fp8 class / 3.0 s broken / 5.69 s cold), **A==A2 True (932b5c22)**, 96/96, 0 NANDETECT, 0 poison, 0 tracebacks, no freeze (watchdog quiet), engine HEALTH=200. Standing combo (kv fp8_e4m3 + mamba float16 + connector ON) now validated at BOTH correctness and the LMCache fast-reuse mechanism.

Pool budget law for P8 (measured, mamba objects ≈ 87.2 KB/token fp16 / 43.6 KB fp8): bench docs are **~5.2k tokens actual** (the generator's 1.3 chars/token estimate is ~4× off for word-soup content — ptok_med=5202 churn 5697), so cold 16 docs ≈ 7.2 GB fp16, churn 112 docs ≈ 55.6 GB fp16, full retention set ≈ 63 GB. **Bench-shape law: N_CHURN=112** (112×5697 ≈ 573k + origs > 506k GPU attn pool → true native eviction; the original 48-doc churn = 246k tokens NEVER evicted → the rehit axis was fake until this fix). Uniform **ratio 1 (60 GB mamba pool)** for both dtypes at bench shape; measured retention at 60 GB holds the full 128-doc set (ON p2/p3 churn 0.77 s = served, vs OFF 3.66 s). Pass-2/3 "cold"/"churn" phases are retention phases (deterministic docs re-hit from pass 1); only p1 phases are true-cold.

## P8 — Perf parity matrix (CLOSED 2026-10-10 01:15 UTC — all 4 legs, ZERO regressions, dtype law discovered)

Method: `leg_p8.sh TAG DTYPE RATIO MODE` — one engine boot per leg, WARM-ENGINE CEILING LAW honored (prefix pass1 = warm-up, passes 2+3 = records N=2), then bench_run7n ×2. Tags: `f16on/f16off` (mamba float16, ratio 1 = 60 GB pool), `f8on/f8off` (mamba fp8_e4m3, ratio 1 = 60 GB pool ≫ needed; uniform with fp16). Artifacts: `bench_{TAG}_prefix_p{1,2,3}.out`, `bench_{TAG}_run7n_{1,2}.out`, `telem_bench_{TAG}_serve.log` (f8on md5 626118ae), `leg_p8_{TAG}.out`.

**fp16 pair verdict (ttft_med seconds, p2/p3 records; p1 in parens where it's the only true-cold sample):**

| phase (docs) | OFF baseline | ON LMCache | verdict |
|---|---|---|---|
| cold 16 orig (~5.2k tok) | 0.34-0.38 (p1 3.49) | 0.35 (p1 3.62) | parity; p1 true-cold +4% = store overhead, noise-class |
| warm 16 orig re-ask | 0.34-0.37 | 0.34-0.35 | parity (both engine-native hits) |
| churn 112 new | 3.66-3.67 (p1 3.67) | 0.76-0.77 (p1 3.89) | **4.7× — CPU tier retains across GPU eviction** |
| rehit 16 orig post-churn | 3.45 (p1 3.45) | 0.58-0.59 (p1 0.62) | **5.6× — THE value axis** |
| run7n agg4 tok/s | 239.85-270.04 | 245.78-269.55 | parity |
| run7n solo tok/s/req | 75.75-84.76 | 76.21-77.35 | parity (OFF solo max 84.8 = single-run noise) |

Mechanism (f16on serve log): 544 seed parked, 256 store(final), 0 depth-clamp lines, 6 LMCNANDETECT / 96 requests in the bench leg (accepted natural rate — sanitizer heals, output clean), 0 Tracebacks, no worker freeze (py-spy stall watchdog quiet), HEALTH=200 at close. Hit distribution healthy (336×5120-class full-depth joint hits).

**fp16 LAW: LMCache is strictly SUPERIOR at the standing combo — 5.6× rehit TTFT, 4.7× retention-churn TTFT, zero regression anywhere (cold +4% is store-write overhead within noise).**

### fp8 pair (COMPLETE) + THE MAMBA-DTYPE DECODE LAW

f8on leg first looked like a decode collapse (agg4 78-90 vs fp16's 240-270, solo 21-27 vs 76-85). Boot config verified correct (mamba fp8_e4m3, pool 60 GB page_bytes 524288 tokens_per_page 512, LMCPOOL tensors confirm fp8 SSM state, spec MTP×4, async, mechanism 544/256 = identical to f16on; `joint lookup`/`hit distribution` grep = probe-only, 0 in ALL bench legs). **f8off leg decided it: agg4 76.4-91.4, solo 22.5-27.1 — identical to f8on within dtype.**

**MAMBA-DTYPE DECODE LAW (new discovery, 2026-10-10): mamba-ssm-cache-dtype float16 = 2.7-3.2× decode throughput vs fp8_e4m3 (solo 76-85 vs 22-27 tok/s/req; agg4 240-270 vs 76-91).** The fp8 rate matches the P84-certified standing-image rate (27-28 tok/s/req) — the lane has ALWAYS run mamba fp8 until this program added float16 support. Mechanism (inferred): fp8-quantized SSM recurrent state degrades MTP drafter/target agreement → spec-acceptance loss; fp16 state = exact. Prefill also dtype-class slower at fp8 (true-cold 3.90 vs 3.49 s/doc; warm 0.54-0.56 vs 0.34-0.37 s; both visible connector-OFF). **LMCache is exonerated and holds parity at BOTH dtypes; float16 mamba dominates fp8 mamba on every axis (decode 2.8×, prefill ~10-15%). The standing combo (kv fp8_e4m3 + mamba float16 + connector ON) is the best configuration measured.** Side-evidence: 4 LMCNANDETECT in the f8off leg — the v138c carry-sanitizer is engine-side (fires without connector), natural-rate class, output clean.

### COMPLETE 4-LEG MATRIX (bench_prefix TTFT med s, p2/p3 records, p1 true-cold in parens; bench_run7n N=2)

| leg (mamba dtype × connector) | cold 16 | warm 16 | churn 112 | rehit 16 | agg4 tok/s | solo tps |
|---|---|---|---|---|---|---|
| f16off (float16, native) | 0.34-0.38 (3.49) | 0.34-0.37 | 3.66-3.67 | 3.45 | 239.9-270.0 | 75.8-84.8 |
| f16on (float16, LMCache) | 0.35 (3.62) | 0.34-0.35 | 0.76-0.77 (3.89) | 0.58-0.59 (0.62) | 245.8-269.6 | 76.2-77.4 |
| f8off (fp8_e4m3, native) | 0.52-0.53 (3.90) | 0.54 | 4.13-4.16 (4.15) | 3.83-3.86 | 76.4-91.4 | 22.5-27.1 |
| f8on (fp8_e4m3, LMCache) | 0.54-0.55 (4.10) | 0.54-0.55 | 0.77 (4.43) | 0.79-0.80 (0.81) | 78.7-90.3 | 21.5-27.5 |

**LMCache effect (the deliverable): rehit TTFT 5.6× faster at fp16 / 4.8× at fp8; churn-phase TTFT 4.7-5.4×; cold +5% store overhead (noise-class); warm and decode exact parity. Zero regressions in all 4 legs. Mechanism counts: 544 seed parked + 256 store(final) both ON legs; 0 Tracebacks; no worker freeze (py-spy watchdog quiet all 4 legs); HEALTH=200 at close of every leg.**

Calibration laws established this round (fed back into bench_prefix.py + this doc's P7.3 pool paragraph): doc-token law (~5.2k real, not 16k), N_CHURN=112 eviction law, pass-2/3 retention-phase law (only p1 is true-cold; true-cold prefill ≈ 3.5 s/doc = 1.5k tok/s class on both connector states).

## P8C — Mid-run corruption + stall ROOT CAUSE (CLOSED 2026-10-10 ~09:30 UTC) — pip-overwritten oneCCL

User report "AI stops in the middle of a run" → 12-leg bisect convicted a **binary-state defect, not any code**: the LMCache bring-up `pip install` (Oct 8 21:07) pulled the oneccl 2021.17.2 wheel, which **replaced `/opt/venv/lib/libccl.so*` symlinks to oneAPI 2021.15 with its own 331 MB files + new `ccl/kernels/kernels.spv`**. `ldd libtorch_xpu.so` resolves CCL from `/opt/venv/lib` first ⇒ every TP=2 all-reduce since Oct 8 ran mismatched to the torch/IPEX build → intermittent 4-10% garbage outputs (`!!!`) + 137-198 s engine-wide self-recovering stalls (collective retries), zero dmesg/kernel signature.

**Evidence chain:** lsv-test corrupt in EVERY config (dtype/prefix/dbg/cache/8-patch-files/caches/adapter/lmcache-package/numpy all exonerated, legs 1a-11); fresh containers from image + the same 8 patched files + lsv-test env = **0/200 clean** (legs 8+8R, runtime-verified via `v138 zeroers` boot markers); discriminative pair: numpy-restored+pip-CCL = 2/100 corrupt vs numpy-restored+image-CCL = **0/100 clean** (legs 11/12, single variable = CCL).

**Fix deployed:** rm pip libccl* + `/opt/venv/lib/ccl/` + oneccl dist-infos → recreate image symlinks → ldd verified → 0/100. numpy restored to image 2.3.5 (pip had downgraded 2.2.6; exonerated but restored). lmcache 0.5.5 + v139 adapter + patched connector re-installed.

**Consequences:** P8 ratios stand (both sides shared broken CCL); absolute perf + the v138 "~1/96 fp8 NaN residual" need clean-CCL re-validation before bake; post-pip ritual law added (verify libccl symlinks + numpy 2.3.5 + nonce 100 = 0 degenerates). **Full writeup: `LMC_PIPEXIT_ONECCL_ROOT_CAUSE_20261010.md` (+ banked /root/build/, legs detail in P8C_CORRUPTION_STALL_FORENSICS_20261010.md).**

## P9a — Clean-CCL perf matrix (CLOSED 2026-10-10 11:04-12:17 UTC) — BAKE GATE MET

Connector-ON boot first re-fixed post-bisect: leg-10a's pristine restore had also clobbered the vllm-side `lmcache_integration` package — `__init__.py` (pristine) imports `multi_process_adapter` → `CudaIPCWrapper` ImportError on lmcache 0.5.5. Fix: adapter v139 (`12cabc38`) → `vllm_v1_adapter.py` + reconstructed patched `__init__.py` (`from . import vllm_v1_adapter` only, banked `/root/build/lmc_init_patched.py`; utils.py/multi_process_adapter.py verified image-stock, untouched); `__pycache__` cleared; import test INIT_OK; boot lmcon5: connector created on TP0+TP1+EngineCore, `LWSDbg` instrumentation live, nonce **0/100**.

Matrix (3 sequential legs, one driver `/root/build/p9_all.sh`, bench_prefix p1 warm-up + p2/p3 records + bench_run7n ×2, py-spy stall watchdog armed, zero fires):

| prefill TTFT med (s) | f16on (LMCache) | f16off (same cont.) | img32 (pristine v1.2.32) | ON vs OFF | ON vs BASE |
|---|---|---|---|---|---|
| cold 16 uniq | 0.30 | 0.30 | 0.73 | 1.0× | 2.4× |
| warm repeat | 0.30 | 0.30 | 0.73 | 1.0× | 2.4× |
| churn 112 | 0.63 | 3.50 | 2.95 | **5.6×** | **4.7×** |
| rehit 16 | 0.47 | 3.29 | 2.69 | **7.0×** | **5.7×** |

Decode: solo200 tps 75.9-76.9 (on) vs 76.5-76.9 (off) vs 75.4-75.9 (base) — parity/+1%; agg4 241-266 vs 235-283 vs 239-271 — parity. Mechanism: seed_parked=544 / store(final)=256 on-leg (exact P8 reproduction), 0/0 off-leg. Quality: 0 NaN detects, 0 Tracebacks, 0 stalls. **Verdict: "LMCache implementation has to be superior" gate MET — no regression on any leg; reuse 5.6-7.0× single-variable; 4.7-5.7× vs pristine baseline.** Cross-lane caveat: img32 cold/warm 0.73 vs lsv-test 0.30 (fresh-container compile warmth vs lane-warm state) while img32 churn/rehit were *faster* than f16off (2.95/2.69 vs 3.50/3.29) — cross-lane absolute deltas carry that lane variance; the same-lane ON/OFF pair is the decisive comparison. Lane restored connector-ON post-matrix (HEALTH 200, 3× connector creation). Artifacts: `/root/build/p9_leg_{f16on,f16off,img32}.out`, `bench_{f16on,f16off}_*` / `bench_img32_*`, `telem_bench_*_serve.log`, `P9_PERF_MATRIX_20261010.out`.

## P9b — Bake (pending)

Strip TEMP debug surface (LWSDbg*, LMCDbgFwd/M0/Bytes/PJ, v138 dbg remnants) + fix `_lmc_flush_pending` double-unpin → docker commit python-only (from P8C §6 verified-clean venv state) → restore-chain update. Include in bake: patched connector (cee1a39) + v139 adapter (12cabc38) + patched `lmcache_integration/__init__.py` + lmcache 0.5.5 wheel contents (site-packages only — the oneccl wheel files must NEVER ride along; P8C law). Re-qualify v138 "~1/96 fp8 NaN residual" + stall-class soak on clean CCL.

---

## Banked artifacts (/root/build/ on .65)

`telem_a.log`, `telem_evict.log`, `base_evict.log`, `telem_b_serve.log` (md5 7c3b6d90 — THE corrupt capture), `telem_c_serve.log` (C4, md5 d4095099), `telem_c{,2,3,4}_leg.log`, `telem_nat_fp8_serve.log` (md5 e804402a) + `telem_nat_fp8_leg.log`, `telem_conn_serve.log` (md5 18a14c65 — freeze capture), `telem_conn2_v139_serve.log` (md5 78d2fcd2 — the A==A2-True proof leg) + `telem_conn2_v139_leg.log`, `fixv138/` (all deployed .py), `fixv139_adapter.py`, `leg_telem_c.sh`, `leg_nat.sh`, `leg_conn.sh`, `leg_conn2.sh`, `probe_evict.py` / `probe_evict0.py`.

## Standing constraints honored

Watchdog paused (`/root/build/lane_watchdog.paused` persists across the 21:2x reboot); host rebooted after the leg_conn engine death (no xe OOPS this time — clean userspace death); orphaned-worker check clean before reboot; plink single-quote discipline; file-based transfers only.
