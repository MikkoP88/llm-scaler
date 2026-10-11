# flashnext-fp8-tier-v1 — Strata memory-handling deep analysis + superiority design

2026-10-11. Sources: full read of `C:\Projects\Strata` — `docs/DETAILS.md` (1906
lines), `src/core/expert_cache.cpp` (778 lines), `src/core/expert_source.cpp`
(structure map), `docs/MULTI_GPU.md`, `docs/KV_PREFETCH.md`, `docs/INTEL.md`
(B70 SYCL sections incl. lines 739–825: **Strata's own run of Flash-Next on the
same Arc Pro B70 we own**). Cross-checked against our P0 measurements on
fn8-g1 (leg-9 serve, 2026-10-11).

---

## Part 1 — verification of the quoted memory table

| Claim | Verdict | Arithmetic / measurement |
|---|---|---|
| FP8 routed experts ~103–112 GB, exceeds 64 GB VRAM by itself | **VERIFIED** | 24,576 experts × 3 mats × 640 × 2560 B = 4,915,200 B/expert → **112.5 GiB weights-only** (120.8 decimal GB); + fp32 block-128 scales ≈ 116 GiB. P0 offloader measured **104.4 GiB pageable** (weights split across offload boundary accounting). Any flavor of the number is 1.6–1.9× the whole 64 GiB card set. |
| Dense parts (GDN/attn/router/shared/embed) BF16 ~13 GB VRAM (P2) | **VERIFIED (consistent)** | leg-9 resident weights = 17.45 GiB on one card = dense + ~4.5 GiB experts the offloader kept resident. Dense alone ≈ 12–13 GiB: embed 248320×2560×2 = 1.27 GB + untied lm_head 1.27 GB + 48 layers × (GDN in/out proj + QKV/O + shared expert 9.8 MB + norms + router 2560×512). |
| PLE n-gram table 51 GB host RAM by design | **VERIFIED EXACTLY** | 320,001,536 rows × 160 B fp8 = 47.687 GiB = **51.22 decimal GB**. Per-token read is only (ngram_size−1)×heads_per_ngram = 16 rows × 160 B = 2.56 KB → H2D rows on demand; the table itself never crosses the bus per step. Host RAM by design is correct. |
| MTP draft layer ~6 GB deferred to P4 | **VERIFIED (consistent)** | one decoder layer + draft head at this width; matches Strata's own on-disk figure for the MTP layer of this model family. |
| 2× B70 = 64 GB VRAM total | **VERIFIED** | 2 × 32 GB (xpu-smi 31.97 GiB usable each). |
| P2 targets ~17–18 GB/rank hot experts (~33%) | **VERIFIED** | 2 × 17.5 GiB = 35 GiB / 112.5 GiB = **31.1%**; at 18/rank → 32%. "~33%" within rounding. |
| P0 = offload run by design: TP1, 104 GB pageable, ~20 GB resident | **VERIFIED EXACTLY** | measured: 104.4 GiB pageable-offloaded, ~20 GiB resident on card 0, TP1. |

All seven figures check out. One refinement worth keeping in mind: the experts
figure is **112.5 GiB in fp8 weights alone** — the "~103 GB" low end is the
measured pageable number after the offloader's boundary accounting, not a
smaller true size.

---

## Part 2 — how Strata turns memory handling into tok/s (their entire speed story)

Strata is a tiered-residency engine: the memory layout **is** the performance
model. Five mechanisms, in order of impact:

### 2.1 VRAM expert cache sized by what the CARD has free, never the request
`expert_cache.cpp::open()` checks `device_free_bytes()` (UMA-aware) **before**
allocating the slot arena, then zeroes it (deterministic-wrong instead of
stale-wrong). Their measured law: **0 MiB free = 7× decode slowdown** (WDDM
silently spills to sysmem; 453 vs 102 tok/s slots difference). This is the same
medicine as our P82/P83 launch-headroom guard — keep a reserve, size the cache
from the free figure after everything else loads.

### 2.2 Per-layer slot quotas + profile-ranked fill (cold-start hit rate)
- **R4.2g per-layer slot ranges**: layer l owns `[l·q, (l+1)·q)` slots. A
  shared free counter previously confined hit rate to 2.97% — a routing-aware
  cache must quota per layer.
- Initial fill from an **STRP v1 profile** (ranked (layer,expert) u16 pairs,
  heat-ranked from prior runs); `rank_learned_profile` = resident-first,
  heat-desc, prior-rank. Cold start is warm start.
- `admit()` has **no eviction of its own** — the adaptive tier above does swaps.

### 2.3 Adaptive swaps that follow the conversation
`--adapt-every N` tokens: a cold expert entering the hot set swaps with the
coldest resident one; the **evicted VRAM blob is copied back into the RAM slot
of the replacement** (exchange, `STRATA_EXCHANGE_ROTATE=1`), so host and device
never disagree. `--adapt-async 1` does it on a helper thread; `STRATA_ADAPT_LAG=2`
bounds staleness. Measured: this is what holds decode flat as topics shift.

### 2.4 The host tier is a COMPUTE tier, not just storage
- **Pinned host mirror** for everything the card lacks; CPU computes uncached
  experts **in place** (AVX-512/AVX2) concurrently with GPU work.
- `--pcie-frac` auto since 0.1.42: `L/(L+1.85R)` clamped 0.10–0.60 — the
  CPU/GPU split tuned from measured lane latencies (+30% on RTX 3060).
- Prefill chunks <1024 tokens get a CPU share (−20..−35% prompt time).
- Read-ahead `madvise(WILLNEED)` on the profile: start 920 s → 70 s.

### 2.5 Skip-and-stream tricks that cut the bus bill
- **Route tail skip** (default on CUDA since 0.1.42): a *missed* expert ranked
  ≥8 of the top-10 is skipped/zeroed, not fetched. +11–24% decode, KL
  0.003–0.006 — quality-neutral.
- **Layer-ahead streaming**: next-layer experts stream over PCIe during the
  current layer's attention (`RouterLookahead` predicts the next router's picks;
  prefill uses a byte-sized streamed ring, `STRATA_RING_BYTES`).
- **KV residency window**: `--kv-resident` keeps the attended window in VRAM,
  the whole KV in pinned host RAM (~13.7 KB/token); k8v4 cuts KV bytes 23%.
- Prompts read in **fewer, bigger chunks** when the cache is cold: a 4K prompt
  touches ~all 512 experts/layer → 13.5k misses × 1.97 MB = 26 GB = 2 s of bus
  per chunk at their hit rate.

### 2.6 Dual GPU = LAYER SPLIT ONLY — their structural ceiling
`MULTI_GPU.md`: each card owns contiguous layers + a full dense copy; decode is
"on par with the faster card alone" (5080+3090: 84–110 tok/s), prompts +18–20%
from pipelining (`--pipeline-windows 2` guessed-window pipelining, +16% mean).
**No expert parallelism exists in Strata.** The second card adds prompt
throughput, not decode. Helper-GPU expert caches (`--expert-cache-remote`)
measured SLOWER. `--batch-mtp` on split −21.9%.

### 2.7 Their run of THIS model on OUR card (the number to beat)
INTEL.md 739+: Flash-Next **IQ3_S** (24,576 experts × 1.97 MB = 46 GB), single
Arc Pro B70, SYCL port 0.1.40-sycl, xe kernel 7.0, **PCIe measured 13.3 GB/s
H2D**, 11.5k experts resident (22 GiB) + pinned host mirror, `--spec 4 --mtp`,
INT8 KV, 32K ctx:

| workload | tok/s |
|---|---|
| prefill 4,095 tok, auto chunks | 730 |
| prefill 4,095 tok, `--prefill 4096` | **980–1,002** |
| prefill 8,169 tok, 8192 own buffers | 1,117 |
| decode, prose (66% draft accept, 1.98 tok/round) | **30.5** |
| decode, code (90% accept, 3.65 tok/round) | **40.8** |

Their decode arithmetic closes: 480 expert-touches/token, per-miss 1.97 MB over
13.3 GB/s ⇒ at near-full miss ≈ 71 ms/round ÷ 1.98 tok ≈ 36 ms/tok ≈ 28 tok/s —
**their decode IS the PCIe bound at their hit rate.** Also documented: xe
2 MiB-page aliasing in big arenas (fixed by tagged re-read `malloc_device_guarded`
— every ≥32 MiB allocation verified), `STRATA_VERIFY_NO_HOST=1` host/GPU
handshake visibility requirement (else NaN logits), bounded spins `kSpinMax`
(their xe wedge law = our P82 family), FP64-free sampler.

---

## Part 3 — the superiority design (why ours wins on tok/s)

### 3.1 The structural win: whole-expert TP2 vs layer split
Strata's dual-GPU decode ≈ one card. Ours: **both B70s compute every layer**,
each owning half the experts (whole-expert partitioning; the **block-128 law**
makes tensor-slicing impossible — 640 moe_intermediate = 5 blocks, 320 = 2.5 —
so whole-expert is not just best, it is the only TP2). Consequences:

- **2× aggregate expert VRAM**: 2 × ~17.5 GiB hot caches vs their single 22 GiB
  (and ours is fp8: 3.56k experts/rank × 4.9 MB, theirs IQ3_S 11.5k × 1.97 MB —
  we hold fewer experts per GiB, so ranking quality matters more, see 3.2).
- **2× aggregate PCIe**: each rank streams only its own misses over its own
  13.3 GB/s lane. Strata cannot do this — a layer-split card fetches for its
  layers alone over one bus.
- Per-layer cost is a small partial-sum exchange of activations (2560-wide, the
  non-MoE parts) over the card-to-card link — vs Strata's whole-activation
  serial handoff between layer groups.
- Decode FLOPs (the dominant MoE GEMV work) genuinely halve per card.

### 3.2 Ceiling math (fp8, dual B70, 17.5 GiB/rank hot cache)
Per token per rank: top-10 of 512 → ~5 experts on this rank's half. With the
hot set covering the usual 70–80% of activations (top-30% experts under Zipf-ish
routing), misses ≈ 1–1.5/layer/rank × 4.9 MB = **4.9–7.4 MB/layer** over
13.3 GB/s ⇒ 0.37–0.55 ms/layer ⇒ **18–26 ms/token ⇒ 38–55 tok/s** before
overlap. Layer-ahead streaming (their RouterLookahead, our P10 helper-thread
experience) hides 40–60% of the remaining bus time → **60–90 tok/s**; MTP×4 at
their observed 2.0–3.6 accepted tok/round → **effective 150–300 tok/s** —
against their 30.5–40.8 on the same silicon, at strictly higher precision
(fp8 e4m3 vs IQ3_S 3-bit). Even the no-overlap floor (38–55) beats their
number, because their single bus is the binding constraint and ours is doubled
and half-loaded.

Prefill honesty: fp8 costs 2.5× IQ3_S bytes per expert, so a fully-cold 4K
prompt that touches all 512 experts/layer streams ~60 GB/rank — the one regime
where we must lean on their tricks (bigger chunks, layer-ahead ring overlap,
CPU share, and pre-warming the profiled hot set so live prompts mostly hit).
Target ≥ their 730 tok/s cold, and hot-set prompts well above.

### 3.3 Adopted from Strata (proven levers, mapped to our stack)
| Strata lever | Ours (P2/P3) |
|---|---|
| STRP profile-ranked initial fill + learned-heat persistence | profile pass on the lane's real traffic (we already run profile-style ranking for LMCache residency) |
| per-layer slot quotas (R4.2g; shared counter = 2.97% hit) | per-(layer,rank) quota in the expert cache table |
| adaptive swaps w/ exchange-back, async helper, lag bound | P3 demand pager; swap copies evicted blob back to host slot |
| pcie-frac CPU/GPU split tuning | tune host-compute share for prefill chunks <1024 tok |
| route tail skip (≥8/10 missed → skip) | adopt verbatim; KL-neutral, +11–24% decode |
| RouterLookahead layer-ahead streaming | prefetch next-layer miss set during current-layer GDN/attn (P10 queue-kick pattern) |
| pinned host mirror under pin limits | per-rank mirror of its own half, pinned window ≤ ~16 GiB/rank (our SYCL USM pin law: >16–32 GiB/process segfaults; torch pin_memory dies 16–20 GiB/process) + pageable remainder with WILLNEED read-ahead |
| KV resident window + pinned host KV | adopt for >32K contexts (their B70 run: whole-KV pinned host, 56 tok/s at 256K, decode holds 42) |
| 0-MiB-free = 7× law | P83 launch-headroom guard already standing |
| xe arena aliasing verify | tag-and-readback on every ≥32 MiB arena (their exact failure mode is on our exact driver) |
| `STRATA_VERIFY_NO_HOST` handshake | our all-reduce/oneCCL discipline already covers cross-card sync; keep slot verify (`verify_slot` analog) in P3 gates |

### 3.4 Deliberately NOT adopted (measured-negative in Strata)
- Filesystem-backed slots: 0.91–0.98× of RAM arena.
- Helper-GPU remote expert caches: made decode slower.
- Layer-split dual GPU: replaced by whole-expert TP2 (3.1).
- `--batch-mtp` on split: −21.9% (moot for us).
- Short first prefill chunk (256) when cache is cold: costs a whole extra
  expert-stream pass — their own fix (first chunk 0 unless fully cached) is the
  law for us too.

### 3.5 Host RAM budget (376 GiB total, lane shares the host)
PLE table 47.7 GiB (day 1, by design) + pageable expert mirror 112.5 GiB +
pinned windows 2×16 GiB + KV-pinned (only when long-ctx mode) ≈ 192 GiB core,
leaving headroom against the lane's ~100 GiB and the 4 GiB-reserve law. Fits;
re-check at P2 bring-up with the lane co-resident.

---

## Part 4 — laws this design must respect (from both codebases)
1. Pin law: pinned per-process ≤ ~16 GiB (SYCL USM), never torch pin_memory
   >16 GiB. Mirrors = pinned window + pageable remainder + madvise.
2. Launch-headroom guard (P83) = Strata's free-check-before-arena law.
3. Tag-readback every big arena on xe (their aliased-2 MiB-pages bug, our
   driver family).
4. Bounded spins everywhere (their kSpinMax = our P82/xe-vm-bind wedge law).
5. Per-layer quotas in any routing-aware cache (their 2.97% confinement bug).
6. Whole-expert TP2 only — block-128 law forbids slicing; per-layer partial-sum
   exchange stays tiny.
7. Watch: `STRATA_VERIFY_NO_HOST` analog — with experts outside VRAM, the
   host/GPU handshake must be explicit per layer or logits NaN (their exact
   B70 failure; our oneCCL/all-reduce discipline is the counterpart).
