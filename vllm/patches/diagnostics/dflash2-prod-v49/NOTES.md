# DFlash2 prod arc v43–v49c — KV-dtype matrix, real-life perf fixes, long-context validation, prod images `llm-scaler-exp:v1` → `v1.2.1`

Server `root@10.20.3.65`, 2× Intel B70 (32.7 GiB) TP=2, target
`qwen3.8-27b-fp8`, DFlash2 drafter `/models/qwen3.8-27b-dflash2` (5 layers,
sliding-window w=2048, 32q/8kv hd128, conv kernel 2, selector rank 256
top_k 16, bf16 internals over fp16 target). This dir closes the user task:
*"Do really deep analyze and search DFlash2. DFlash2 has realy bad real live
performance and crash ussually, validate all --kv-cache-dtype types are
supported and tested like example fp8_e4m3. Fix DFlash2 poor real live
performance, use on testing DFlash2 k=7 vs MTP k=4 and prompt 'Write a html
car game.', DFlash2 has to be superior, include concurrent testings fix
issues. Final task is build new prod image llm-scaler-exp:v* ... include to
testing/validation long context testing."*

## Outcome (honest)

DFlash2 k=7 does **not** beat MTP k=4 on raw short-context tok/s after all
fixes (single 36.7 vs 46.1; conc 71.2 vs 79.4 combined). It **is** the only
lane that survives concurrent long context: MTP k=4 stalls at 2×28k and
**dies** (wedge → `TimeoutError: RPC call to sample_tokens`) at 2×56k,
while DFlash2 completes every concurrent long-context workload cleanly —
its draft attention is O(2048-window), not O(context). DFlash2 was fixed
from 27.9 → 36.7 tok/s single (+31%) / 52.3 → 71.2 combined concurrent
(+36%) / 68.2 → 77.4 (+13%) via two convicted root causes. Prod image and
lane: `llm-scaler-exp:v1.2.1`, draft KV pinned `turboquant_k8v4`.

## Prod image lineage + naming scheme

Image tags follow the user's scheme: `v1 -> v2` = full new features,
`v1 -> v1.1` = big fixes/improvements, `v1.1 -> v1.1.1` = small fixes.

| image | id | content |
|---|---|---|
| `llm-scaler-exp:v1` | `f02fa561e900` | first DFlash2-capable prod image (v40–v42 era) |
| `llm-scaler-exp:v1.1` | `468a61a7e79f` | + v43 descale-assert fix, KV-matrix-era overlays |
| `llm-scaler-exp:v1.2` | `821b576f0642` | + v49 TQ rebase, **v49b perf fix**, v48 overlays (5 files, md5-verified, full battery PASS) |
| `llm-scaler-exp:v1.2.1` | `4a10b5f6c81b` | + **v49c** draft-KV dtype policy (default = match `--kv-cache-dtype`, optional override) |

Prod boot (certified cell) pins the override:

```
cd /root/build && DFLASH2_IMG=llm-scaler-exp:v1.2.1 DFLASH2_TP1=0 \
  DFLASH2_MEMUTIL=0.92 MAXLEN=262144 \
  VLLM_DFLASH_DRAFT_KV_DTYPE=turboquant_k8v4 \
  nohup bash dt_dflash2_serve2.sh > <boot.out> 2>&1 &
```

## The comm-stomper defect (KNOWN_ISSUES #11, fixed v27, carried here)

Two host-side sites enqueued oneCCL collectives onto the SAME
ProcessGroupXCCL communicator during steady decode: the target's
verify/decode step and the eager drafter (~6–11 colls/step). Under
`--async-scheduling` the drafter of step N+1 overlaps the target of step N,
per-rank issue order inverts, oneCCL matching inverts, both ranks spin in
collective kernels that never retire (>=32k serve wedge; py-spy: one rank
in align-mode D2H, peer a step ahead in the drafter; engines 100% at ~22%
EU). Fix = dedicated drafter communicator for the duration of
`propose`/`dummy_run` (`drafter_comm_v48.py`, `VLLM_XPU_DRAFTER_PG=0`
restores stock). This is the ancestor of the MTP long-context crash below:
the wedge family survives in the MTP full-context draft-attention regime.

## KV-cache dtype matrix (validated on DFlash2 k=7 lane)

| `--kv-cache-dtype` | verdict |
|---|---|
| `turboquant_4bit_nc` | ✓ prod cell (draft pool k8v4) |
| `turboquant_k8v4` | ✓ |
| `turboquant_k3v4_nc` | ✓ |
| `turboquant_3bit_nc` | ✓ only @ MAXLEN ≤ 258048 (VRAM) |
| `fp8_e4m3` | ✓ (draft inherits fp8_e4m3) |
| `fp8_e5m2` | ✗ upstream kernel gap |
| `auto` | ✗ DFlash2 defect (uncompressed draft pool: 13.09 GiB needed vs 6.67 available @262144; and `auto`-target boot defect) |

## Perf root causes (convicted, measured, fixed)

1. **v49 — batched-rebase base** (correctness/cleanliness, no perf delta).
2. **v49b — per-layer synchronizing D2H in the draft cam.**
   `dflash.py:set_inputs_first_pass` built the draft
   `CommonAttentionMetadata` without `seq_lens_cpu_upper_bound`;
   `TurboQuantMetadataBuilder.build` copies that field into
   `TurboQuantMetadata.seq_lens_cpu`; when None, every TQ
   `_prefill_attention` call ran `attn_metadata.seq_lens.tolist()` — a
   synchronizing D2H, 5 draft layers per propose. Under async scheduling
   the first drain per propose blocks the host behind the overlapped target
   queue: measured propose h−d = 36.2 ms == tforward d exactly.
   Fix: source `cad.seq_lens_cpu_upper_bound[:num_reqs] + num_query_per_req`
   (the runner captures optimistic seq_lens into the upper bound BEFORE
   async mode nulls `_seq_lens_cpu` — sourcing `_seq_lens_cpu` was the
   failed first attempt). Upper bound is safe: the only host consumers are
   the `q_len == seq_len` first-chunk check (upper bound steers validly to
   the continuation path) and the `cached_len` debug/gate (gate unreachable
   at q_len=8 ≤ 128).
   Effect: TQTIME fwd 7.5 → 0.43–0.51 ms/call (MTP parity 0.45); step
   81.7 → 66.4–70.1 ms; propose h 63.7 → 28–35, d 27.5 → 11.2; single
   31.7 → 36.7 tok/s; concurrent 52.3 → 71.2; conc2 68.2 → 77.4; numerics
   clean (acceptance histogram 444/254/148/80/44/20/7, yield 2.44).
   Diagnosis method: py-spy on the live lane (7/8 dumps leaf at
   turboquant_attn `:1024` = the tolist) + the h−d == tforward-d identity.

**Residual DFlash2-vs-MTP deficit (structural, measured):** tforward
+2.6 ms (8-row verify vs 5-row) + dforward host +18.5 ms (5-layer eager
drafter: python `_grouped_conv` taps=2 loop, native rotary, ~11 drafter
collectives). Yield 2.44 vs 2.55–2.59.

3. **v49c — draft-KV dtype policy flipped to match-target.**
   Old v21c auto-policy rewrote the draft pool dtype only for
   `turboquant_4bit_nc` targets (k8v4 above 131072, `auto` below); all
   other dtypes inherited. New default: **draft KV = `--kv-cache-dtype`
   for every dtype**; `VLLM_DFLASH_DRAFT_KV_DTYPE` remains the optional
   override (any value, e.g. `auto`, `turboquant_k8v4`).
   Consequences: tq4nc@262144 default-draft is now tq4nc (equal measured
   acceptance to k8v4 — re-validated live: b65 deltas 87/112 @2k and
   116/224 @16k vs k8v4's 87/112 and 120/217; single 38.2 tok/s; pool
   373,824 tokens) — the certified prod boot still pins
   `VLLM_DFLASH_DRAFT_KV_DTYPE=turboquant_k8v4`; tq4nc@<=131072 loses the
   old `auto` (bf16 flash) default — override to `auto` to restore.

## Why DFlash2 draft KV cannot "ride target KV" like MTP

Pool assignment is by KV-spec matching. The MTP head is one more
target-family layer (same attention class, full causal, same kv-head
geometry, same block size, same dtype lane) → identical spec → allocated
into the target's unified KV pool ("rides target KV" = shares the pool
format/allocation, not the values). DFlash2 is a separate 5-layer model:
sliding-window w=2048 on all layers, non-causal block attention
(`causal=False`, block = 1+k), different cell width/geometry, different
dtype lane → allocator must create a second pool, and the drafter must
re-encode the context itself (`precompute_and_store_context_kv` at
prefill). Consequence measured below: MTP's draft attends full context at
every spec step (wedge regime at 28k–56k concurrent); DFlash2's draft is
O(2048) forever (all long-context concurrent runs clean).

## Corrected benchmarks (same-day A/B, certified)

Prompt `Write a html car game.` (real-life path: server default sampling,
reasoning counted; `dt_cargame.py`).

| lane | single | concurrent (comb) | concurrent2 (comb) | bench |
|---|---|---|---|---|
| DFlash2 k=7 pre-fix (v1.1 era) | 27.9 | 52.3 | 68.2 | – |
| DFlash2 k=7 **v49b2** | 36.7/36.8 | **71.2** | **77.4** | 34.1 |
| DFlash2 k=7 **v1.2 baked** | 41.2/36.8 | 69.9 | 83.7 | 37.2 |
| MTP k=4 (control) | 46.1/42.0 | 79.4 | 90.6 | 44.3 |

v1.2 full battery: mixed crash-repro CLEAN (35.0+39.4); b65 acceptance
87/112 @2k, 120/217 @16k; bench3 decode 482.5/280.6/149.1 tok/s @2k/16k/65k,
conc8 aggregate 198.8 tok/s, blended acceptance 0.596; coh_probe
`Paris -0.452` ×3 distinct=1 (bit-stable, matches certified baseline).

## Long-context validation (dt_longctx.py; 8k/16k/32k → 13,970/27,954/55,830 prompt toks)

Singles — MTP wins (decays slower): MTP 38.5/24.6/17.8 vs DFlash2
23.5/15.9/9.8 tok/s.

Concurrent (2 clients, distinct prompts):
- DFlash2: **all clean** — 14.5+14.6 / 10.2+9.5 / 5.7+5.6.
- MTP k=4: 14k OK (24.2+24.0); **28k STALL** (15 toks in 313 s, no usage
  chunk); **56k DEATH**: gen throughput 0.0 with 2 reqs running, 4×60 s
  shm_broadcast starvation, then `TimeoutError: RPC call to sample_tokens
  timed out` → EngineCore fatal → container exit. The 56k reqs had resumed
  from the ~28k shared filler prefix (53.6% prefix-cache hit) and wedged at
  ~28k depth mid-spec-decode — the #11 wedge family under MTP
  full-context draft attention; the 4-minute RPC timeout converts the
  wedge into engine suicide.

## v48 negative result (drafter TP1 replication)

Full DFlash2 drafter weights per rank (world=1 for the drafter, target
still TP2) convicted **CORRUPT**: acceptance ~0.2 in all 3 tested combos,
and no perf benefit (collectives saved < eager overhead added). Dropped;
both drafters run TP2-sharded on both GPUs. Corruption never root-caused
(drafter_comm_v48.py docstring carries the attempt log).

## Files

- `turboquant_attn_v49.py` — TQ attention backend, v49 batched rebase
  (overlays site-packages `vllm/v1/attention/backends/turboquant_attn.py`).
- `dflash_v49.py` — DFlashProposer with the v49b seq_lens upper-bound fix
  (baked in v1.2; md5 5e5b491c3e45cafb41ca94092bc0b0a3).
- `dflash_v49c.py` — v49 + v49c dtype policy (baked in v1.2.1; md5
  162366b29f3ee3d96f24b6956b14fa64).
- `drafter_comm_v48.py` — v27 dedicated drafter communicator (comm-stomper
  fix) + v48 TP1 attempt notes.
- `dflash2_v48.py`, `scheduler_v48.py` — era overlays baked since v1.1/v1.2.
- `dt_dflash2_serve2.sh` — boot matrix (SPEC=dflash7/dflash4/mtp*/none,
  KVDTYPE, EMIT_K guard, dbg knobs).
- `dt_cargame.py` — real-life bench client (single/concurrent/concurrent2/
  bench/mixed).
- `dt_longctx.py` — long-context A/B client (single/concurrent 8k/16k/32k).
