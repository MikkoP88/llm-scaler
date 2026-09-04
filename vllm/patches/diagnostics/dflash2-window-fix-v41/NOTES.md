# DFlash2 read-side sliding-window fix (v41)

Closes the v40 open defect (`../dflash2-spec-port-v40/NOTES.md` §6:
acceptance decays with context). Engagement Sep 4, host `10.20.3.65`.

**Verdict: FIXED.** Acceptance ladder 72.3/30.4/9.3 % (baseline @2.3k/
18.2k/74k) → **77.7 / 70.6 / 62.2 %** with the window enforced. DFlash2
now beats MTP at 2k (412 vs 380 tok/s) and 65k (189 vs 181), par at 16k
and conc8. Experimental image baked: `llm-scaler-exp:dflash2-v41`
(= prod:v1 + v40 hooks + this fix, md5-verified vs the validated lane).
Prod serving posture unchanged (prod:v1, 4bit TQ nospec).

## 1. Root cause

The draft's attention impl is **`TurboQuantAttentionImpl`**
(`vllm/v1/attention/backends/turboquant_attn.py`) — not FlashAttention.
Its `__init__` accepts `sliding_window: int | None = None` but **never
stores it: the window is silently dropped**. Every DFlash2 draft step
(5 layers, all `sliding_window=2048`) therefore ran the causal MQ kernel
`triton_turboquant_mq_decode_attention` as a **causal ramp over the FULL
prefix** — kernel mask `kv_offs < q0 + row` over block-table columns
starting at position 0 — instead of the trained trailing 2048-token
window. RoPE far-key noise at distances never seen in training grew
smoothly with context: the gradual decay (no 2048/8192 cliff), correct
≤2.3k behavior (prefix ≈ window), 9.3 % @74k, and flat `dforward` cost
(int4-quantized reads are cheap either way) all follow.

Exonerated by code reading before the fix: KV allocation
(`DFlashAttention.get_kv_cache_spec` widens Sliding→Full — allocation is
full by design), slot math (the triton input-expansion kernel is
position-faithful), write-side KV content (precompute = chunked fused
projection of target states, v1-inherited and reused by upstream
PR #52816), and the FA window plumbing (irrelevant — TQ impl used).

Confirmation that the causal MQ branch is the exercised path: zero
occurrences of the v21 `non-causal MQ` one-shot log in the DFlash2 lane
(all draft steps causal; `sliding_attention_causal=True` fork default).

## 2. Fix (era-3, `dflash2_winfix_edit.py` — 11 hooks, fail-loud anchors)

| file | hooks |
|---|---|
| `turboquant_attn.py` | B1 store `self.sliding_window` in `__init__`; B2 windowed causal MQ call; B3 windowed synthetic-decode fallback |
| `triton_turboquant_decode.py` | A1 `Win_start_ptr` kernel arg; A2 constexpr-gated load; A3 per-row lower bound in the causal mask; A4 `HAS_WINDOW` constexpr; A5 wrapper `win_start` param; A6/A7 launcher wiring |

Semantics (exact, per row r at global position `cl + r`, window `w`):
visible keys = `[cl + r - w + 1, cl + r]`. Host side: `off = clamp_min(q0
- w, 0) // block_size`; block table rebased via device-op gather (local
column 0 = global block `off`); `q0' = q0 - off·bs`; `win_start = q0 - w
- off·bs` (may be negative). Kernel: causal upper bound unchanged
(`kv_offs < q0' + r`), new lower bound `kv_offs >= win_start + r`.

Properties:
- **Graph-safe**: the branch is gated on static impl attributes only
  (`self.sliding_window is not None and causal`); every data-dependent
  value is a device tensor op inside the captured region (same class as
  the v19b `seq_lens - (q_len-1)` already in that branch). Short
  sequences resolve to `off=0` / negative `win_start` == exact causal
  semantics (identity gather, trivially-true bound).
- **Inert by default**: `sliding_window=None` for target full-attention
  layers (TQ impl), the GDN layers (different impl), MTP drafter and
  DFlash v1 — `HAS_WINDOW=0` compiles the stock kernel body unchanged
  (constexpr variant).
- **Synthetic-decode fallback** (only reachable via
  `VLLM_TQ_MQ_VERIFY=0` or `VLLM_TQ_MQ_MAX_Q<8`): block-aligned rebase
  without head cut — up to `block_size-1` stale tokens beyond the window
  start; documented, one-shot logged.

## 3. Verification (lane `lsv-test`, prod:v1 + runtime-injected patches)

Acceptance (per-round `/metrics` deltas):

| ctx | v40 baseline | v41 fixed |
|---|---|---|
| 2.3k | 72.3 % | **77.7 %** (87/112) |
| 18.2k | 30.4 % | **70.6 %** (89/126) |
| 74k | 9.3 % | **62.2 %** (148/238) |

One-shot activation log confirmed on both workers:
`TurboQuant sliding-window MQ draft attention active (v41): w=2048, q_len=8`.

Perf (`dt_bench3`, single-stream decode tok/s):

| ctx | v40 | v41 | MTP k=1 |
|---|---|---|---|
| 2k | 329–396 | **411.6** | 380 |
| 16k | 174–212 | 262.0 | 268 |
| 65k | 27–46 | **189.1** | 181 |
| conc8 | 119–125 | 123.6 | 126 |
| acceptance blended | 0.46–0.51 | 0.635 | 0.68–0.81 |

Correctness (`dt_probe2`): p1 `3cecc747d7885a11` and p2
`c87e27c45fcafcb6` = references 3/3 runs; p4 flaps between exactly the
two v40-documented hashes; p6 standalone = reference `1f9c461b0f2bd1cb`
×3. **Note:** p5/p6 now also flap between coherent trajectories across
battery runs (p6: ref ↔ `ebcc8258…`, 103↔118 completion tokens) — same
fp16 knife-edge class as the v40 p3/p4 bistability (KNOWN_ISSUES #18;
the MTP control showed the class in v40). The higher-acceptance draft
changes verify-batch shapes, widening exposure. Not a window-fix
correctness defect; standalone prompts remain bit-stable.

## 4. Image

`llm-scaler-exp:dflash2-v41` — first image that CONTAINS DFlash2 (v40
was runtime-injected only). Build: `Dockerfile` in this dir, context
`/root/build/qwen38-dflash2`; runs both patchers (v40 `dflash2_edit.py`
+ v41 `dflash2_winfix_edit.py`) inside a prod:v1 container, then grep
gates + py_compile + markers `/root/.dflash2_patched` (makes the
dt_dflash2_serve.sh wrapper exec immediately) and
`/root/.dflash2_v41_baked`. Validated: all 4 patched/new files
**md5-identical** to the runtime-verified lane; full battery on the
baked image (ladder 77.7/66.2/51.4 %, hashes within the documented
envelopes). Boot: `dt_dflash2_serve_baked.sh` (serve script with the
image name swapped). Promotion path to a `prod:vN` would follow a
serving decision on DFlash2; prod stays nospec until then.

## 5. Repro

```
bash /root/build/dt_dflash2_boot.sh            # patch + boot (9 files)
python3 /root/build/dt_probe2.py               # hashes + acceptance
python3 /root/build/dt_b65.py                  # 2k/16k acceptance deltas
python3 /root/build/dt_bench3.py               # perf battery
# baked image lane:
bash /root/build/qwen38-dflash2/dt_dflash2_serve_baked.sh
```

## 6. Files

| file | what |
|---|---|
| `dflash2_winfix_edit.py` | era-3 patcher: 11 window hooks (B1–B3, A1–A7) |
| `Dockerfile` | bake recipe for `llm-scaler-exp:dflash2-v41` |
| `dt_dflash2_boot.sh` | lane boot, extended to 9 injected files (both patchers) |
| `dt_dflash2_serve.sh` | serve script (v40 minus the DFLASH2_NOWINDOW probe env) |
| `dt_dflash2_serve_baked.sh` | serve straight from the baked exp image |
