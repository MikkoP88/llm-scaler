# tq-nibble-unpack v39 — bit-exact, REJECTED ON PERF

Attempt to close the TQ-vs-fp8 decode gap (4bit_nc decode 14.3 tok/s @65k
vs fp8 22.4) by eliminating the MSE nibble-unpack's redundant byte loads.

## What it does

`v39_tq_nibble.py` (era-3 patcher, 6 anchor sites in
`triton_turboquant_decode.py`): for `MSE_BITS == 4`, replace the 16-bit
assembly (`mse_raw0 | mse_raw1 << 8` then per-element shift/mask — two
byte loads per output element) with a single load of `BLOCK_D // 2` bytes
plus `tl.interleave(nb & 0xF, (nb >> 4) & 0xF)`. Bit-exactness is a
theorem for all 256 byte values (even d = low nibble, odd d = high nibble
in both formulations); V4 sites analogous via `val_bases`/`VAL_DATA_BYTES`.

`v39_check.py` — in-container rig: loads the pristine module from the
`.pre39` backup side-by-side with the patched live module
(`SourceFileLoader` — `spec_from_file_location` cannot infer a loader for
the `.pre39` suffix), drives every launcher entry point (decode, MQ,
MQ non-causal, full-dequant K/V) over all TQ presets
(4bit_nc ± norm_correction, k3v4_nc, 3bit_nc, k8v4) on identical
synthetic data, `torch.equal` compare, plus a 16k-context kernel timing
preview. `v39_apply.sh` — double-cycle live A/B boot (fresh container →
patch → `docker restart`; the image itself is never modified).

`v39b_qrot.py` — designed but NEVER APPLIED: rotate q+new-K (O(q_len))
in `_continuation_prefill` instead of inverse-rotating the whole cached
prefix (O(cached_len)). Orthogonally equivalent, not bit-exact.
Deprioritized: 4bit 65k prefill already beats fp8 (1685 vs 1167 tok/s).

## Verdict (2026-09-03, llm-scaler-prod:v1, turboquant_4bit_nc)

Correctness: PASSED, completely.
- rig: 25/25 bit-exact across 5 presets × 5 entry points
- live probe `harm_probe10.py`: `0ce080630035` ×10 (prod 4bit ref)
- full dt_loop battery: 8/8 hashes identical to the unpatched matrix lane

Perf: REGRESSION, growing with context depth:

| metric            | 4bit baseline | v39a    | delta |
|-------------------|---------------|---------|-------|
| prefill 65k tok/s | 1685          | 1487    | −12%  |
| decode 2k tok/s   | 32.68         | 29.57   | −9.5% |
| decode 16k tok/s  | 25.27         | 16.65   | −34%  |
| decode 65k tok/s  | 14.32         | 6.69    | −53%  |

Root cause of the failure: `tl.interleave` lowers poorly on XPU/triton
here — it costs more than the L1-cached second byte load it replaces.
The original "redundant load" cost model was wrong: at BLOCK_D=128 the
MSE bytes are contiguous and L1-resident; the +1 load was effectively
free. Lesson recorded for future kernel work: the decode deficit is NOT
in the unpack loads (see the study in
`diagnostics/kv-dtype-loop-study-v39/` — the gap is triton-1-warp-tiled
vs fp8's native ESIMD flash decode, architectural).

Committed as documented negative result; do NOT bake. Prod untouched
(restored + probe-certified `0ce080630035` ×10 after the experiment).
