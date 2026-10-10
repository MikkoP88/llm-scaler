# Mission: CONDUCTOR — self-driving test program for v78 R1b

Read `../CONTEXT.md` first. You own the load-generation program. Work in
this directory only. Produce `REPORT.md` + `requests.jsonl` + per-leg logs.

## Pre-flight (retry up to 15 min total)
1. Poll `curl -s -o /dev/null -w "%{http_code}" http://10.20.3.65:8000/health`
   until 200. If it never reaches 200, write REPORT.md with verdict
   BOOT_FAILED + the last 100 serve log lines and exit.
2. `GET /v1/models` → record model id.
3. Confirm R1b markers in serve log (plink, read-only):
   grep for `v78 R1b: MTP head loading REPLICATED` and
   `TP1 spoof active over` — record the binding count.

## Program (sequential legs; between legs wait for all in-flight requests
to drain and 60s idle)

### LEG 1 — smoke + correctness sanity (~10 min)
- 20 sequential short chat requests (varied 1-3 turn prompts, temp 0.3).
- Record per-request TTFT/total to requests.jsonl.
- Sanity: outputs must be non-empty, coherent, not degenerate repetition.
- Any 5xx or empty body → record, continue to 20 attempts max.

### LEG 2 — standard storm (~20 min)
- 3 waves × 24 concurrent standard prompts (≈500-800 output tokens).
- Record all timings. Watch for stalls per CONTEXT definitions.

### LEG 3 — deep-context MTP×3 (the hazard shape, ~40 min)
- 6 concurrent conversations, each 6+ turns with large cumulative context
  (aim ≥ 30k computed tokens per conversation before finishing; grow by
  requesting ~4-6k-token generations per turn and feeding history back).
- This is the historical crash shape. If health dies → CONTEXT rule 4.

### LEG 4 — mixed + idle gaps (~15 min)
- 8 concurrent standard requests, then 120s full idle, then 8 more, then
  300s idle, then 4 deep-ish (10k) requests. Catches idle-transition
  stalls (the user's v52f fingerprint class).

## Report (REPORT.md)
Verdict line first: `VERDICT: PASS | STALL | CRASH | BOOT_FAILED | MIXED`.
Then: per-leg summary tables (count, ok, p50/p95 TTFT, p50/p95 total),
stall events with timestamps + log fingerprint lines found, crash evidence
if any, R1b marker lines, and a 5-line conclusion. requests.jsonl keeps
one line per request: leg, ts, ttft_s, total_s, status, tokens.
