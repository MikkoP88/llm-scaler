# Mission: VERIFIER — independent R1b measurement verdict

Read `../CONTEXT.md` first. You do light, independent probing and you
analyze the SPECTIMING evidence. Work in this directory only. Produce
`VERDICT.md`. Two passes; total runtime ≤ 90 minutes.

## Pass 1 (start ~15 min in, after lane is healthy)
1. Health + models check (GET only).
2. Acceptance sanity: 10 short chat requests at temp 0.3. Checks:
   non-empty, coherent, no obvious degeneration, no 5xx. Record raw
   outputs to outputs_pass1.txt.
3. Spec-decode liveliness: grep serve log (plink read-only) for recent
   `SPECTIMING` lines → save last 40 to spectiming_pass1.log.
   Parse `propose=` and `greedy=` device times if present in the format.

## Pass 2 (start ~60 min in)
1. Repeat acceptance sanity (10 requests) → outputs_pass2.txt.
2. Harvest last 200 SPECTIMING lines → spectiming_pass2.log.
3. Analysis (the core deliverable):
   - Baseline (pre-R1b, measured earlier on this lane): propose device
     ≈ 12ms, greedy device ≈ 2.2ms, step_wall ≈ 72-94ms pure-decode;
     mixed prefill+decode steps eager ~146ms (structural, unchanged).
   - R1b PASS bar: propose device ≈ ≤ 7ms AND greedy device ≈ ≤ 1.0ms
     (collectives gone), with no regression in step_wall_ms beyond noise.
   - Compare medians from your harvests; state PASS/FAIL per metric.
4. VERDICT.md: `VERDICT: R1B_PASS | R1B_FAIL | INCONCLUSIVE` + metric
   table (baseline vs measured), acceptance sanity result, and any
   anomaly lines from the log (resets, wedge fingerprints).

## Rules
- Zero load beyond your sanity probes (small, sequential).
- If health is 000 in either pass: record, mark CRASH in VERDICT.md,
  save `docker exec lsv-test tail -200 /root/serve_full.log` locally as
  crash_evidence.txt, skip remaining probing, finish the analysis you can.
