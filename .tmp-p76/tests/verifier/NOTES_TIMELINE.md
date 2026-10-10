# VERIFIER interim evidence notes — v78 R1b validation (written after Pass 1)

Local clock = FLEDT (UTC+3). All serve-log timestamps below are container/UTC.
Verifier leg started ≈ 22:49 UTC Oct 2 (lane already healthy).

## Run timeline (from serve_full.log banner/init-failure markers)
| Boot (UTC) | APIServer pid | VLLM_XPU_MTP_TP1 | Outcome |
|---|---|---|---|
| 15:12:19 | 208 | — | long control era; DEVICE_LOST crash 20:32:05 (EngineCore 331); rebooted after |
| 21:27:55 | 2018 | — | served ~21:14+ SPECTIMING era (workers 1580/1586); replaced |
| 21:58:51 | 49 | SET=1 | DIED at engine init (line 11873) |
| 22:09:38 | 824 | SET=1 | DIED at engine init (line 12938) |
| 22:21:47 | 1669 | SET=1 | DIED at engine init (line 14017) |
| 22:28:08 | 2427 | NOT SET | CURRENT run, serving (this is the run my probes hit) |

## Decisive evidence chain
1. All 3 R1b-flagged boots logged `Unknown vLLM environment variable detected:
   VLLM_XPU_MTP_TP1` (lines 11311, 11884, 12949) — and all 3 died with
   `RuntimeError: Engine core initialization failed` (lines 11873, 12938, 14017).
2. Root cause (EngineCore pid=1942, 22:23:06, line ~13922):
   `ValueError: To serve at least one request with the model's max seq len
   (262144), 4.97 GiB KV cache is needed, which is larger than the available
   KV cache memory (3.31 GiB). ... estimated maximum model length is 163840.`
   → The replicated MTP head cost ≈ 8.28 − 3.31 = **4.97 GiB/rank** (current
   no-flag run reports "Available KV cache memory: 8.28 GiB" at 22:29:23).
   R1b flag therefore DID take effect (weights replicated) but blew the KV
   budget → R1b never served a single request anywhere in this log.
3. Current run (22:28) has NO MTP_TP1 in /proc/2427/environ (checked: only
   ALLOW_ASYNC, E5M2, ALLREDUCE, XPU_GRAPH, FP8_MQ, SPEC_DRAFT_BARRIER*).
   No R1b marker in its log section; only v22 eager-head marker
   (qwen3_5_mtp.py:531, VLLM_XPU_MTP_EAGER_HEAD).
4. Current-run SPECTIMING (Pass 1, 40 lines, 22:46:32→22:49:25):
   propose_d median 10.2 ms (p10 9.9 / p90 11.8); greedy_d median 2.2 flat;
   step_wall median 60.9 (p10 58.6 / p90 97.0, one 119 spike pair).
   = baseline posture. greedy d=2.2 exactly baseline ⇒ collectives still
   present. Old-era spot check (21:14): propose d=11.0/20.1, greedy 2.2 — same.

## Acceptance sanity Pass 1 (10 req, temp 0.3, max_tokens 64, sequential)
10/10 HTTP 200; walls 0.44–5.12 s (max at req1 cold + entropy req);
contents coherent (Paris, 12, Shakespeare, Mercury, Au, nails, 3 sides, Hola;
"Entropy is" truncated by max_tokens inside reasoning — not degeneration).
outputs_pass1.txt.

## Anomalies
- v52f stall fingerprints IN CURRENT RUN: 372 lines from 22:30:19 through
  ≥22:51:09 (196 PRE-COPY / 176 DEF-PP), continuous during my leg.
- DEVICE_LOST crashes at 20:32:05 — OLD era (pre-R1b-attempts), not current.
- Current run: no DEVICE_LOST / EngineCore died / ccs reset lines.
- step_wall p90 97 / spike 119 ms — modest; mission baseline band 72–94
  pure-decode; mixed prefill+decode eager ~146 (structural).

## Verdict direction (to finalize after Pass 2)
R1B_FAIL: (a) metrics FAIL bar on the only serving posture (10.2 / 2.2 vs
≤7 / ≤1.0), and (b) that posture is not R1b — R1b cannot boot at lane config
(replicated head −4.97 GiB/rank → KV init check fail at max_model_len 262144;
fix path: gpu_memory_utilization ↑ or max_model_len ≤163840).
