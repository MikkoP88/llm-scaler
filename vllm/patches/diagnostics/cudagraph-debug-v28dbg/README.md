# cudagraph-debug v28dbg — in-engine flight recorder (wedge localization)

Debug overlay on `llm-scaler-vllm-adv:v27` for KNOWN_ISSUES #11 (the
graphs-x-spec wedge residual). NOT a serving image: it writes one
timestamped line per collective / full-graph replay so the NEXT live wedge
names the phase that never returned. No L0 tracer exists in the image
(onetrace/vtune absent; ZE loader debug trace unsupported by the loader
build) — this is the substitute.

## What it instruments

| file (overlaid in-image) | markers |
|---|---|
| `fr.py` -> site-packages root | the recorder itself; `/tmp/fr_<pid>.log`, line-buffered, `VLLM_XPU_FR=0` disables |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `replay begin desc=...` / `replay end` around `graph.replay()` in `run_fullgraph` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | `AR begin numel=... via_env=...` / `AR end` around `all_reduce`; `AG begin/end` around `all_gatherv` |

## The load-bearing guard

Every `_fr.log` call AND every f-string that builds its argument is behind
`torch.compiler.is_compiling()`. Dynamo traces through `all_reduce` during
profile_run / piece tracing and faults on the recorder's file I/O and on
f-string tensor methods (`numel=` interpolation) — this crashed the first
build deterministically at boot. Do not remove the guards.

## Verified facts (2026-08-31, two boots, k4 + VIA_ALLGATHER)

- Full-graph replay NEVER runs in the spec decode path: **0 replay
  markers** vs **43,552 AR markers per rank**. The graphs surface under
  spec = torch.compile PIECEWISE pieces + EAGER collectives between them
  (`vllm::all_reduce` is a splitting op unless
  `VLLM_XPU_ALLOW_COMM_IN_GRAPH=1`). The residual therefore lives at the
  piece-replay x eager-collective boundary, not in captured-collective
  replay.
- Healthy baseline: exact `AR begin`/`AR end` pairing (43,552/43,552, zero
  orphans), ~200 us per eager AR, ~3 ms gaps between consecutive ARs (the
  replayed piece time) at 65k decode.
- Throughput cost of the recorder at serve time was not measurable at
  battery granularity (identical walls to the un-instrumented v27 boot).

## Decision rule at a live wedge

Read the tail of BOTH `/tmp/fr_<pid>.log` (one per TP rank):

| fr tail | verdict |
|---|---|
| last line `AR begin` (no `AR end`) | the eager oneCCL collective hangs — on the VIA path that is `all_gather` |
| `AR end` then silence | a replayed/compiled PIECE spins on device |
| no markers at all since request start | the hang is OUTSIDE collectives+replay (host scheduling) |

Cross-check with the host watcher capture (`/root/build/wedge_cap/`):
xpu-smi engine stats + py-spy stacks of both workers.

## Runbook (host 10.20.3.65)

```bash
# build (never while serving)
docker build -t llm-scaler-vllm-adv:v28dbg /root/build/v28dbg/

# boot the wedge-susceptible config with the recorder live
nohup bash /root/build/serve_boot_var.sh "" "VLLM_XPU_ALLREDUCE_VIA_ALLGATHER=1" v28dbg "" 512 llm-scaler-vllm-adv:v28dbg > /root/build/v28_boot.out 2>&1 &

# arm the host watcher (captures xpu-smi + py-spy + fr tails on a 50 s stall)
nohup bash /root/build/frw3.sh > /root/build/watcher.out 2>&1 &
# (frw3.sh source: ../wedge-endgame-v27/frw3.sh; adjust the engine-log tail
#  path inside to the boot's log name)

# fire the battery (stage via docker cp after each fresh container)
docker exec lsv-test python /tmp/deep_repro_fox.py 8   # 65k fox probes
docker exec lsv-test python /tmp/long_exp.py 4         # 2048-tok ignore_eos @65k
docker exec lsv-test python /tmp/conc_repro.py 10      # 2 concurrent streams
```

## Session outcome (2026-08-31)

Two instrumented boots (incl. one deliberate re-roll) ran the full battery
clean — 10/10 short-ctx, 14/14 65k probes, 4/4 long-exposure (twice), 20/20
concurrent (twice) — vs a ~1/2 wedge rate on the same shape the night
before. Susceptibility appears decided per-boot at capture time; the
recorder + watcher are armed for the next susceptible boot. Untried levers
and the full mechanism discussion: KNOWN_ISSUES.md #11 (v28dbg update).
