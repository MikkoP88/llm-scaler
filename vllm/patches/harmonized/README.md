# harmonized/ — the era-1/2 tree delta as explicit patches

The production vLLM tree diverges from its wheel manifest in exactly
**16 modified + 3 added files** (audited by `../harm-vaudit.py` against the
pip RECORD; `fr.py` sits at site-packages root, outside the `vllm/` package,
so it only shows up in a root-level audit). Era-3 (v32–v38) changes were
already surgical scripts in `../prod/`; everything older accumulated as
in-image edits with no patch script of their own. This directory captures
the **entire** remaining delta as reviewable, md5-gated unified diffs:

```
pristine wheel (adv:v14) + patches/01..16 + added/  ==  prod tree (adv:v38)
```

## Contents

| file | purpose |
|------|---------|
| `patches/01-config-tq-spec-enable.patch` | #05b: TurboQuant KV no longer auto-disables the spec drafter (v19; rollback `VLLM_ALLOW_TQ_SPEC=0`) — ORPHAN EDIT until now |
| `patches/02-comm-via-allgather-stablebuf.patch` | v29 FIX-1: per-shape stable output buffers for the VIA-allgather decode path (row-capped), + v28dbg `fr` import |
| `patches/03-attn-e5m2-guard.patch` | v34: e5m2-with-fp8-ckpt opt-in (`VLLM_XPU_ALLOW_E5M2_FP8_CKPT`; #19) |
| `patches/04-mtp-eager-head-compile-guard.patch` | v22: keep the MTP eagle head eager — torch.compile'd head issues oneCCL allgathers from dynamo code and segfaults TP>1 boot (`VLLM_XPU_MTP_EAGER_HEAD`) |
| `patches/05-xpu-unsafe-spec-graphs-gate.patch` | v30/v31.1/v35: force eager for spec×TP×piecewise-graphs (the #11 livelock posture); k>3 clamp removed (v35) |
| `patches/06-flash-attn-f8-scale-fixes.patch` | v19c: static fp8 scale read (capture-safe); v33 MQ3D varlen shim; v34 cached descale + FA2 split pin; v38 fp8 decode ESIMD→vxk FA2 reroute (#18 fix) |
| `patches/07-tq-mq-verify-backend.patch` | v19: multi-query verify fast path in the TQ backend (one MQ kernel per verify step; `VLLM_TQ_MQ_VERIFY`); v19b graphs capture; v21 non-causal draft |
| `patches/08-tq-mq-verify-kernel.patch` | v19: the MQ verify Triton kernel itself (flash-decoding style, per-row limits); v32 per-row 2D accumulation fix |
| `patches/09-unified-attn-scalefold.patch` | v33: fold per-tensor fp8 scales onto S/P tiles (identical math, 16x fewer scaled elements) + allow 3D split path |
| `patches/10-sched-ignore-eos-guard.patch` | #05d: ignore_eos must never early-finish (guards both check paths + FINISH_DIAG log) — ORPHAN EDIT until now |
| `patches/11-dflash-parity-stall-guard.patch` | v21c: draft-KV VRAM parity threshold; v20 spec_timing instrumentation; host-side drafter stall guard |
| `patches/12-proposer-drafter-comm.patch` | v25: local-argmax default for XPU MTP drafters (kills the full-vocab allgather per draft step; #11); v27 dedicated drafter communicator hook; v29 fr |
| `patches/13-cudagraph-replay-fr-logs.patch` | v28dbg: flight-recorder marks around graph replay |
| `patches/14-runner-async-counts-barrier-off.patch` | v32: async accepted-counts + deferred postprocess; v37 draft-barrier default OFF; v25 barrier machinery (env-only now) |
| `patches/15-worker-capture-stabilizer.patch` | v29 FIX-2: quiesce XPU allocator before compile/capture (`VLLM_XPU_CAPTURE_STAB`) + faulthandler |
| `patches/16-mamba-deferred-postprocess.patch` | v32: `run_deferred_postprocess` (N-step snapshots, dead/migrated-row skip) |
| `added/fr.py` | v28dbg flight recorder, LIVE at site-packages root (imported by 02/12/13) |
| `added/vllm/v1/spec_decode/drafter_comm.py` | v27: dedicated oneCCL communicator context for the drafter — LIVE (imported by llm_base_proposer) |
| `added/vllm/v1/spec_decode/spec_timing.py` | v20 A2: spec segment timing — LIVE (imported by dflash/llm_base_proposer/gpu_model_runner) |
| `apply.py` | md5-gated fail-loud applier (see doc header) |
| `MANIFEST.md5` | pristine + patched md5 for all 19 files |
| `_work/harm-extract.sh` | host-side generator (extracts v14/v38 file pairs from the images, regenerates diffs + manifest) |

## Apply / verify

```bash
# against a v14-base site-packages tree (in container or a scratch copy):
python3 apply.py --sp /opt/venv/lib/python3.12/site-packages
```

Every target must be exactly pristine (patch applied, md5 re-verified) or
already patched (skipped) — anything else aborts with exit 4. Diffs are
`git apply`-compatible (`a/… b/…` headers, `-p1`).

Scratch verification (2026-09-03, host /root/build/harm-scratch): fresh
v14 file copies + `apply.py` ⇒ **19/19 byte-identical** to the v38 tree
(see MANIFEST.md5 md5s; flash_attn.py = `1000effd71318a29b7e3279114084e4d`
matches the certified prod tree).

## Notes

- `spec_timing.py` was long believed dead; it is import-live in three
  modules (the v20 instrumentation calls are cheap no-ops when disabled).
  Removing it would require touching 02/11/12/14 — do that in a dedicated
  cleanup, not by deleting the file.
- `fr.py` (flight recorder) is v28dbg residue that rode into the prod
  lineage via the v31.1 bake. It is ON by default (`VLLM_XPU_FR=0`
  disables): one timestamped line per graph-replay/collective/propose to
  `/tmp/fr_<pid>.log`. All certified prod perf numbers were measured with
  it active — leave it or gate it off explicitly, but do not delete it
  (import-load-bearing).
- This set is the RECORD of what eras 1–2 did. Reproducing prod images
  still goes through the era-3 bake chain (v31.1 → v37 → v38); regenerating
  from v14+harmonized is possible but unexercised beyond the scratch check.
