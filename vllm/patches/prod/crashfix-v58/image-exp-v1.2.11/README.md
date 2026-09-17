# image-exp-v1.2.11 — `llm-scaler-exp:v1.2.11`: 26.18-TUNED contingency image (NOT crash-free — events #10/#11)

Image ID `sha256:35003605bf1416369b3df64f3e31985b10f1d0f21ce1597538e6eaa69306dc96`
(23.2 GB, built 2026-09-17, host 10.20.3.65). Bake marker in-image:
`/root/.llm_scaler_exp_v1211_baked`. Naming per user directive 2026-09-17
("next available version `llm-scaler-exp:v*`"). **Verdict (same-day
revision): the image was built as a crash-free contingency; that premise
was REFUTED by its own sustain battery (event #10) and re-refuted on a
user-directed clean-host reboot control (event #11). NEO 26.18 does NOT
eliminate the §14 race. The image remains the best-CONFIGURED 26.18 lane
(max-tok/s draft-3, deterministic source-true numerics) with NO proven
crash-avoidance benefit; the standing 26.14 lane stays the default.**

## What it is

The standing lane (stock NEO 26.14, `llm-scaler-exp:v1.2.10` + boot patchers)
delivers full 50-class performance and carries §14-class crash exposure
(GSD-12919 / compute-runtime #939: xe/GuC submission race, `Reason: LR job
cleanup`, 11/11 identical devcoredump/dmesg class across every
kernel/fw/runtime cell ever tested — INCLUDING the NEO 26.18 baked here,
see "Crash evidence"). This image bakes the 26.18 stack plus the best
measured 26.18 configuration. Its original purpose (crash-free serving at
accepted degradation) is refuted; retained value: the tuned+ deterministic
26.18 lane is one boot command away if 26.18 semantics are ever wanted
(e.g. as an A/B lane for the upstream fix when Intel ships it).

- Base: `llm-scaler-exp:v1.2.10` (latest exp lineage tree, crashfix-v58 standing)
- NEO 26.18 runtime overlay baked (6 debs from `/root/build/neodl`:
  `intel-opencl-icd` + `libze-intel-gpu1` + `intel-ocloc` 26.18.38308.1-0,
  `intel-igc-core-2`/`intel-igc-opencl` 2.34.4, `libigdgmm12` 22.10.0 —
  the exact mix P2V0B/Q22b certified numerics on)
- Boot patchers baked (no runtime patching): `patch_f15b.py` +
  `patch_arstage.py` + `patch_v58_p1.py`, all grep-verified in-build
- `__pycache__` purged → source-true execution (see numerics contract)
- Serve config baked at `/root/serve_user.sh`: MTP spec **draft-3**
  (`num_speculative_tokens:3`) — the 26.18 tax-shifted optimum (below)
- py-spy preinstalled; mamba logger.info sed applied; identical env/device/
  mount contract as the standing lane (`boot_exp2618.sh`)

## Why draft-3 (max-tok/s campaign, 2026-09-17)

NEO 26.18's cost = per-op host-side blocking on every eager submission
(§24 H2e). Fewer eager proposer ops per accepted token → shorter drafts pay
more. Single-variable arms on 26.18 (each f8ref-gated):

| arm | draft | solo warm tok/s | conc4 warm tok/s | f8ref |
|---|---|---|---|---|
| A0 | 4 (standing) | 38.0 | 117.8 | EXACT 3/3 |
| **A1** | **3** | **42.3** | **137.8** | 2/3 (probe3 drift) |
| A2 | 2 | 40.2 | 135.0 | 2/3 (probe3 drift) |

Interior optimum at 3: **+11% solo / +17% conc4 vs draft-4 on 26.18**.
Deficit vs the 26.14 standing lane shrinks from −24%/−20% to **−16%/−6%**.
(Levers checked and dead: sampler kernel already ON by default, no
whole-step-capture knob, no clock-pin sysfs, cmdlist env inert §22.)

## Certified performance (2026-09-17, host 10.20.3.65)

| metric | exp:v1.2.11 (draft-3, 26.18) | standing 26.14 (draft-4) | delta |
|---|---|---|---|
| boot→health | ~190-220 s | ~170-200 s | — |
| solo decode | 42.3-42.7 | 50.2-50.4 | −16% |
| conc4 warm (agg) | 135.0-138.9 | 144-148 | −6% |
| bench3 2k decode | 400.6-400.9 | 401.9 | parity |
| bench3 16k decode | 309.4 cold / 321.9 warm | 479.5 | −33% |
| bench3 65k decode | 289.2-289.6 | 344.4 | −16% |
| bench3 conc8 (agg) | 306.9-308.3 | 357.9 | −14% |
| acceptance | 0.817-0.819 | 0.740-0.747 | draft-3-calibrated |
| engine resets (screens) | 0 | 0 | — |

Long-context cells (16k/65k) carry the largest 26.18 deficit — expected
under the per-op host-blocking mechanism.

## Crash evidence — the crash-free premise REFUTED (2026-09-17)

Two §14-class events ON this exact image lane, same class as all 9 prior
26.14 events (11/11 total):

- **EVENT #10** (first battery; host had been up through all §24 testing
  since 09-16 20:06): 18-round sustain + q33 watcher armed 05:43:45
  base_resets=0 → crash at **TTF 1h00m04s** (watcher fire 06:43:51).
  da:00.0 ccs `guc_id=32` first — devcoredump captured in TTL:
  `Reason: LR job cleanup, guc_id=32` (GuC 70.44.1) — + b1:00.0 `guc_id=22`;
  worker died `UR_RESULT_ERROR_DEVICE_LOST` in sample_tokens→copy_to_gpu.
- **EVENT #11** (user-directed CLEAN-HOST control to exclude residue from
  earlier crashes): host REBOOTED 07:34:47 (fresh boot, resets=0 verified);
  image lane re-booted, watcher + 18-round sustain armed 07:43:25
  base_resets=0. Rounds 1-12 clean; **onset ~10:28-10:30 round 13**
  (f15b detector: sample_tokens/propose_begin stalled 186 s, last completed
  op an AR); 10:32:15 `TimeoutError: RPC call to sample_tokens timed out` →
  `EngineDeadError`; 10:33:35 GuC LR-cleanup BOTH cards `b1:00.0 guc_id=22`
  + `da:00.0 guc_id=32` (SAME guc_ids as #10); capture 10:33:37 (devcoredump
  TTL expired — no fresh bin; class evidence = dmesg + app signature).
  **TTF 2h46m onset / 2h49m hard-fail.**

**Verdict: the contamination hypothesis is disproved — 26.18 crashes from
a pristine host.** Honest nuance: both 26.18 failures are this image's
config (draft-3 + purged); the earlier Q31 battery on 26.18 draft-4 (pyc
path) passed 18/18 — the crash-free claim was n=1 sample luck; the race
is probabilistic and persists on 26.18 runtime. 26.18 n=2 TTFs {1h00m,
2h49m} overlap the 26.14 historical range (14 min–2h52m): no demonstrated
rate reduction, at −16% solo decode. A lane-watchdog is MANDATORY on this
lane exactly as on the standing lane. This is also the strongest data
point for the upstream report (GSD-12919 / compute-runtime #939).

## Numerics contract

f8ref (3 fixed prompts, temp 0, mt 160, 2 reps): **deterministic
`{0b21bb2d3c6a, 68332ec7c31b, 95e24129958b}`** — stable across 3 runs on
the image lane + reproduced on an overlay lane with pycache purge (P2EG).

- probe-2 EXACT vs standing (`68332ec7c31b`).
- probe-3 drift is draft-length-inherent: verify batch shape changes → fp16
  kernel tiling → one near-tie argmax flip (draft-4 `05c88ff03b0c`,
  draft-3 `95e24129958b`, draft-2 `85b0ec22a6a`; probes 1-2 byte-identical
  at every length on the pyc path).
- probe-1 `0b21…` vs certified `cb8c3851b897` = the `__pycache__` purge:
  base v1.2.10 ships only 6 vllm `.pyc`; 3 diverge from fresh compiles of
  their own `.py` (`v1/worker/gpu_model_runner`, `v1/worker/mamba_utils`,
  `v1/attention/ops/triton_fp8_mq`); the shipped bytecode is what every
  no-purge lane (incl. the certified 26.14 standing lane) executes.
  Single-variable proof: overlay boot + purge alone reproduces the image
  hashes (P2EG, deterministic 2/2); without purge, draft-3 is
  rep-nondeterministic (f8ref `distinct=MISMATCH` on both draft-3 pyc-path
  lanes — near-tie wobble run-to-run).
- Shipped contract = source-true + deterministic: preferred over matching
  the wobbly pyc-path attractor (precedent: adv:v19 carried its own
  self-consistent hash set; conc-dependent temp-0 divergence is documented
  wheel-native). Per-module attribution of the probe-1 flip (restore-one-pyc
  test): optional, deferred — requires lane reboots.
- If EXACT certified hashes are required at the cost of −24%: boot the same
  image with draft-4? NO — draft-4 exactness was shown on the PYC path
  (A0/overlay); on this purged image draft-4 hashes are unverified. For a
  hash-exact 26.18 lane, rebuild without the purge line (then expect
  draft-4 EXACT + draft-3 wobble). Documented deliberately.

## Deploy / swap / rollback runbook

```bash
# PAUSE the standing-lane watchdog first (it auto-relaunches the 26.14 lane):
touch /root/build/lane_watchdog.paused
# Swap in this image lane:
bash /root/build/boot_exp2618.sh PRODV1211     # boots lsv-test from exp:v1.2.11
# health: curl http://localhost:8000/health    (port 8000, name lsv-test —
#                                         all tooling/watchdog contracts hold)
# MANDATORY on this lane too (NOT crash-free — event #11):
# keep a crash watcher armed (q33_watch.sh pattern) or re-point lane_watchdog.
# Rollback to the standing 26.14 lane (draft-4):
sed -i 's/num_speculative_tokens":3/num_speculative_tokens":4/' /root/build/serve_user.sh
bash /root/build/repro_bootQ21b.sh RESTORE     # + rm the pause flag to re-arm
```

Notes:
- The image carries its own `/root/serve_user.sh` (draft-3). The HOST
  `/root/build/serve_user.sh` is shared by the repro_boot* scripts for the
  26.14 lane — keep it at draft-4 when the standing lane is the default.
- `lane_watchdog.sh` currently relaunches `repro_bootQ21b.sh` (26.14
  lineage). If exp:v1.2.11 ever becomes the default lane, edit the lineage
  line in `/root/build/lane_watchdog.sh` to call `boot_exp2618.sh`. The
  watchdog must stay armed on ANY lane (this one included) — §14 exposure
  is lane-independent.
- No host prerequisites beyond the standing ones (kernel 6.17.0-1010-intel,
  GuC 70.44.1 — image is host-state independent across the tested matrix;
  events #10/#11 landed on contaminated AND pristine hosts alike).

## Limitations / known issues

1. **NOT crash-free** (events #10/#11 above; 11/11 identical §14 class).
   −16% solo / −6% conc4 / −33% 16k / −14% conc8 vs the standing 26.14
   lane buys no demonstrated crash-avoidance — hence the standing lane
   (max perf + watchdog-bounded §14 exposure) remains the default, and
   this image is a contingency/A-B lane only.
2. f8ref hash set is image-specific (above); A/B comparisons vs the
   standing lane must use probes 1-2 + perf, not probe-1/3 equality.
3. NEO 26.18 ≠ newest (26.31): 26.18 is the #939-bisect-adjacent
   version with measured behavior here; newer = unverified.
4. Root fix is still Intel's (GSD-12919 / #939). When a fixed NEO/kernel
   ships, re-run the §24 matrix; this image is the natural A/B baseline
   and then retires.

## Evidence (host /root/build/lce1/ unless noted)

`boot_P2E0/P2E1/P2E2/P2E1F/P2EG/EXPV1211/EXPV1211C.out`,
`f8ref_nv_{P2E0,P2E1,P2E2,P2E1F,PRODV2,PRODV2B,PRODV2C,P2EG_R1,P2EG_R2,
EXPV1211}.out(.txt)`, `nv_{P2E0,P2E1,P2E2,PRODV2,EXPV1211}_q17_{cold,warm}.out`,
`nvscreen_P2E1F.out`, `bench3_NVP2E1F_{cold,warm}.out`,
`build_prodv2.out`, `build_expv1211.out`,
`sustain_EXPV1211.out` + `expv1211_watch.out` (event #10),
`sustain_EXPV1211C.out` + `expv1211c_watch.out` (event #11 clean-host
control), `Q22_crash/` event #10 (`devcoredump_card2_Q22.bin`,
`Reason: LR job cleanup, guc_id=32`) + event #11 (10:33 set:
`first_errors.txt`, `dmesg` resets guc22+32, `f15b_dump_520/526.log`,
`fr_520/526.log`, `serve_full_Q22.log`, `dump_lines/section.txt`),
`/tmp/pyc_base.txt` (6 shipped pyc md5s);
host `/root/build/{p2e_ctx/, boot_exp2618.sh, repro_bootP2EG.sh, p2eg_mk.sh,
cleanhost_launch.sh, cleanhost_arm.sh, sustain_wait.sh, sustain_wait_c.sh}`;
repo: this dir (`Dockerfile`, `boot_exp2618.sh`) + campaign scripts in
`crashfix-v58/`.
