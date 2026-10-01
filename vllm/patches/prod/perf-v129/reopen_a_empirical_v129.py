#!/usr/bin/env python3
"""reopen_a_empirical_v129.py — perf-v129 P53b REOPEN-A empirical leg.

Companion to reopen_a_spectrum_v129.py (P53a analytic). P53a read the
checkpoint's real decay spectrum and predicted, via the OU model
(ss err = eps * sqrt(1/(1-gamma^2))), that int8-per-feature roundtrip
error (eps ~= 0.39%) clears the 18% rejection bar for 99.26% of heads
but not the long-memory tail (max 44.6%). This leg tests that
prediction EMPIRICALLY on the M2 harness kernel — the same op family
whose raw-e4m3 pool collapsed at N~6 in the v127 M1 forensics.

Design: N-step recurrence on the production spec kernel. Four arms
share identical per-step inputs; only the SSM pool storage differs:

  REF       fp16 pool, no injection            (truth)
  E4M3      raw e4m3 pool — the kernel roundtrips it internally on
            every launch; this arm MUST blow up (v127 collapse
            comparand: validates the harness against known physics)
  I8DYN     fp16 pool + external int8 per-(hv,k)-feature roundtrip
            after every launch, scales recomputed per step (the
            numerics-isolated REOPEN-A question)
  I8STATIC  same roundtrip but scales fixed from the initial pool
            with x4 headroom (practicality signal: saturation as the
            state grows under static per-request scales)

Head ladder: HV=16 heads with target per-token gamma log-spaced
0.95..0.99996 (dt_bias=0 -> dt=softplus(0); A_log=ln(ln(1/g)/dt)) so
the spectrum spans P53a's p50..max. The EFFECTIVE per-launch decay
phi is estimated from the REF trajectory itself (lag-1 autocorrelation
/ Yule-Walker on the dynamic slots) — the verdict never relies on the
nominal ladder mapping.

OU validation: eps_eff calibrated per head at N=1 (amp=1 there);
predicted err(N) = eps_eff * sqrt((1-phi^(2N))/(1-phi^2)); the model
is VALID iff measured/predicted stays within [0.5, 2] at every
checkpoint N>=8 (dynamic slots 0..7; static slots 8..11 are a
pure-accumulation control — noise there never decays, sqrt(N) by
construction, so they are reported but exempt).

Verdict caps: E4M3_COLLAPSE=REPRODUCED|NOT, OU_MODEL=VALID|INVALID,
I8_TAIL_SPLIT=CLEAN|MUDDY, REOPEN_A_VERDICT=OPEN_HYBRID|CLOSED
(OPEN_HYBRID iff collapse reproduced AND model valid AND the per-head
over/under-bar classification at N_final matches the OU prediction —
i.e. int8 clears the bar exactly where the spectrum says it should).

Run in a THROWAWAY GPU container, never on the lane:
  docker run --rm --device /dev/dri -v /root/build/v132_so:/so \
    -v /root/build/v129_stage:/h intel/omix:0.1.0-devel-ubuntu24.04 \
    bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; \
             /opt/venv/bin/python /h/reopen_a_empirical_v129.py \
             /so/custom_esimd_kernels_lgrf.so'
"""
import json
import math
import sys
import time

import torch

SO = sys.argv[1] if len(sys.argv) > 1 else "/so/custom_esimd_kernels_lgrf.so"
OUT = "/h/reopen_a_empirical_v129.json"
torch.ops.load_library(SO)
OP = torch.ops.custom_esimd_kernels_vllm
DEV = torch.device("xpu")

# Qwen3.5/3.6 TP=2 geometry accepted by the kernel's TORCH_CHECK (M2).
H, HV, K, V = 8, 16, 128, 128
NSD, NST = 2, 4
DIM = 2 * H * K + HV * V
NTOK = NSD * NST
NSS = NTOK + 4
SCALE = 1.0 / (H ** 0.5)

N_STEPS = 4096                       # 8 tokens/launch -> 32k decode tokens
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BAR = 0.18                           # v127 rejection bar (18%)
GAMMAS = [0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.9965, 0.9975,
          0.998, 0.9985, 0.999, 0.9993, 0.9996, 0.9998, 0.99996]

DT_NOM = math.log1p(1.0)             # softplus(0) = 0.6931


def step_inputs(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {"qkvz": torch.randn(NTOK, DIM, generator=g).to(torch.float16),
            "ba": torch.randn(NTOK, 2 * HV, generator=g).to(torch.float16)}


def main():
    t_start = time.time()
    g = torch.Generator(device="cpu").manual_seed(1234)

    # fixed inputs
    conv_state0 = torch.randn(NSS, 3, DIM, generator=g).to(torch.float16)
    conv_weight = (torch.randn(DIM, 4, generator=g) * 0.25).to(torch.float16)
    conv_bias = (torch.randn(DIM, generator=g) * 0.1).to(torch.float16)
    spec_idx = torch.arange(NTOK, dtype=torch.int32)
    token_indx = torch.arange(NTOK, dtype=torch.int32)
    accepted = torch.full((NSD,), NST, dtype=torch.int32)
    dt_bias = torch.zeros(HV, dtype=torch.float16)
    a_log = torch.tensor([math.log(math.log(1.0 / gam) / DT_NOM)
                          for gam in GAMMAS], dtype=torch.float16)

    # stratified initial pool (P23F regime; per-(hv,k) mag 1e-3..10)
    mag = torch.pow(10.0, torch.rand(1, HV, 1, K, generator=g) * 4.0 - 3.0)
    pool0 = (torch.randn(NSS, HV, V, K, generator=g) * mag).to(torch.float16)

    dev = {k: v.to(DEV) for k, v in
           (("cw", conv_weight), ("cb", conv_bias), ("si", spec_idx),
            ("ti", token_indx), ("ac", accepted), ("al", a_log),
            ("db", dt_bias))}

    # four arms: pools + private conv states (evolve identically — no
    # noise is injected there; asserted at the end)
    p_ref = pool0.to(DEV).clone()
    p_e4 = pool0.to(DEV).to(torch.float8_e4m3fn).clone()
    p_i8d, _ = i8_roundtrip(pool0.to(DEV).clone())
    runmax0 = pool0.abs().amax(dim=(0, 2)).float().clamp_min(1e-30)
    sc_stat = (127.0 / (4.0 * runmax0)).to(DEV)
    p_i8s, _ = i8_roundtrip(pool0.to(DEV).clone(), sc_stat)
    pools = {"ref": p_ref, "e4m3": p_e4, "i8dyn": p_i8d, "i8static": p_i8s}
    convs = {a: conv_state0.to(DEV).clone() for a in pools}
    outs = {a: (torch.zeros(NTOK, HV, V, dtype=torch.float16, device=DEV),
                torch.zeros(NTOK, HV, V, dtype=torch.float16, device=DEV))
            for a in pools}

    # per-step drives (iid across steps — matches the OU model's input
    # assumption; 4096 x 13KB fits CPU RAM easily)
    drives = [step_inputs(10_000 + n) for n in range(N_STEPS)]

    # autocorr accumulators (Yule-Walker) on dynamic slots of REF
    s1 = torch.zeros(HV, device=DEV)
    s2 = torch.zeros(HV, device=DEV)
    prev_dyn = None
    clip_hist = []                    # i8static saturation frac/step (cap)
    clip_acc = 0.0
    cks = {}

    def launch(arm, drive):
        o, z = outs[arm]
        OP.esimd_gdn_conv_fused_seq_spec(
            qkvz=drive["qkvz"].to(DEV), conv_state=convs[arm],
            conv_weight=dev["cw"], conv_bias=dev["cb"],
            spec_state_indices=dev["si"], A_log=dev["al"],
            dt_bias=dev["db"], ba=drive["ba"].to(DEV),
            ssm_state=pools[arm], output=o, z_out=z,
            token_indx=dev["ti"], num_accepted_tokens=dev["ac"],
            num_spec_decodes=NSD, num_spec_tokens=NST,
            H=H, HV=HV, K=K, V=V, scale=SCALE)

    def head_rel(arm):
        r = pools["ref"][0:NTOK].float()
        a = pools[arm][0:NTOK].float()
        num = (a - r).pow(2).sum(dim=(0, 2, 3))
        den = r.pow(2).sum(dim=(0, 2, 3)).clamp_min(1e-30)
        dyn = (num / den).sqrt()
        rs = pools["ref"][NTOK:].float()
        as_ = pools[arm][NTOK:].float()
        stat = (((as_ - rs).pow(2).sum(dim=(0, 2, 3))
                 / rs.pow(2).sum(dim=(0, 2, 3)).clamp_min(1e-30)).sqrt()
                .mean())
        return dyn.cpu(), float(stat)

    for n in range(N_STEPS):
        d = drives[n]
        launch("ref", d)
        # autocorr on REF dynamic slots (h_n vs h_{n+1})
        cur = pools["ref"][0:NTOK].float()
        if prev_dyn is not None:
            s1 += (prev_dyn * cur).sum(dim=(0, 2, 3))
            s2 += (prev_dyn * prev_dyn).sum(dim=(0, 2, 3))
        prev_dyn = cur
        launch("e4m3", d)
        launch("i8dyn", d)
        launch("i8static", d)
        pools["i8dyn"], _ = i8_roundtrip(pools["i8dyn"])
        pools["i8static"], clip = i8_roundtrip(pools["i8static"], sc_stat)
        clip_acc += clip
        if (n + 1) in CHECKPOINTS:
            torch.xpu.synchronize()
            if not torch.isfinite(pools["ref"].float()).all():
                print("ABORT: REF pool diverged at step %d" % (n + 1))
                break
            cks[n + 1] = {a: head_rel(a) for a in
                          ("e4m3", "i8dyn", "i8static")}
            if len(clip_hist) < 64:
                clip_hist.append(round(clip_acc / (n + 1), 6))

    torch.xpu.synchronize()
    wall = time.time() - t_start

    # isolation assert: conv states identical across arms (no injected
    # noise outside the ssm pool)
    conv_ok = all(torch.equal(convs[a].view(torch.uint8),
                              convs["ref"].view(torch.uint8))
                  for a in pools)

    phi = (s1 / s2.clamp_min(1e-30)).clamp(1e-6, 0.999999).cpu()
    phi_list = [round(float(x), 6) for x in phi]
    n_fin = max(cks) if cks else 0
    heads = []
    model_ok = True
    split_ok = True
    for h in range(HV):
        ph = float(phi[h])
        row = {"head": h, "gamma_target": GAMMAS[h], "phi_hat": ph,
               "eps_eff": None, "e4m3_first_over": None,
               "pred_final": None, "meas_final": None,
               "over_pred": None, "over_meas": None}
        e1 = cks.get(1, {}).get("i8dyn")
        if e1 is not None:
            row["eps_eff"] = round(float(e1[0][h]), 5)
        for ck, dd in sorted(cks.items()):
            if dd["e4m3"][0][h] >= BAR and row["e4m3_first_over"] is None:
                row["e4m3_first_over"] = ck
        if row["eps_eff"] and n_fin:
            amp_fin = math.sqrt(max((1.0 - ph ** (2 * n_fin))
                                    / max(1.0 - ph * ph, 1e-12), 0.0))
            row["pred_final"] = round(row["eps_eff"] * amp_fin, 4)
            row["meas_final"] = round(float(cks[n_fin]["i8dyn"][0][h]), 4)
            row["over_pred"] = row["pred_final"] > BAR
            row["over_meas"] = row["meas_final"] > BAR
            if row["over_pred"] != row["over_meas"]:
                split_ok = False
            for ck, dd in sorted(cks.items()):
                if ck < 8:
                    continue
                amp = math.sqrt(max((1.0 - ph ** (2 * ck))
                                    / max(1.0 - ph * ph, 1e-12), 0.0))
                pred = row["eps_eff"] * amp
                if pred > 1e-9 and not 0.5 <= dd["i8dyn"][0][h] / pred <= 2.0:
                    model_ok = False
        heads.append(row)

    hi_phi = [r for r in heads if r["phi_hat"] >= 0.99]
    collapse = bool(hi_phi and any(r["e4m3_first_over"] is not None
                                   and r["e4m3_first_over"] <= 16
                                   for r in hi_phi))
    verdict = "OPEN_HYBRID" if (collapse and model_ok and split_ok
                                and conv_ok) else "CLOSED"

    print("\nphi_hat per head:", phi_list)
    print("conv isolation: %s" % ("OK" if conv_ok else "VIOLATED"))
    print("i8static clip frac (running mean, capped): %s" % clip_hist[-3:])
    for r in heads:
        print("  h%-2d g_tgt=%.5f phi=%.6f eps=%.4f%% e4m3_N18=%s "
              "i8 fin pred=%.3f meas=%.3f over=%s/%s"
              % (r["head"], r["gamma_target"], r["phi_hat"],
                 100.0 * (r["eps_eff"] or 0), r["e4m3_first_over"],
                 r["pred_final"] or -1, r["meas_final"] or -1,
                 r["over_pred"], r["over_meas"]))
    print("E4M3_COLLAPSE=%s OU_MODEL=%s I8_TAIL_SPLIT=%s" % (
        "REPRODUCED" if collapse else "NOT",
        "VALID" if model_ok else "INVALID",
        "CLEAN" if split_ok else "MUDDY"))
    print("REOPEN_A_VERDICT=%s wall=%.1fs steps=%d" % (verdict, wall, n_fin))

    with open(OUT, "w") as f:
        json.dump({"n_steps": n_fin, "wall_s": round(wall, 1),
                   "checkpoints": [ck for ck in sorted(cks)],
                   "phi_hat": phi_list, "gammas_target": GAMMAS,
                   "conv_isolation_ok": conv_ok,
                   "i8static_clip_frac": clip_hist,
                   "curves": {str(ck): {a: {"dyn": [round(float(x), 5)
                                                  for x in cks[ck][a][0]],
                                          "static_mean":
                                              round(cks[ck][a][1], 5)}
                              for a in cks[ck]} for ck in sorted(cks)},
                   "heads": heads,
                   "e4m3_collapse": collapse, "ou_model_valid": model_ok,
                   "i8_tail_split_clean": split_ok,
                   "verdict": verdict}, f, indent=1)
    print("EMPIRICAL_DONE verdict=%s json=%s" % (verdict, OUT))


def i8_roundtrip(pool, scale=None):
    """int8 per-(hv,k)-feature storage roundtrip on device.

    Dynamic arm: scale recomputed from the CURRENT pool runmax (the
    numerics-isolated question — no saturation, pure eps injection).
    Static arm: caller passes frozen scales; returns clip frac too.
    """
    if scale is None:
        runmax = pool.abs().amax(dim=(0, 2)).float().clamp_min(1e-30)
        scale = 127.0 / runmax
    sc = scale[None, :, None, :]
    qf = pool.float() * sc
    q = torch.clamp(torch.round(qf), -127.0, 127.0)
    clip = float((qf.abs() > 127.0).float().mean())
    return (q / sc).to(torch.float16), clip


if __name__ == "__main__":
    main()
