#!/usr/bin/env python3
"""reopen_a_spectrum_v129.py — perf-v129 P53 REOPEN-A analytic leg.

RECURRENCE-COMPOUNDING LAW (v127, M1 rejection): quantization error
per state roundtrip compounds because the GDN SSM state recurses every
decode step. e4m3 half-ULP relative error ~= 2^-4 = 6.25%/roundtrip ->
collapse at N ~= 6 observed. int8 with per-feature scales:
2^-8 ~= 0.39%/roundtrip (16x better). The OPEN question is the growth
REGIME, which is governed by the recurrence's decay spectrum:

  h_{n+1} = gamma * h_n + dt*B*x_n,   gamma = exp(dt*A) in (0,1)

Injected roundtrip noise eps_n (independent, |eps| ~= 0.39% relative)
propagates as an Ornstein-Uhlenbeck random walk: steady-state
amplification = sqrt(1/(1-gamma^2))  (bounded, NOT unbounded linear).
Features with gamma -> 1 (long memory) amplify most.

This leg reads the ACTUAL checkpoint GDN parameters (A_log, dt_bias —
read-only safetensors access, CPU-only, no engine interaction) and
computes the gamma distribution per layer/head:

  A = -exp(A_log)   (mamba convention, negative)
  dt_nominal = softplus(dt_bias)     (input-independent component)
  gamma = exp(-dt_nominal * exp(A_log))

Verdict inputs reported:
  - gamma percentiles across all layers x heads
  - OU amplification sqrt(1/(1-gamma^2)) percentiles
  - steady-state relative error = 0.39% * amplification
  - FRACTION of heads whose steady-state error exceeds the 18%
    rejection bar, and the equivalent pure-sqrt-N N@18% for reference
Caps verdict: SPECTRUM_DONE OVER18_FRAC=<x> MAX_SS_ERR_PCT=<x>.

Run IN-CONTAINER (read-only): docker exec lsv-test /opt/venv/bin/python3
/root/reopen_a_spectrum_v129.py
"""
import glob
import json
import math

import numpy as np
import torch
from safetensors import safe_open

CKPT = "/models/target"
OUT = "/root/reopen_a_spectrum_v129.json"


def read_f64(shard, name):
    """bf16/fp32-safe tensor read -> float64 numpy (numpy cannot decode
    bfloat16; the venv torch can)."""
    with safe_open(shard, framework="pt") as f:
        t = f.get_tensor(name)
    return t.to(torch.float64).cpu().numpy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def main():
    idx = glob.glob(CKPT + "/*.safetensors.index.json")
    shards = sorted(glob.glob(CKPT + "/*.safetensors"))
    a_logs, dt_biases = {}, {}
    if idx:
        with open(idx[0]) as f:
            wmap = json.load(f)["weight_map"]
        for name, shard in wmap.items():
            if name.endswith(".linear_attn.A_log"):
                a_logs[name] = CKPT + "/" + shard
            elif name.endswith(".linear_attn.dt_bias"):
                dt_biases[name] = CKPT + "/" + shard
    else:
        print("no index json; scanning shards directly")
        for shard in shards:
            with safe_open(shard, framework="numpy") as f:
                for name in f.keys():
                    if name.endswith(".linear_attn.A_log"):
                        a_logs[name] = shard
                    elif name.endswith(".linear_attn.dt_bias"):
                        dt_biases[name] = shard

    def layer_of(n):
        return n.rsplit(".", 1)[0]          # strip .A_log / .dt_bias

    layers = sorted(set(map(layer_of, a_logs))
                    & set(map(layer_of, dt_biases)))
    print("GDN layers found: %d (A_log %d, dt_bias %d)"
          % (len(layers), len(a_logs), len(dt_biases)), flush=True)

    gammas, rows = [], []
    for lpfx in layers:
        lname = lpfx + ".A_log"
        dname = lpfx + ".dt_bias"
        a_log = read_f64(a_logs[lname], lname)
        dt_bias = read_f64(dt_biases[dname], dname)
        if a_log.shape != dt_bias.shape:
            print("shape mismatch on %s: %s vs %s — pairing by "
                  "position anyway" % (lname, a_log.shape,
                                       dt_bias.shape))
        n = min(a_log.size, dt_bias.size)
        A = -np.exp(a_log.ravel()[:n])
        dt = softplus(dt_bias.ravel()[:n])
        g = np.exp(dt * A)           # in (0,1)
        gammas.append(g)
        rows.append({"layer": lpfx.split(".")[3] if len(
            lpfx.split(".")) > 4 else lpfx,
                     "n_heads": int(n),
                     "gamma_p50": float(np.percentile(g, 50)),
                     "gamma_p99": float(np.percentile(g, 99)),
                     "gamma_max": float(g.max())})
    g = np.concatenate(gammas)

    def pct(a, q):
        return float(np.percentile(a, q))

    ampl = np.sqrt(1.0 / np.maximum(1.0 - g * g, 1e-12))
    ss_err_pct = 0.39 * ampl                      # int8 per-roundtrip eps
    e4m3_ss_pct = 6.25 * ampl                     # comparand arm
    over18 = float(np.mean(ss_err_pct > 18.0))
    n_at_18_sqrtN = (18.0 / 0.39) ** 2            # pure random-walk ref

    print("\ngamma:        p50=%.6f p90=%.6f p99=%.6f max=%.6f"
          % (pct(g, 50), pct(g, 90), pct(g, 99), g.max()))
    print("OU amp:       p50=%.2f  p90=%.2f  p99=%.2f  max=%.2f"
          % (pct(ampl, 50), pct(ampl, 90), pct(ampl, 99), ampl.max()))
    print("int8 SS err:  p50=%.2f%% p90=%.2f%% p99=%.2f%% max=%.2f%%"
          % (pct(ss_err_pct, 50), pct(ss_err_pct, 90),
             pct(ss_err_pct, 99), ss_err_pct.max()))
    print("e4m3 SS err:  p50=%.2f%% max=%.2f%%  (comparand; must be "
          "far over bar — validates the model against the observed "
          "N~6 collapse)" % (pct(e4m3_ss_pct, 50), e4m3_ss_pct.max()))
    print("OVER18_FRAC=%.4f  MAX_SS_ERR_PCT=%.2f  "
          "sqrtN N@18%%=%.0f (no-decay reference)"
          % (over18, ss_err_pct.max(), n_at_18_sqrtN))

    with open(OUT, "w") as f:
        json.dump({"n_layers": len(layers),
                   "n_heads_total": int(g.size),
                   "gamma": {"p50": pct(g, 50), "p90": pct(g, 90),
                             "p99": pct(g, 99), "max": float(g.max())},
                   "amplification": {"p50": pct(ampl, 50),
                                     "p90": pct(ampl, 90),
                                     "p99": pct(ampl, 99),
                                     "max": float(ampl.max())},
                   "int8_ss_err_pct": {"p50": pct(ss_err_pct, 50),
                                       "p90": pct(ss_err_pct, 90),
                                       "p99": pct(ss_err_pct, 99),
                                       "max": float(ss_err_pct.max())},
                   "e4m3_ss_err_pct_max": float(e4m3_ss_pct.max()),
                   "over18_frac": over18,
                   "sqrtN_N_at_18": n_at_18_sqrtN,
                   "per_layer": rows}, f, indent=1)
    print("SPECTRUM_DONE OVER18_FRAC=%.4f MAX_SS_ERR_PCT=%.2f"
          % (over18, ss_err_pct.max()))


if __name__ == "__main__":
    main()
