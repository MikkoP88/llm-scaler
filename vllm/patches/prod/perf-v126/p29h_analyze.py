#!/usr/bin/env python3
"""p29h_analyze.py — analyze /root/p29h_diag_<pid>.pt ring dumps from the
P29H v8 instrumented lane.

Ring semantics: buf is (8192, 8) fp32; slot (ctr % 8192) is written on
every captured spec replay of the OWNER layer (layer-0 GDN), so after ctr
replays the first min(ctr, 8192) slots are valid, in replay order.

Columns: [md_out, md_pool, md_conv, nan_out, nan_pool, nan_conv,
          qkvz_absmax, nsd]
  md_out   = |native-fp8-pool output - kernel-on-fp16-snapshot output|.max
  md_pool  = |post-call live pool rows - pre-call fp16 snapshot|.max
  md_conv  = same for conv rows (conv cache is fp16 everywhere)
  nan_*    = 1.0 if any NaN/Inf appeared
  nsd      = num_spec_decodes baked into the replayed graph (static per
             graph; the column identifies WHICH graph size wrote the row)

Verdict logic:
  - md_out > tol on some rows -> layer-0 native fp8 spec output diverges
    from the fp16-state reference under REAL traffic -> op/state path
    convicted at the recorded nsd values.
  - all md_out ~ 0 while text is bang-salad -> layer-0 fp8 spec call is
    clean; corruption lives elsewhere (commit path, other layers,
    sampling/verify, KV) -> widen telemetry next.
"""
import sys

import torch

TOL = 1e-3
BAD = 0.05


def analyze(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("meta", {})
    ctr = int(d.get("ctr", 0))
    buf = d["buf"].float()
    n = min(ctr, buf.shape[0])
    print("=== %s ===" % path)
    print("meta: %s" % meta)
    print("saves=%d ctr=%d valid_rows=%d" % (d.get("saves", -1), ctr, n))
    if n == 0:
        print("EMPTY RING")
        return
    v = buf[:n]
    cols = meta.get("cols", ["md_out", "md_pool", "md_conv", "nan_out",
                             "nan_pool", "nan_conv", "qkvz_absmax", "nsd"])
    print("col maxima:")
    for j, c in enumerate(cols):
        print("  %-10s max=%.6g  mean=%.6g" % (c, v[:, j].max(), v[:, j].mean()))
    md_o, md_p, md_c = v[:, 0], v[:, 1], v[:, 2]
    nan_bad = (v[:, 3] > 0) | (v[:, 4] > 0) | (v[:, 5] > 0)
    div = (md_o > BAD) | (md_p > BAD) | (md_c > BAD) | nan_bad
    print("rows md_out>%.3g: %d | md_pool>%.3g: %d | md_conv>%.3g: %d"
          % (BAD, int((md_o > BAD).sum()), BAD, int((md_p > BAD).sum()),
             BAD, int((md_c > BAD).sum())))
    print("rows any nan/inf: %d | rows any divergence: %d / %d"
          % (int(nan_bad.sum()), int(div.sum()), n))
    if int(div.sum()):
        idx = torch.nonzero(div).flatten()
        first = int(idx[0])
        print("FIRST divergent row #%d: %s" % (
            first, ["%.6g" % x for x in v[first].tolist()]))
        nsds = v[idx, 7]
        uniq, cnts = torch.unique(nsds, return_counts=True)
        print("nsd distribution of divergent rows:",
              {float(u): int(c) for u, c in zip(uniq, cnts)})
        print("last divergent row #%d: %s" % (
            int(idx[-1]), ["%.6g" % x for x in v[int(idx[-1])].tolist()]))
    else:
        print("NO divergence above %.3g in md_* — layer-0 native fp8 spec"
              " call matches fp16-state reference on every recorded"
              " replay" % BAD)
    # nsd coverage of the whole ring (which graph sizes ran)
    uniq, cnts = torch.unique(v[:, 7], return_counts=True)
    print("ring nsd coverage:", {float(u): int(c)
                                 for u, c in zip(uniq, cnts)})
    qmax = v[:, 6].max()
    print("qkvz_absmax max=%.6g (inf-free=%s)" % (qmax, bool(torch.isfinite(
        v[:, 6]).all())))
    print("last 3 rows:")
    for i in range(max(0, n - 3), n):
        print("  #%d %s" % (i, ["%.6g" % x for x in v[i].tolist()]))


def main():
    paths = sys.argv[1:] or ["/root/p29h_diag_%d.pt" % p
                             for p in (869, 875)]
    for p in paths:
        try:
            analyze(p)
        except Exception as e:  # noqa: BLE001
            print("=== %s === ANALYZE-ERROR %r" % (p, e))
    print("P29H_ANALYZE_DONE")


if __name__ == "__main__":
    sys.exit(main())
