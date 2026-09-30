#!/usr/bin/env python3
"""p29f_probe.py — v126 P29F: serve-exact graph padding emulation.

P29D conviction: native fp8 spec kernel RED only at nsd>1 inside XPU decode
graphs (LEG C), GREEN eager (LEG B) and GREEN nsd==1-under-graphs (LEG D).
P29E: single-graph capture/replay with all-real inputs is bitwise-correct
— so the serve-only delta is GRAPH PADDING: real batch < captured size,
dummy tail slots sanitized by gdn_attn.py exactly as:
    spec_state_indices tail <- NULL_BLOCK_ID (0)
    spec_sequence_masks tail <- False
    num_accepted_tokens tail <- 1
    spec_token_indx: NO tail fill (stale arange from capture/prev step)
    qkvz/ba pad rows: arbitrary garbage activations

This probe reproduces that EXACTLY at op level (capture nsd=8, replay
real=3 + 5 dummies) and measures:
  1. real rows: outputs + SSM/conv checkpoint rows vs eager-on-same-tensors
  2. row-0 victim: does the dummy walk (idx=0, acc=1) rewrite pool/conv
     row 0? (upstream never state-updates dummies — new kernel exposure)
  3. fp16 control (same exposure class, fp16 gate never reaches it in prod)
"""
import sys

import torch

DEV = "xpu"
K = V = 128
H = 8
HV = 24
NC = 512
NST = 5
NSD_CAP = 8
NSD_REAL = 3
SCALE = float(K ** -0.5)
DIM = 2 * H * K + 2 * HV * V

DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]


def bv(t):
    return t.view(torch.uint8) if t.dtype != torch.float16 else t.view(torch.int16)


def main():
    print("=== P29F serve-exact padding emulation (cap=%d real=%d) ==="
          % (NSD_CAP, NSD_REAL))
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec as op
    torch.manual_seed(77)
    cw = (torch.randn(DIM, 4, device=DEV) * 0.1).half()
    cb = torch.zeros(DIM, device=DEV, dtype=torch.float16)
    A_log = (torch.randn(HV, device=DEV) * 0.1).half()
    dtb = (torch.randn(HV, device=DEV) * 0.1).half()

    for name, dt in DTYPES:
        torch.manual_seed(4242)
        pool16 = (torch.randn(NC, HV, V, K, device=DEV) * 0.05).half()
        conv = (torch.randn(NC, 3, DIM, device=DEV) * 0.01).half()
        pool = pool16 if dt == torch.float16 else pool16.to(dt).contiguous()

        # victim marker state on row 0 (the NULL_BLOCK_ID target)
        if dt == torch.float16:
            pool[0] = (torch.randn(HV, V, K, device=DEV) * 0.123).half()
        else:
            pool.view(torch.uint8)[0] = torch.arange(
                pool[0].numel(), dtype=torch.uint8, device=DEV).reshape(
                pool[0].shape)
        conv[0] = (torch.randn(3, DIM, device=DEV) * 0.117).half()

        # static buffers (capture-sized)
        idx_s = torch.zeros(NSD_CAP, NST, dtype=torch.int32, device=DEV)
        tok_s = torch.arange(NSD_CAP * NST, dtype=torch.int32, device=DEV)
        acc_s = torch.full((NSD_CAP,), 4, dtype=torch.int32, device=DEV)
        qkvz_s = (torch.randn(NSD_CAP * NST, DIM, device=DEV) * 0.5).half()
        ba_s = (torch.randn(NSD_CAP * NST, 2 * HV, device=DEV) * 0.5).half()
        out_s = torch.zeros(NSD_CAP * NST, HV, V, dtype=torch.float16,
                            device=DEV)
        z_s = torch.zeros_like(out_s)

        # capture-time content: full dummy batch, distinct ring rows 300+
        for s in range(NSD_CAP):
            idx_s[s] = torch.arange(300 + s * NST, 300 + s * NST + NST,
                                    dtype=torch.int32, device=DEV)

        def call():
            op(qkvz_s, conv, cw, cb, idx_s, A_log, dtb, ba_s, pool,
               out_s, z_s, tok_s, acc_s, NSD_CAP, NST, H, HV, K, V, SCALE)

        call()
        torch.xpu.synchronize()
        pool0_pre_capture = pool[0].clone()
        conv0_pre_capture = conv[0].clone()
        try:
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                call()
        except Exception as e:  # noqa: BLE001
            print("  %s: CAPTURE-ERROR %r" % (name, e))
            continue
        torch.xpu.synchronize()
        pool0_post_capture = pool[0].clone()
        conv0_post_capture = conv[0].clone()
        cap_wrote_row0 = not bool(torch.equal(bv(pool0_post_capture),
                                              bv(pool0_pre_capture)))
        print("  %-5s capture-exec wrote row0=%s conv0=%s"
              % (name, cap_wrote_row0,
                 not bool(torch.equal(conv0_post_capture, conv0_pre_capture))))

        # ---- replay-time content: real=3 + serve-style sanitized tail ----
        torch.manual_seed(9999)
        rings = [list(range(100, 105)), list(range(105, 110)),
                 list(range(110, 115))]
        for s in range(NSD_REAL):
            idx_s[s] = torch.tensor(rings[s], dtype=torch.int32, device=DEV)
        idx_s[NSD_REAL:] = 0                              # NULL_BLOCK_ID
        acc_s[:NSD_REAL] = torch.tensor([4, 2, 3], dtype=torch.int32,
                                        device=DEV)
        acc_s[NSD_REAL:] = 1
        tok_s[:NSD_REAL * NST] = torch.arange(NSD_REAL * NST,
                                              dtype=torch.int32, device=DEV)
        # tok_s tail LEFT at capture-time arange (stale) — serve behavior
        qkvz_s[:NSD_REAL * NST] = (torch.randn(
            NSD_REAL * NST, DIM, device=DEV) * 0.5).half()
        qkvz_s[NSD_REAL * NST:] = (torch.randn(
            (NSD_CAP - NSD_REAL) * NST, DIM, device=DEV) * 300.0).half()
        ba_s[:NSD_REAL * NST] = (torch.randn(
            NSD_REAL * NST, 2 * HV, device=DEV) * 0.5).half()
        ba_s[NSD_REAL * NST:] = (torch.randn(
            (NSD_CAP - NSD_REAL) * NST, 2 * HV, device=DEV) * 300.0).half()

        # eager reference on identical contents (incl. dummy walk)
        pool_r = pool.clone()
        conv_r = conv.clone()
        out_r = torch.zeros_like(out_s)
        z_r = torch.zeros_like(out_s)
        op(qkvz_s.clone(), conv_r, cw, cb, idx_s.clone(), A_log, dtb,
           ba_s.clone(), pool_r, out_r, z_r, tok_s.clone(), acc_s.clone(),
           NSD_CAP, NST, H, HV, K, V, SCALE)
        torch.xpu.synchronize()

        g.replay()
        torch.xpu.synchronize()

        rt = NSD_REAL * NST
        out_ok = bool(torch.equal(out_s[:rt], out_r[:rt]))
        z_ok = bool(torch.equal(z_s[:rt], z_r[:rt]))
        rows_t = torch.tensor(sum(rings, []), device=DEV)
        pool_ok = bool(torch.equal(bv(pool[rows_t]), bv(pool_r[rows_t])))
        conv_ok = bool(torch.equal(conv[rows_t], conv_r[rows_t]))
        row0_replay_wrote = not bool(torch.equal(bv(pool[0]),
                                                 bv(pool0_post_capture)))
        row0_matches_eager = bool(torch.equal(bv(pool[0]), bv(pool_r[0])))
        conv0_wrote = not bool(torch.equal(conv[0], conv0_post_capture))
        conv0_matches_eager = bool(torch.equal(conv[0], conv_r[0]))
        r0 = pool[0].to(torch.float16).float()
        r0e = pool_r[0].to(torch.float16).float()
        print("  %-5s real out=%s z=%s pool=%s conv=%s | "
              "ROW0: replay-wrote=%s ==eager=%s conv0-wrote=%s ==eager=%s"
              " | row0 nan/inf replay=%d/%d eager=%d/%d"
              % (name, "OK" if out_ok else "BAD", "OK" if z_ok else "BAD",
                 "OK" if pool_ok else "BAD", "OK" if conv_ok else "BAD",
                 row0_replay_wrote, row0_matches_eager, conv0_wrote,
                 conv0_matches_eager,
                 int(torch.isnan(r0).sum()), int(torch.isinf(r0).sum()),
                 int(torch.isnan(r0e).sum()), int(torch.isinf(r0e).sum())))
        print("P29F %-5s out=%d z=%d pool=%d conv=%d row0_wrote=%d "
              "row0_eager_same=%d conv0_wrote=%d conv0_eager_same=%d "
              "row0_nan=%d row0_inf=%d row0e_nan=%d row0e_inf=%d"
              % (name, out_ok, z_ok, pool_ok, conv_ok, row0_replay_wrote,
                 row0_matches_eager, conv0_wrote, conv0_matches_eager,
                 int(torch.isnan(r0).sum()), int(torch.isinf(r0).sum()),
                 int(torch.isnan(r0e).sum()), int(torch.isinf(r0e).sum())))
        del pool, conv, pool_r, conv_r, out_r, z_r
    print("P29F_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
