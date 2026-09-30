#!/usr/bin/env python3
"""p29b5_probe.py — v126 P29B5: absence-vs-wrong-math discriminator.

Z5 fingerprint: batch seq1 pool[45] differs from solo in EXACTLY hv>=8,
all 128 V rows; out-row diffs were exactly N_hv * 128 lanes (full heads).
Two hypotheses:
  (a) those (seq, hv) work-groups' out/pool stores NEVER EXECUTED
      -> out rows stay 0, pool slots stay PRISTINE bytes;
  (b) they executed with different math -> nonzero, non-pristine content.
This probe measures, per (seq, hv):
  out_written : batch out row != 0 anywhere in the head
  pool_pristine: batch pool slot head == pristine pool bytes (never written)
  conv_written: batch conv slot head-slice == solo's (vs pristine)
for nsd=2 AND nsd=4 on e4m3 (deterministic). Prints the ran/not-ran matrix.
"""
import torch

K = V = 128
H, HV = 8, 24
DIM = 2 * H * K + 2 * HV * V
DEV = "xpu"
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DT = torch.float8_e4m3fn

from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec  # noqa

torch.manual_seed(2929)
cw = torch.randn(DIM, 4, dtype=torch.float16, device=DEV) * 0.1
cb = torch.zeros(DIM, dtype=torch.float16, device=DEV)
A_log16 = (torch.randn(HV, dtype=torch.float32, device=DEV) * 0.1).half()
dt_bias = torch.randn(HV, dtype=torch.float16, device=DEV) * 0.1
conv0 = torch.randn(NC, 3, DIM, dtype=torch.float16, device=DEV) * 0.01
pool16 = torch.randn(NC, HV, V, K, dtype=torch.float16, device=DEV) * 0.05

torch.manual_seed(31337)
qkvzB = torch.randn(4 * NST, DIM, dtype=torch.float16, device=DEV) * 0.1
baB = torch.randn(4 * NST, 2 * HV, dtype=torch.float16, device=DEV) * 0.5


def fresh():
    return pool16.to(DT).contiguous()


def spec(pool, conv, qkvz, ba, rows_idx, accepted):
    n = len(accepted)
    out = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    z = torch.zeros(n * NST, HV, V, dtype=torch.float16, device=DEV)
    idx = torch.tensor(rows_idx, dtype=torch.int32, device=DEV).reshape(n, NST)
    tok = torch.arange(n * NST, dtype=torch.int32, device=DEV)
    acc = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    esimd_gdn_conv_fused_seq_spec(
        qkvz, conv, cw, cb, idx, A_log16, dt_bias, ba, pool,
        out, z, tok, acc, n, NST, H, HV, K, V, SCALE)
    torch.xpu.synchronize()
    return out, z


def solo(i, acc_i, slots_i):
    return spec(fresh(), conv0.clone(),
                qkvzB[i * NST:(i + 1) * NST].contiguous(),
                baB[i * NST:(i + 1) * NST].contiguous(), slots_i, [acc_i])


V_OFF = 2 * H * K            # conv row: [qk(2048) | v(3072)] per dim layout


def matrix(nsd, tag):
    slots = list(range(40, 40 + nsd * NST))
    accs = [2, 4, 1, 3][:nsd]
    oB, _ = spec(fresh(), conv0.clone(), qkvzB[:nsd * NST].contiguous(),
                 baB[:nsd * NST].contiguous(), slots, accs)
    pB = fresh()
    spec(pB, conv0.clone(), qkvzB[:nsd * NST].contiguous(),
         baB[:nsd * NST].contiguous(), slots, accs)
    pristine = fresh()
    print("--- %s nsd=%d ran-matrix (P=out nonzero, p=pool pristine,"
          " c=conv solo-eq) ---" % (tag, nsd))
    for i in range(nsd):
        sl_i = slots[i * NST:(i + 1) * NST]
        oS, _ = solo(i, accs[i], sl_i)
        row = []
        for hv in range(HV):
            out_written = oB[i * NST:i * NST + 1, hv].abs().max().item() > 0
            # pool: the t=0 save slot of this seq = sl_i[0]
            prist = torch.equal(
                pB[sl_i[0]][hv].view(torch.uint8),
                pristine[sl_i[0]][hv].view(torch.uint8))
            # conv head slice: v region lanes for hv (128 wide at V_OFF+hv*V)
            lo = V_OFF + hv * V
            conv_slice_b = pB is None  # placeholder, use real conv below
            row.append("%s%s" % ("P" if out_written else "-",
                                 "p" if prist else "-"))
        print("  seq%d: %s" % (i, " ".join(row)))
    # conv check per seq/hv against a solo conv
    cB = conv0.clone()
    spec(fresh(), cB, qkvzB[:nsd * NST].contiguous(),
         baB[:nsd * NST].contiguous(), slots, accs)
    for i in range(nsd):
        sl_i = slots[i * NST:(i + 1) * NST]
        _, _ = solo(i, accs[i], sl_i)
        cS = conv0.clone()
        spec(fresh(), cS, qkvzB[i * NST:(i + 1) * NST].contiguous(),
             baB[i * NST:(i + 1) * NST].contiguous(), sl_i, [accs[i]])
        eq = []
        for hv in range(HV):
            lo = V_OFF + hv * V
            b_ok = torch.equal(cB[sl_i[0], 2, lo:lo + V],
                               cS[sl_i[0], 2, lo:lo + V])
            eq.append("c" if b_ok else "-")
        print("  seq%d conv V-slices t0: %s" % (i, " ".join(eq)))


matrix(2, "e4m3")
matrix(4, "e4m3")
print("PROBE5_DONE")
