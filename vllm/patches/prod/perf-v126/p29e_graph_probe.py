#!/usr/bin/env python3
"""p29e_graph_probe.py — v126 P29E: XPU-graph capture/replay probe for the
native fp8 spec path (LEG-C conviction: graphs ON + native fp8 = RED 68/80;
async OFF changed nothing; eager = GREEN; bridge-under-graphs = GREEN;
fp16-under-graphs (same v131 wrapper) = GREEN).

Everything inside a vLLM decode graph is CAPTURED ONCE at boot and REPLAYED
with mutated static inputs. Two op classes inside that region differ between
the green fp16 lane and the red fp8 lane:
  (a) the v131 P29B-FIX snapshot gather  at::index_select(conv_state / ssm_state)
      — conv_state is ALWAYS fp16 (green); ssm_state is fp8 ONLY on the red lane
  (b) the ESIMD spec kernel reading fp8 state pointers
      (P28 bridge proved the kernel under graphs green with fp16 pool clones)

This probe replays the serve mechanism at op level:
  P0 graph machinery control         (y = x*2 must replay fresh)
  P1 index_select capture freshness  per dtype fp16/e4m3/e5m2, plain + out=
  P2 full wrapper capture            fp16 vs e4m3/e5m2: mutate ALL static
                                     inputs in place, replay, bitwise-compare
                                     outputs + checkpointed pool rows vs eager
Verdict logic prints P29E lines; exit 0 always (measurement).
"""
import sys

import torch

DEV = "xpu"
K = V = 128
H = 8
HV = 24
NC = 512
NST = 5
SCALE = float(K ** -0.5)
DIM = 2 * H * K + 2 * HV * V
ROWS = list(range(100, 100 + 8 * NST))      # 40 snap rows (nsd=8 geometry)

DTYPES = [("fp16", torch.float16),
          ("e4m3", torch.float8_e4m3fn),
          ("e5m2", torch.float8_e5m2)]


def bv(t):
    return t.view(torch.uint8) if t.dtype != torch.float16 else t.view(torch.int16)


def graph_ok():
    """P0: basic capture/replay freshness sanity for the API usage."""
    x = torch.randn(1024, device=DEV, dtype=torch.float16)
    y = torch.empty_like(x)

    def work():
        y.copy_(x * 2)

    work()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(g):
            work()
        x.copy_(torch.randn(1024, device=DEV, dtype=torch.float16))
        g.replay()
        torch.xpu.synchronize()
        ok = bool(torch.allclose(y, x * 2))
        print("P0 graph machinery control: %s" % ("PASS" if ok else "FAIL"))
        return ok
    except Exception as e:  # noqa: BLE001
        print("P0 graph machinery control: CAPTURE-ERROR %r" % (e,))
        return False


def p1_index_select():
    """index_select under capture: after mutating pool + idx in place, a
    correctly-captured gather must return FRESH values at replay."""
    print("\n=== P1 index_select capture freshness ===")
    E = HV * V * K
    verdict = {}
    for name, dt in DTYPES:
        torch.manual_seed(11)
        pool = (torch.randn(NC, E, device=DEV) * 0.05).half()
        pool = pool if dt == torch.float16 else pool.to(dt).contiguous()
        idx = torch.tensor(ROWS[:20], dtype=torch.int64, device=DEV)
        snap = torch.zeros(20, E, dtype=dt, device=DEV)

        def gather_plain():
            return torch.index_select(pool, 0, idx)

        def gather_out():
            torch.index_select(pool, 0, idx, out=snap)

        # warmup (eager) so kernels exist before capture
        gather_out()
        torch.xpu.synchronize()
        try:
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                r = gather_plain()          # noqa: F841 (python ref held)
                gather_out()
            # mutate ALL captured inputs in place (replay-time contents)
            new_vals = (torch.randn(NC, E, device=DEV) * 0.05).half()
            pool.copy_(new_vals if dt == torch.float16
                       else new_vals.to(dt))
            new_rows = [(x + 7) % NC for x in ROWS[:20]]
            idx.copy_(torch.tensor(new_rows, dtype=torch.int64, device=DEV))
            g.replay()
            torch.xpu.synchronize()
            ref = torch.index_select(pool, 0, idx)
            plain_ok = bool(torch.equal(bv(r), bv(ref)))
            out_ok = bool(torch.equal(bv(snap), bv(ref)))
            del r
        except Exception as e:  # noqa: BLE001
            print("  %s: CAPTURE-ERROR %r" % (name, e))
            verdict[name] = ("error", "error")
            continue
        verdict[name] = (plain_ok, out_ok)
        print("  %-5s plain=%-7s out=-%s" %
              (name, "FRESH" if plain_ok else "STALE/BAD",
                      "FRESH" if out_ok else "STALE/BAD"))
        del pool, snap, idx, ref
    print("P1_RESULT " + " ".join(
        "%s:plain=%s,out=%s" % (n, *v) for n, v in verdict.items()))
    return verdict


def build_lane(dt, seed):
    torch.manual_seed(seed)
    pool16 = (torch.randn(NC, HV, V, K, device=DEV) * 0.05).half()
    conv16 = (torch.randn(NC, 3, DIM, device=DEV) * 0.01).half()
    pool = pool16 if dt == torch.float16 else pool16.to(dt).contiguous()
    return pool, conv16, pool16


def p2_wrapper():
    """Full v131 wrapper under capture: mutate every static input in place,
    replay, compare outputs + checkpointed pool rows bitwise vs eager."""
    from custom_esimd_kernels_vllm import esimd_gdn_conv_fused_seq_spec as op
    print("\n=== P2 spec wrapper capture (nsd=8 geometry) ===")
    nsd = 8
    torch.manual_seed(77)
    cw = (torch.randn(DIM, 4, device=DEV) * 0.1).half()
    cb = torch.zeros(DIM, device=DEV, dtype=torch.float16)
    A_log = (torch.randn(HV, device=DEV) * 0.1).half()
    dtb = (torch.randn(HV, device=DEV) * 0.1).half()

    def params():
        n = nsd * NST
        return ((torch.randn(n, DIM, device=DEV) * 0.5).half(),
                (torch.randn(n, 2 * HV, device=DEV) * 0.5).half(),
                torch.randint(1, 6, (nsd,), device=DEV, dtype=torch.int32))

    def idx_tensor(rows):
        return torch.tensor(rows, dtype=torch.int32,
                            device=DEV).reshape(nsd, NST)

    verdict = {}
    for name, dt in DTYPES:
        pool, conv16, pool16 = build_lane(dt, 4242)
        rows = ROWS
        qkvz, ba, acc = params()
        idx = idx_tensor(rows)
        tok = torch.arange(nsd * NST, dtype=torch.int32, device=DEV)
        out = torch.zeros(nsd * NST, HV, V, dtype=torch.float16, device=DEV)
        z = torch.zeros_like(out)

        def call():
            op(qkvz, conv16, cw, cb, idx, A_log, dtb, ba, pool,
               out, z, tok, acc, nsd, NST, H, HV, K, V, SCALE)

        # eager warmup + capture
        call()
        torch.xpu.synchronize()
        try:
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                call()
        except Exception as e:  # noqa: BLE001
            print("  %s: CAPTURE-ERROR %r" % (name, e))
            verdict[name] = "error"
            del pool, conv16, pool16
            continue

        # mutate ALL static inputs in place (fresh content, same addresses)
        torch.manual_seed(9999)
        np16 = (torch.randn(NC, HV, V, K, device=DEV) * 0.05).half()
        pool.copy_(np16 if dt == torch.float16 else np16.to(dt))
        conv16.copy_((torch.randn(NC, 3, DIM, device=DEV) * 0.01).half())
        qkvz2, ba2, acc2 = params()
        qkvz.copy_(qkvz2)
        ba.copy_(ba2)
        acc.copy_(acc2)

        # eager reference with the SAME fresh inputs from pristine clones
        pool_r = pool.clone()
        conv_r = conv16.clone()
        out_r = torch.zeros_like(out)
        z_r = torch.zeros_like(out)
        op(qkvz.clone(), conv_r, cw, cb, idx.clone(), A_log, dtb, ba.clone(),
           pool_r, out_r, z_r, tok.clone(), acc.clone(), nsd, NST, H, HV, K,
           V, SCALE)
        torch.xpu.synchronize()

        # replay the graph and compare
        g.replay()
        torch.xpu.synchronize()
        rows_t = torch.tensor(rows, device=DEV)
        out_ok = bool(torch.equal(out, out_r))
        z_ok = bool(torch.equal(z, z_r))
        pool_ok = bool(torch.equal(bv(pool[rows_t]), bv(pool_r[rows_t])))
        conv_ok = bool(torch.equal(conv16[rows_t], conv_r[rows_t]))
        verdict[name] = (out_ok, z_ok, pool_ok, conv_ok)
        print("  %-5s out=%-7s z=%-7s pool-ckpt=%-7s conv-ckpt=%s" %
              (name, "OK" if out_ok else "BAD", "OK" if z_ok else "BAD",
                      "OK" if pool_ok else "BAD",
                      "OK" if conv_ok else "BAD"))
        del pool, conv16, pool16, out_r, z_r, pool_r, conv_r

    print("P2_RESULT " + " ".join(
        "%s:%s" % (n, "/".join("OK" if x else "BAD" for x in v))
        for n, v in verdict.items() if isinstance(v, tuple)))
    return verdict


def main():
    print("=== P29E XPU-graph capture probe (v131 wrapper ops) ===")
    print("torch", torch.__version__, "dev", torch.xpu.get_device_name(0))
    ok0 = graph_ok()
    if not ok0:
        print("P29E_VERDICT: PROBE_INVALID (graph machinery control failed)")
        return 0
    p1 = p1_index_select()
    p2 = p2_wrapper()
    print("\n=== P29E summary ===")
    for n, v in p1.items():
        print("P1 %-5s plain=%s out=%s" % (n, v[0], v[1]))
    for n, v in p2.items():
        print("P2 %-5s %s" % (n, v))
    print("P29E_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
