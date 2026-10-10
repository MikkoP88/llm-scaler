#!/usr/bin/env python3
"""llm-scaler P81: GDN-core argument freeze-frame (wedge mechanism probe).

P80 sealed WHERE the wedge is (gdn_attention_core_xpu launch boundary at
deep context, all compute units idle). P81 discriminates the mechanism
class by snapshotting the host-visible argument image of every GDN core
dispatch:

  args dirty  (ptr drift / shape anomaly at the hung call)
      -> the fork's GDN call construction is wrong at depth
  args clean  (stable shapes/ptrs)
      -> launch/batch machinery behind the op (extension state registry
         or runtime/driver surface) -- next probe surface

Two dispatch sites in gdn_linear_attn.py (both P80-instrumented; P81
anchors on the P80 rung lines, so apply patch_p80_event_ladder.py image
FIRST):
  sycl  upstream torch.ops.vllm.gdn_attention_core_xpu (the P80 verdict
        site -- fires all storm long at x4 width, verify rows > 128)
  conv  the fused ESIMD _gdn_conv_decode gate (covers the width-2 /
        small-batch branch)

Metadata ONLY: shape / dtype / stride / data_ptr / repr of scalars --
zero device syncs, zero tensor reads, so the race under probe is not
perturbed. Line-buffered per-pid file readable mid-run (buffered-flush
law). Env-gated P81_ARGSNAP=1; inert otherwise.

Usage (in container, AFTER the P80 image patcher):
  /opt/venv/bin/python /tmp/patch_p81_argsnap.py image
"""
import hashlib
import os
import py_compile
import sys

MARK = "llm-scaler P81"

MODULE = '''"""llm-scaler P81 argument snapshot (vllm/_p81_argsnap.py).

Freeze-frames the host-visible argument image of every GDN core dispatch.
Host-side metadata ONLY (shape/dtype/stride/data_ptr) -- no device syncs,
no tensor reads -- so the wedge race under probe is not perturbed.
Line-buffered per-pid file, readable mid-run.

Env: P81_ARGSNAP=1 arms; P81_ARGSNAP_PATH (default /root/p81_argsnap).
Inert (no-op) unless armed.
"""
import os
import threading
import time

_ON = os.environ.get("P81_ARGSNAP", "") == "1"
_PATH = os.environ.get("P81_ARGSNAP_PATH", "/root/p81_argsnap")

_lock = threading.Lock()
_f = None
_n = 0


def _meta(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return "T%s,%s,s%s,p%x" % (
                str(tuple(x.shape)),
                str(x.dtype).replace("torch.", ""),
                str(tuple(x.stride())),
                x.data_ptr(),
            )
    except Exception:
        pass
    try:
        return repr(x)[:48]
    except Exception:
        return "<unrepr>"


def p81_snap(site, *items):
    global _f, _n
    if not _ON:
        return
    try:
        with _lock:
            if _f is None:
                _f = open("%s.%d" % (_PATH, os.getpid()), "a", buffering=1)
            _n += 1
            _f.write("%d %.3f %s %s\\n"
                     % (_n, time.time(), site,
                        " ".join(_meta(x) for x in items)))
    except Exception:
        pass
'''

P = ("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
     "mamba/gdn_linear_attn.py")

SITES = [
    # (anchor line in the P80-patched file, rung lines to insert after)
    ('            p80_ev("gdn_core")  # llm-scaler P80\n',
     '            from vllm._p81_argsnap import p81_snap  # llm-scaler P81\n'
     '            p81_snap("sycl", core_attn_out, z, projected_states_qkvz,'
     ' projected_states_ba, self.prefix, num_tokens, _use_buf)'
     '  # llm-scaler P81\n'),
    ('        p80_ev("gdn_conv")  # llm-scaler P80\n',
     '        from vllm._p81_argsnap import p81_snap  # llm-scaler P81\n'
     '        p81_snap("conv", projected_states_qkvz, projected_states_ba,'
     ' core_attn_out, z, self._gdn_conv_esimd_ok)  # llm-scaler P81\n'),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode != "image":
        print("usage: patch_p81_argsnap.py image")
        sys.exit(2)

    vllm_dir = "/opt/venv/lib/python3.12/site-packages/vllm"
    tp = os.path.join(vllm_dir, "_p81_argsnap.py")
    open(tp, "w").write(MODULE)
    py_compile.compile(tp, doraise=True, cfile="/tmp/p81c_mod.pyc")
    print("MODULE_WRITTEN %s md5 %s"
          % (tp, hashlib.md5(open(tp, "rb").read()).hexdigest()))

    src = open(P).read()
    if MARK in src:
        print("SKIP(already) %s" % P)
        return
    for anchor, rung in SITES:
        n = src.count(anchor)
        assert n == 1, "anchor count %d != 1 for %r" % (n, anchor[:48])
        src = src.replace(anchor, anchor + rung, 1)
    open(P, "w").write(src)
    py_compile.compile(P, doraise=True, cfile="/tmp/p81c_gdn.pyc")
    print("PATCHED %s md5 %s markers=%d"
          % (P, hashlib.md5(open(P, "rb").read()).hexdigest(),
             src.count(MARK)))


if __name__ == "__main__":
    main()
