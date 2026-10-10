#!/usr/bin/env python3
"""llm-scaler P82: device memory-pressure freeze-frame at the GDN core sites.

Extends the P81 argument snap with HOST-SIDE device memory pressure on every
SYCL GDN core dispatch (the wedge surface): torch allocator view (allocated/
reserved MB) + DRIVER view (free/total MB via mem_get_info -- captures
non-torch driver submit temporaries too, which is the allocation the L0
OOM-at-event-record tell implicates). Metadata only; the queries are host-side
allocator/driver stats -- no stream commands are issued, so the race is not
perturbed. On wedge, the last sycl line = the hung call's OWN pressure reading
-> directly confirms/refutes the allocation-ceiling mechanism and picks the
fix lever (launch-headroom guard vs state-registry pinning).

Sites (both in gdn_linear_attn.py; anchors are the same base anchors the P80
v2 patcher uses, so apply patch_p80_event_ladder.py FIRST, then this):
  sycl  torch.ops.vllm.gdn_attention_core_xpu(   (12-space)  -> enriched line
  conv  handled = self._gdn_conv_esimd_ok and    (8-space)   -> lean P81 line

Usage (in container):
  /opt/venv/bin/python /tmp/patch_p82_argsnap.py image
"""
import hashlib
import os
import py_compile
import sys

MARK = "llm-scaler P82"

MODULE = '''"""llm-scaler P82 device memory-pressure freeze-frame.

Per-call argument metadata (P81 format) +, on sycl-site calls only, the
device memory pressure the hung launch would see: torch allocator
(allocated/reserved) and the DRIVER view (free/total via mem_get_info; -1 =
driver query unavailable in this build). Host-side stats only -- zero stream
commands, zero device syncs -- so the wedge race is not perturbed. The
measured memquery latency is written once as a header line (perturbation
audit: if the driver query turns out slow, the number says so).

Env: P82_ARGSNAP=1 to arm, P82_ARGSNAP_PATH (default /root/p82_argsnap),
one line-buffered file per process (pid suffix). Inert unless armed.
"""
import os
import threading
import time

_ON = os.environ.get("P82_ARGSNAP", "") == "1"
_PATH = os.environ.get("P82_ARGSNAP_PATH", "/root/p82_argsnap")

_lock = threading.Lock()
_n = 0
_f = None


def _mem():
    try:
        import torch
        a = torch.xpu.memory_allocated() >> 20
        r = torch.xpu.memory_reserved() >> 20
        try:
            free, total = torch.xpu.mem_get_info()
            return (a, r, free >> 20, total >> 20)
        except Exception:
            return (a, r, -1, -1)
    except Exception:
        return None


def _meta(x):
    try:
        return "T%s,%s,s%s,p%x" % (str(tuple(x.shape)),
                                   str(x.dtype).replace("torch.", ""),
                                   str(tuple(x.stride())), x.data_ptr())
    except Exception:
        try:
            return repr(x)[:48]
        except Exception:
            return "?"


def p82_snap(site, *items):
    global _n, _f
    if not _ON:
        return
    try:
        with _lock:
            if _f is None:
                _f = open("%s.%d" % (_PATH, os.getpid()), "a", buffering=1)
                t0 = time.perf_counter()
                _mem()
                _f.write("# memquery_us=%.1f\\n"
                         % ((time.perf_counter() - t0) * 1e6))
            m = _mem() if site == "sycl" else None
            _n += 1
            tail = (" mem[a=%d,r=%d,f=%d,t=%d]" % m) if m else ""
            _f.write("%d %.3f %s %s%s\\n"
                     % (_n, time.time(), site,
                        " ".join(_meta(x) for x in items), tail))
    except Exception:
        pass
'''

SITES = [
    ("sycl",
     "            torch.ops.vllm.gdn_attention_core_xpu(",
     '            from vllm._p82_argsnap import p82_snap  # llm-scaler P82\n'
     '            p82_snap("sycl", core_attn_out, z, '
     'projected_states_qkvz, projected_states_ba, self.prefix, num_tokens, '
     '_use_buf)  # llm-scaler P82\n'),
    ("conv",
     "        handled = self._gdn_conv_esimd_ok and "
     "self._gdn_conv_decode(",
     '        from vllm._p82_argsnap import p82_snap  # llm-scaler P82\n'
     '        p82_snap("conv", projected_states_qkvz, '
     'projected_states_ba, core_attn_out, z, self._gdn_conv_esimd_ok)  '
     '# llm-scaler P82\n'),
]


def _ins_before(src, anchor, lines, tag):
    n = src.count(anchor)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(anchor, lines + anchor, 1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode != "image":
        print("usage: patch_p82_argsnap.py image")
        sys.exit(2)
    vllm_dir = "/opt/venv/lib/python3.12/site-packages/vllm"
    mp = os.path.join(vllm_dir, "_p82_argsnap.py")
    open(mp, "w").write(MODULE)
    py_compile.compile(mp, doraise=True, cfile="/tmp/p82c_module.pyc")
    print("MODULE_WRITTEN %s md5 %s"
          % (mp, hashlib.md5(open(mp, "rb").read()).hexdigest()))
    gp = os.path.join(vllm_dir,
                      "model_executor/layers/mamba/gdn_linear_attn.py")
    src = open(gp).read()
    if MARK in src:
        print("SKIP(already) %s" % gp)
    else:
        for tag, anchor, lines in SITES:
            src = _ins_before(src, anchor, lines, tag)
        open(gp, "w").write(src)
        py_compile.compile(gp, doraise=True, cfile="/tmp/p82c_gdn.pyc")
        n = src.count(MARK)
        assert n == 4, "marker count %d != 4" % n
        print("PATCHED %s md5 %s markers=%d"
              % (gp, hashlib.md5(open(gp, "rb").read()).hexdigest(), n))


if __name__ == "__main__":
    main()
