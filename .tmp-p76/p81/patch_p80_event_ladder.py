#!/usr/bin/env python3
"""llm-scaler P80 v2: event-ladder tracer for the spec verify-path wedge.

v2 = this morning's proven patcher with the two site bugs fixed at source
(d2h 4-space anchor matched a wrong site -> 8-space parse_output body;
fattn bare anchor appears 3x -> del kv_cache_dummy_dep 2-line anchor), so a
FRESH container gets the full instrument in one pass with no repair scripts.

Deploys /opt/venv/.../vllm/_p80_tracer.py and instruments the rung sites.
At wedge time the tracer's head-of-line STALL line names the earliest op
whose completion never arrived = the hung kernel's Python-side boundary,
plus the pending list behind it.

Rungs (name -> site):
  d2h        rejection_sampler.py parse_output, before output_token_ids.cpu()
             (the known py-spy park site) -- 8-space body anchor
  gdn_core   gdn_linear_attn.py, before torch.ops.vllm.gdn_attention_core_xpu
             (the op the R1 device capture saw never retiring)
  gdn_conv   gdn_linear_attn.py, before the ESIMD conv+core decode attempt
  gdn_chunk  gdn_linear_attn.py fi_chunk_gated_delta_rule, before the
             flashinfer chunk_gated_delta_rule launch
  fattn      attention.py unified_attention_with_output entry (2-line anchor)

Usage (in container):
  /opt/venv/bin/python /tmp/patch_p80_event_ladder.py image
             # writes tracer + patches the three in-image modules
(bind mode unchanged from v1; the bind mount is inert on the chain-restored
container -- in-image only.)

Design laws honored:
  - P80_TRACE env-gated; every tracer call is exception-swallowing; capture
    guard (is_current_stream_capturing) so cudagraph capture never embeds
    ladder events.
  - Ladder files are SEPARATE from serve_full.log and opened line-buffered
    with explicit flush -> immune to the SPECTIMING BUFFERED-FLUSH law
    (lines are readable MID-RUN, before worker exit).
  - Idempotent: every inserted line carries the marker 'llm-scaler P80'.
"""
import hashlib
import os
import py_compile
import sys

MARK = "llm-scaler P80"

TRACER = '''"""llm-scaler P80 event ladder (vllm/_p80_tracer.py).

Records an XPU event per instrumented rung on the current stream; a daemon
thread walks the deque head-first completing events and heartbeats to a
line-buffered log. If the head event stops completing for > P80_STALL_SECS
the STALL line names it (the earliest unretired kernel boundary) and dumps
the pending list behind it -- exact wedge attribution.

Env: P80_TRACE=1 to arm, P80_TRACE_PATH (default /root/p80_ladder),
P80_STALL_SECS (default 60). One log per process (pid suffix): both TP
ranks trace independently. Inert (no-op) unless armed.
"""
import collections
import os
import threading
import time

_ON = os.environ.get("P80_TRACE", "") == "1"
_PATH = os.environ.get("P80_TRACE_PATH", "/root/p80_ladder")
_STALL = float(os.environ.get("P80_STALL_SECS", "60"))

_lock = threading.Lock()
_q = collections.deque()
_seq = 0
_f = None
_th = None
_stalled = set()
_done = 0
_last_done = "-"


def _log(s):
    global _f
    try:
        if _f is None:
            _f = open("%s.%d" % (_PATH, os.getpid()), "a", buffering=1)
        _f.write("%.3f %s\\n" % (time.time(), s))
        _f.flush()
    except Exception:
        pass


def _worker():
    global _done, _last_done
    while True:
        time.sleep(0.25)
        with _lock:
            now = time.time()
            n_done = 0
            while _q:
                name, seq, ev, ts = _q[0]
                try:
                    ok = ev.query()
                except Exception:
                    ok = True
                if not ok:
                    if now - ts > _STALL and seq not in _stalled:
                        _stalled.add(seq)
                        pend = ", ".join(
                            "%s#%d" % (p[0], p[1]) for p in list(_q)[:6]
                        )
                        _log("STALL head=%s#%d age=%.1f pend=[%s]"
                             % (name, seq, now - ts, pend))
                    break
                _q.popleft()
                n_done += 1
                _done += 1
                _last_done = "%s#%d" % (name, seq)
            if n_done:
                _log("HB done=%d last=%s pend=%d"
                     % (_done, _last_done, len(_q)))


def p80_ev(name):
    if not _ON:
        return
    try:
        import torch
        try:
            if torch.xpu.is_current_stream_capturing():
                return
        except Exception:
            pass
        global _seq
        ev = torch.xpu.Event()
        ev.record()
        with _lock:
            _seq += 1
            _q.append((name, _seq, ev, time.time()))
            if _th is None:
                _start_thread()
    except Exception:
        pass


def _start_thread():
    global _th
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    _th = t
    _log("TRACER_START pid=%d stall=%.0fs" % (os.getpid(), _STALL))
'''


def _ins_before(src, anchor, lines, tag):
    n = src.count(anchor)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(anchor, lines + anchor, 1)


def _ins_after(src, anchor, lines, tag):
    n = src.count(anchor)
    assert n == 1, "%s: anchor count %d != 1" % (tag, n)
    return src.replace(anchor, anchor + "\n" + lines, 1)


def _rung(indent, name):
    return (
        "%sfrom vllm._p80_tracer import p80_ev  # %s\n"
        '%sp80_ev("%s")  # %s\n' % (indent, MARK, indent, name, MARK)
    )


def patch_file(path, ops):
    src = open(path).read()
    if MARK in src:
        print("SKIP(already) %s" % path)
        return
    for kind, anchor, rung, tag in ops:
        lines = _rung(rung[0], rung[1])
        if kind == "before":
            src = _ins_before(src, anchor, lines, tag)
        else:
            src = _ins_after(src, anchor, lines, tag)
    open(path, "w").write(src)
    cfile = "/tmp/p80c_%d.pyc" % (abs(hash(path)) % 100000)
    py_compile.compile(path, doraise=True, cfile=cfile)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print("PATCHED %s md5 %s" % (path, md5))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "image":
        vllm_dir = "/opt/venv/lib/python3.12/site-packages/vllm"
        tp = os.path.join(vllm_dir, "_p80_tracer.py")
        open(tp, "w").write(TRACER)
        py_compile.compile(tp, doraise=True, cfile="/tmp/p80c_tracer.pyc")
        print("TRACER_WRITTEN %s md5 %s"
              % (tp, hashlib.md5(open(tp, "rb").read()).hexdigest()))
        patch_file(
            os.path.join(vllm_dir,
                         "model_executor/layers/mamba/gdn_linear_attn.py"),
            [
                ("before",
                 "        handled = self._gdn_conv_esimd_ok and "
                 "self._gdn_conv_decode(",
                 ("        ", "gdn_conv"), "gdn_conv"),
                ("before",
                 "            torch.ops.vllm.gdn_attention_core_xpu(",
                 ("            ", "gdn_core"), "gdn_core"),
                ("before",
                 "    result = chunk_gated_delta_rule_fi(",
                 ("    ", "gdn_chunk"), "gdn_chunk"),
            ],
        )
        patch_file(
            os.path.join(vllm_dir, "v1/sample/rejection_sampler.py"),
            [
                # v2 fix: 8-space parse_output body anchor (v1's 4-space
                # form matched a wrong site and needed repair_p80_rejsampler)
                ("before",
                 "        output_token_ids_np = output_token_ids"
                 ".cpu().numpy()",
                 ("        ", "d2h"), "d2h"),
            ],
        )
        patch_file(
            os.path.join(vllm_dir,
                         "model_executor/layers/attention/attention.py"),
            [
                # v2 fix: del kv_cache_dummy_dep 2-line anchor (bare
                # layer_name = _resolve... appears 3x in the file)
                ("after",
                 "    del kv_cache_dummy_dep\n"
                 "    layer_name = _resolve_layer_name(layer_name)",
                 ("    ", "fattn"), "fattn"),
            ],
        )
    elif mode == "bind":
        patch_file(
            "/root/build/p77/code/gpu_model_runner.py",
            [
                ("after",
                 "        ) = self.execute_model_state\n"
                 "        # Clear ephemeral state.\n"
                 "        self.execute_model_state = None",
                 ("        ", "fwd"), "fwd"),
            ],
        )
    else:
        print("usage: patch_p80_event_ladder.py image|bind")
        sys.exit(2)


if __name__ == "__main__":
    main()
