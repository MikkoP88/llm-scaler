#!/usr/bin/env python3
"""patch_perreq_v128.py — v128 WS-D D4 per-request split telemetry.

The D4 aggregate measurement (P37: recompute excess 24.7% of computed,
>= 15% escalation bar) cannot attribute the excess between the two
candidate causes:
  - GDN-snap recompute: the hybrid cache manager snaps prefix-cache hits
    to the mamba/GDN state boundary (4096-token granularity), so a
    full-KV-hit multi-turn request still FIRES up to ~4k already-seen
    tokens (they land in local_compute);
  - eviction re-prefill: a request whose prefix was evicted between
    turns re-fires the whole miss span (also local_compute).

Per-request (prompt_len, cached, computed=prompt-cached) splits them:
full-hit cohort (hit_ratio high) -> computed ~= new suffix + GDN snap,
and the snap shows as a mod-4096 signature; low-hit cohort -> eviction.

Patch site: OutputProcessor._update_stats_from_finished — runs once per
finished request in the FRONTEND process, where RequestState already
carries prompt_len, num_cached_tokens (set from engine prefill_stats at
first-token time), stats.num_generation_tokens and
stats.first_token_latency.

DORMANT unless /root/.v128_perreq exists (marker-file law: env vars do
not reliably reach every process in the TP/multiproc tree; the m0live
collector uses the same law). When armed, appends one JSON line per
finished request to /root/v128_perreq.jsonl. The write path is wrapped
so telemetry can never break the output path (standing lesson).
"""
import sys

P = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/"
     "output_processor.py")
MARKER = "llm-scaler v128 PERREQ"

with open(P) as f:
    t = f.read()

if MARKER in t:
    print("patch_perreq_v128: already applied, no-op")
    sys.exit(0)

# --- anchor 1: module-level handle slots (must exist BEFORE the first
# read; `global` alone does not initialize a missing module attribute)
anchor_mod = "EMPTY_CPU_TENSOR = torch.empty(0, device=\"cpu\")\n"
assert t.count(anchor_mod) == 1, "module anchor count=%d" % t.count(anchor_mod)
mod_block = (
    anchor_mod
    + "\n# llm-scaler v128 PERREQ: lazy file handle for the D4 per-request\n"
    "# split telemetry (dormant unless /root/.v128_perreq — marker-file law)\n"
    "_v128_perreq_fh = None\n"
    "_v128_perreq_next_check = 0.0\n"
)
t = t.replace(anchor_mod, mod_block)

# --- anchor 2: method head through the asserts (unique; full-line match
# through end of block so nothing can split the insertion mid-block)
anchor = (
    "    def _update_stats_from_finished(\n"
    "        self,\n"
    "        req_state: RequestState,\n"
    "        finish_reason: FinishReason | None,\n"
    "        iteration_stats: IterationStats | None,\n"
    "    ):\n"
    "        if iteration_stats is None:\n"
    "            return\n"
    "\n"
    "        assert finish_reason is not None\n"
    "        assert req_state.stats is not None\n"
)
assert t.count(anchor) == 1, "method anchor count=%d" % t.count(anchor)

block = '''
        # llm-scaler v128 PERREQ: D4 per-request split telemetry (dormant
        # unless /root/.v128_perreq; wrapped so it can never raise into
        # the output path). One JSON line per finished request.
        try:
            import json as _json
            import os as _os
            import time as _time
            global _v128_perreq_fh, _v128_perreq_next_check
            _now = _time.time()
            if _v128_perreq_fh is None and _now >= _v128_perreq_next_check:
                _v128_perreq_next_check = _now + 5.0
                if _os.path.exists("/root/.v128_perreq"):
                    _v128_perreq_fh = open("/root/v128_perreq.jsonl", "a",
                                           buffering=1)
                    _v128_perreq_fh.write("# perreq armed %s\\n"
                                          % _time.strftime("%Y-%m-%dT%H:%M:%S"))
            if _v128_perreq_fh is not None:
                _plen = int(req_state.prompt_len or 0)
                _cached = int(req_state.num_cached_tokens or 0)
                _v128_perreq_fh.write(_json.dumps({
                    "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "rid": req_state.external_req_id or req_state.request_id,
                    "prompt": _plen,
                    "cached": _cached,
                    "computed": max(_plen - _cached, 0),
                    "generated": int(req_state.stats.num_generation_tokens or 0),
                    "hit_ratio": round(_cached / _plen, 4) if _plen else 0.0,
                    "finish": getattr(finish_reason, "name", str(finish_reason)),
                    "ttft": round(req_state.stats.first_token_latency or 0.0, 3),
                }) + "\\n")
        except Exception:
            pass

'''
t = t.replace(anchor, anchor + block)

with open(P, "w") as f:
    f.write(t)

# verify: reread from disk (the file the engine will actually import)
with open(P) as f:
    t2 = f.read()
print("patch_perreq_v128 applied")
print("marker count:", t2.count(MARKER))
print("module slots:", t2.count("_v128_perreq_fh = None"))
print("py-compile check:", end=" ")
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print("OK")
except py_compile.PyCompileError as e:
    print("FAIL", e)
    sys.exit(1)
