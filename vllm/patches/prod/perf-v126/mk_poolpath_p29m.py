#!/usr/bin/env python3
"""mk_poolpath_p29m.py — build patch_v126_poolpath_p29m.py from the P29C
poolpath patch: MAGNITUDE-MINIMAL diagnostic (v10, supersedes P29H v9 for
the state-magnitude ground-truth leg).

P29H v9 post-mortem (e5m2 lane, 09-29 22:42-22:52) — three lessons:
  1. SNAPSHOT SEMANTICS: the FLUSH dumps are CAPTURE-TIME snapshots.
     ctr/buf/pmX read 0 because the recorded device ops only execute on
     REPLAY, and the FLUSH site (eager prefill gate inside
     _gdn_conv_decode) never fires during serial traffic — pure-prefill
     steps take the prefill path and never enter the decode gate at all.
     saves advanced 1->65 exclusively on capture-ladder mixed dummies,
     then froze for the whole probe. Post-replay values died with the
     wedged worker, undumped.
  2. WEDGE EXTENSION: e5m2+P29H wedged at 22:52:14 with the SAME
     sample_tokens RPC-timeout signature as every fp16+P29H leg — P29H
     convicted on BOTH pool dtypes (diagnostic-only, never ships).
  3. READOUT LAW: never read device state from the forward path —
     FLUSH's float(device-scalar)/torch.save host syncs are
     async-scheduling-hostile, and the gate site is unreachable on
     serial lanes anyway. Readout belongs to a TIMER THREAD.

P29M therefore collects ONLY the magnitude ground truth:
  PRE  (E2C site, owner-gated, one layer): premx = running max of
       |ssm_state[spec rows]| PRE-call (fp8 pool cast to fp32 FIRST —
       abs() on float8 dtypes is not universally supported), bmx =
       running max of |ba|.
  POST (E5 site, same gate): pmx = running max of the same rows
       POST-call (after the native kernel updated them in place).
  TIMER THREAD (daemon, started once at DIAG_INIT): every 20 s dumps
       {t, pid, pmx, premx, bamx} to /root/p29m_mag_<pid>.pt + a TICK
       print — wedge-proof by construction (dumps land before death).
No ring, no reference kernel, no FLUSH, no forward-path host syncs:
the replay overhead is 3 index_select/max ops on one layer.

Activation reuses the P29H marker file /root/.v126_p29h (p29c_boot.sh
P29H=1 branch). Run: P29H=1 POOLPATCH=/root/build/patch_v126_poolpath_p29m.py
  sh p29c_boot.sh fp8_e5m2 8000 fp8_e5m2
then p29_serial_mag.py; harvest /root/p29m_mag_*.pt + TICK lines.
The e5m2-rounding caveat stands: pool values carry ~2 sig bits, which
is sufficient for the RANGE question (does |state| exceed e4m3's 448 /
approach e5m2's 57344?) that the P29I format verdict documents.
"""
import os

SRC = "/root/build/patch_v126_poolpath.py"
DST = "/root/build/patch_v126_poolpath_p29m.py"

DIAG_MIN = (
    '        self._gdn_conv_state_ok = False\n'
    '        # v126 P29M: magnitude-minimal diagnostic. Readout NEVER\n'
    '        # happens on the forward path (P29H v9 lessons: FLUSH host\n'
    '        # syncs are async-scheduling-hostile AND the eager-prefill\n'
    '        # gate is unreachable on serial lanes) — a daemon timer\n'
    '        # thread dumps the running maxes every 20 s instead.\n'
    '        if os.path.exists("/root/.v126_p29h"):\n'
    '            _p29m_g = globals()\n'
    '            if "_g_dn_p29m_pmx" not in _p29m_g:\n'
    '                for _p29m_n in ("pmx", "premx", "bmx"):\n'
    '                    _p29m_g["_g_dn_p29m_" + _p29m_n] = torch.zeros(\n'
    '                        (), dtype=torch.float32, device="xpu")\n'
    '                _p29m_g["_g_dn_p29m_owner"] = id(self)\n'
    '                print("v126 P29M ARMED owner=%s pid=%d" % (\n'
    '                    hex(id(self)), os.getpid()), flush=True)\n'
    '                import threading as _p29m_th\n'
    '                import time as _p29m_time\n'
    '\n'
    '                def _p29m_tick():\n'
    '                    while True:\n'
    '                        _p29m_time.sleep(20)\n'
    '                        try:\n'
    '                            _g = globals()\n'
    '                            _d = {\n'
    '                                "t": _p29m_time.time(),\n'
    '                                "pid": os.getpid(),\n'
    '                                "pmx": float(_g["_g_dn_p29m_pmx"]),\n'
    '                                "premx": float(\n'
    '                                    _g["_g_dn_p29m_premx"]),\n'
    '                                "bamx": float(_g["_g_dn_p29m_bmx"]),\n'
    '                            }\n'
    '                            torch.save(\n'
    '                                _d,\n'
    '                                "/root/p29m_mag_%d.pt" % os.getpid())\n'
    '                            print("v126 P29M TICK pmx=%.2f premx=%.2f "\n'
    '                                  "bamx=%.2f" % (\n'
    '                                      _d["pmx"], _d["premx"],\n'
    '                                      _d["bamx"]), flush=True)\n'
    '                        except Exception:\n'
    '                            pass\n'
    '\n'
    '                _p29m_t = _p29m_th.Thread(\n'
    '                    target=_p29m_tick, daemon=True)\n'
    '                _p29m_t.start()\n')

PRE_MIN = (
    '            _p29m_g = globals()\n'
    '            _p29m_on = _p29m_g.get("_g_dn_p29m_owner", 0) == id(self)\n'
    '            if _p29m_on:\n'
    '                _p29m_rows = (\n'
    '                    spec_state_indices[:num_spec_decodes, '
    ':num_spec_tokens]\n'
    '                    .reshape(-1).to(torch.int64).clamp_min(0))\n'
    # cast the fp8 pool to fp32 BEFORE abs() — pointwise abs() on
    # float8_e5m2 tensors is not universally supported on this build
    '                _p29m_p = torch.index_select(\n'
    '                    ssm_state, 0, _p29m_rows).to(torch.float32)\n'
    '                _p29m_g["_g_dn_p29m_premx"].copy_(torch.maximum(\n'
    '                    _p29m_g["_g_dn_p29m_premx"],\n'
    '                    _p29m_p.abs().max()))\n'
    '                _p29m_b = projected_states_ba[\n'
    '                    :num_spec_decodes * num_spec_tokens]\n'
    '                _p29m_g["_g_dn_p29m_bmx"].copy_(torch.maximum(\n'
    '                    _p29m_g["_g_dn_p29m_bmx"],\n'
    '                    _p29m_b.to(torch.float32).abs().max()))\n')

POST_MIN = (
    '            if _p29m_on:\n'
    '                _p29m_pp = torch.index_select(\n'
    '                    ssm_state, 0, _p29m_rows).to(torch.float32)\n'
    '                _p29m_g["_g_dn_p29m_pmx"].copy_(torch.maximum(\n'
    '                    _p29m_g["_g_dn_p29m_pmx"],\n'
    '                    _p29m_pp.abs().max()))\n')

CALL_TAIL_OLD = (
    '                num_accepted_tokens[:num_spec_decodes],\n'
    '                num_spec_decodes,\n'
    '                num_spec_tokens,\n'
    '                self.num_k_heads // self.tp_size,\n'
    '                self.num_v_heads // self.tp_size,\n'
    '                self.head_k_dim,\n'
    '                self.head_v_dim,\n'
    '                self._gdn_attn_scale,\n'
    '            )\n'
    '            return True')
CALL_TAIL_NEW = (
    '                num_accepted_tokens[:num_spec_decodes],\n'
    '                num_spec_decodes,\n'
    '                num_spec_tokens,\n'
    '                self.num_k_heads // self.tp_size,\n'
    '                self.num_v_heads // self.tp_size,\n'
    '                self.head_k_dim,\n'
    '                self.head_v_dim,\n'
    '                self._gdn_attn_scale,\n'
    '            )\n' + POST_MIN + '            return True')


def triple(s):
    return "'''" + s + "'''"


E5_DEF = (
    "E5M_OLD = " + triple(CALL_TAIL_OLD) + "\n\n"
    "E5M_NEW = " + triple(CALL_TAIL_NEW) + "\n\n")

with open(SRC) as f:
    txt = f.read()

# 1) DIAG_MIN into the E1_NEW literal (before its closing quotes)
a1 = '        self._gdn_conv_state_ok = False"""'
if txt.count(a1) != 1:
    raise SystemExit("E1 anchor count %d != 1" % txt.count(a1))
txt = txt.replace(a1, DIAG_MIN + '"""')

# 2) PRE_MIN into the E2C_NEW literal (between the one-shot marker print
#    and the native call — anchored on the print tail so E2C_OLD can't
#    match)
a2 = ('                      % ssm_state.dtype, flush=True)\n'
      '            esimd_gdn_conv_fused_seq_spec(\n'
      '                projected_states_qkvz,"""')
if txt.count(a2) != 1:
    raise SystemExit("E2C anchor count %d != 1" % txt.count(a2))
txt = txt.replace(
    a2,
    '                      % ssm_state.dtype, flush=True)\n' + PRE_MIN +
    '            esimd_gdn_conv_fused_seq_spec(\n'
    '                projected_states_qkvz,"""')

# 3) define the E5 pair right before the EDITS list
a3 = "\nEDITS = ["
if txt.count(a3) != 1:
    raise SystemExit("EDITS anchor count %d != 1" % txt.count(a3))
txt = txt.replace(a3, "\n" + E5_DEF + "\nEDITS = [")

# 4) extend the EDITS list with the POST pair
a4 = '    ("E3 state gate", E3_OLD, E3_NEW),\n]'
if txt.count(a4) != 1:
    raise SystemExit("EDITS list anchor count %d != 1" % txt.count(a4))
txt = txt.replace(
    a4,
    '    ("E3 state gate", E3_OLD, E3_NEW),\n'
    '    ("E5 p29m post", E5M_OLD, E5M_NEW),\n'
    ']')

with open(DST, "w") as f:
    f.write(txt)
print("wrote", DST)
print("markers P29C:", txt.count("v126 P29C"))
print("markers P29M:", txt.count("v126 P29M"))
print("P29H leftovers (must be 0):", txt.count("P29H"))
print("dot/norm in metrics:", txt.count("torch.dot") + txt.count(".norm("))
print("float( outside thread (eyeball):", txt.count("float("))
print("size:", os.path.getsize(os.path.devnull) and len(txt))
