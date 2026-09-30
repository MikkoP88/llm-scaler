#!/usr/bin/env python3
"""mk_poolpath_p29h.py — build patch_v126_poolpath_p29h.py from the P29C
poolpath patch (ORIGINAL any-nsd posture): adds the P29H comprehensive
in-pipeline telemetry (v7: capture-point expansion per user directive
"add comprehensive telemetry capture points to locate issues from
pipelines and etc").

P29H (marker file /root/.v126_p29h, diagnostic lanes only). Capture
points, all following the capture-legality law below:
  T1 ARMED print   — DIAG_INIT ran; owner/pid (answers: did the marker
                     file reach worker __init__ at all?)
  T2 CALL# trace   — every EAGER _gdn_conv_decode entry: self/owner ids
                     + match + npref/ndec/spec. Graph replays never
                     print, so npref>0 lines prove prefill routing;
                     match=False convicts the owner guard.
  T3 SPEC trace    — first spec-branch executions (incl. the capture
                     pass): nsd/nst/dtypes (host values only).
  T4 metrics ring  — in-graph per-replay 8-col trajectory on the layer-0
                     native call vs bridge reference:
                     [maxdiff_out, maxdiff_pool, maxdiff_conv, nan_out,
                     nan_pool, nan_conv, qkvz_absmax, nsd]
  T5 FLUSH + dump  — eager-prefill-only torch.save of the ring +
                     self-describing meta (pid, pool/conv dtypes); the
                     first divergent slot = convicted stage, nsd column
                     = convicted graph size.
  v9 running maxes — pmx/premx/bmx scalars (touched-rows pool absmax
                     post/pre-call, |ba|), replay-updated via in-place
                     copy_(maximum(..)); flush prints + saves them =
                     the REAL state-magnitude ground truth per lane.

The layer-0
GDN spec call is mirrored by a bridge reference on IDENTICAL live inputs
(fresh fp16 snapshot of the touched pool rows + conv rows + input
clones); per-call metrics accumulate in persistent eagerly-allocated
device tensors (pure device ops -> recorded by graph capture, updated on
every replay): 8 columns (see T4 above).

Activation is a FILE, not an env var: the serve process' env (402) had
V126_P29H=1 while the spawned EngineCore (675) and TP workers (874/880)
did NOT — plain env inheritance between vLLM v1 process tiers is broken
on this build (root cause not run down; not worth it). A marker file is
visible to every process in the container, no propagation needed.

Capture-legality law (learned v1-v3, all boots died in capture):
  - host syncs inside capture are ILLEGAL (.item()/.cpu() -> "wait method
    cannot be used for an event associated with a command graph")
  - torch.dot / .norm() inside capture are ILLEGAL on this XPU backend
    (reduction dispatch does an event wait); abs().max() reductions are
    PROVEN legal -> all metrics are max-based, no dot/norm/sum
  - cross-device D2H inside capture is avoided entirely: the flush reads
    the device buffer ONLY inside the eager prefill branch, which decode-
    graph capture dummies never enter
  - pin_memory under the worker XPU device context needs explicit
    device="cpu" (v2 lesson; moot now — no pinned tensors remain)
  - FUNCTION-SCOPE law (v7 boot death): at the prefill-gate site only
    module globals and self attrs are referenceable — spec-branch locals
    (ssm_state, conv_state, ...) are UNBOUND there (UnboundLocalError).
    Gate-site code must stash anything it needs into globals from the
    spec branch first (PRE stashes the dtype strings).
"""
SRC = "/root/build/patch_v126_poolpath.py"
DST = "/root/build/patch_v126_poolpath_p29h.py"

DIAG_INIT = (
    '        self._gdn_conv_state_ok = False\n'
    '        # v126 P29H: in-situ native-vs-bridge diagnostic buffers\n'
    '        # (persistent, eagerly allocated; updates are pure device ops\n'
    '        # recorded by graph capture — max-reductions only, no dot/norm\n'
    '        # (event-wait reductions are capture-illegal), no host syncs).\n'
    '        if os.path.exists("/root/.v126_p29h"):\n'
    '            _p29h_g = globals()\n'
    '            if "_g_dn_p29h_buf" not in _p29h_g:\n'
    '                _p29h_g["_g_dn_p29h_buf"] = torch.zeros(\n'
    '                    8192, 8, dtype=torch.float32, device="xpu")\n'
    '                _p29h_g["_g_dn_p29h_ctr"] = torch.zeros(\n'
    '                    (), dtype=torch.int64, device="xpu")\n'
    '                _p29h_g["_g_dn_p29h_saves"] = 0\n'
    # v9: running-max scalars — updated in-place via copy_(maximum(..))
    # so graph replays keep updating the SAME buffer; abs().max() is the
    # proven capture-legal reduction. pmx = touched-rows pool absmax
    # POST-call, premx = PRE-call, bmx = |ba|. Read only at flush
    # (eager) — the REAL fp16-lane state magnitude ground truth.
    '                _p29h_g["_g_dn_p29h_pmx"] = torch.zeros(\n'
    '                    (), dtype=torch.float32, device="xpu")\n'
    '                _p29h_g["_g_dn_p29h_premx"] = torch.zeros(\n'
    '                    (), dtype=torch.float32, device="xpu")\n'
    '                _p29h_g["_g_dn_p29h_bmx"] = torch.zeros(\n'
    '                    (), dtype=torch.float32, device="xpu")\n'
    '                _p29h_g["_g_dn_p29h_owner"] = id(self)\n'
    '                print("v126 P29H ARMED owner=%s pid=%d" % (\n'
    '                    hex(id(self)), os.getpid()), flush=True)')

PRE = (
    '            _p29h_g = globals()\n'
    '            _p29h_on = _p29h_g.get("_g_dn_p29h_owner", 0) == id(self)\n'
    '            if _p29h_on:\n'
    '                _p29h_n = num_spec_decodes * num_spec_tokens\n'
    '                _p29h_rows = (\n'
    '                    spec_state_indices[:num_spec_decodes, '
    ':num_spec_tokens]\n'
    '                    .reshape(-1).to(torch.int64).clamp_min(0))\n'
    '                _p29h_pool_h = torch.index_select(\n'
    '                    ssm_state, 0, _p29h_rows).to(torch.float16)'
    '.contiguous()\n'
    '                _p29h_conv_h = torch.index_select(\n'
    '                    conv_state, 0, _p29h_rows).contiguous()\n'
    '                _p29h_qkvz = projected_states_qkvz[:_p29h_n].clone()\n'
    '                _p29h_ba = projected_states_ba[:_p29h_n].clone()\n'
    '                _p29h_idx = spec_state_indices[\n'
    '                    :num_spec_decodes, :num_spec_tokens].contiguous()\n'
    '                _p29h_tok = spec_token_indx[:_p29h_n].contiguous()\n'
    '                _p29h_acc = num_accepted_tokens[\n'
    '                    :num_spec_decodes].contiguous()\n'
    '                _p29h_qabs = projected_states_qkvz[:_p29h_n].abs().max()\n'
    # v9 running maxes (recorded device ops — replay-updated, same law
    # as the ring: in-place copy_ into the persistent scalar)
    '                _p29h_g["_g_dn_p29h_premx"].copy_(torch.maximum(\n'
    '                    _p29h_g["_g_dn_p29h_premx"],\n'
    '                    _p29h_pool_h.abs().max().to(torch.float32)))\n'
    '                _p29h_g["_g_dn_p29h_bmx"].copy_(torch.maximum(\n'
    '                    _p29h_g["_g_dn_p29h_bmx"],\n'
    '                    _p29h_ba.abs().max().to(torch.float32)))\n'
    # stash dtypes as module globals HERE (spec branch, states bound) —
    # the FLUSH site (prefill gate) CANNOT reference ssm_state/conv_state:
    # v7 died there with UnboundLocalError (states are locals bound only
    # deeper in the function). Gate-site code may only use globals/attrs.
    '                _p29h_g["_g_dn_p29h_pdt"] = str(ssm_state.dtype)\n'
    '                _p29h_g["_g_dn_p29h_cdt"] = str(conv_state.dtype)\n'
    # T3 SPEC trace: host values only (nsd/nst/dtypes) — prints during
    # the capture pass and eager spec, never on replay
    '                _p29h_sp = _p29h_g.get("_g_dn_p29h_spass", 0)\n'
    '                if _p29h_sp < 8:\n'
    '                    _p29h_g["_g_dn_p29h_spass"] = _p29h_sp + 1\n'
    '                    print("v126 P29H SPEC#%d nsd=%d nst=%d pool=%s "\n'
    '                          "conv=%s qkvz=%s pid=%d"\n'
    '                          % (_p29h_sp, num_spec_decodes,\n'
    '                             num_spec_tokens, ssm_state.dtype,\n'
    '                             conv_state.dtype,\n'
    '                             projected_states_qkvz.dtype, os.getpid()),\n'
    '                          flush=True)\n')

POST = (
    '            if _p29h_on:\n'
    '                _p29h_hv = self.num_v_heads // self.tp_size\n'
    '                _p29h_out = torch.zeros(\n'
    '                    _p29h_n, _p29h_hv, self.head_v_dim,\n'
    '                    dtype=torch.float16, device=core_attn_out.device)\n'
    '                _p29h_z = torch.zeros_like(_p29h_out)\n'
    '                esimd_gdn_conv_fused_seq_spec(\n'
    '                    _p29h_qkvz, _p29h_conv_h, self._gdn_conv_weight_2d,\n'
    '                    self._gdn_conv_bias_zeros, _p29h_idx,\n'
    '                    self._gdn_A_log_fp16, self._gdn_dt_bias_fp16, '
    '_p29h_ba,\n'
    '                    _p29h_pool_h, _p29h_out, _p29h_z, _p29h_tok, '
    '_p29h_acc,\n'
    '                    num_spec_decodes, num_spec_tokens,\n'
    '                    self.num_k_heads // self.tp_size, _p29h_hv,\n'
    '                    self.head_k_dim, self.head_v_dim, '
    'self._gdn_attn_scale)\n'
    '                _a = core_attn_out[:_p29h_n].to(torch.float32)\n'
    '                _b = _p29h_out.to(torch.float32)\n'
    '                _md_o = (_a - _b).abs().max()\n'
    '                _pp = torch.index_select(\n'
    '                    ssm_state, 0, _p29h_rows).to(torch.float16)\n'
    '                _md_p = (_pp.to(torch.float32)\n'
    '                         - _p29h_pool_h.to(torch.float32)).abs().max()\n'
    '                _cc = torch.index_select(\n'
    '                    conv_state, 0, _p29h_rows)\n'
    '                _md_c = (_cc.to(torch.float32)\n'
    '                         - _p29h_conv_h.to(torch.float32)).abs().max()\n'
    '                _bad_o = (torch.isnan(_a) | torch.isinf(_a)).to(\n'
    '                    torch.float16).max()\n'
    '                _bad_p = (torch.isnan(_pp) | torch.isinf(_pp)).to(\n'
    '                    torch.float16).max()\n'
    '                _bad_c = (torch.isnan(_cc) | torch.isinf(_cc)).to(\n'
    '                    torch.float16).max()\n'
    '                _p29h_g["_g_dn_p29h_pmx"].copy_(torch.maximum(\n'
    '                    _p29h_g["_g_dn_p29h_pmx"],\n'
    '                    _pp.abs().max().to(torch.float32)))\n'
    '                _vals = torch.stack([\n'
    '                    _md_o.to(torch.float32), _md_p.to(torch.float32),\n'
    '                    _md_c.to(torch.float32),\n'
    '                    _bad_o.to(torch.float32), _bad_p.to(torch.float32),\n'
    '                    _bad_c.to(torch.float32),\n'
    '                    _p29h_qabs.to(torch.float32),\n'
    '                    torch.full((), float(num_spec_decodes),\n'
    '                               dtype=torch.float32,\n'
    '                               device=core_attn_out.device)])\n'
    '                _p29h_g["_g_dn_p29h_buf"].index_copy_(\n'
    '                    0, (_p29h_g["_g_dn_p29h_ctr"] % 8192).reshape(1),\n'
    '                    _vals.reshape(1, 8))\n'
    '                _p29h_g["_g_dn_p29h_ctr"] += 1\n')

FLUSH = (
    '            # v126 P29H: flush the diagnostic trajectory. This branch\n'
    '            # runs ONLY on eager prefill/mixed batches — decode-graph\n'
    '            # capture dummies take the spec branch below — so reading\n'
    '            # the device buffer here NEVER happens inside capture.\n'
    '            _p29h_g = globals()\n'
    '            if _p29h_g.get("_g_dn_p29h_owner", 0) == id(self):\n'
    '                _p29h_g["_g_dn_p29h_saves"] = (\n'
    '                    _p29h_g.get("_g_dn_p29h_saves", 0) + 1)\n'
    '                if _p29h_g["_g_dn_p29h_saves"] % 8 == 1:\n'
    '                    torch.save(\n'
    '                        {"saves": _p29h_g["_g_dn_p29h_saves"],\n'
    '                         "ctr": int(_p29h_g["_g_dn_p29h_ctr"]),\n'
    '                         "buf": _p29h_g["_g_dn_p29h_buf"].cpu().clone(),\n'
    '                         "meta": {\n'
    '                             "pid": os.getpid(),\n'
    '                             "pool_dtype": _p29h_g.get(\n'
    '                                 "_g_dn_p29h_pdt", "?"),\n'
    '                             "conv_dtype": _p29h_g.get(\n'
    '                                 "_g_dn_p29h_cdt", "?"),\n'
    '                             "cols": ["md_out", "md_pool", "md_conv",\n'
    '                                      "nan_out", "nan_pool", "nan_conv",\n'
    '                                      "qkvz_absmax", "nsd"],\n'
    '                             "pool_absmax_runmax": float(\n'
    '                                 _p29h_g["_g_dn_p29h_pmx"]),\n'
    '                             "pool_pre_runmax": float(\n'
    '                                 _p29h_g["_g_dn_p29h_premx"]),\n'
    '                             "ba_runmax": float(\n'
    '                                 _p29h_g["_g_dn_p29h_bmx"])}},\n'
    '                        "/root/p29h_diag_%d.pt" % os.getpid())\n'
    '                    print("v126 P29H FLUSH saves=%d ctr=%d pid=%d "\n'
    '                          "pmx=%.1f premx=%.1f bamx=%.1f" % (\n'
    '                        _p29h_g["_g_dn_p29h_saves"],\n'
    '                        int(_p29h_g["_g_dn_p29h_ctr"]), os.getpid(),\n'
    '                        float(_p29h_g["_g_dn_p29h_pmx"]),\n'
    '                        float(_p29h_g["_g_dn_p29h_premx"]),\n'
    '                        float(_p29h_g["_g_dn_p29h_bmx"])),\n'
    '                        flush=True)\n')

DBG = (
    '        # v126 P29H: call trace. Graph replays re-execute recorded\n'
    '        # device ops but NEVER this Python — so a line here = an EAGER\n'
    '        # call through _gdn_conv_decode (capture pass, prefill, mixed).\n'
    '        if "_g_dn_p29h_buf" in globals():\n'
    '            _p29h_dbg_g = globals()\n'
    '            _p29h_np = getattr(attn_metadata, "num_prefills", 0)\n'
    '            _p29h_c = _p29h_dbg_g.get("_g_dn_p29h_calls", 0)\n'
    '            _p29h_pc = _p29h_dbg_g.get("_g_dn_p29h_pcall", 0)\n'
    '            if _p29h_c < 4 or (_p29h_np and _p29h_pc < 12):\n'
    '                _p29h_dbg_g["_g_dn_p29h_calls"] = _p29h_c + 1\n'
    '                if _p29h_np:\n'
    '                    _p29h_dbg_g["_g_dn_p29h_pcall"] = _p29h_pc + 1\n'
    '                print("v126 P29H CALL#%d self=%s own=%s match=%s "\n'
    '                      "npref=%s ndec=%s spec=%s"\n'
    '                      % (_p29h_c, hex(id(self)),\n'
    '                         hex(_p29h_dbg_g.get("_g_dn_p29h_owner", 0)),\n'
    '                         _p29h_dbg_g.get("_g_dn_p29h_owner", 0)\n'
    '                         == id(self),\n'
    '                         _p29h_np,\n'
    '                         getattr(attn_metadata, "num_decodes", "?"),\n'
    '                         getattr(attn_metadata,\n'
    '                                 "spec_sequence_masks", None)\n'
    '                         is not None), flush=True)\n')

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
    '            )\n' + POST + '            return True')

GATE_OLD = (
    '        if (\n'
    '            attn_metadata.num_prefills != 0\n'
    '            or (\n'
    '                not is_spec_batch\n'
    '                and (\n'
    '                    attn_metadata.num_decodes <= 0\n'
    '                    or attn_metadata.num_decodes > 128\n'
    '                )\n'
    '            )\n'
    '        ):\n'
    '            return False')
GATE_NEW = (
    DBG +
    '        if (\n'
    '            attn_metadata.num_prefills != 0\n'
    '            or (\n'
    '                not is_spec_batch\n'
    '                and (\n'
    '                    attn_metadata.num_decodes <= 0\n'
    '                    or attn_metadata.num_decodes > 128\n'
    '                )\n'
    '            )\n'
    '        ):\n' + FLUSH + '            return False')


def triple(s):
    return "'''" + s + "'''"


E5E6_DEF = (
    "E5_OLD = " + triple(CALL_TAIL_OLD) + "\n\n"
    "E5_NEW = " + triple(CALL_TAIL_NEW) + "\n\n"
    "E6_OLD = " + triple(GATE_OLD) + "\n\n"
    "E6_NEW = " + triple(GATE_NEW) + "\n\n")

with open(SRC) as f:
    txt = f.read()

# 1) E1_NEW literal: append diag init before its closing quotes
a1 = '        self._gdn_conv_state_ok = False"""'
if txt.count(a1) != 1:
    raise SystemExit("E1 anchor count %d != 1" % txt.count(a1))
txt = txt.replace(a1, DIAG_INIT + '"""')

# 2) E2C_NEW literal: insert PRE block between the one-shot print and the
#    call (anchored on the print tail so E2C_OLD cannot match)
a2 = ('                      % ssm_state.dtype, flush=True)\n'
      '            esimd_gdn_conv_fused_seq_spec(\n'
      '                projected_states_qkvz,"""')
if txt.count(a2) != 1:
    raise SystemExit("E2C anchor count %d != 1" % txt.count(a2))
txt = txt.replace(
    a2,
    '                      % ssm_state.dtype, flush=True)\n' + PRE +
    '            esimd_gdn_conv_fused_seq_spec(\n'
    '                projected_states_qkvz,"""')

# 3) DBG (T2 CALL trace) rides GATE_NEW below — inserted at the head of
#    the gate block inside _gdn_conv_decode, after the metadata lookup,
#    which the gate itself dereferences. No separate anchor needed.

# 4) define E5/E6 edit pairs right before the EDITS list
a3 = "\nEDITS = ["
if txt.count(a3) != 1:
    raise SystemExit("EDITS anchor count %d != 1" % txt.count(a3))
txt = txt.replace(a3, "\n" + E5E6_DEF + "\nEDITS = [")

# 4) extend the EDITS list with the new pairs
a4 = '    ("E3 state gate", E3_OLD, E3_NEW),\n]'
if txt.count(a4) != 1:
    raise SystemExit("EDITS list anchor count %d != 1" % txt.count(a4))
txt = txt.replace(
    a4,
    '    ("E3 state gate", E3_OLD, E3_NEW),\n'
    '    ("E5 p29h post", E5_OLD, E5_NEW),\n'
    '    ("E6 p29h flush", E6_OLD, E6_NEW),\n'
    ']')

with open(DST, "w") as f:
    f.write(txt)
print("wrote", DST)
print("markers P29C:", txt.count("v126 P29C"))
print("markers P29H:", txt.count("v126 P29H"))
print("dot/norm in metrics:", txt.count("torch.dot") + txt.count(".norm("))
print("pin_memory left:", txt.count("pin_memory"))
