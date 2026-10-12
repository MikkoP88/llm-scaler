#!/usr/bin/env python3
"""FN8_05 (DEC40 D7): static-shape, sync-free PLE spec path + telemetry
capture gate.  [surgical revision — dash-free anchors]

D4 py-spy verdict (pyspy_d4_tp0/tp1.raw, 2026-10-12, 5000 samples each):
the 250 ms eager step is HOST-bound —
  * 13-14% PLE path (this module; _ple_forward_spec runs up to FOUR device
           syncs per layer per step: int(sq[-1].item()), lens.max().item(),
           torch.all(his).item(), fresh.any().item() — 12 PLE layers
           = up to 48 syncs/step)
Spec verify widths are UNIFORM k+1 per seq (gdn_attn build() law), so the
spec group needs no dynamic padding at all: static gather -> view -> core
-> scatter, zero syncs, capture-safe (required for the D8 graphs leg).
The non-spec group (mixed prefill-joined steps, eager-only) is untouched.

Edits (qwen4_exp.py):
  E1a docstring marker 04g -> 04g/05 (idempotency key)
  E1b drop the int(sq[-1].item()) sync at method head
  E1c spec-group body: static rewrite (S*L uniform law, view instead of
      scatter-padding, torch.full lens, static j-loop bound L)
  E2a _FN8_XPU_CAPTURING helper next to the diag-seen set
  E2b _ple_telemetry isfinite bool() sync gated on capture state

Idempotent; py_compile gate; backup .fn8_05.bak; EOL aware; anchor==1 gate.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_05"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "DEC40 D1 (04g/05)"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1a: docstring marker ---------------------------------------------------
OLD_A_LF = '''        """DEC40 D1 (04g): PLE under MTP spec verify.
'''
NEW_A_LF = '''        """DEC40 D1 (04g/05): PLE under MTP spec verify — static spec group.
'''

# --- E1b: drop head sync ------------------------------------------------------
OLD_B_LF = '''        n_spec_toks = int(sq[-1].item())
        T = hidden_states.shape[0]
'''
NEW_B_LF = '''        T = hidden_states.shape[0]
'''

# --- E1c: spec-group body, static rewrite -------------------------------------
OLD_C_LF = '''        S = sq.shape[0] - 1
        if S > 0 and n_spec_toks > 0:
            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None:
                s_indx = torch.arange(n_spec_toks, device=device)
            s_indx = s_indx.to(torch.long)
            hs = hidden_states[s_indx]
            ids = input_ids[s_indx]
            lens = sq[1:] - sq[:-1]
            max_len = int(lens.max().item())
            if s_slots.shape[1] < max_len:
                raise RuntimeError(
                    f"Qwen4Exp PLE spec: {max_len} queries but "
                    f"{s_slots.shape[1]} state slots"
                )
            tok_pos = torch.arange(n_spec_toks, device=device, dtype=torch.long)
            req_of_tok = torch.searchsorted(sq[1:], tok_pos, right=True)
            local_of_tok = tok_pos - sq[req_of_tok]
            hidden_pad = hs.new_zeros(S, max_len, hs.shape[-1])
            hidden_pad[req_of_tok, local_of_tok] = hs
            ids_pad = torch.full((S, max_len), eos, dtype=torch.long, device=device)
            ids_pad[req_of_tok, local_of_tok] = ids

            prev_ctx = hist_pool[s_slots[:, 0], :ctx_len, 0].clone()
            conv_state = conv_pool[s_slots[:, 0], :S_len].clone()
            token_history = torch.cat([prev_ctx, ids_pad], dim=-1)
            _t0 = _perf()
            out_pad, _, _ = ple(hidden_pad, token_history, lens, conv_state)
            _ple_telemetry(True, S, max_len, n_spec_toks, _perf() - _t0, out_pad)
            outs[s_indx] = out_pad[req_of_tok, local_of_tok]

            # per-position state scatter: slot j = state AFTER token j
            seq_all = torch.cat([conv_state, hidden_pad], dim=1)  # (S,S_len+L,D)
            tok_all = torch.cat([prev_ctx, ids_pad], dim=1)       # (S,ctx+L)
            for j in range(max_len):
                conv_pool[s_slots[:, j], :S_len] = seq_all[:, j + 1 : j + 1 + S_len]
                hist_pool[s_slots[:, j], :ctx_len, 0] = tok_all[
                    :, j + 1 : j + 1 + ctx_len
                ]
'''
NEW_C_LF = '''        S = sq.shape[0] - 1
        L = s_slots.shape[1]                 # static k+1 (shape-only python)
        n_spec_toks = S * L if S > 0 else 0  # 05: uniform-width law (gdn build)
        if S > 0:
            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None:
                s_indx = torch.arange(n_spec_toks, device=device)
            s_indx = s_indx.to(torch.long)
            if s_indx.shape[0] != n_spec_toks:
                raise RuntimeError(
                    "Qwen4Exp PLE spec: non-uniform verify widths "
                    f"({s_indx.shape[0]} tokens vs S*L={n_spec_toks})"
                )
            D = hidden_states.shape[-1]
            # uniform k+1 per seq: physical order IS (S, L) — static view,
            # no scatter-padding, no searchsorted, no .item() syncs
            hs_pad = hidden_states[s_indx].view(S, L, D)
            ids_pad = input_ids[s_indx].view(S, L)
            lens = torch.full((S,), L, dtype=torch.long, device=device)

            prev_ctx = hist_pool[s_slots[:, 0], :ctx_len, 0].clone()
            conv_state = conv_pool[s_slots[:, 0], :S_len].clone()
            token_history = torch.cat([prev_ctx, ids_pad], dim=-1)
            _t0 = _perf()
            out_pad, _, _ = ple(hs_pad, token_history, lens, conv_state)
            _ple_telemetry(True, S, L, n_spec_toks, _perf() - _t0, out_pad)
            outs[s_indx] = out_pad.reshape(S * L, D)

            # per-position state scatter: slot j = state AFTER token j
            seq_all = torch.cat([conv_state, hs_pad], dim=1)  # (S,S_len+L,D)
            tok_all = torch.cat([prev_ctx, ids_pad], dim=1)   # (S,ctx+L)
            for j in range(L):
                conv_pool[s_slots[:, j], :S_len] = seq_all[:, j + 1 : j + 1 + S_len]
                hist_pool[s_slots[:, j], :ctx_len, 0] = tok_all[
                    :, j + 1 : j + 1 + ctx_len
                ]
'''

# --- E2a: capturing helper -----------------------------------------------------
OLD_D_LF = '''_FN8_PLE_DIAG_SEEN = set()
'''
NEW_D_LF = '''_FN8_PLE_DIAG_SEEN = set()
_FN8_XPU_CAPTURING = getattr(torch.xpu, "is_current_stream_capturing", None)
'''

# --- E2b: telemetry isfinite sync gated on capture ------------------------------
OLD_E_LF = '''    if w["calls"] >= 200:
        try:
            if not bool(out_pad.isfinite().all()):
'''
NEW_E_LF = '''    if w["calls"] >= 200:
        # 05: bool(isfinite().all()) syncs the device — illegal under
        # command-graph capture; reset the window and skip while capturing.
        if _FN8_XPU_CAPTURING is not None and _FN8_XPU_CAPTURING():
            w.update(calls=0, reqs=0, toks=0, s=0.0, max_ms=0.0)
            return
        try:
            if not bool(out_pad.isfinite().all()):
'''

EDITS = [
    ("docstring_marker", OLD_A_LF, NEW_A_LF),
    ("head_sync_drop", OLD_B_LF, NEW_B_LF),
    ("spec_group_static", OLD_C_LF, NEW_C_LF),
    ("capturing_helper", OLD_D_LF, NEW_D_LF),
    ("telem_gate", OLD_E_LF, NEW_E_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_05.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if MARKER in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old_lf, _ in EDITS:
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_05.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src
for _, old_lf, new_lf in EDITS:
    out = out.replace(ed(old_lf), ed(new_lf))

with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8", newline=""
) as tf:
    tf.write(out)
    tmp = tf.name
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print(f"{TAG} FAIL: compile error -> {e}")
    sys.exit(3)
finally:
    os.unlink(tmp)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(out)
print(
    f"{TAG} [{REL}] edited "
    f"(eol={'CRLF' if eol == chr(13) + chr(10) else 'LF'}), md5={md5(path)}"
)
print(f"{TAG} ALL_EDITS_OK")
