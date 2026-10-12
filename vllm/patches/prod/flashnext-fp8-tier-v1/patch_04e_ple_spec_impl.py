#!/usr/bin/env python3
"""FN8_04e (DEC40 D1): the PLE spec-verify implementation proper.

04d fell the P4 guard but fed spec steps the WRONG fields
(GDNAttentionMetadata carries spec_*/non_spec_* names, not the generic
query_start_loc/state_indices_tensor) -> "missing query_start_loc/state
indices" at 03:53.

Fork layout law (gdn_attn.py + reorder_batch_to_split_decodes_and_prefills):
  * global token tensor T = [spec-decode tokens | non-spec tokens]
    (reorder_batch guarantees spec decodes first; the builder's mixed-case
    cumsum qsls are rebased to exactly this layout, and the pure-spec case
    slices query_start_loc[:S+1] directly);
  * spec seq i: spec_state_indices_tensor[i] = (k+1,) PER-POSITION slots
    (block-table row) — read state from slot[i,0], process the (k+1)-token
    mini-sequence causally, write position-j state to slot[i,j]; the eagle
    pool machinery then keeps slots 0..L_a-1 (accepted length) and frees
    the rest, so garbage in positions >= L_a is harmless;
  * non-spec group (regular decodes + extends + prefills): same flat path
    as before, offset by n_spec_toks, non-spec fields, FINAL state written
    to the seq's single slot.

The spec mini-sequence rides the same padded core prefill uses (it IS a
short prefill); causality of n-gram keys holds because token_history is
[prev_ctx | ids 0..j-1] per position inside the core.

Per-position state scatter (the only new math):
  conv slot j  = last S_len slices of [conv_state | tokens 0..j]
  hist slot j  = last ctx_len  ids    of [prev_ctx  | tokens 0..j]
  -> sliding windows over cat(old, padded-new); a j-loop of k+1 = 5
  scatter writes, (S, S_len, D) / (S, ctx_len) each.

Idempotent; py_compile gate; backup .fn8_04e.bak; EOL aware.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04E"
REL = "model_executor/models/qwen4_exp.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- edit 1: spec branch in _ple_forward_flat -------------------------------
OLD1_LF = (
    '        if kv_cache is None or len(kv_cache) < 4:\n'
    '            raise RuntimeError("Qwen4Exp PLE: 4-slot mamba state not wired '
    'to the layer")\n'
    "\n"
    "        qsl, block_ids_t, has_initial_state = self._ple_metadata()\n"
)
NEW1_LF = (
    '        if kv_cache is None or len(kv_cache) < 4:\n'
    '            raise RuntimeError("Qwen4Exp PLE: 4-slot mamba state not wired '
    'to the layer")\n'
    "\n"
    "        # DEC40 D1 (04e): spec verify takes the dedicated spec path\n"
    "        if getattr(attn_md, \"spec_sequence_masks\", None) is not None:\n"
    "            return self._ple_forward_spec(hidden_states, input_ids, attn_md)\n"
    "\n"
    "        qsl, block_ids_t, has_initial_state = self._ple_metadata()\n"
)

# --- edit 2: _ple_forward_spec method before the stateless path -------------
OLD2_LF = "    def _ple_forward_stateless(\n"
NEW2_LF = '''    def _ple_forward_spec(
        self,
        hidden_states: torch.Tensor,  # (T, hc_hidden) global token order
        input_ids: torch.Tensor,      # (T,)
        md,                           # GDNAttentionMetadata (spec build)
    ) -> torch.Tensor:
        """DEC40 D1 (04e): PLE under MTP spec verify.

        Fork layout law: T = [spec tokens | non-spec tokens].  Spec seq i
        reads its state from slot[i, 0], runs the (k+1)-token mini-sequence
        through the prefill-shaped padded core, then scatters per-position
        states into slots [i, j] -- the eagle pool machinery keeps
        0..L_a-1 (positions past the accepted length are freed, so their
        garbage content is harmless).  Non-spec seqs take the flat path on
        the global suffix.
        """
        ple = self.ple
        device = hidden_states.device
        eos = ple.ple_embedding.eos_token_id
        ctx_len = ple.ple_embedding.context_len
        S_len = ple.short_conv_state_len
        input_ids = input_ids.to(torch.long)
        kv_cache = getattr(self.linear_attn, "kv_cache", None)
        conv_pool, hist_pool = kv_cache[2], kv_cache[3]

        sq = md.spec_query_start_loc.to(torch.long)
        s_slots = md.spec_state_indices_tensor.to(torch.long)  # (S, k+1)
        n_spec_toks = int(sq[-1].item())
        T = hidden_states.shape[0]
        outs = torch.empty_like(hidden_states)

        # ---- spec group (global prefix) --------------------------------
        S = sq.shape[0] - 1
        if S > 0 and n_spec_toks > 0:
            hs = hidden_states[:n_spec_toks]
            ids = input_ids[:n_spec_toks]
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
            outs[:n_spec_toks] = out_pad[req_of_tok, local_of_tok]

            # per-position state scatter: slot j = state AFTER token j
            seq_all = torch.cat([conv_state, hidden_pad], dim=1)  # (S,S_len+L,D)
            tok_all = torch.cat([prev_ctx, ids_pad], dim=1)       # (S,ctx+L)
            for j in range(max_len):
                conv_pool[s_slots[:, j], :S_len] = seq_all[:, j + 1 : j + 1 + S_len]
                hist_pool[s_slots[:, j], :ctx_len, 0] = tok_all[
                    :, j + 1 : j + 1 + ctx_len
                ]

        # ---- non-spec group (global suffix) ----------------------------
        nq = getattr(md, "non_spec_query_start_loc", None)
        n_slots = getattr(md, "non_spec_state_indices_tensor", None)
        if nq is not None and n_slots is not None and n_spec_toks < T:
            nq = nq.to(torch.long)
            n_slots = n_slots.to(torch.long)
            N = nq.shape[0] - 1
            if N > 0:
                hs2 = hidden_states[n_spec_toks:]
                ids2 = input_ids[n_spec_toks:]
                lens2 = nq[1:] - nq[:-1]
                max_len2 = int(lens2.max().item())
                n_toks2 = T - n_spec_toks
                tok_pos2 = torch.arange(n_toks2, device=device, dtype=torch.long)
                req2 = torch.searchsorted(nq[1:], tok_pos2, right=True)
                loc2 = tok_pos2 - nq[req2]
                hp2 = hs2.new_zeros(N, max_len2, hs2.shape[-1])
                hp2[req2, loc2] = hs2
                ip2 = torch.full((N, max_len2), eos, dtype=torch.long, device=device)
                ip2[req2, loc2] = ids2
                prev2 = hist_pool[n_slots, :ctx_len, 0].clone()
                his = getattr(md, "has_initial_state", None)
                if his is not None and not bool(torch.all(his[:N]).item()):
                    fresh = ~his[:N].to(torch.bool)
                    if bool(fresh.any().item()):
                        prev2[fresh] = eos
                cs2 = conv_pool[n_slots, :S_len].clone()
                th2 = torch.cat([prev2, ip2], dim=-1)
                _t0 = _perf()
                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2)
                _ple_telemetry(False, N, max_len2, n_toks2, _perf() - _t0, o2)
                outs[n_spec_toks:] = o2[req2, loc2]
                conv_pool[n_slots, :S_len] = nc2
                hist_pool[n_slots, :ctx_len, 0] = nw2
        elif n_spec_toks < T:
            raise RuntimeError(
                "Qwen4Exp PLE spec: non-spec tokens present but "
                "non-spec metadata missing"
            )
        return outs

    def _ple_forward_stateless(
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04e.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if "DEC40 D1 (04e)" in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old_lf in (("flat_branch", OLD1_LF), ("spec_method", OLD2_LF)):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_04e.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ed(OLD1_LF), ed(NEW1_LF)).replace(ed(OLD2_LF), ed(NEW2_LF))

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
