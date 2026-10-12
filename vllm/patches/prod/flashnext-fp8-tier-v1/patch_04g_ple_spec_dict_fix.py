#!/usr/bin/env python3
"""FN8_04g (DEC40 D1): the dict-dispatch fix + index-map gather for PLE spec.

04F DIAG GROUND TRUTH (04:46:23, first verify step):
  counts={num_prefills: 0, num_decodes: 0, num_spec_decodes: 1,
          num_spec_decode_tokens: 5, num_actual_tokens: 5}
  fields: spec_* set, non_spec_* None, masks=True, token_indx set
  ctx = DICT of 50 per-prefix entries (hybrid model)
  spec_mode=True -> the raise came from _ple_metadata's 04d spec branch

Root cause: _ple_forward_flat tested spec_sequence_masks on the CONTEXT
DICT (getattr on dict -> None always), so the 04e spec path never engaged;
_ple_metadata then dug the real GDN object out (masks=True), took its spec
branch and read query_start_loc/state_indices_tensor — fields that do not
exist on GDNAttentionMetadata.

Second law baked in: the builder's mixed case builds spec_token_indx /
non_spec_token_indx via argsort(masks, stable) — a PERMUTATION the runner
applies, i.e. physical token layout under spec is index-map-defined
(non-spec first in the argsort order), NOT [spec | non-spec] by position.
So the spec path GATHERS tokens by spec_token_indx (pure-spec case it is
just arange) and SCATTERS outputs back by the same index; non-spec group
likewise via non_spec_token_indx. No positional assumptions remain.

Edits:
  G1  flat dispatch: extract md = ctx.attn_metadata[prefix] (mirror of
      _ple_metadata), test masks on md, pass md to _ple_forward_spec.
  G2  _ple_forward_spec: gather/scatter via spec_token_indx and
      non_spec_token_indx (arange fallback = old positional behavior).
  G3  _ple_metadata 04d spec branch: read the REAL spec fields
      (defensive; unreachable once G1 routes spec steps away).

Idempotent; py_compile gate; backup .fn8_04g.bak; EOL aware.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_04G"
REL = "model_executor/models/qwen4_exp.py"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- G1: flat dispatch block (current = 04f-wrapped 04e branch) -------------
OLD1_LF = """        # DEC40 D1 (04e): spec verify takes the dedicated spec path
        _masks = getattr(attn_md, \"spec_sequence_masks\", None)
        if _FN8_PLE_DIAG_SEEN.add(
            (
                _masks is not None,
                getattr(attn_md, \"num_prefills\", -1),
                getattr(attn_md, \"num_decodes\", -1),
                getattr(attn_md, \"num_spec_decodes\", -1),
            )
        ):
            logger.info(
                \"FN8_PLE_DIAG_ENTRY masks=%s npf=%s ndec=%s nspec=%s T=%d\",
                _masks is not None,
                getattr(attn_md, \"num_prefills\", -1),
                getattr(attn_md, \"num_decodes\", -1),
                getattr(attn_md, \"num_spec_decodes\", -1),
                hidden_states.shape[0],
            )
        if _masks is not None:
            return self._ple_forward_spec(hidden_states, input_ids, attn_md)
"""
NEW1_LF = """        # DEC40 D1 (04g): the forward-context attr is a per-prefix DICT for
        # hybrid models — extract THIS layer's GDN metadata before any spec
        # field test (04e/04f bug: masks checked on the dict = always None,
        # so spec steps fell through to _ple_metadata's broken 04d branch).
        _md = (
            attn_md.get(self.linear_attn.prefix)
            if isinstance(attn_md, dict)
            else attn_md
        )
        if _md is None:
            raise RuntimeError(
                \"Qwen4Exp PLE: no attention metadata for \"
                f\"{self.linear_attn.prefix}\"
            )
        _masks = getattr(_md, \"spec_sequence_masks\", None)
        if _FN8_PLE_DIAG_SEEN.add(
            (
                _masks is not None,
                getattr(_md, \"num_prefills\", -1),
                getattr(_md, \"num_decodes\", -1),
                getattr(_md, \"num_spec_decodes\", -1),
            )
        ):
            logger.info(
                \"FN8_PLE_DIAG_ENTRY masks=%s npf=%s ndec=%s nspec=%s T=%d\",
                _masks is not None,
                getattr(_md, \"num_prefills\", -1),
                getattr(_md, \"num_decodes\", -1),
                getattr(_md, \"num_spec_decodes\", -1),
                hidden_states.shape[0],
            )
        if _masks is not None:
            return self._ple_forward_spec(hidden_states, input_ids, _md)
"""

# --- G2: full _ple_forward_spec replace (anchor = exact 04e method) ---------
OLD2_LF = """    def _ple_forward_spec(
        self,
        hidden_states: torch.Tensor,  # (T, hc_hidden) global token order
        input_ids: torch.Tensor,      # (T,)
        md,                           # GDNAttentionMetadata (spec build)
    ) -> torch.Tensor:
        \"\"\"DEC40 D1 (04e): PLE under MTP spec verify.

        Fork layout law: T = [spec tokens | non-spec tokens].  Spec seq i
        reads its state from slot[i, 0], runs the (k+1)-token mini-sequence
        through the prefill-shaped padded core, then scatters per-position
        states into slots [i, j] -- the eagle pool machinery keeps
        0..L_a-1 (positions past the accepted length are freed, so their
        garbage content is harmless).  Non-spec seqs take the flat path on
        the global suffix.
        \"\"\"
        ple = self.ple
        device = hidden_states.device
        eos = ple.ple_embedding.eos_token_id
        ctx_len = ple.ple_embedding.context_len
        S_len = ple.short_conv_state_len
        input_ids = input_ids.to(torch.long)
        kv_cache = getattr(self.linear_attn, \"kv_cache\", None)
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
                    f\"Qwen4Exp PLE spec: {max_len} queries but \"
                    f\"{s_slots.shape[1]} state slots\"
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
        nq = getattr(md, \"non_spec_query_start_loc\", None)
        n_slots = getattr(md, \"non_spec_state_indices_tensor\", None)
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
                his = getattr(md, \"has_initial_state\", None)
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
                \"Qwen4Exp PLE spec: non-spec tokens present but \"
                \"non-spec metadata missing\"
            )
        return outs
"""
NEW2_LF = """    def _ple_forward_spec(
        self,
        hidden_states: torch.Tensor,  # (T, hc_hidden) physical token order
        input_ids: torch.Tensor,      # (T,)
        md,                           # GDNAttentionMetadata (spec build)
    ) -> torch.Tensor:
        \"\"\"DEC40 D1 (04g): PLE under MTP spec verify.

        Token groups are identified by INDEX MAPS (spec_token_indx /
        non_spec_token_indx — the builder's argsort permutation), never by
        position.  Spec seq i reads its state from slot[i, 0], runs the
        (k+1)-token mini-sequence through the prefill-shaped padded core,
        then scatters per-position states into slots [i, j] — the eagle
        pool machinery keeps 0..L_a-1 (positions past the accepted length
        are freed, so their garbage content is harmless).  Non-spec seqs
        take the flat path over their index map.
        \"\"\"
        ple = self.ple
        device = hidden_states.device
        eos = ple.ple_embedding.eos_token_id
        ctx_len = ple.ple_embedding.context_len
        S_len = ple.short_conv_state_len
        input_ids = input_ids.to(torch.long)
        kv_cache = getattr(self.linear_attn, \"kv_cache\", None)
        conv_pool, hist_pool = kv_cache[2], kv_cache[3]

        sq = md.spec_query_start_loc.to(torch.long)
        s_slots = md.spec_state_indices_tensor.to(torch.long)  # (S, k+1)
        n_spec_toks = int(sq[-1].item())
        T = hidden_states.shape[0]
        outs = torch.empty_like(hidden_states)

        # ---- spec group ---------------------------------------------------
        S = sq.shape[0] - 1
        if S > 0 and n_spec_toks > 0:
            s_indx = getattr(md, \"spec_token_indx\", None)
            if s_indx is None:
                s_indx = torch.arange(n_spec_toks, device=device)
            s_indx = s_indx.to(torch.long)
            hs = hidden_states[s_indx]
            ids = input_ids[s_indx]
            lens = sq[1:] - sq[:-1]
            max_len = int(lens.max().item())
            if s_slots.shape[1] < max_len:
                raise RuntimeError(
                    f\"Qwen4Exp PLE spec: {max_len} queries but \"
                    f\"{s_slots.shape[1]} state slots\"
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

        # ---- non-spec group ------------------------------------------------
        nq = getattr(md, \"non_spec_query_start_loc\", None)
        n_slots = getattr(md, \"non_spec_state_indices_tensor\", None)
        n_nonspec_toks = T - n_spec_toks
        if nq is not None and n_slots is not None and n_nonspec_toks > 0:
            nq = nq.to(torch.long)
            n_slots = n_slots.to(torch.long)
            N = nq.shape[0] - 1
            if N > 0:
                ns_indx = getattr(md, \"non_spec_token_indx\", None)
                if ns_indx is None:
                    ns_indx = torch.arange(
                        n_spec_toks, T, device=device, dtype=torch.long
                    )
                ns_indx = ns_indx.to(torch.long)
                hs2 = hidden_states[ns_indx]
                ids2 = input_ids[ns_indx]
                lens2 = nq[1:] - nq[:-1]
                max_len2 = int(lens2.max().item())
                tok_pos2 = torch.arange(
                    n_nonspec_toks, device=device, dtype=torch.long
                )
                req2 = torch.searchsorted(nq[1:], tok_pos2, right=True)
                loc2 = tok_pos2 - nq[req2]
                hp2 = hs2.new_zeros(N, max_len2, hs2.shape[-1])
                hp2[req2, loc2] = hs2
                ip2 = torch.full((N, max_len2), eos, dtype=torch.long, device=device)
                ip2[req2, loc2] = ids2
                prev2 = hist_pool[n_slots, :ctx_len, 0].clone()
                his = getattr(md, \"has_initial_state\", None)
                if his is not None and not bool(torch.all(his[:N]).item()):
                    fresh = ~his[:N].to(torch.bool)
                    if bool(fresh.any().item()):
                        prev2[fresh] = eos
                cs2 = conv_pool[n_slots, :S_len].clone()
                th2 = torch.cat([prev2, ip2], dim=-1)
                _t0 = _perf()
                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2)
                _ple_telemetry(False, N, max_len2, n_nonspec_toks, _perf() - _t0, o2)
                outs[ns_indx] = o2[req2, loc2]
                conv_pool[n_slots, :S_len] = nc2
                hist_pool[n_slots, :ctx_len, 0] = nw2
        elif n_nonspec_toks > 0:
            raise RuntimeError(
                \"Qwen4Exp PLE spec: non-spec tokens present but \"
                \"non-spec metadata missing\"
            )
        return outs
"""

# --- G3: _ple_metadata spec branch reads the real fields --------------------
OLD3_LF = """            self._ple_spec_mode = True
            qsl = getattr(md, \"query_start_loc\", None)
            block_ids = getattr(md, \"state_indices_tensor\", None)
"""
NEW3_LF = """            self._ple_spec_mode = True
            qsl = getattr(md, \"spec_query_start_loc\", None)
            block_ids = getattr(md, \"spec_state_indices_tensor\", None)
"""


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_04g.bak"
    if os.path.exists(bak):
        shutil.copyfile(bak, path)
        print(f"{TAG} restored {REL} md5={md5(path)}")
    else:
        print(f"{TAG} no backup")
        sys.exit(4)
    sys.exit(0)

with open(path, "r", encoding="utf-8", newline="") as f:
    src = f.read()

if "DEC40 D1 (04g)" in src:
    print(f"{TAG} edit already present, skip (md5={md5(path)})")
    sys.exit(0)

eol = "\r\n" if "\r\n" in src else "\n"


def ed(lf_text):
    return lf_text.replace("\n", eol)


for name, old_lf in (("flat_dispatch", OLD1_LF), ("spec_method", OLD2_LF), ("meta_fields", OLD3_LF)):
    n = src.count(ed(old_lf))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_04g.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src
for old_lf, new_lf in ((OLD1_LF, NEW1_LF), (OLD2_LF, NEW2_LF), (OLD3_LF, NEW3_LF)):
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
