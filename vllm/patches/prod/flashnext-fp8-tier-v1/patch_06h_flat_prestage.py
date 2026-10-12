#!/usr/bin/env python3
"""FN8_06h (DEC40): extend the 06f prestage to FLAT (non-spec) steps.

Root cause (d9b try-1 boot fatal, 2026-10-12 11:49:57, banked
/stage/bank/serve_d9b_fatal.log):

    RuntimeError: Worker failed with error 'wait method cannot be used
    for an event associated with a command graph.'

with PLE telemetry FN8_PLE_OP {calls: 2, rows: 65600, stub: 0} and
FN8_PLE_PRE {calls: 0}.  The 06f design (patch_06f) tagged only the
SPEC-path ple() calls ("spec"/"ns" groups, _ple_forward_spec) and the
model-level prestage hook returned early for non-spec steps:

    if md is None or getattr(md, "spec_sequence_masks", None) is None:
        return  # non-spec step: prefill/flat keeps the eager op path

On an MTP boot (d9) every DECODE step carries spec metadata, so capture
dummies took the tagged path and capture survived.  On a NO-MTP boot
(d9b isolation leg) every decode step — including the FULL_DECODE_ONLY
capture dummy runs — is FLAT (_ple_forward_flat), whose ple() call was
UNTAGGED -> fell through to the 06c host-gather op -> blocking D2H
event wait inside graph capture -> worker death at boot.

Edits:
  E1  fast-path guard: serve the bank only when m = ngram_ids.numel()
      fits (<= cap).  Oversized eager steps (the 65600-row stateless
      profile run vs cap 65536) keep the eager op path; without the
      guard the silently-clamped slice would raise on the view.
  E2  _ple_forward_flat ple() call: fn8_group="ns".
  E3  _ple_forward_stateless ple() call: fn8_group="ns" (profile run
      overflows to the eager op via E1; small capture dummies use the
      seeded bank).
  E4  model-level fn8_ple_prestage: flat-step branch mirroring
      _ple_forward_flat's history construction EXACTLY (same qsl /
      block-id fallback chain, eos padding, fresh-seed) so the staged
      row order equals the in-graph compute_ngram_ids order, then the
      shared staging tail (duplicated 4-liner; spec body untouched).

Idempotent; py_compile gate; backup .fn8_06h.bak; EOL aware; anchor==1
per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06h"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06h"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# E1: fast path — add the m<=cap guard (bank lookup restructure).
OLD1_LF = '''        if (
            fn8_group is not None
            and self._fn8_dev16
            and fn8_group in self._fn8_dev16
        ):
            # 06f fast path: this step's rows were staged host-side
            # (fn8_prestage via the 06g runner hook, EAGER pre-dispatch);
            # the captured graph only slices + views the fixed device bank.
            # Slice length is static per graph (S/L baked at capture);
            # view of a contiguous 2D slice is capture-legal.
            return self._fn8_dev16[fn8_group][:ngram_ids.numel()].view(
                *ngram_ids.shape[:-1], self.embedding_dim)
'''
NEW1_LF = '''        _fn8_bank = (
            self._fn8_dev16.get(fn8_group)
            if fn8_group is not None else None)
        if _fn8_bank is not None and ngram_ids.numel() <= _fn8_bank.shape[0]:
            # 06f fast path: this step's rows were staged host-side
            # (fn8_prestage via the 06g runner hook, EAGER pre-dispatch);
            # the captured graph only slices + views the fixed device bank.
            # Slice length is static per graph (S/L baked at capture);
            # view of a contiguous 2D slice is capture-legal.
            # 06h: m<=cap guard added — oversized steps (65600-row
            # stateless profile run vs cap 65536) must keep the EAGER
            # op path; a silently-clamped slice would raise on the view.
            return _fn8_bank[:ngram_ids.numel()].view(
                *ngram_ids.shape[:-1], self.embedding_dim)
'''

# E2: flat-path ple() call — tag "ns".
OLD2_LF = '''        out_pad, new_conv_state, new_ctx = ple(hidden_pad, token_history, lens, conv_state)
'''
NEW2_LF = '''        # 06h: tag the bank group — the flat path runs under
        # FULL_DECODE_ONLY capture on no-MTP boots (d9b try-1 law:
        # untagged -> 06c op -> capture-illegal D2H event wait).
        out_pad, new_conv_state, new_ctx = ple(
            hidden_pad, token_history, lens, conv_state, fn8_group="ns")
'''

# E3: stateless ple() call — tag "ns" (profile overflows to op via E1).
OLD3_LF = '''        lens = torch.tensor([T], dtype=torch.long, device=device)
        out_pad, _, _ = ple(
            hidden_states.unsqueeze(0), token_history, lens, conv_state
        )
        return out_pad.squeeze(0)
'''
NEW3_LF = '''        lens = torch.tensor([T], dtype=torch.long, device=device)
        # 06h: tag the bank group — capture dummy runs with
        # attn_metadata=None route here; the eager profile run (m > cap)
        # still takes the 06c op via the E1 guard.
        out_pad, _, _ = ple(
            hidden_states.unsqueeze(0), token_history, lens, conv_state,
            fn8_group="ns")
        return out_pad.squeeze(0)
'''

# E4: model-level prestage — flat-step branch.
OLD4_LF = '''        if md is None or getattr(md, "spec_sequence_masks", None) is None:
            return  # non-spec step: prefill/flat keeps the eager op path
        kv = getattr(l0.linear_attn, "kv_cache", None)
        if not kv or len(kv) < 4:
            return
        hist_pool = kv[3]
        device = input_ids.device
        eos = l0.ple.ple_embedding.eos_token_id
        ctx_len = l0.ple.ple_embedding.context_len
        input_ids = input_ids.to(torch.long)
        ths = {}
        sq = md.spec_query_start_loc.to(torch.long)
'''
NEW4_LF = '''        if md is None:
            return
        _masks = getattr(md, "spec_sequence_masks", None)
        kv = getattr(l0.linear_attn, "kv_cache", None)
        if not kv or len(kv) < 4:
            return
        hist_pool = kv[3]
        device = input_ids.device
        eos = l0.ple.ple_embedding.eos_token_id
        ctx_len = l0.ple.ple_embedding.context_len
        input_ids = input_ids.to(torch.long)
        ths = {}
        if _masks is None:
            # FN8_06h: flat step (no-spec decode / prefill / mixed).  The
            # flat forward now tags fn8_group="ns", so its rows must
            # stage here.  Mirror _ple_forward_flat's history
            # construction EXACTLY (same qsl/block-id fallback chain,
            # eos padding, fresh-seed) so the staged row order equals
            # the in-graph compute_ngram_ids order.
            qsl = getattr(md, "non_spec_query_start_loc", None)
            if qsl is None:
                qsl = getattr(md, "query_start_loc", None)
            blk = getattr(md, "non_spec_state_indices_tensor", None)
            if blk is None:
                blk = getattr(md, "state_indices_tensor", None)
            if qsl is None or blk is None or input_ids.shape[0] == 0:
                return
            qsl = qsl.to(torch.long)
            _N = qsl.shape[0] - 1
            if _N <= 0:
                return
            _T = input_ids.shape[0]
            tok_pos = torch.arange(_T, device=device, dtype=torch.long)
            req = torch.searchsorted(qsl[1:], tok_pos, right=True)
            loc = tok_pos - qsl[req]
            lens_f = qsl[1:] - qsl[:-1]
            max_len_f = int(lens_f.max().item())  # eager hook — sync OK
            ip = torch.full((_N, max_len_f), eos, dtype=torch.long,
                            device=device)
            ip[req, loc] = input_ids
            prev = hist_pool[blk[:_N].to(torch.long), :ctx_len, 0].clone()
            his = getattr(md, "has_initial_state", None)
            if his is not None:
                prev[~his[:_N].to(torch.bool)] = eos
            ths["ns"] = torch.cat([prev, ip], dim=-1)
            for ly in layers:
                emb = ly.ple.ple_embedding
                for g, th in ths.items():
                    emb.fn8_prestage(emb.compute_ngram_ids(th), g)
            return
        sq = md.spec_query_start_loc.to(torch.long)
'''

EDITS = [
    ("E1_fastpath_guard", OLD1_LF, NEW1_LF),
    ("E2_flat_tag", OLD2_LF, NEW2_LF),
    ("E3_stateless_tag", OLD3_LF, NEW3_LF),
    ("E4_prestage_flat", OLD4_LF, NEW4_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06h.bak"
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


out = src
for name, old, new in EDITS:
    n = out.count(ed(old))
    if n != 1:
        print(f"{TAG} FAIL: anchor {name} occurs {n}x (want 1)")
        sys.exit(2)
    out = out.replace(ed(old), ed(new))

bak = path + ".fn8_06h.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

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
