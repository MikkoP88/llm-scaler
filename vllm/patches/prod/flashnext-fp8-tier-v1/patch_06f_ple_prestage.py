#!/usr/bin/env python3
"""FN8_06f (DEC40): PLE prestage — host-side row gather into fixed device
banks that captured decode graphs slice+view.

Problem: the PLE n-gram table (~47.7 GB fp8, host-resident BY DESIGN) makes
every decode step host-gather rows (06c op: blocking D2H of ids -> host
index_select -> H2D -> device dequant).  That op can never capture (D8
crash-2 law), so decode runs eager ~250 ms/step = ~12 tok/s.  Graphs need
the gather OUT of the captured region.

Design (n-gram ids are a pure function of TOKEN HISTORY — token ids only,
no hidden states):
  * 06g runner hook (separate patch, gpu_model_runner.py) calls
    ForConditionalGeneration.fn8_ple_prestage(input_ids, attn_metadata)
    EAGER right after _preprocess, before set_forward_context/dispatch.
    It mirrors _ple_forward_spec's spec/ns grouping from layer-0 pools
    (token-id stream identical across PLE layers; this lane has exactly
    ONE PLE layer, ple_layer_ids [2]) and per group computes
    compute_ngram_ids on device, blocking-D2Hs the ~KB ids, gathers +
    dequants REAL rows host-side into a pinned fp16 bank, async-H2Ds into
    the fixed device bank.  H2D precedes the replay launch on the same
    stream -> race-free by ordering.
  * The captured graph's PLE call sites (group tag "spec"/"ns" baked per
    call site) hit the 06f FAST PATH: slice+view of the device bank.
    Zero host work, zero gathers in-graph.
  * Banks allocated + seeded EAGER at the end of _assemble_ple_table
    (post-load, pre-profile/capture) so EVERY capture bakes the fast
    path.  Seed = dequantized REAL row 0 (d8s crash-5 law: a zero-magnitude
    seed NaN-drifted to an embedding-gather OOB at decode token ~18).
  * Prefill / stateless / non-spec steps keep group=None -> 06c op path
    (eager, unchanged).

Numerics: host dequant mirrors _dequant_rows exactly (per-tensor scale or
[128,128]-block scales; uint8-view index_select for fp8, same as the op).

Gates: env FN8_PLE_PRESTAGE=1 enables (banks only allocated when set);
FN8_PLE_PRESTAGE_MAX_TOKENS (default 4096) sizes the banks.  Fast path
silently absent when banks are empty -> stub/op behavior unchanged.

Idempotent; py_compile gate; backup .fn8_06f.bak; EOL aware; anchor==1
gates everywhere; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06f"
REL = "model_executor/models/qwen4_exp.py"
MARKER = "FN8_06f"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# --- E1: telemetry record + atexit dump (beside the 06c one) --------------------
OLD_E1 = '''def _fn8_ple_op_dump():
    import sys as _s
    print(f"FN8_PLE_OP {_FN8_PLE_OP_REC}", file=_s.stderr, flush=True)
'''
NEW_E1 = '''def _fn8_ple_op_dump():
    import sys as _s
    print(f"FN8_PLE_OP {_FN8_PLE_OP_REC}", file=_s.stderr, flush=True)
    print(f"FN8_PLE_PRE {_FN8_PLE_PRE_REC}", file=_s.stderr, flush=True)


# 06f: prestage telemetry (see Qwen4ExpNGramEmbedding.fn8_prestage)
_FN8_PLE_PRE_REC = {"calls": 0, "rows": 0, "overflow": 0, "ms": 0.0,
                    "ms_max": 0.0}
'''

# --- E2: embedding __init__ tail — bank attrs (PLAIN: offloader law) ------------
OLD_E2 = '''        # column block split derived from the scale layout at load time;
        # None => passthrough (unquantized table)
        self._col_block_size: int | None = None
'''
NEW_E2 = '''        # column block split derived from the scale layout at load time;
        # None => passthrough (unquantized table)
        self._col_block_size: int | None = None
        # 06f: prestage banks, group -> pinned host / fixed device fp16 row
        # banks ("spec" / "ns").  PLAIN ATTRS (offloader law, class
        # docstring).  Allocated+seeded EAGER by fn8_prestage_init (called
        # from _assemble_ple_table, pre-profile/capture) so every captured
        # decode graph bakes the fast path, never the host-gather op.
        self._fn8_pin16: dict = {}
        self._fn8_dev16: dict = {}
'''

# --- E3: forward signature + fast path -------------------------------------------
OLD_E3 = '''    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather host table rows for (R, L, heads) ids -> (R, L, embedding_dim)
        fp16 on the ids' device.
'''
NEW_E3 = '''    def forward(self, ngram_ids: torch.Tensor,
                fn8_group: str | None = None) -> torch.Tensor:
        """Gather host table rows for (R, L, heads) ids -> (R, L, embedding_dim)
        fp16 on the ids' device.
'''
OLD_E3B = '''        if _FN8_PLE_STUB:
'''
NEW_E3B = '''        if (
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
        if _FN8_PLE_STUB:
'''

# --- E4: fn8_prestage_init + fn8_prestage methods --------------------------------
OLD_E4 = '''# ---------------------------------------------------------------------------
# PLE layer (reference Qwen4ExpTextPLELayer, byte-exact math)
'''
NEW_E4 = '''    def fn8_prestage_init(self, device) -> None:
        """06f: allocate per-group fp16 row banks (pinned host mirror +
        fixed device bank) and seed with a REAL table row (d8s crash-5 law:
        zero-magnitude seed NaN-drifted to an embedding-gather OOB).  Runs
        EAGER at the end of _assemble_ple_table — before profile/capture —
        so captures bake the fast path, never the uncapturable host op."""
        floor = int(os.environ.get("FN8_PLE_PRESTAGE_MAX_TOKENS", "4096"))
        cap = max(1, floor) * self.ngram_heads
        row = self._host_table[:1].to(device).to(torch.float32)
        if self._col_block_size is None:
            if self._host_scale_inv.numel() == 1:
                row = row * self._host_scale_inv.to(device).to(torch.float32)
            elif self._host_scale.numel() == 1:
                row = row * self._host_scale.to(device).to(torch.float32)
        else:
            _sc = self._host_scale_inv.to(device).to(torch.float32)
            row = row * _sc[0].view(1, -1)
        row = row.to(torch.float16)
        for g in ("spec", "ns"):
            pin = torch.empty((cap, self.head_dim_per_ngram),
                              dtype=torch.float16, device="cpu").pin_memory()
            dev = torch.empty((cap, self.head_dim_per_ngram),
                              dtype=torch.float16, device=device)
            dev.copy_(row)  # broadcast seed (in-distribution magnitudes)
            self._fn8_pin16[g] = pin
            self._fn8_dev16[g] = dev
        logger.info(
            "FN8_06f: prestage banks ready (cap=%d rows/group, seeded real "
            "row 0)", cap)

    def fn8_prestage(self, ngram_ids: torch.Tensor, group: str) -> None:
        """06f: stage THIS step's rows for `group` — blocking D2H of ids
        (~KB), host gather (uint8-view law) + dequant into the pinned bank,
        async H2D into the fixed device bank the captured graph reads.
        Runs EAGER from the 06g runner hook; the H2D precedes the replay
        launch on the same stream -> race-free by ordering."""
        dev = self._fn8_dev16.get(group)
        if dev is None:
            return
        ids = ngram_ids.reshape(-1).to(torch.long)
        m = int(ids.numel())
        if m == 0:
            return
        if m > dev.shape[0]:
            _FN8_PLE_PRE_REC["overflow"] += 1
            return
        import time as _fn8_t
        _t0 = _fn8_t.time()
        ids_h = ids.to("cpu", non_blocking=False)
        if self._host_table.dtype == torch.float8_e4m3fn:
            rows = self._host_table.view(torch.uint8).index_select(
                0, ids_h).view(torch.float8_e4m3fn)
        else:
            rows = self._host_table.index_select(0, ids_h)
        f = rows.to(torch.float32)
        if self._col_block_size is None:
            if self._host_scale_inv.numel() == 1:
                f = f * self._host_scale_inv.to(torch.float32)
            elif self._host_scale.numel() == 1:
                f = f * self._host_scale.to(torch.float32)
        else:
            srows = self._host_scale_inv[ids_h // self._col_block_size]
            for k, st in enumerate(range(0, f.shape[-1],
                                         self._col_block_size)):
                en = min(st + self._col_block_size, f.shape[-1])
                f[:, st:en] *= srows[..., k, None]
        self._fn8_pin16[group][:m].copy_(f)
        dev[:m].copy_(self._fn8_pin16[group][:m], non_blocking=True)
        _FN8_PLE_PRE_REC["calls"] += 1
        _FN8_PLE_PRE_REC["rows"] += m
        _dt = _fn8_t.time() - _t0
        _FN8_PLE_PRE_REC["ms"] += _dt
        if _dt > _FN8_PLE_PRE_REC["ms_max"]:
            _FN8_PLE_PRE_REC["ms_max"] = _dt


# ---------------------------------------------------------------------------
# PLE layer (reference Qwen4ExpTextPLELayer, byte-exact math)
'''

# --- E5: PLELayer.forward signature gains fn8_group -------------------------------
OLD_E5 = '''        lengths: torch.Tensor,        # (R,) true lengths
        conv_state: torch.Tensor,     # (R, short_conv_state_len, hc_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
'''
NEW_E5 = '''        lengths: torch.Tensor,        # (R,) true lengths
        conv_state: torch.Tensor,     # (R, short_conv_state_len, hc_hidden)
        fn8_group: str | None = None,  # 06f: prestage bank tag ("spec"/"ns")
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
'''

# --- E6: pass the tag through to the embedding -----------------------------------
OLD_E6 = '''        embeddings = self.ple_embedding(ngram_ids)  # (R, L, ple_embed_dim)
'''
NEW_E6 = '''        embeddings = self.ple_embedding(
            ngram_ids, fn8_group=fn8_group)  # (R, L, ple_embed_dim)
'''

# --- E7/E8: bake the group tag per call site in _ple_forward_spec -----------------
OLD_E7 = '''            out_pad, _, _ = ple(hs_pad, token_history, lens, conv_state)
'''
NEW_E7 = '''            out_pad, _, _ = ple(hs_pad, token_history, lens, conv_state,
                                fn8_group="spec")
'''
OLD_E8 = '''                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2)
'''
NEW_E8 = '''                o2, nc2, nw2 = ple(hp2, th2, lens2, cs2, fn8_group="ns")
'''

# --- E9: ForConditionalGeneration.fn8_ple_prestage --------------------------------
OLD_E9 = '''    def _assemble_ple_table(self) -> None:
'''
NEW_E9 = '''    @torch.no_grad()
    def fn8_ple_prestage(self, input_ids, attn_metadata) -> None:
        """06f: stage this step's PLE rows host-side.  The 06g runner hook
        calls this EAGER after _preprocess, before dispatch.  Mirrors
        _ple_forward_spec's spec/ns grouping from layer-0 pools (the
        token-id stream is identical across PLE layers; ple_layer_ids [2]
        -> exactly one PLE layer on this lane).  Non-spec steps (prefill /
        flat) keep group=None -> eager 06c op path, untouched."""
        if not _ple_env_flag("FN8_PLE_PRESTAGE"):
            return
        m = self.language_model.model
        layers = getattr(m, "_fn8_ple_layers", None)
        if layers is None:
            layers = [ly for ly in m.layers
                      if getattr(ly, "ple", None) is not None]
            m._fn8_ple_layers = layers
        if not layers:
            return
        l0 = layers[0]
        md = (attn_metadata.get(l0.linear_attn.prefix)
              if isinstance(attn_metadata, dict) else attn_metadata)
        if md is None or getattr(md, "spec_sequence_masks", None) is None:
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
        s_slots = md.spec_state_indices_tensor.to(torch.long)
        S = sq.shape[0] - 1
        L = s_slots.shape[1]
        n_spec = S * L if S > 0 else 0
        T = input_ids.shape[0]
        if S > 0:
            s_indx = getattr(md, "spec_token_indx", None)
            if s_indx is None or s_indx.shape[0] != n_spec:
                s_indx = torch.arange(n_spec, device=device, dtype=torch.long)
            else:
                s_indx = s_indx.to(torch.long)
            ids_pad = input_ids[s_indx].view(S, L)
            prev_ctx = hist_pool[s_slots[:, 0], :ctx_len, 0].clone()
            ths["spec"] = torch.cat([prev_ctx, ids_pad], dim=-1)
        nq = getattr(md, "non_spec_query_start_loc", None)
        n_slots = getattr(md, "non_spec_state_indices_tensor", None)
        n_non = T - n_spec
        if (nq is not None and n_slots is not None and n_non > 0
                and nq.shape[0] > 1):
            nq = nq.to(torch.long)
            n_slots = n_slots.to(torch.long)
            N = nq.shape[0] - 1
            if N > 0:
                ns_indx = getattr(md, "non_spec_token_indx", None)
                if ns_indx is None or ns_indx.shape[0] != n_non:
                    ns_indx = torch.arange(
                        n_spec, T, device=device, dtype=torch.long)
                else:
                    ns_indx = ns_indx.to(torch.long)
                ids2 = input_ids[ns_indx]
                max_len2 = n_non
                tok_pos2 = torch.arange(
                    n_non, device=device, dtype=torch.long)
                req2 = torch.searchsorted(nq[1:], tok_pos2, right=True)
                loc2 = tok_pos2 - nq[req2]
                ip2 = torch.full((N, max_len2), eos, dtype=torch.long,
                                 device=device)
                ip2[req2, loc2] = ids2
                prev2 = hist_pool[n_slots, :ctx_len, 0].clone()
                his = getattr(md, "has_initial_state", None)
                if his is not None:
                    prev2[~his[:N].to(torch.bool)] = eos
                ths["ns"] = torch.cat([prev2, ip2], dim=-1)
        for ly in layers:
            emb = ly.ple.ple_embedding
            for g, th in ths.items():
                emb.fn8_prestage(emb.compute_ngram_ids(th), g)

    def _assemble_ple_table(self) -> None:
'''

# --- E10: assemble-tail bank init --------------------------------------------------
OLD_E10 = '''        logger.info("Qwen4Exp PLE: host table ready (%s layout)",
                    "block128" if emb._col_block_size else "per-tensor/passthrough")
'''
NEW_E10 = '''        logger.info("Qwen4Exp PLE: host table ready (%s layout)",
                    "block128" if emb._col_block_size else "per-tensor/passthrough")
        if _ple_env_flag("FN8_PLE_PRESTAGE"):
            # 06f: banks must exist before profile/capture so every decode
            # graph bakes the fast path (device at post-load = model device).
            emb.fn8_prestage_init(next(self.parameters()).device)
'''

EDITS = [
    ("E1_atexit", OLD_E1, NEW_E1),
    ("E2_init_attrs", OLD_E2, NEW_E2),
    ("E3_fwd_sig", OLD_E3, NEW_E3),
    ("E3B_fastpath", OLD_E3B, NEW_E3B),
    ("E4_methods", OLD_E4, NEW_E4),
    ("E5_ple_sig", OLD_E5, NEW_E5),
    ("E6_passthru", OLD_E6, NEW_E6),
    ("E7_spec_site", OLD_E7, NEW_E7),
    ("E8_ns_site", OLD_E8, NEW_E8),
    ("E9_model_prestage", OLD_E9, NEW_E9),
    ("E10_assemble_init", OLD_E10, NEW_E10),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06f.bak"
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


for name, old, _new in EDITS:
    n = src.count(ed(old))
    if n != 1:
        print(f"{TAG} FAIL: anchor '{name}' occurs {n}x (want 1)")
        sys.exit(2)

bak = path + ".fn8_06f.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src
for name, old, new in EDITS:
    out = out.replace(ed(old), ed(new))

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
