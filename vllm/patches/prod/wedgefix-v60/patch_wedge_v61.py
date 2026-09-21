#!/usr/bin/env python3
"""llm-scaler v61 wedge patcher (WEDGEFIX-F): REPLICATED MTP DRAFTER +
spec draft-barrier default OFF. The #11 root fix.

Root cause (KNOWN_ISSUES #11, GSD-12919; 2026-09-21 v60 drills): with
TP>1 the eager MTP drafter issues ~12 oneCCL collectives per propose
(4 passes x [fc gather + o_proj AR + mlp AR]) + vocab-parallel
top-token gathers BETWEEN the target's graphed verify replays. When
the two ranks reach those eager collectives skewed (one host still
draining the post-verify D2H, the peer already submitting head
collectives), oneCCL's non-preemptible spin-wait kernels miss the
rendezvous, the UR in-flight window fills, both engines burn
(Compute 100% + Copy 100%, EU stalled; f15b ring: drafter ARs
numel=25600 DONE then silence, propose_gpu_done PENDING; xe ccs/bcs
engine resets both tiles; 0 tok/s -> v55.3 fast-clean SIGKILL).
Trigger is probabilistic per step, NOT context-gated (drill3 died at
computed=3994). The v60 mitigation (WEDGEFIX-B: full-device
torch.xpu.synchronize() before the drafter EVERY step) prevents it by
forcing empty, symmetric device queues — at a measured -9..-31% decode
throughput.

ROOT FIX (this patch): construct, load, and run the MTP head
REPLICATED on every rank — a TP=1 view during Qwen3_5MTP construction
and weight loading bakes tp_size=1 semantics into every parallel layer
(full vocab embedding, full fc, full qkv/o_proj/down_proj, full
lm_head, full attention heads). The draft forward then issues ZERO
eager oneCCL collectives: the fc all-gather, the 12 layer ARs, and the
sharded top-token gathers all disappear by construction (layers skip
collectives when their baked tp_size == 1; get_top_tokens early-
returns on a full-vocab local argmax). With no eager draft collectives
there is no rendezvous to skew — the wedge TRIGGER CLASS is gone, and
the per-step drain barrier becomes unnecessary:

  F1 (qwen3_5_mtp.py):  TP1-view machinery + wrap Qwen3_5MTP.__init__
                        and .load_weights (recurse-once guard); logs
                        REPLICATED-DRAFTER shape proof at build.
                        VLLM_XPU_REPLICATED_DRAFT=0 restores the stock
                        TP-sharded drafter.
  F2 (logits_processor.py): get_top_tokens derives tp_size from the
                        lm_head's baked tp_size (replicated head =>
                        local argmax IS global; no gather). Stock
                        sharded heads keep identical behavior.
  F3 (gpu_model_runner.py): VLLM_XPU_SPEC_DRAFT_BARRIER default 1->0
                        (barrier no longer required; =1 restores the
                        v60 drain for diagnosis).

Costs of replication (documented, measured at boot): +~1.5 GB weights
per rank (full embed + lm_head + MTP layer vs sharded) and the draft
KV pool doubles per rank (num_key_value_heads 4: 2/rank sharded ->
4/rank replicated). KV capacity delta is logged by the standard pool
sizing lines.

Speculative decoding (MTP x4), XGrammar-2 (0.2.7), STALFIX, v60 A/C/E
all untouched. Spec and XGrammar-2 remain fully supported on every
image (standing directive).

Usage:  python3 patch_wedge_v61.py [--check|--apply|--revert]
Idempotent; anchor-count==1 asserted; backups <file>.v61bak;
py_compile after write with auto-restore on failure.
"""

import os
import py_compile
import shutil
import sys

SP = "/opt/venv/lib/python3.12/site-packages/vllm"
F_MTP = f"{SP}/model_executor/models/qwen3_5_mtp.py"
F_LP = f"{SP}/model_executor/layers/logits_processor.py"
F_GMR = f"{SP}/v1/worker/gpu_model_runner.py"
F_PRP = f"{SP}/v1/spec_decode/llm_base_proposer.py"

# --------------------------------------------------------- F1: qwen3_5_mtp
# F1a: module-scope machinery, inserted BEFORE the @support_torch_compile
# decorator (never between a decorator and its class).
F1A_OLD = """\
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
"""

F1A_NEW = '''\
# llm-scaler v61 (WEDGEFIX-F) ------------------------------------------------
# Replicated MTP drafter: build/load the head under a TP=1 view so every
# parallel layer bakes full-width semantics (see patch_wedge_v61.py for the
# full root-cause record). No eager oneCCL collectives remain in the draft
# step => the KNOWN_ISSUES #11 rendezvous wedge trigger class is gone and
# the per-step VLLM_XPU_SPEC_DRAFT_BARRIER drain is no longer required.
def _v61_active() -> bool:
    import os as _os

    from vllm.platforms import current_platform as _cp

    try:
        return (
            _os.environ.get("VLLM_XPU_REPLICATED_DRAFT", "1") == "1"
            and _cp.is_xpu()
        )
    except Exception:
        return False


class _V61Tp1View:
    """Present TP=1 semantics to every module that imported parallel_state's
    tensor-parallel getters, matched by function-object identity across
    sys.modules (covers from-vllm.distributed and from-...parallel_state
    import styles alike). Restore-on-exit; nest-safe (inner no-op when the
    global TP world size is already 1)."""

    def __enter__(self):
        import sys as _sys

        import vllm.distributed.parallel_state as _ps

        self._ws = _ps.get_tensor_model_parallel_world_size
        self._rk = _ps.get_tensor_model_parallel_rank
        self._saved: list = []
        if self._ws() == 1:
            return self
        _seen: set = set()
        for _mod in list(_sys.modules.values()):
            _d = getattr(_mod, "__dict__", None)
            if not isinstance(_d, dict) or id(_d) in _seen:
                continue
            _seen.add(id(_d))
            if _d.get("get_tensor_model_parallel_world_size") is self._ws:
                _d["get_tensor_model_parallel_world_size"] = lambda: 1
                self._saved.append(
                    (_d, "get_tensor_model_parallel_world_size", self._ws))
            if _d.get("get_tensor_model_parallel_rank") is self._rk:
                _d["get_tensor_model_parallel_rank"] = lambda: 0
                self._saved.append(
                    (_d, "get_tensor_model_parallel_rank", self._rk))
        return self

    def __exit__(self, *exc):
        for _d, _name, _orig in self._saved:
            _d[_name] = _orig
        return False


def _v61_log_shapes(m) -> None:
    try:
        _parts = []
        _pred = getattr(m, "model", None)
        _emb = getattr(_pred, "embed_tokens", None)
        if _emb is not None and getattr(_emb, "weight", None) is not None:
            _parts.append(f"embed={tuple(_emb.weight.shape)}")
        _lm = getattr(m, "lm_head", None)
        if _lm is not None and getattr(_lm, "weight", None) is not None:
            _parts.append(f"lm_head={tuple(_lm.weight.shape)}")
        _fc = getattr(_pred, "fc", None)
        if _fc is not None and getattr(_fc, "weight", None) is not None:
            _parts.append(f"fc={tuple(_fc.weight.shape)}")
        _layers = getattr(_pred, "layers", None)
        if _layers is not None and len(_layers):
            _attn = getattr(_layers[0], "self_attn", None)
            if _attn is not None:
                _parts.append(
                    "attn_heads=%d/%d kv=%d/%d"
                    % (getattr(_attn, "num_heads", -1),
                       getattr(_attn, "total_num_heads", -1),
                       getattr(_attn, "num_kv_heads", -1),
                       getattr(_attn, "total_num_kv_heads", -1)))
        logger.info(
            "llm-scaler v61 REPLICATED-DRAFTER shapes: %s",
            " ".join(_parts) if _parts else "(none)")
    except Exception:
        pass


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
'''

# F1b: wrap Qwen3_5MTP.__init__ (recurse-once under the TP1 view).
F1B_OLD = """\
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
"""

F1B_NEW = """\
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # llm-scaler v61 (WEDGEFIX-F): construct the MTP head REPLICATED
        # (TP=1 view) on every rank — zero eager draft collectives, the
        # KNOWN_ISSUES #11 wedge trigger class removed. Guard re-invokes
        # this __init__ exactly once under the view; the nested call runs
        # the original body. VLLM_XPU_REPLICATED_DRAFT=0 restores stock.
        if _v61_active() and not getattr(self, "_v61_in_tp1", False):
            self._v61_in_tp1 = True
            try:
                with _V61Tp1View():
                    self.__init__(vllm_config=vllm_config, prefix=prefix)
                _v61_log_shapes(self)
            finally:
                self._v61_in_tp1 = False
            return
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
"""

# F1c: wrap Qwen3_5MTP.load_weights the same way (the parallel-layer
# weight loaders slice by their baked tp_rank/tp_size).
F1C_OLD = """\
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
"""

F1C_NEW = """\
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # llm-scaler v61 (WEDGEFIX-F): load under the same TP=1 view used
        # for construction so every rank materializes the FULL head weights
        # (the layer objects baked tp_size=1 at construction).
        if _v61_active() and not getattr(self, "_v61_in_tp1", False):
            self._v61_in_tp1 = True
            try:
                with _V61Tp1View():
                    return self.load_weights(weights)
            finally:
                self._v61_in_tp1 = False

        def remap_weight_names(weights):
"""

# F1d: llm_base_proposer.py — do NOT swap the target's TP-sharded
# embed_tokens/lm_head into the (replicated) MTP drafter; sharing would
# reintroduce the draft-side vocab collectives v61 removes.
F1D_OLD = """\
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )
"""

F1D_NEW = """\
            else:
                # MTP model
                from vllm.model_executor.models.qwen3_5_mtp import _v61_active
                if _v61_active():
                    share_embeddings = False
                    logger.info(
                        "llm-scaler v61 (WEDGEFIX-F): draft keeps its own "
                        "full-vocab replicated embed_tokens; sharing the "
                        "target's TP-sharded module would reintroduce draft "
                        "collectives."
                    )
                else:
                    share_embeddings = True
                    logger.info(
                        "Detected MTP model. "
                        "Sharing target model embedding weights with the draft model."
                    )
"""

F1E_OLD = """\
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )
"""

F1E_NEW = """\
        else:
            # MTP model
            from vllm.model_executor.models.qwen3_5_mtp import _v61_active
            if _v61_active():
                share_lm_head = False
                logger.info(
                    "llm-scaler v61 (WEDGEFIX-F): draft keeps its own "
                    "full-vocab replicated lm_head; sharing the target's "
                    "TP-sharded module would reintroduce draft collectives."
                )
            else:
                share_lm_head = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model lm_head weights with the draft model."
                )
"""

# -------------------------------------------------- F2: logits_processor.py
F2_OLD = """\
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()
"""

F2_NEW = """\
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        # llm-scaler v61 (WEDGEFIX-F): derive tp_size from the lm head's own
        # baked-in tp_size rather than the global TP group — the v61
        # replicated MTP drafter runs a tp_size=1 lm_head inside a TP>1 job,
        # so its local argmax over the full vocabulary IS the global argmax
        # (early return below, no gather). Stock sharded heads carry
        # tp_size == global TP and keep identical behavior.
        tp_size = (
            getattr(lm_head, "tp_size", None)
            or get_tensor_model_parallel_world_size()
        )
"""

# F4: gpu_model_runner.py get_kv_cache_spec — the replicated drafter has
# double the KV heads per rank, so its page is 2x every target page and
# unify_kv_cache_spec_page_size cannot reconcile it (MambaSpec's
# page_size_padded is a fixed contract; growing its block cannot raise the
# padded value -> assert). Halve the DRAFT layer's block size instead: its
# page then equals the unified target/mamba page, unify early-returns, and
# the stock mamba/attention alignment dance stays untouched. Draft-only and
# purely arithmetic: with the stock 2-head drafter the page already matches
# and this is a no-op (also under VLLM_XPU_REPLICATED_DRAFT=0).
F4_OLD = """\
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec
"""

F4_NEW = """\
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        # llm-scaler v61 (WEDGEFIX-F): the replicated MTP drafter carries
        # double the KV heads per rank (full 4 instead of sharded 2), so its
        # page is 2x every target page and the page-unification pass cannot
        # reconcile the two (the mamba page_size_padded set by the XPU
        # alignment hook is a fixed allocation contract; growing its block
        # cannot raise the padded value -> unify assert). Halve the draft
        # layer's block size instead: the draft page then equals the
        # unified target/mamba page, unification early-returns untouched,
        # and physical KV bytes per token are unchanged (half the tokens per
        # page, double the bytes per token — the drafter holds 4 kv
        # heads/rank by design). No-op for the stock sharded drafter.
        _draft_names = set(
            getattr(getattr(self, "drafter", None), "_draft_attn_layer_names", ())
        )
        if kv_cache_spec and _draft_names:
            _tgt_pages = [
                s.page_size_bytes
                for n, s in kv_cache_spec.items()
                if n not in _draft_names
            ]
            if _tgt_pages:
                _tgt_page = min(_tgt_pages)
                for _n in sorted(_draft_names & set(kv_cache_spec)):
                    _s = kv_cache_spec[_n]
                    _r = _s.page_size_bytes // _tgt_page
                    if (
                        _s.page_size_bytes > _tgt_page
                        and _s.page_size_bytes % _tgt_page == 0
                        and _r > 1
                        and _s.block_size % _r == 0
                    ):
                        kv_cache_spec[_n] = _s.copy_with_new_block_size(
                            _s.block_size // _r
                        )
                        logger.info(
                            "llm-scaler v61 (WEDGEFIX-F): draft KV pool block "
                            "halved for %s: %d -> %d tokens (page %d -> %d, "
                            "matches target/mamba unified page)",
                            _n,
                            _s.block_size,
                            _s.block_size // _r,
                            _s.page_size_bytes,
                            _tgt_page,
                        )

        return kv_cache_spec
"""

# ------------------------------------------------- F3: gpu_model_runner.py
F3_OLD = """\
_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "1") == "1"
)
"""

F3_NEW = """\
# llm-scaler v61 (WEDGEFIX-F): barrier DEFAULT OFF. The v61 replicated
# MTP drafter issues ZERO eager oneCCL collectives in the draft step
# (full-width head baked under a TP=1 view), removing the rank-skew
# rendezvous hazard the v60 drain was protecting against — the barrier
# is no longer required and cost -9..-31% decode throughput.
# VLLM_XPU_SPEC_DRAFT_BARRIER=1 restores the v60 every-step drain for
# diagnosis; VLLM_XPU_REPLICATED_DRAFT=0 restores the stock sharded
# drafter (barrier strongly recommended in that mode).
_SPEC_DRAFT_BARRIER = (
    os.environ.get("VLLM_XPU_SPEC_DRAFT_BARRIER", "0") == "1"
)
"""

PATCHES = [
    ("F1a", F_MTP, "WEDGEFIX-F replicated-drafter machinery (v61)",
     [(F1A_OLD, F1A_NEW, "llm-scaler v61 (WEDGEFIX-F) ---")]),
    ("F1b", F_MTP, "WEDGEFIX-F MTP __init__ TP1 wrap (v61)",
     [(F1B_OLD, F1B_NEW, "llm-scaler v61 (WEDGEFIX-F): construct the MTP head")]),
    ("F1c", F_MTP, "WEDGEFIX-F MTP load_weights TP1 wrap (v61)",
     [(F1C_OLD, F1C_NEW, "llm-scaler v61 (WEDGEFIX-F): load under the same TP=1 view")]),
    ("F1d", F_PRP, "WEDGEFIX-F keep draft embed_tokens (v61)",
     [(F1D_OLD, F1D_NEW, "llm-scaler v61 (WEDGEFIX-F): draft keeps its own ")]),
    ("F1e", F_PRP, "WEDGEFIX-F keep draft lm_head (v61)",
     [(F1E_OLD, F1E_NEW, "llm-scaler v61 (WEDGEFIX-F): draft keeps its own ")]),
    ("F2", F_LP, "WEDGEFIX-F get_top_tokens lm_head tp_size (v61)",
     [(F2_OLD, F2_NEW, "llm-scaler v61 (WEDGEFIX-F): derive tp_size from the lm head")]),
    ("F4", F_GMR, "WEDGEFIX-F draft KV page unify (v61)",
     [(F4_OLD, F4_NEW, "llm-scaler v61 (WEDGEFIX-F): draft KV pool block ")]),
    ("F3", F_GMR, "WEDGEFIX-F draft-barrier default OFF (v61)",
     [(F3_OLD, F3_NEW, "llm-scaler v61 (WEDGEFIX-F): barrier DEFAULT OFF")]),
]


def die(msg: str) -> None:
    print(f"[v61] FAIL: {msg}")
    sys.exit(1)


def check_apply(revert: bool) -> None:
    for tag, path, label, pairs in PATCHES:
        try:
            src = open(path, encoding="utf-8").read()
        except OSError as e:
            die(f"{label}: cannot read {path}: {e}")
        bak = path + ".v61bak"
        if revert:
            if os.path.exists(bak):
                shutil.copyfile(bak, path)
                print(f"[v61] {tag} REVERTED {path}")
            else:
                print(f"[v61] {tag} no-backup (unchanged) {path}")
            continue
        applied_any = False
        ok = True
        for old, new, nmark in pairs:
            if nmark in src:
                print(f"[v61] {tag} ALREADY-APPLIED ({nmark[:40]}…)")
                continue
            n = src.count(old)
            if n != 1:
                print(f"[v61] {tag} ANCHOR-COUNT {n} (need 1) in {path}")
                ok = False
                continue
            if not os.path.exists(bak):
                shutil.copyfile(path, bak)
            src = src.replace(old, new)
            applied_any = True
        if not ok:
            die(f"{label}: anchor check failed")
        if not applied_any:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            shutil.copyfile(bak, path)
            die(f"{label}: py_compile failed (restored): {e}")
        print(f"[v61] {tag} APPLIED {path}")
    print("[v61] pass complete")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--apply"
    if mode == "--check":
        for tag, path, label, pairs in PATCHES:
            try:
                src = open(path, encoding="utf-8").read()
            except OSError as e:
                die(f"{label}: {e}")
            for old, new, nmark in pairs:
                state = "ALREADY" if nmark in src else (
                    "ANCHOR-OK" if src.count(old) == 1 else "ANCHOR-BAD")
                print(f"[v61] {tag} {state} {path}")
        sys.exit(0)
    check_apply(revert=(mode == "--revert"))
