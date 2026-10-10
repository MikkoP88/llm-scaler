#!/usr/bin/env python3
"""P85 F1 patcher — rid-keyed accepted-count remap (python-only, zero-perf).

Target lineage: v1.2.31 gpu_model_runner.py (md5 6269b2133b013a8b1fd55c78909a585d).
Patches gpu_model_runner.py.orig -> gpu_model_runner.py on the STAGE dir only;
never touches the live container. Strict anchored replaces (assert count==1),
py_compile gate, unified diff + md5 ledger output.

Root cause addressed (P85 Phase-1 verdict): async carry-index desync. The
wait-side remap of the pinned num_accepted_tokens side-channel trusts the
POSITIONAL prev_positions mapping to describe exactly the record->wait batch
transition. A finish/preempt/condense in between describes a different
transition -> a row receives another request's accepted count -> the GDN
kernel loads/rolls the wrong rollback column (v52e NaN class) and the
num_computed GPU correction is torn (deep-forward metadata corruption route).

Fix (three coordinated CPU-only pieces):
  1. Record stash (both align/non-align branches): rid->row index of the
     recording step + mirror of the vsc epoch (see 3).
  2. Wait-side rid-keyed remap with envelope clamp [1, num_spec_tokens+1],
     PLUS per-row repair of prev_positions to the rid-truthful record row
     (-1 when the request was absent). prev_positions[i] for a
     broken-lineage row is a FALSE positional claim; every consumer of it
     gathers another request's data: the num_computed GPU correction
     (metadata tear / M2), the accepted-count where() overwrite (v52e
     rollback column), and _prepare_input_ids' prev_sampled_token_ids
     injection (cross-request token contamination). The repair makes the
     tensor's contract ("current pos -> prev pos") true per row; it is a
     no-op when lineages are intact (rec row == claimed row by
     construction).
  3. vsc freshness gate: valid_sampled_token_count_gpu is only reassigned
     on the drafter FALLBACK path (not input_fits_in_drafter). After a
     fits-path step the tensor holds an OLDER step's row order while the
     record stash advanced -> the correction gather reads stale rows into
     corrected = prev_computed + valid_counts. Fresh := the epoch
     advanced between the previous record-step mirror and this record
     step's stash (a write happened during the recording step). On stale
     steps the gate routes to the existing CPU fallback branch. Epoch is
     bumped where the tensor is assigned in _copy_valid_sampled_token_count.

Zero GPU ops, zero barriers, zero events: <=64 dict inserts at record +
<=64 dict lookups/repairs at wait per step + one int compare.
"""
import difflib
import hashlib
import py_compile
import sys

SRC = "/root/build/p85_stage/gpu_model_runner.py.orig"
DST = "/root/build/p85_stage/gpu_model_runner.py"

with open(SRC, "r", encoding="utf-8") as f:
    ORIG = f.read()

# Pre-flight asserts on the lineage we were built against.
assert hashlib.md5(ORIG.encode()).hexdigest() == \
    "6269b2133b013a8b1fd55c78909a585d", "orig md5 drift — re-extract"
assert "logger = init_logger" in ORIG, "module logger missing"
assert "self.num_spec_tokens" in ORIG, "runner spec-token attr missing"
assert "\nimport os\n" in ORIG, "os import missing (ledger env read)"


def rep(text: str, old: str, new: str) -> str:
    n = text.count(old)
    assert n == 1, "anchor count %d != 1 for: %r" % (n, old[:90])
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# Patch 1a — record side, ALIGN branch: rid->row index stash of the
# recording step (sibling of the existing _v32_pending_pp snapshot).
# ---------------------------------------------------------------------------
P1A_OLD = """                for _rid in list(self.input_batch.req_ids)[:num_reqs]
            ]
        else:"""

P1A_NEW = """                for _rid in list(self.input_batch.req_ids)[:num_reqs]
            ]
            # llm-scaler v85 F1 (P85): rid-keyed row index of the recording
            # step. The wait-side remap must key accepted counts by request
            # identity, not by prev_positions alone: finish/preempt/condense
            # between record and wait can describe a different row
            # transition than the one these counts were recorded under,
            # feeding the next step's GDN metadata another request's
            # rollback column (v52e NaN class) and tearing the num_computed
            # GPU correction (deep-forward metadata route).
            self._v85_rec_index = {
                _rid: _i
                for _i, _rid in enumerate(
                    list(self.input_batch.req_ids)[:num_reqs]
                )
            }
            self._v85_rec_vsc_epoch = getattr(self, "_v85_vsc_epoch", None)
            # llm-scaler v85 F1 (P85): fresh = a vsc write happened during
            # THIS recording step (epoch advanced past the previous
            # record-step mirror). Simple mirror equality would hold
            # trivially on later fits-path steps — the stale case itself.
            self._v85_vsc_fresh = (
                getattr(self, "_v85_vsc_epoch", None) is not None
                and getattr(self, "_v85_prev_mirror", None) is not None
                and self._v85_vsc_epoch != self._v85_prev_mirror
            )
            self._v85_prev_mirror = self._v85_rec_vsc_epoch
        else:"""

# ---------------------------------------------------------------------------
# Patch 1b — record side, NON-ALIGN branch: same stash (the pinned tensor
# carries the same step-N row order there).
# ---------------------------------------------------------------------------
P1B_OLD = """            # llm-scaler v55.3: recording-step shape stash (non-align).
            self._v55_rec_step_tokens = (
                scheduler_output.total_num_scheduled_tokens)"""

P1B_NEW = """            # llm-scaler v55.3: recording-step shape stash (non-align).
            self._v55_rec_step_tokens = (
                scheduler_output.total_num_scheduled_tokens)
            # llm-scaler v85 F1 (P85): rid-keyed row index of the recording
            # step (see the align-branch twin for the full rationale).
            self._v85_rec_index = {
                _rid: _i
                for _i, _rid in enumerate(
                    list(self.input_batch.req_ids)[:num_reqs]
                )
            }
            self._v85_rec_vsc_epoch = getattr(self, "_v85_vsc_epoch", None)
            # llm-scaler v85 F1 (P85): fresh = a vsc write happened during
            # THIS recording step (epoch advanced past the previous
            # record-step mirror). Simple mirror equality would hold
            # trivially on later fits-path steps — the stale case itself.
            self._v85_vsc_fresh = (
                getattr(self, "_v85_vsc_epoch", None) is not None
                and getattr(self, "_v85_prev_mirror", None) is not None
                and self._v85_vsc_epoch != self._v85_prev_mirror
            )
            self._v85_prev_mirror = self._v85_rec_vsc_epoch"""

# ---------------------------------------------------------------------------
# Patch 1c — vsc epoch bump where valid_sampled_token_count_gpu is
# assigned (drafter fallback path only). Lets the wait side prove the
# tensor belongs to the recording step (fits-path steps leave it stale).
# ---------------------------------------------------------------------------
P1C_OLD = """        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count"""

P1C_NEW = """        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count
            # llm-scaler v85 F1 (P85): epoch of this assignment — the GPU
            # correction may only gather from it while the recording step's
            # stash mirrors this epoch (fallback-path-only assignment means
            # fits-path steps leave an older step's row order in place).
            self._v85_vsc_epoch = getattr(self, "_v85_vsc_epoch", 0) + 1"""

# ---------------------------------------------------------------------------
# Patch 2 — wait side (_update_states): rid-keyed remap with envelope clamp,
# replacing the positional prev_positions gather. The legacy positional
# gather is retained for the non-async reordered case (unchanged behavior).
# ---------------------------------------------------------------------------
P2_OLD = """            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )"""

P2_NEW = """            # Async mode: llm-scaler v85 F1 (P85) — rid-keyed remap +
            # rid-truthful prev_positions repair.
            # prev_positions[i] is a POSITIONAL claim about the recording
            # step; after finish/preempt/condense it can describe a
            # different transition than the one the pinned counts and the
            # GPU-side tensors were recorded under. Every positional
            # consumer then gathers another request's data: the accepted
            # count selects a wrong GDN rollback column (v52e NaN class),
            # the num_computed correction tears next-step metadata (M2),
            # and _prepare_input_ids injects another request's sampled
            # tokens. Repair each row to the rid-truthful record row (-1
            # when absent: the truthful new-row value, and the code's own
            # safe default). No-op when lineages are intact. Also clamp
            # accepted counts to the structural envelope [1, num_spec+1].
            # CPU-only, <= num_reqs dict lookups per step.
            if self.use_async_scheduling:
                if not globals().get("_v85_f1_armed"):
                    globals()["_v85_f1_armed"] = True
                    logger.info(
                        "llm-scaler v85 F1: rid-keyed accepted remap ACTIVE")
                _v85_idx = getattr(self, "_v85_rec_index", None) or {}
                _v85_cpu = self.input_batch.num_accepted_tokens_cpu
                _v85_lim = self.num_spec_tokens + 1
                _v85_prev = self.prev_positions.np[:num_reqs]
                _v85_rep = 0
                for _i, _rid in enumerate(
                    self.input_batch.req_ids[:num_reqs]
                ):
                    _v85_j = _v85_idx.get(_rid, -1)
                    if _v85_prev[_i] != _v85_j:
                        _v85_prev[_i] = _v85_j
                        _v85_rep += 1
                    _v85_v = _v85_cpu[_v85_j] if _v85_j >= 0 else 1
                    if _v85_v < 1:
                        _v85_v = 1
                    elif _v85_v > _v85_lim:
                        _v85_v = _v85_lim
                    self.num_accepted_tokens.np[_i] = _v85_v
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
                if _v85_rep and not globals().get("_v85_rep_logged"):
                    globals()["_v85_rep_logged"] = True
                    logger.info(
                        "llm-scaler v85 F1: prev_positions repaired "
                        "(first occurrence: %d/%d rows)", _v85_rep, num_reqs)
            elif prev_req_id_to_index:
                # Async off but batch reordered: legacy positional remap.
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )"""

# ---------------------------------------------------------------------------
# Patch 4 — forensic step ledger (env-gated VLLM_V85_LEDGER=<path>): one
# line per async spec step with the rid-keyed accepted vector, repair
# count and freshness. Validation soak builds only; a single truthiness
# check per step when unset. Post-mortem: any value outside
# [1, num_spec_tokens+1] or a repair burst preceding a reset = direct
# M1/M2 confirmation from host side.
# ---------------------------------------------------------------------------
P4_OLD = """logger = init_logger(__name__)"""

P4_NEW = """logger = init_logger(__name__)
# llm-scaler v85 (P85): forensic step ledger — env-gated, validation only.
_V85_LEDGER_PATH = os.environ.get("VLLM_V85_LEDGER", "")"""

# ---------------------------------------------------------------------------
# Patch 5 — ledger write at the end of the async remap loop.
# ---------------------------------------------------------------------------
P5_OLD = """                if _v85_rep and not globals().get("_v85_rep_logged"):"""

P5_NEW = """                if _V85_LEDGER_PATH:
                    self._v85_step_no = getattr(self, "_v85_step_no", 0) + 1
                    with open(_V85_LEDGER_PATH, "a", encoding="utf-8") as _lf:
                        _lf.write("%d n=%d rep=%d acc=%s\\n" % (
                            self._v85_step_no, num_reqs, _v85_rep,
                            " ".join(map(str, self.num_accepted_tokens.np[:num_reqs]))))
                if _v85_rep and not globals().get("_v85_rep_logged"):"""

# ---------------------------------------------------------------------------
# Patch 3 — GPU-correction gate: require vsc freshness (the fallback-path
# side copy fired in the recording step). Stale steps take the existing
# CPU fallback branch — the upstream-consistent value, never torn.
# ---------------------------------------------------------------------------
P3_OLD = """        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):"""

P3_NEW = """        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
            # llm-scaler v85 F1 (P85): gather only from a same-step
            # valid_sampled_token_count_gpu (see the epoch bump where it
            # is assigned). Default True preserves the first-step legacy
            # behavior before any stash exists.
            and getattr(self, "_v85_vsc_fresh", True)
        ):"""


def main() -> int:
    text = ORIG
    text = rep(text, P2_OLD, P2_NEW)   # wait side first: its anchor text
    text = rep(text, P1A_OLD, P1A_NEW)  # is disjoint from the record stashes
    text = rep(text, P1B_OLD, P1B_NEW)
    text = rep(text, P1C_OLD, P1C_NEW)
    text = rep(text, P4_OLD, P4_NEW)
    text = rep(text, P5_OLD, P5_NEW)
    text = rep(text, P3_OLD, P3_NEW)

    with open(DST, "w", encoding="utf-8") as f:
        f.write(text)

    py_compile.compile(DST, doraise=True)

    diff = list(difflib.unified_diff(
        ORIG.splitlines(True), text.splitlines(True),
        fromfile="gpu_model_runner.py.orig", tofile="gpu_model_runner.py"))
    sys.stdout.writelines(diff)
    print("\n--- md5 ledger ---")
    print("orig:    %s" % hashlib.md5(ORIG.encode()).hexdigest())
    print("patched: %s" % hashlib.md5(text.encode()).hexdigest())
    print("lines:   %d -> %d" % (len(ORIG.splitlines()), len(text.splitlines())))
    print("PY_COMPILE: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
