#!/usr/bin/env python3
"""FN8_06k (DEC40): spec-verify padded-tail probe + harden.

Forensics (d9 MTP boots, banked /stage/serve_d9.log; d8s3 crash-5 era
identical): worker dies at the FIRST spec verify step with

    Indexing.h:622: global id [5..24]: index out of bounds

= a 25-wide index op where only slots 0..4 hold valid indices
(bs=1 x (k+1)=5 real, 20 padded garbage).  d9b (identical stack, NO
spec) survived 1536 decode tokens => tier gather + PLE bank fast path
+ flat-path capture fully exonerated; the bug is spec-only machinery.
Drafter runs graph-mode NONE (v22 eager-head law) => unpadded => the
25-wide op is on the TARGET side: the captured FULL_DECODE_ONLY verify
graph replays over PADDED token slots, indexing shared buffers whose
padded tails are only initialized at capture time (dummies) — or
never.  Prime suspects, all indexed ops with padded tails:
  - input_ids persistent buffer tail past num_actual_tokens
  - attn_metadata.slot_mapping tail (in-graph KV scatter index!)
  - input_batch.prev_sampled_token_ids [padded_bs, k+1] (=5x5=25
    slots on this posture; async-scheduling path only) — the exact
    assert signature if rows 1..4 are garbage.

Edits:
  K1  Insert at the 06g prestage hook site (eager pre-dispatch, runs
      before "# Run the model." => before the verify graph replay;
      .item() probes are legal here — 06h flat prestage has synced
      here every step since without issue):
      - ONE-SHOT SPEC_PROBE INFO line: shapes + global min/max of
        input_ids, slot_mapping, prev_sampled_token_ids,
        cu_num_draft_tokens.  Prints BEFORE the crash => banked
        either way.
      - PER-STEP TAIL HARDEN (spec steps only): zero-fill input_ids
        and slot_mapping tails past num_actual_tokens, and
        prev_sampled_token_ids rows past input_batch.num_reqs.
        Zero = padding_token_id / slot 0 semantics already used by
        the capture dummies; padded rows are masked downstream, so
        this changes no real token.  If the garbage-tail hypothesis
        is right, this alone survives the first spec step.

Idempotent; py_compile gate; backup .fn8_06k.bak; EOL aware;
anchor==1; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06k"
REL = "v1/worker/gpu_model_runner.py"
MARKER = "FN8_06k"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

OLD_LF = '''        _fn8_ple_ps = getattr(self.model, "fn8_ple_prestage", None)
        if _fn8_ple_ps is not None:
'''
NEW_LF = '''        # FN8_06k (DEC40): spec-verify padded-tail probe + harden.
        # d9 dies at the FIRST verify replay with Indexing.h:622 ids
        # [5..24] of a 25-wide index op (bs1 x (k+1)=5 real + 20 padded
        # garbage).  Probe once (banked before any crash), then zero the
        # padded tails every spec step: 0 = padding_token_id / slot-0
        # semantics the capture dummies already used; padded rows are
        # masked downstream, so no real token changes.
        try:
            _fn8_smd = getattr(attn_metadata, "spec_decode_metadata", None)
            if _fn8_smd is not None:
                _fn8_nreal = int(getattr(attn_metadata, "num_actual_tokens",
                                         input_ids.shape[0]))
                if not globals().get("_FN8_06K_PROBED", False):
                    globals()["_FN8_06K_PROBED"] = True
                    _fn8_sm = getattr(attn_metadata, "slot_mapping", None)
                    _fn8_pst = getattr(self.input_batch,
                                       "prev_sampled_token_ids", None)
                    _fn8_cnd = getattr(_fn8_smd, "cu_num_draft_tokens", None)
                    logger.info(
                        "FN8_06k SPEC_PROBE nreal=%d padded=%d ii_dtype=%s "
                        "ii_min=%s ii_max=%s sm_len=%s sm_min=%s sm_max=%s "
                        "pst=%s cnd=%s",
                        _fn8_nreal, input_ids.shape[0], input_ids.dtype,
                        int(input_ids.to(torch.int64).min().item()),
                        int(input_ids.to(torch.int64).max().item()),
                        (_fn8_sm.shape[0] if _fn8_sm is not None else -1),
                        (int(_fn8_sm.to(torch.int64).min().item())
                         if _fn8_sm is not None else -1),
                        (int(_fn8_sm.to(torch.int64).max().item())
                         if _fn8_sm is not None else -1),
                        (f"{tuple(_fn8_pst.shape)}min="
                         f"{int(_fn8_pst.min().item())}max="
                         f"{int(_fn8_pst.max().item())}"
                         if _fn8_pst is not None else "NA"),
                        (f"{tuple(_fn8_cnd.shape)}min="
                         f"{int(_fn8_cnd.min().item())}max="
                         f"{int(_fn8_cnd.max().item())}"
                         if _fn8_cnd is not None else "NA"))
                # Tail harden (every spec step; padded rows masked out).
                if input_ids.shape[0] > _fn8_nreal:
                    input_ids[_fn8_nreal:].zero_()
                _fn8_sm2 = getattr(attn_metadata, "slot_mapping", None)
                if _fn8_sm2 is not None and _fn8_sm2.shape[0] > _fn8_nreal:
                    _fn8_sm2[_fn8_nreal:].zero_()
                _fn8_pst2 = getattr(self.input_batch,
                                    "prev_sampled_token_ids", None)
                _fn8_nreq = getattr(self.input_batch, "num_reqs", None)
                if (_fn8_pst2 is not None and _fn8_nreq is not None
                        and _fn8_pst2.shape[0] > _fn8_nreq):
                    _fn8_pst2[_fn8_nreq:].zero_()
        except Exception as _fn8_6k_e:  # noqa: BLE001 (telemetry mandate)
            logger.warning("FN8_06k probe/harden failed: %s", _fn8_6k_e)
        _fn8_ple_ps = getattr(self.model, "fn8_ple_prestage", None)
        if _fn8_ple_ps is not None:
'''


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06k.bak"
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


n = src.count(ed(OLD_LF))
if n != 1:
    print(f"{TAG} FAIL: anchor 'ple_ps hook' occurs {n}x (want 1)")
    sys.exit(2)

bak = path + ".fn8_06k.bak"
if not os.path.exists(bak):
    shutil.copyfile(path, bak)
    print(f"{TAG} backup {bak} md5={md5(bak)}")

out = src.replace(ed(OLD_LF), ed(NEW_LF))

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
