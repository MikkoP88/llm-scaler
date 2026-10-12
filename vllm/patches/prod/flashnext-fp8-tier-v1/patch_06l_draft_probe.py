#!/usr/bin/env python3
"""FN8_06l (DEC40): staged drafter probes for the first-spec-step abort.

Forensics closed so far (all banked /stage/bank/*, serve_d9.log 13:06
boot): d9 (MTP x4 + FULL graphs) dies at the FIRST spec decode step
with SYCL Indexing.h:622 — a 25-wide index op where ids 0..4 (bs x
(k+1) = 1 x 5) are valid and ids 5..24 assert.  06k's runner-side
probe NEVER printed => the abort fires inside propose(), before the
"Run the model." hook.  The four kernels JIT'ing at the death moment
(_zero_kv_blocks, _compute_slot_mapping, eagle_prepare_next_token_
padded, eagle_step_slot_mapping_metadata) are all DRAFT-phase.  The
drafter runs graph-mode NONE (v22 eager-head law) and unpadded, so
the garbage-index buffer is one the drafter feeds to an indexed op:
draft input_ids, positions, draft slot mapping, or token_indices.

Probe plan — staged ONE-SHOT prints (assert kills the worker, so the
LAST printed line pins the stage; eager .item() syncs are legal here):
  S1  after set_inputs_first_pass: num_tokens, token_indices_to_sample
      min/max, positions min/max.
  S2  before the FIRST head call: model_kwargs input_ids/positions
      min/max + slot_mapping_size.
  S3  before each draft-loop head call (pass 0..2): input_ids /
      positions / draft slot-mapping min/max + hidden finite.

Raw prints (flush=True) — worker stdout IS the serve log (the native
SYCL assert messages land there the same way).

Idempotent; py_compile gate; backup .fn8_06l.bak; EOL aware;
anchor==1 per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06l"
REL = "v1/spec_decode/llm_base_proposer.py"
MARKER = "FN8_06L"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)
path = os.path.join(PKG, REL)

# L1+S1: probe after set_inputs_first_pass (anchor: the no-draft_index
# build_per_group call, unique to the first-pass section).
OLD1_LF = '''        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
'''
NEW1_LF = '''        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        # FN8_06L S1: first-pass inputs staged (one-shot).
        try:
            if not globals().get("_FN8_06L_S1", False):
                globals()["_FN8_06L_S1"] = True
                print(
                    "FN8_06L S1 first_pass bs=%d num_tokens=%s tis=%s "
                    "tis_min=%s tis_max=%s pos_min=%s pos_max=%s" % (
                        batch_size, num_tokens,
                        tuple(token_indices_to_sample.shape),
                        int(token_indices_to_sample.to(torch.int64)
                            .min().item()),
                        int(token_indices_to_sample.to(torch.int64)
                            .max().item()),
                        int(self.positions[:num_tokens].to(torch.int64)
                            .min().item()),
                        int(self.positions[:num_tokens].to(torch.int64)
                            .max().item())),
                    flush=True)
        except Exception as _fn8_6l_e:  # noqa: BLE001 (telemetry mandate)
            print(f"FN8_06L S1 probe failed: {_fn8_6l_e}", flush=True)

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
'''

# L2+S2: probe around the FIRST head call (spec_seg block).
OLD2_LF = '''            with spec_seg("dforward"):  # llm-scaler v20 A2
                ret_hidden_states = self.model(**model_kwargs)
'''
NEW2_LF = '''            with spec_seg("dforward"):  # llm-scaler v20 A2
                # FN8_06L S2: first head-call inputs (one-shot).
                try:
                    if not globals().get("_FN8_06L_S2", False):
                        globals()["_FN8_06L_S2"] = True
                        _fn8_ii1 = model_kwargs.get("input_ids")
                        print(
                            "FN8_06L S2 first_head ii=%s ii_min=%s "
                            "ii_max=%s pos_min=%s pos_max=%s smsize=%s "
                            "sm_min=%s sm_max=%s" % (
                                (tuple(_fn8_ii1.shape)
                                 if _fn8_ii1 is not None else None),
                                (int(_fn8_ii1.to(torch.int64).min().item())
                                 if _fn8_ii1 is not None else -1),
                                (int(_fn8_ii1.to(torch.int64).max().item())
                                 if _fn8_ii1 is not None else -1),
                                int(model_kwargs["positions"]
                                    .to(torch.int64).min().item()),
                                int(model_kwargs["positions"]
                                    .to(torch.int64).max().item()),
                                slot_mapping_size,
                                int(self._slot_mapping_buffer[
                                    :slot_mapping_size].min().item()),
                                int(self._slot_mapping_buffer[
                                    :slot_mapping_size].max().item())),
                            flush=True)
                except Exception as _fn8_6l_e:  # noqa: BLE001
                    print(f"FN8_06L S2 probe failed: {_fn8_6l_e}",
                          flush=True)
                ret_hidden_states = self.model(**model_kwargs)
'''

# L3+S3: probe before each draft-loop head call.
OLD3_LF = '''                slot_mapping=self._get_slot_mapping(model_input_tokens),
            ):
                ret_hidden_states = self.model(**model_kwargs)
'''
NEW3_LF = '''                slot_mapping=self._get_slot_mapping(model_input_tokens),
            ):
                # FN8_06L S3: draft-pass head-call inputs (one-shot/pass).
                try:
                    if not globals().get("_FN8_06L_S3_%d" % token_index,
                                         False):
                        globals()["_FN8_06L_S3_%d" % token_index] = True
                        _fn8_sm3 = self._get_slot_mapping(model_input_tokens)
                        print(
                            "FN8_06L S3 pass=%d input_batch_size=%s "
                            "ii=%s ii_min=%s ii_max=%s pos_min=%s pos_max=%s "
                            "sm_len=%s sm_min=%s sm_max=%s hs_finite=%s" % (
                                token_index, input_batch_size,
                                (tuple(model_input_ids.shape)
                                 if model_input_ids is not None else None),
                                (int(model_input_ids.to(torch.int64)
                                     .min().item())
                                 if model_input_ids is not None else -1),
                                (int(model_input_ids.to(torch.int64)
                                     .max().item())
                                 if model_input_ids is not None else -1),
                                int(self._get_positions(model_input_tokens)
                                    .to(torch.int64).min().item()),
                                int(self._get_positions(model_input_tokens)
                                    .to(torch.int64).max().item()),
                                _fn8_sm3.shape[0],
                                int(_fn8_sm3.to(torch.int64).min().item()),
                                int(_fn8_sm3.to(torch.int64).max().item()),
                                bool(torch.isfinite(
                                    hidden_states.float()).all().item())),
                            flush=True)
                except Exception as _fn8_6l_e:  # noqa: BLE001
                    print(f"FN8_06L S3 probe failed: {_fn8_6l_e}",
                          flush=True)
                ret_hidden_states = self.model(**model_kwargs)
'''

EDITS = [
    ("L1_S1_first_pass", OLD1_LF, NEW1_LF),
    ("L2_S2_first_head", OLD2_LF, NEW2_LF),
    ("L3_S3_draft_pass", OLD3_LF, NEW3_LF),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


if "--restore" in sys.argv:
    bak = path + ".fn8_06l.bak"
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

bak = path + ".fn8_06l.bak"
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
