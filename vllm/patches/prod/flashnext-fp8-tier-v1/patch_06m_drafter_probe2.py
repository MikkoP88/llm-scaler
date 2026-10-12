#!/usr/bin/env python3
"""FN8_06m (DEC40): stage-map completion for the first-spec-step abort.

06l verdicts (13:20 boot, banked serve_d9.log): S1/S2 clean (bs=1,
num_tokens=18, ii 11..22373, pos 0..17); drafter's slot buffer holds
65536+ (its own GDN state-slot region — valid, 3 draft passes entered);
S3 probe crashed ('dict' has no .shape — _get_slot_mapping returns a
per-group dict) so pass inputs were never dumped; 06k's runner probe
never fired (spec_decode_metadata attr absent on this metadata build)
=> verify-step tensors still unseen.  Death stage remains ambiguous:
inside a draft head call, or after propose() (verify replay/sampler).

Edits:
  FILE A  v1/spec_decode/llm_base_proposer.py
  A1  S3 fix: dict-aware slot-mapping probe (first group's tensor).
  A2  S3 fix: None-guarded sm_len/sm_min/sm_max print args.
  A3  S4: one-shot POST-head-call marker per draft pass — its presence
      proves the head call survived; absence pins the death inside
      that pass.
  A4  S5: one-shot post-propose marker (stack+return survived).

  FILE B  v1/worker/gpu_model_runner.py
  B1  06M step dump at the 06k site but attr-agnostic: fires on the
      first 6 execute steps AND on any padded step (input_ids.shape[0]
      != num_actual_tokens); dumps input_ids / slot_mapping /
      block_table / seq_lens / spec attr shapes + min/max; zero-fills
      padded input_ids + slot_mapping tails (same padding-0 semantics
      as capture dummies; masked rows unaffected).

Idempotent; py_compile gate; backups .fn8_06m.bak; EOL aware;
anchor==1 per edit; --restore mode.
"""
import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

import importlib.util

TAG = "FN8_06m"
MARKER = "FN8_06M"

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.origin:
    print(f"{TAG} FAIL: vllm not importable")
    sys.exit(1)
PKG = os.path.dirname(spec.origin)

PATH_A = os.path.join(PKG, "v1/spec_decode/llm_base_proposer.py")
PATH_B = os.path.join(PKG, "v1/worker/gpu_model_runner.py")

# A1: dict-aware slot mapping in S3.
OLD_A1 = '''                        _fn8_sm3 = self._get_slot_mapping(model_input_tokens)
'''
NEW_A1 = '''                        _fn8_sm3 = self._get_slot_mapping(model_input_tokens)
                        if isinstance(_fn8_sm3, dict):  # FN8_06M per-group
                            _fn8_sm3 = (list(_fn8_sm3.values())[0]
                                        if _fn8_sm3 else None)
'''

# A2: None-guarded sm print args in S3.
OLD_A2 = '''                                _fn8_sm3.shape[0],
                                int(_fn8_sm3.to(torch.int64).min().item()),
                                int(_fn8_sm3.to(torch.int64).max().item()),
'''
NEW_A2 = '''                                (_fn8_sm3.shape[0]
                                 if _fn8_sm3 is not None else -1),
                                (int(_fn8_sm3.to(torch.int64).min().item())
                                 if _fn8_sm3 is not None else -1),
                                (int(_fn8_sm3.to(torch.int64).max().item())
                                 if _fn8_sm3 is not None else -1),
'''

# A3: S4 post-head-call marker.
OLD_A3 = '''                except Exception as _fn8_6l_e:  # noqa: BLE001
                    print(f"FN8_06L S3 probe failed: {_fn8_6l_e}",
                          flush=True)
                ret_hidden_states = self.model(**model_kwargs)
'''
NEW_A3 = '''                except Exception as _fn8_6l_e:  # noqa: BLE001
                    print(f"FN8_06L S3 probe failed: {_fn8_6l_e}",
                          flush=True)
                ret_hidden_states = self.model(**model_kwargs)
                # FN8_06M S4: this head call SURVIVED (post-call).
                if not globals().get("_FN8_06M_S4_%d" % token_index, False):
                    globals()["_FN8_06M_S4_%d" % token_index] = True
                    print(f"FN8_06M S4 pass={token_index} head_done "
                          f"ib={input_batch_size}", flush=True)
'''

# A4: S5 post-propose marker.
OLD_A4 = '''        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
'''
NEW_A4 = '''        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        # FN8_06M S5: propose returned clean (one-shot).
        if not globals().get("_FN8_06M_S5", False):
            globals()["_FN8_06M_S5"] = True
            print("FN8_06M S5 propose_done bs=%d" % batch_size, flush=True)
        return draft_token_ids
'''

EDITS_A = [
    ("A1_s3_dict", OLD_A1, NEW_A1),
    ("A2_s3_guards", OLD_A2, NEW_A2),
    ("A3_s4_postcall", OLD_A3, NEW_A3),
    ("A4_s5_postpropose", OLD_A4, NEW_A4),
]

# B1: attr-agnostic verify-step dump (inserted before the 06k block).
OLD_B1 = '''        # FN8_06k (DEC40): spec-verify padded-tail probe + harden.
'''
NEW_B1 = '''        # FN8_06M (DEC40): verify-step tensor dump — attr-agnostic
        # (06k never fired: spec_decode_metadata absent on this build).
        # Dumps the first 6 steps + every padded step, then zero-fills
        # padded input_ids / slot_mapping tails (padding-0 semantics of
        # the capture dummies; masked rows unaffected).
        try:
            _fn8_m_n = int(getattr(attn_metadata, "num_actual_tokens",
                                   input_ids.shape[0]))
            _fn8_m_pad = input_ids.shape[0]
            _fn8_m_step = globals().setdefault("_FN8_06M_STEP", 0)
            globals()["_FN8_06M_STEP"] = _fn8_m_step + 1
            if _fn8_m_step < 6 or _fn8_m_pad != _fn8_m_n:
                _fn8_m_sm = getattr(attn_metadata, "slot_mapping", None)
                _fn8_m_bt = getattr(attn_metadata, "block_table_tensor",
                                    None)
                _fn8_m_sl = getattr(attn_metadata, "seq_lens", None)
                _fn8_m_sq = getattr(attn_metadata, "spec_query_start_loc",
                                    None)
                print(
                    "FN8_06M step=%d nreal=%d padded=%d ii_min=%s "
                    "ii_max=%s sm=%s sm_min=%s sm_max=%s bt=%s bt_min=%s "
                    "bt_max=%s sl_min=%s sl_max=%s sq=%s" % (
                        _fn8_m_step, _fn8_m_n, _fn8_m_pad,
                        int(input_ids.to(torch.int64).min().item()),
                        int(input_ids.to(torch.int64).max().item()),
                        (tuple(_fn8_m_sm.shape)
                         if _fn8_m_sm is not None else None),
                        (int(_fn8_m_sm.to(torch.int64).min().item())
                         if _fn8_m_sm is not None else -1),
                        (int(_fn8_m_sm.to(torch.int64).max().item())
                         if _fn8_m_sm is not None else -1),
                        (tuple(_fn8_m_bt.shape)
                         if _fn8_m_bt is not None else None),
                        (int(_fn8_m_bt.to(torch.int64).min().item())
                         if _fn8_m_bt is not None else -1),
                        (int(_fn8_m_bt.to(torch.int64).max().item())
                         if _fn8_m_bt is not None else -1),
                        (int(_fn8_m_sl.min().item())
                         if _fn8_m_sl is not None else -1),
                        (int(_fn8_m_sl.max().item())
                         if _fn8_m_sl is not None else -1),
                        (tuple(_fn8_m_sq.shape)
                         if _fn8_m_sq is not None else None)),
                    flush=True)
            if _fn8_m_pad > _fn8_m_n:
                input_ids[_fn8_m_n:].zero_()
                _fn8_m_sm2 = getattr(attn_metadata, "slot_mapping", None)
                if _fn8_m_sm2 is not None and _fn8_m_sm2.shape[0] > _fn8_m_n:
                    _fn8_m_sm2[_fn8_m_n:].zero_()
        except Exception as _fn8_6m_e:  # noqa: BLE001 (telemetry mandate)
            print(f"FN8_06M runner probe failed: {_fn8_6m_e}", flush=True)
        # FN8_06k (DEC40): spec-verify padded-tail probe + harden.
'''

EDITS_B = [
    ("B1_step_dump", OLD_B1, NEW_B1),
]


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def apply(path, edits):
    if MARKER in open(path, "r", encoding="utf-8", newline="").read():
        print(f"{TAG} {os.path.basename(path)} already edited, skip "
              f"(md5={md5(path)})")
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        src = f.read()
    eol = "\r\n" if "\r\n" in src else "\n"

    def ed(t):
        return t.replace("\n", eol)

    out = src
    for name, old, new in edits:
        n = out.count(ed(old))
        if n != 1:
            print(f"{TAG} FAIL {os.path.basename(path)}: anchor {name} "
                  f"occurs {n}x (want 1)")
            sys.exit(2)
        out = out.replace(ed(old), ed(new))
    bak = path + ".fn8_06m.bak"
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"{TAG} backup {os.path.basename(path)} md5={md5(bak)}")
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
    print(f"{TAG} [{os.path.basename(path)}] edited md5={md5(path)}")


if "--restore" in sys.argv:
    for p in (PATH_A, PATH_B):
        bak = p + ".fn8_06m.bak"
        if os.path.exists(bak):
            shutil.copyfile(bak, p)
            print(f"{TAG} restored {os.path.basename(p)} md5={md5(p)}")
    sys.exit(0)

apply(PATH_A, EDITS_A)
apply(PATH_B, EDITS_B)
print(f"{TAG} ALL_EDITS_OK")
