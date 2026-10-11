#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llm-scaler flashnext-fp8-tier-v1 P0 patch_00b — QWEN4-EXP MODEL
# REGISTRATION (Qwen3.8-Flash-Next-FP8, Qwen4ExpForConditionalGeneration).
#
# Installs the vendored model implementation + wires it into the model
# registry on the baked vllm image (llm-scaler-exp:v1.2.35 lineage):
#
#   1. NEW FILE   <root>/model_executor/models/qwen4_exp.py
#                 (content = sibling files/qwen4_exp_model.py — located
#                  next to this patcher or under ./files/)
#   2. EDIT       <root>/model_executor/models/registry.py
#                 _TEXT_GENERATION_MODELS +=
#                 "Qwen4ExpForConditionalGeneration":
#                     ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
#
# _TEXT_GENERATION_MODELS is the correct surface: _VLLM_MODELS = {**_TEXT_
# GENERATION_MODELS, **_MULTIMODAL_MODELS, ...} so a text-gen entry is
# reachable WITHOUT dragging in vLLM multimodal preprocessing (this port
# serves text only; checkpoint vision weights are skipped at load).
# Requires patch_00a (config registration) to be applied first — the model
# module imports nothing from it at import time, but model loading does.
#
# Usage:
#   patch_00b_qwen4exp_model.py            # patch the in-container root
#   patch_00b_qwen4exp_model.py ROOT       # explicit vllm package root
#   patch_00b_qwen4exp_model.py --restore [ROOT]
#
# Idempotent (re-run prints ALREADY and exits 0 when every marker is in
# place).  Backups *.q4bbak hold the pre-patch pristine.  Patched text is
# py_compile-checked on temp files before atomic os.replace.  Order: new
# module file first (inert until the registry references it), then
# registry.py; an anchor failure aborts with the unreferenced new file
# left harmless (cleaned by --restore).

import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

ROOT_DEFAULT = "/opt/venv/lib/python3.12/site-packages/vllm"
BACKUP_SUFFIX = ".q4bbak"
MODULE_FILENAME = "qwen4_exp_model.py"  # looked up in ./files/ then ./

# --- site 1: NEW FILE ---------------------------------------------------
NEW_REL = "model_executor/models/qwen4_exp.py"

# --- site 2: model_executor/models/registry.py ---------------------------
# Anchor verified byte-exact against llm-scaler-exp:v1.2.35 (unique match
# inside the _TEXT_GENERATION_MODELS literal).
REG_REL = "model_executor/models/registry.py"
A_REG_ANCHOR = '''    "Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),
'''
A_REG_REPL = '''    "Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),
    "Qwen4ExpForConditionalGeneration": ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
'''
# Normalize away accidental CRLF (Windows-side edits of THIS script).
A_REG_ANCHOR = A_REG_ANCHOR.replace("\r\n", "\n")
A_REG_REPL = A_REG_REPL.replace("\r\n", "\n")
REG_MARKER = '"Qwen4ExpForConditionalGeneration": ("qwen4_exp", "Qwen4ExpForConditionalGeneration")'

TAG = "FN8_00B"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def locate_module() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "files", MODULE_FILENAME),
        os.path.join(here, MODULE_FILENAME),
    ):
        if os.path.exists(cand):
            return cand
    print(f"{TAG} FAIL: module payload {MODULE_FILENAME} not found next to the patcher (./files/ or ./)")
    sys.exit(9)


def compile_ok(path: str) -> bool:
    try:
        py_compile.compile(path, doraise=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{TAG} FAIL: py_compile {path}: {exc!r}")
        return False


def write_new(root: str, src_path: str) -> int:
    dst = os.path.join(root, NEW_REL)
    content = read_bytes(src_path)
    if os.path.exists(dst):
        if md5(read_bytes(dst)) == md5(content):
            print(f"{TAG} ALREADY: {dst} (md5 {md5(content)}) matches payload")
            return 0
        print(f"{TAG} FAIL: {dst} exists with different content — refusing to overwrite")
        return 2
    # compile-check the payload on a temp file in the same dir
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="q4b_", dir=os.path.dirname(dst) or ".")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(content)
        if not compile_ok(tmp):
            os.unlink(tmp)
            return 3
        with open(dst + BACKUP_SUFFIX, "wb") as f:
            f.write(content)  # record what we wrote; --restore unlinks both
        os.replace(tmp, dst)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.unlink(tmp)
        print(f"{TAG} FAIL: write {dst}: {exc!r}")
        return 4
    print(f"{TAG} NEWFILE: {dst} md5 {md5(content)} size {len(content)}")
    return 0


def patch_registry(root: str) -> int:
    path = os.path.join(root, REG_REL)
    raw = read_bytes(path)
    src = raw.decode("utf-8")
    if REG_MARKER in src:
        print(f"{TAG} ALREADY: {path} carries the registry entry")
        return 0
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        with open(backup, "wb") as f:
            f.write(raw)
        print(f"{TAG} backup written: {backup} (md5 {md5(raw)})")
    cnt = src.count(A_REG_ANCHOR)
    if cnt != 1:
        print(f"{TAG} FAIL: registry anchor matches {cnt} times (need exactly 1) — file untouched")
        return 3
    patched = src.replace(A_REG_ANCHOR, A_REG_REPL, 1)
    if REG_MARKER not in patched:
        print(f"{TAG} FAIL: post-patch marker missing — file untouched")
        return 4
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="q4b_", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(patched.encode("utf-8"))
        py_compile.compile(tmp, doraise=True)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.unlink(tmp)
        print(f"{TAG} FAIL: compile/replace {path}: {exc!r} — file untouched")
        return 5
    print(
        f"{TAG} PATCHED: {path} md5 {md5(raw)} -> {md5(patched.encode('utf-8'))} "
        f"size {len(raw)} -> {len(patched.encode('utf-8'))}"
    )
    return 0


def restore(root: str) -> int:
    rc = 0
    newf = os.path.join(root, NEW_REL)
    newbak = newf + BACKUP_SUFFIX
    if os.path.exists(newf):
        if os.path.exists(newbak) and md5(read_bytes(newf)) == md5(read_bytes(newbak)):
            os.unlink(newf)
            os.unlink(newbak)
            print(f"{TAG} RESTORED: removed {newf}")
        else:
            print(f"{TAG} RESTORE WARN: {newf} content drifted from what we wrote — left in place")
            rc = 1
    path = os.path.join(root, REG_REL)
    backup = path + BACKUP_SUFFIX
    if os.path.exists(backup):
        shutil.copyfile(backup, path)
        got, want = md5(read_bytes(path)), md5(read_bytes(backup))
        if got != want:
            print(f"{TAG} RESTORE FAIL: md5 mismatch after copy for {path}")
            rc = 2
        else:
            print(f"{TAG} RESTORED: {path} from backup (md5 {got})")
    else:
        print(f"{TAG} RESTORE: no backup for {path} (nothing to do)")
    return rc


def main() -> int:
    args = sys.argv[1:]
    do_restore = "--restore" in args
    root = ROOT_DEFAULT
    for a in args:
        if not a.startswith("--"):
            root = a
    if do_restore:
        return restore(root)
    src = locate_module()
    # inert-first order: module file, then registry
    rc = write_new(root, src)
    if rc:
        return rc
    return patch_registry(root)


if __name__ == "__main__":
    sys.exit(main())
