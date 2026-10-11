#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# llm-scaler flashnext-fp8-tier-v1 P0 patch_00a — QWEN4-EXP CONFIG
# REGISTRATION (Qwen3.8-Flash-Next-FP8, model_type qwen4_exp).
#
# Registers the vendored Qwen4-Exp config trio on the baked vllm image
# (llm-scaler-exp:v1.2.35 lineage, vllm v0.21.1.dev0+gad7125a43 class):
#
#   1. NEW FILE   <root>/transformers_utils/configs/qwen4_exp.py
#                 (content = sibling files/qwen4_exp_config.py — located
#                  next to this patcher or under ./files/)
#   2. EDIT       <root>/transformers_utils/configs/__init__.py
#                 _CLASS_TO_MODULE += Qwen4ExpConfig / Qwen4ExpTextConfig /
#                 Qwen4ExpVisionConfig -> ...configs.qwen4_exp
#                 __all__ += the same three names
#   3. EDIT       <root>/transformers_utils/config.py
#                 _CONFIG_REGISTRY LazyConfigDict literal +=
#                 qwen4_exp="Qwen4ExpConfig",
#
# The load path (config.py AutoConfig.register(model_type, config_class,
# exist_ok=True)) then resolves "qwen4_exp" from the checkpoint's
# config.json through OUR vendored classes — the stock transformers 5.8.0
# on the image has no qwen4_exp.
#
# Usage:
#   patch_00a_qwen4exp_config.py            # patch the in-container root
#   patch_00a_qwen4exp_config.py ROOT       # explicit vllm package root
#   patch_00a_qwen4exp_config.py --restore [ROOT]
#
# Idempotent (re-run prints ALREADY and exits 0 when every marker is in
# place).  Backups *.q4abak hold the pre-patch pristine.  Patched text is
# py_compile-checked on temp files before atomic os.replace — a target is
# never left half-written.  Order: new module file first (inert until
# referenced), then __init__.py, then config.py; any anchor failure aborts
# leaving the remaining targets untouched (the unreferenced new file is
# harmless and is cleaned by --restore).

import hashlib
import os
import py_compile
import shutil
import sys
import tempfile

ROOT_DEFAULT = "/opt/venv/lib/python3.12/site-packages/vllm"
BACKUP_SUFFIX = ".q4abak"
MODULE_FILENAME = "qwen4_exp_config.py"  # looked up in ./files/ then ./

# --- site 1: NEW FILE -------------------------------------------------
NEW_REL = "transformers_utils/configs/qwen4_exp.py"

# --- site 2: configs/__init__.py ---------------------------------------
# Anchors verified byte-exact against llm-scaler-exp:v1.2.35
# (_CLASS_TO_MODULE dict entry + __all__ list entry, both unique).
INIT_REL = "transformers_utils/configs/__init__.py"
A_DICT_ANCHOR = '''    "Qwen3_5TextConfig": "vllm.transformers_utils.configs.qwen3_5",
'''
A_DICT_REPL = '''    "Qwen3_5TextConfig": "vllm.transformers_utils.configs.qwen3_5",
    "Qwen4ExpConfig": "vllm.transformers_utils.configs.qwen4_exp",
    "Qwen4ExpTextConfig": "vllm.transformers_utils.configs.qwen4_exp",
    "Qwen4ExpVisionConfig": "vllm.transformers_utils.configs.qwen4_exp",
'''
A_ALL_ANCHOR = '''    "Qwen3_5TextConfig",
'''
A_ALL_REPL = '''    "Qwen3_5TextConfig",
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
'''

# --- site 3: transformers_utils/config.py --------------------------------
CFG_REL = "transformers_utils/config.py"
A_REG_ANCHOR = '''    qwen3_5_moe="Qwen3_5MoeConfig",
'''
A_REG_REPL = '''    qwen3_5_moe="Qwen3_5MoeConfig",
    qwen4_exp="Qwen4ExpConfig",
'''

# Normalize away accidental CRLF so a Windows-side edit of THIS script
# cannot desynchronize the templates from the LF target files.
A_DICT_ANCHOR, A_DICT_REPL = A_DICT_ANCHOR.replace("\r\n", "\n"), A_DICT_REPL.replace("\r\n", "\n")
A_ALL_ANCHOR, A_ALL_REPL = A_ALL_ANCHOR.replace("\r\n", "\n"), A_ALL_REPL.replace("\r\n", "\n")
A_REG_ANCHOR, A_REG_REPL = A_REG_ANCHOR.replace("\r\n", "\n"), A_REG_REPL.replace("\r\n", "\n")

TAG = "FN8_00A"


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
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="q4a_", dir=os.path.dirname(dst) or ".")
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


def patch_text_file(path: str, sites: list[tuple[str, str, str]], marker: str) -> int:
    raw = read_bytes(path)
    src = raw.decode("utf-8")
    if marker in src:
        print(f"{TAG} ALREADY: {path} carries {marker!r}")
        return 0
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        with open(backup, "wb") as f:
            f.write(raw)
        print(f"{TAG} backup written: {backup} (md5 {md5(raw)})")
    patched = src
    for name, anchor, repl in sites:
        cnt = patched.count(anchor)
        if cnt != 1:
            print(f"{TAG} FAIL: anchor {name} matches {cnt} times (need exactly 1) in {path} — file untouched")
            return 3
        patched = patched.replace(anchor, repl, 1)
    if marker not in patched:
        print(f"{TAG} FAIL: post-patch marker {marker!r} missing in {path} — file untouched")
        return 4
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="q4a_", dir=os.path.dirname(path) or ".")
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
    for rel in (INIT_REL, CFG_REL):
        path = os.path.join(root, rel)
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

    # inert-first order: module file, __init__, config registry
    rc = write_new(root, src)
    if rc:
        return rc
    rc = patch_text_file(
        os.path.join(root, INIT_REL),
        [
            ("init _CLASS_TO_MODULE", A_DICT_ANCHOR, A_DICT_REPL),
            ("init __all__", A_ALL_ANCHOR, A_ALL_REPL),
        ],
        marker='"Qwen4ExpConfig": "vllm.transformers_utils.configs.qwen4_exp"',
    )
    if rc:
        return rc
    rc = patch_text_file(
        os.path.join(root, CFG_REL),
        [("config _CONFIG_REGISTRY", A_REG_ANCHOR, A_REG_REPL)],
        marker='qwen4_exp="Qwen4ExpConfig"',
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
