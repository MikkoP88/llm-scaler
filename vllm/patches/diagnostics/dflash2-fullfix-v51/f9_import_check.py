# f9_import_check.py — llm-scaler v51 Phase 4a:
# Static import-compatibility check for F9 overlay files against the
# target image's vLLM tree. Catches the ImportError class that killed the
# first v51f9 bake (foreign-revision overlay importing names/modules the
# image tree does not have) WITHOUT executing any vllm code (no device,
# no torch init): every top-level `from vllm.<mod> import <names>` in the
# overlay is resolved to a file in the image tree, and each name is
# searched for a definition (`def name(`, `class name`, `name =`,
# `name: `, or a re-export via `import ... as name` / plain `name,` in a
# from-import list).
#
# Usage: python3 f9_import_check.py <vllm_root> <overlay.py> [overlay2.py ...]
import ast
import os
import re
import sys

FORBIDDEN_HINTS = ("__pycache__",)


def module_to_path(root: str, mod: str):
    # mod like "vllm.v1.engine.core" -> root/vllm/v1/engine/core.py
    # or package dir -> root/.../core/__init__.py
    rel = mod.replace(".", "/")
    p1 = os.path.join(root, rel + ".py")
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(root, rel, "__init__.py")
    if os.path.isfile(p2):
        return p2
    return None


def find_name_def(path: str, name: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except OSError:
        return False
    pats = [
        "def " + name + "(",
        "def " + name + " (",
        "class " + name + ":",
        "class " + name + "(",
        "class " + name + "((",
        name + " = ",
        name + ": ",
        name + " =",
        name + ",",
        name + ")",
    ]
    for p in pats:
        if p in src:
            return True
    # relative re-export inside __init__: from .x import name
    if re.search(r"import[^\n]*\b" + re.escape(name) + r"\b", src):
        return True
    return False


def check(root: str, overlay: str) -> int:
    with open(overlay, "r", encoding="utf-8", errors="replace") as f:
        tree = ast.parse(f.read(), filename=overlay)
    problems = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 0 or not node.module:
            continue
        top = node.module.split(".")[0]
        if top != "vllm":
            continue
        mpath = module_to_path(root, node.module)
        if mpath is None:
            print("MISSING-MODULE %s: vllm module '%s' not in image tree (%s)"
                  % (os.path.basename(overlay), node.module, overlay))
            problems += 1
            continue
        for alias in node.names:
            nm = alias.asname or alias.name
            if alias.name == "*":
                continue
            if not find_name_def(mpath, alias.name):
                print("MISSING-NAME    %s: '%s' not found in image %s"
                      % (os.path.basename(overlay), alias.name, mpath))
                problems += 1
    return problems


def main() -> None:
    root = sys.argv[1]
    overlays = sys.argv[2:]
    total = 0
    for ov in overlays:
        print("== checking %s ==" % ov)
        total += check(root, ov)
    print("TOTAL-PROBLEMS %d" % total)
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
