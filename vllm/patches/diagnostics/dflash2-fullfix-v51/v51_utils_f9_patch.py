# v51_utils_f9_patch.py — llm-scaler v51 Phase 4a (F9 suite), take 2.
#
# Take-1 lesson (2026-09-06): this image's tree (v0.21.1.dev0+gad7125a43)
# does NOT define merge_async_iterators in vllm/v1/engine/utils.py (that
# file holds CoreEngineProcManager and must be left image-native). The
# F9 utils delta belongs in vllm/utils/async_utils.py, whose single-
# iterator fast path (lines 281-285) matches the anchor below EXACTLY.
#
# Delta (F9, crash-3 zombie leak): when the merge wrapper is closed, the
# wrapped generator was previously only closed by GC — leaving the
# async_llm generate() task (and its engine request) alive for a
# nondeterministic window when a serving consumer abandoned the stream.
# The multi-iterator path below already acloses its children; this patch
# gives the fast path the same guarantee, which is what propagates
# aclose() into async_llm_v51.generate()'s finally -> fire-and-forget
# abort(request_id, internal=True).
#
# Run INSIDE the bake container as: python3 /root/v51_utils_f9_patch.py
import py_compile
import sys

TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/utils/async_utils.py"

OLD = (
    "    if len(iterators) == 1:\n"
    "        # Fast-path single iterator case.\n"
    "        async for item in iterators[0]:\n"
    "            yield 0, item\n"
    "        return\n"
)

NEW = (
    "    if len(iterators) == 1:\n"
    "        # Fast-path single iterator case.\n"
    "        # llm-scaler v51 (F9): close the wrapped generator when this\n"
    "        # merge wrapper is closed -- previously the inner generator\n"
    "        # (async_llm generate()) was only closed by GC, leaving the\n"
    "        # engine request alive for a nondeterministic window when a\n"
    "        # serving consumer abandoned the stream (crash-3 zombie\n"
    "        # leak; the multi-iterator path below already acloses its\n"
    "        # children).\n"
    "        try:\n"
    "            async for item in iterators[0]:\n"
    "                yield 0, item\n"
    "        finally:\n"
    "            aclose = getattr(iterators[0], \"aclose\", None)\n"
    "            if aclose is not None:\n"
    "                with contextlib.suppress(BaseException):\n"
    "                    await aclose()\n"
    "        return\n"
)


def main() -> None:
    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()
    if NEW in src:
        print("already-patched: F9 fast-path present, nothing to do")
        return
    n = src.count(OLD)
    if n != 1:
        print("FATAL: fast-path anchor count = %d (expected 1); aborting" % n)
        sys.exit(1)
    out = src.replace(OLD, NEW, 1)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(out)
    py_compile.compile(TARGET, doraise=True)
    print("PATCHED-OK %s (fast-path aclose, F9)" % TARGET)


if __name__ == "__main__":
    main()
