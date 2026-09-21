#!/usr/bin/env python3
"""patch_stall_v59.py — v59 §1 tool-call stream STALL fix (crashfix-v58 line).

ROOT CAUSE (proven, 05:36 crash log):
  Qwen3XMLToolParser streaming can emit a FIRST tool-call delta with
  function.name=None (nameless <function>, parameter-start-before-name,
  tail/auto-close deltas). The Responses API simple-streaming processor
  opens its tool-call output item from that first delta:
      emit_simple_tool_call_open -> ResponseFunctionToolCallItem(name=None)
  -> pydantic ValidationError INSIDE the SSE generator -> the entire
  response stream for that request dies mid-flight -> Claude Code sees a
  dead stream = STALL. Path: /v1/responses auto-tools (no tool_choice) ->
  DelegatingParser.parse_delta -> Qwen3XMLToolParser.extract_tool_calls_streaming.

FIXES:
  A (belt)  responses/streaming_events.py: emit_simple_tool_call_open
            coerces name=None -> "" + WARNING. Stream never dies.
  B (root)  tool_parsers/qwen3xml_tool_parser.py: first delta per
            tool-call index always carries a name (known current name,
            single-tool fallback, else ""). Continuation deltas untouched.
  C (telemetry) responses/serving.py: RESPROBE START/FIRST-EVENT/TOOL-OPEN/
            DONE lifecycle lines with request_id + counters so stalls are
            attributable in serve_full.log -> host serve_live.log.

Usage:  patch_stall_v59.py --apply (default) | --check | --revert
Backups: <file>.stalfix.bak written on first apply; --revert restores.
Idempotent: skips any file already carrying its STALFIX marker.
Anchor-count asserted == 1 before any write; py_compile after write,
backup restored automatically on compile failure.
"""

import os
import py_compile
import shutil
import sys

VENV = "/opt/venv/lib/python3.12/site-packages/vllm"

F_EVENTS = f"{VENV}/entrypoints/openai/responses/streaming_events.py"
F_PARSER = f"{VENV}/tool_parsers/qwen3xml_tool_parser.py"
F_SERVING = f"{VENV}/entrypoints/openai/responses/serving.py"

# ---------------------------------------------------------------------------
# FIX A — streaming_events.py  emit_simple_tool_call_open
# ---------------------------------------------------------------------------

A_OLD = """    state.current_state = _StateType.TOOL_CALL
    state.current_item_id = random_uuid()
"""

A_NEW = """    state.current_state = _StateType.TOOL_CALL
    state.current_item_id = random_uuid()
    # STALFIX-A (v59): a tool-call item must never open with name=None —
    # ResponseFunctionToolCallItem.name is pydantic str-validated and a None
    # raises inside this generator, killing the whole SSE stream (the
    # tool-call stall). Coerce to '' + warn; the stream survives.
    if name is None:
        import logging as _sf_logging

        _sf_logging.getLogger(__name__).warning(
            "STALFIX-A: tool_call open with name=None -> '' (item_id=%s, index=%s)",
            state.current_item_id,
            index,
        )
        name = ""
"""

# ---------------------------------------------------------------------------
# FIX B — qwen3xml_tool_parser.py
# ---------------------------------------------------------------------------

# B1: __init__ — name-emitted tracker attribute
B1_OLD = """        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
"""

B1_NEW = """        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        # STALFIX-B (v59): per-index flag — a tool-call index must carry its
        # function NAME in its FIRST delta (the Responses API opens the
        # tool-call output item from the first delta; name=None there kills
        # the whole stream).
        self._stalfix_name_emitted: dict[int, bool] = {}
"""

# B2: extract_tool_calls_streaming reset block ("streaming session" comment,
# 12-space indent — distinct from non-streaming "new extraction" reset)
B2_OLD = """            self.parser.reset_streaming_state()
            # Reset tool call tracking arrays for new streaming session
            self.prev_tool_call_arr = []
"""

B2_NEW = """            self.parser.reset_streaming_state()
            # Reset tool call tracking arrays for new streaming session
            self.prev_tool_call_arr = []
            # STALFIX-B (v59): reset per-request name-emitted tracking
            self._stalfix_name_emitted = {}
"""

# B3: first-delta-carries-name guard right after the parser call
B3_OLD = """        # Parse the delta text and get the result
        delta = self.parser.parse_single_streaming_chunks(delta_text)
"""

B3_NEW = """        # Parse the delta text and get the result
        delta = self.parser.parse_single_streaming_chunks(delta_text)

        # STALFIX-B (v59): enforce first-delta-carries-name. The XML state
        # machine emits continuation deltas with name=None; if such a delta
        # is the FIRST for its index (nameless <function>, param-start
        # before name, tail/auto-close delta), attach the known name so the
        # Responses API can open the tool-call item safely.
        if delta is not None and delta.tool_calls:
            for _sf_tc in delta.tool_calls:
                if not _sf_tc.function:
                    continue
                _sf_idx = _sf_tc.index if _sf_tc.index is not None else 0
                if _sf_tc.function.name:
                    self._stalfix_name_emitted[_sf_idx] = True
                elif not self._stalfix_name_emitted.get(_sf_idx):
                    _sf_name = getattr(self.parser, "current_function_name", None)
                    if not _sf_name and self.tools:
                        _sf_names = [
                            t.function.name
                            for t in self.tools
                            if getattr(t, "function", None) and t.function.name
                        ]
                        if len(_sf_names) == 1:
                            _sf_name = _sf_names[0]
                    _sf_tc.function.name = _sf_name or ""
                    self._stalfix_name_emitted[_sf_idx] = True
                    logger.warning(
                        "STALFIX-B: first tool_call delta had no name -> attached %r (index=%s)",
                        _sf_tc.function.name,
                        _sf_idx,
                    )
"""

# ---------------------------------------------------------------------------
# FIX C — responses/serving.py  _process_simple_streaming_events
# ---------------------------------------------------------------------------

# C1: probe init right after parser construction
C1_OLD = """        processor = SimpleStreamingEventProcessor()
        parser = self.parser(tokenizer, request.tools) if self.parser else None
"""

C1_NEW = """        processor = SimpleStreamingEventProcessor()
        parser = self.parser(tokenizer, request.tools) if self.parser else None

        # STALFIX-C (v59): per-request lifecycle probe for tool-call streams
        # — makes stalls/deaths attributable in serve logs.
        import logging as _sf_logging
        import time as _sf_time

        _sf_lg = _sf_logging.getLogger(__name__)
        _sf_rid = getattr(request, "request_id", "?")
        _sf_t0 = _sf_time.monotonic()
        _sf_first = None
        _sf_n = {"deltas": 0, "tc_named": 0, "tc_delta": 0, "reason": 0, "content": 0}
        _sf_lg.info(
            "RESPROBE START rid=%s tools=%s parser=%s",
            _sf_rid,
            len(request.tools) if request.tools else 0,
            type(parser).__name__ if parser else "none",
        )
"""

# C2: first engine event latency
C2_OLD = """            output = ctx.last_output.outputs[0]
            self._raise_if_error(output.finish_reason, request.request_id)
"""

C2_NEW = """            output = ctx.last_output.outputs[0]
            self._raise_if_error(output.finish_reason, request.request_id)
            # STALFIX-C (v59): first engine event latency
            if _sf_first is None:
                _sf_first = _sf_time.monotonic() - _sf_t0
                _sf_lg.info(
                    "RESPROBE FIRST-EVENT rid=%s latency=%.3fs",
                    _sf_rid,
                    _sf_first,
                )
"""

# C3: counters + tool-open trace on each non-empty delta
C3_OLD = """            if not delta_message:
                continue

            target_state, tool_call = processor.resolve_target_state(delta_message)
"""

C3_NEW = """            if not delta_message:
                continue

            # STALFIX-C (v59) counters + tool-open trace
            _sf_n["deltas"] += 1
            if delta_message.tool_calls:
                _sf_n["tc_delta"] += len(delta_message.tool_calls)
                for _sf_tc in delta_message.tool_calls:
                    if _sf_tc.function and _sf_tc.function.name:
                        _sf_n["tc_named"] += 1
                        _sf_lg.info(
                            "RESPROBE TOOL-OPEN rid=%s index=%s name=%s",
                            _sf_rid,
                            _sf_tc.index,
                            _sf_tc.function.name,
                        )
            elif delta_message.reasoning:
                _sf_n["reason"] += 1
            elif delta_message.content:
                _sf_n["content"] += 1

            target_state, tool_call = processor.resolve_target_state(delta_message)
"""

# C4: DONE line before the final close loop
C4_OLD = """        for event in processor.close_current():
            yield _increment_sequence_number_and_return(event)

    async def _process_harmony_streaming_events(
"""

C4_NEW = """        # STALFIX-C (v59): completion line — a START without DONE (and no
        # ASGI traceback) or TOOL-OPEN without DONE = mid-stream death.
        _sf_lg.info(
            "RESPROBE DONE rid=%s deltas=%s tc_named=%s tc_delta=%s reason=%s content=%s dur=%.1fs",
            _sf_rid,
            _sf_n["deltas"],
            _sf_n["tc_named"],
            _sf_n["tc_delta"],
            _sf_n["reason"],
            _sf_n["content"],
            _sf_time.monotonic() - _sf_t0,
        )
        for event in processor.close_current():
            yield _increment_sequence_number_and_return(event)

    async def _process_harmony_streaming_events(
"""

# patch spec: (tag, file, marker, [(old, new), ...])
PATCHES = [
    ("A", F_EVENTS, "STALFIX-A (v59)", [(A_OLD, A_NEW)]),
    ("B", F_PARSER, "STALFIX-B (v59)", [(B1_OLD, B1_NEW), (B2_OLD, B2_NEW), (B3_OLD, B3_NEW)]),
    ("C", F_SERVING, "STALFIX-C (v59)", [(C1_OLD, C1_NEW), (C2_OLD, C2_NEW), (C3_OLD, C3_NEW), (C4_OLD, C4_NEW)]),
]


def apply_patches(mode: str) -> int:
    rc = 0
    for tag, path, marker, subs in PATCHES:
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError as e:
            print(f"[{tag}] MISSING {path}: {e}")
            rc = 1
            continue
        bak = path + ".stalfix.bak"
        if mode == "revert":
            try:
                shutil.copyfile(bak, path)
                py_compile.compile(path, doraise=True)
                print(f"[{tag}] REVERTED+COMPILED from {bak}")
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] REVERT-FAIL {e}")
                rc = 1
            continue
        present = marker in src
        if mode == "check":
            print(f"[{tag}] {'APPLIED' if present else 'CLEAN'} {path}")
            continue
        if present:
            print(f"[{tag}] ALREADY-APPLIED {path}")
            continue
        # verify every anchor matches exactly once
        ok = True
        for old, _new in subs:
            n = src.count(old)
            if n != 1:
                tail = old.strip().splitlines()[-1]
                print(f"[{tag}] ANCHOR-COUNT {n} != 1 at ...{tail!r} — SKIP {path}")
                ok = False
        if not ok:
            rc = 1
            continue
        if not os.path.exists(bak):
            shutil.copyfile(path, bak)
            print(f"[{tag}] BACKUP {bak}")
        out = src
        for old, new in subs:
            out = out.replace(old, new, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        try:
            py_compile.compile(path, doraise=True)
            print(f"[{tag}] PATCHED+COMPILED {path}")
        except py_compile.PyCompileError as e:
            shutil.copyfile(bak, path)
            print(f"[{tag}] COMPILE-FAIL restored backup: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "--apply").lstrip("-")
    sys.exit(apply_patches(mode))
