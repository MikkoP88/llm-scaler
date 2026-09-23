"""custom_callbacks.py — Claude Code compatibility repair for the qwen3.8 lane.

The model stringifies deeply-nested tool arguments (e.g. AskUserQuestion
emits {"questions": "[{...}]"} — a JSON array inside a string). CC's tool
input validation rejects that, breaking the AskUserQuestion dropdown and any
tool whose schema nests arrays inside arrays. Prompt-level nudges do not fix
it (3/3 failures), so repair server-side:

- async_post_call_streaming_iterator_hook: an async GENERATOR function (this
  litellm build calls it as hook(response=...) inside
  _wrap_streaming_iterator_with_enrichment and iterates the result directly).
  For /v1/messages the stream is raw SSE frames. Buffer each tool_use
  block's input_json_delta, un-stringify container-shaped string values,
  re-emit as one delta before content_block_stop. Per-request state lives in
  the generator frame (the instance is shared across concurrent requests).
- async_post_call_success_hook: awaited as hook(user_api_key_dict=...,
  data=..., response=...) — same repair for non-streaming responses,
  defensively handling anthropic-dict and openai ModelResponse shapes.

Only touches strings that strip to '[' or '{' AND json-parse to a
list/dict, inside tool arguments. Everything else passes through untouched.
"""
import json
import time
from typing import Optional

from litellm.integrations.custom_logger import CustomLogger

MAX_STRINGIFIED_LEN = 262144
LOG_PATH = "/tmp/custom_cb.log"


def _log(msg: str) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    except Exception:
        pass


def _try_loads(s: str):
    t = s.strip()
    if len(t) < 2 or t[0] not in "[{" or len(t) > MAX_STRINGIFIED_LEN:
        return None
    try:
        v = json.loads(t)
    except Exception:
        return None
    return v if isinstance(v, (list, dict)) else None


def _repair_obj(o):
    if isinstance(o, dict):
        def _val(v):
            # v2 fix: _try_loads returns None for plain strings — that None used to
            # REPLACE the value, nulling every non-container string in tool args
            # ({"command": "echo hi"} -> {"command": null}). Pass the original
            # string through unless it parses to a list/dict.
            if isinstance(v, str):
                parsed = _try_loads(v)
                return v if parsed is None else parsed
            return _repair_obj(v)
        return {k: _val(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_repair_obj(x) for x in o]
    return o


def _repair_args_str(s: str) -> Optional[str]:
    """Repair a raw tool-arguments JSON string; None if unchanged/unparseable."""
    try:
        parsed = json.loads(s)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    repaired = _repair_obj(parsed)
    if repaired == parsed:
        return None
    return json.dumps(repaired, separators=(",", ":"))


class CCCompatRepairHandler(CustomLogger):
    def _repair_input_dict(self, inp: dict) -> bool:
        try:
            new = _repair_obj(inp)
        except Exception:
            return False
        if new != inp:
            inp.clear()
            inp.update(new)
            return True
        return False

    async def async_post_call_success_hook(self, user_api_key_dict=None, data=None, response=None):
        try:
            if isinstance(response, dict) and isinstance(response.get("content"), list):
                for b in response["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and isinstance(b.get("input"), dict):
                        if self._repair_input_dict(b["input"]):
                            _log("repaired nonstream anthropic tool_use %s" % b.get("name"))
            else:
                ch = getattr(response, "choices", None)
                msg = ch[0].message if ch else None
                for tc in getattr(msg, "tool_calls", None) or []:
                    fn = getattr(tc, "function", None)
                    raw = getattr(fn, "arguments", None)
                    if isinstance(raw, str):
                        fixed = _repair_args_str(raw)
                        if fixed is not None:
                            fn.arguments = fixed
                            _log("repaired nonstream openai tool_call %s" % getattr(fn, "name", None))
        except Exception as e:
            _log("success_hook error %r" % (e,))
        return response

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict=None, response=None, request_data=None):
        buf = ""
        tool_idx = {}  # index -> accumulated partial_json

        def handle_event_text(text: str) -> str:
            """Process one complete SSE frame; return text to emit (may prepend a synthetic delta)."""
            out = ""
            data = None
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("data:"):
                    try:
                        data = json.loads(ls[5:].strip())
                    except Exception:
                        data = None
                    break
            if not isinstance(data, dict):
                return text  # not a JSON frame — pass through
            t = data.get("type")
            idx = data.get("index")
            if t == "content_block_start":
                cb = data.get("content_block") or {}
                if isinstance(cb, dict) and cb.get("type") == "tool_use":
                    tool_idx[idx] = ""
            elif t == "content_block_delta":
                delta = data.get("delta") or {}
                if idx in tool_idx and isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                    tool_idx[idx] += delta.get("partial_json") or ""
                    return ""  # suppress; re-emitted at block stop
            elif t == "content_block_stop":
                if idx in tool_idx:
                    acc = tool_idx.pop(idx)
                    fixed = _repair_args_str(acc)
                    emit = fixed if fixed is not None else acc
                    if emit:
                        syn = {"type": "content_block_delta", "index": idx,
                               "delta": {"type": "input_json_delta", "partial_json": emit}}
                        out += "event: content_block_delta\ndata: %s\n\n" % json.dumps(syn)
                        if fixed is not None:
                            _log("repaired stream tool_use idx=%s len %d->%d" % (idx, len(acc), len(emit)))
            return out + text

        async for chunk in response:
            if isinstance(chunk, bytes):
                try:
                    chunk = chunk.decode("utf-8")
                except Exception:
                    yield chunk
                    continue
            if not isinstance(chunk, str):
                yield chunk
                continue
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                res = handle_event_text(frame + "\n\n")
                if res:
                    yield res
        if buf:
            res = handle_event_text(buf)
            if res:
                yield res
        if tool_idx:  # stream ended mid-block — flush best effort
            for idx, acc in tool_idx.items():
                if acc:
                    syn = {"type": "content_block_delta", "index": idx,
                           "delta": {"type": "input_json_delta", "partial_json": acc}}
                    yield "event: content_block_delta\ndata: %s\n\n" % json.dumps(syn)


custom_callback = CCCompatRepairHandler()
