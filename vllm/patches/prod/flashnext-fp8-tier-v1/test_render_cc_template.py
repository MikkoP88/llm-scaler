#!/usr/bin/env python3
"""Offline render test for the flash-next CC chat template.

Runs INSIDE fn8-d1: python3 /stage/test_render_cc_template.py
Loads /stage/chat_template_flashnext_cc.jinja, renders CC-shaped
conversations with the model's own tokenizer, and asserts the CC
deltas behave (high->xhigh, string tool args, content parts) plus
native parity (think scaffold, tool format, bogus effort raises).
"""
import sys

from transformers import AutoTokenizer

TMPL = open("/stage/chat_template_flashnext_cc.jinja").read()
tok = AutoTokenizer.from_pretrained("/models/flash-next-fp8")

fails = []


def render(msgs, **kw):
    tools = kw.pop("tools", None)
    return tok.apply_chat_template(
        msgs, chat_template=TMPL, tokenize=False,
        add_generation_prompt=kw.pop("add_generation_prompt", True),
        tools=tools, **kw)


def case(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:
        fails.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")


XHIGH = "Reasoning effort is set to xhigh."
LOW = "Reasoning effort is set to low."


def c_basic():
    out = render([{"role": "user", "content": "hello"}])
    assert out.endswith("<|im_start|>assistant\n<think>\n"), repr(out[-60:])
    assert XHIGH in out, "default xhigh instruction missing"


def c_high():
    out = render([{"role": "user", "content": "hi"}], reasoning_effort="high")
    assert XHIGH in out, "high did not map to xhigh"


def c_minimal():
    out = render([{"role": "user", "content": "hi"}], reasoning_effort="minimal")
    assert LOW in out, "minimal did not map to low"


def c_parts():
    out = render([{"role": "user", "content": [
        {"type": "text", "text": "plain part"}, "bare string"]}])
    assert "plain part" in out and "bare string" in out, out


def c_tools():
    import copy
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function", "function": {
             "name": "get_weather",
             "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"},
        {"role": "user", "content": "thanks"},
    ]
    # serve path: vLLM _postprocess_messages parses string args to dicts
    import json as _json
    msgs2 = copy.deepcopy(msgs)
    for m in msgs2:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                a = tc["function"]["arguments"]
                tc["function"]["arguments"] = (
                    _json.loads(a) if isinstance(a, str) else a)
    out = render(msgs2, tools=tools)
    assert "<function=get_weather>" in out, "tool_call name missing"
    assert "<parameter=city>" in out, "dict arguments not expanded"
    assert "Paris" in out, "dict arguments value missing"
    assert "<tool_response>" in out and "18C sunny" in out, "tool response missing"
    assert out.endswith("<|im_start|>assistant\n<think>\n"), repr(out[-60:])
    assert "<|image_pad|>" not in out, "false-positive vision pad"
    # raw-string fallback (no preprocessing): must not die, must carry value
    out_raw = render(msgs, tools=tools)
    assert "<function=get_weather>" in out_raw, "fallback tool name missing"
    assert "<parameter=arguments>" in out_raw, "fallback parameter missing"
    assert "Paris" in out_raw, "fallback value missing"


def c_bogus():
    try:
        render([{"role": "user", "content": "hi"}], reasoning_effort="bogus")
    except Exception:
        return
    raise AssertionError("bogus reasoning_effort did not raise")


case("basic_think_xhigh", c_basic)
case("high_maps_xhigh", c_high)
case("minimal_maps_low", c_minimal)
case("content_parts_and_bare_strings", c_parts)
case("cc_tool_loop_string_args", c_tools)
case("bogus_effort_raises", c_bogus)

print("RESULT", "ALL_PASS" if not fails else f"FAILED={fails}")
sys.exit(0 if not fails else 1)
