#!/usr/bin/env python3
"""kwarg_render_test — ground-truth check of which chat_template_kwargs
values the qwen3.8-27b-fp8 template actually honors. Renders with the
same HF code path vLLM uses (apply_chat_template)."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/models/target")
msgs = [{"role": "user", "content": "Hi"}]
multi = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1",
     "reasoning_content": "deep thought about q1"},
    {"role": "user", "content": "q2"},
]

def render(kw, tag, messages=msgs):
    try:
        s = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, **kw)
        eff = "Reasoning effort is set" in s
        tail = s[-34:].replace("\n", "\\n")
        print(f"{tag:34s} effort_instr={str(eff):5s} tail={tail!r}")
    except Exception as e:
        print(f"{tag:34s} EXCEPTION: {str(e)[:90]}")

print("== enable_thinking variants (1-turn) ==")
render({}, "no kwargs (undefined)")
render({"enable_thinking": "True", "reasoning_effort": "xhigh"},
       'STRING "True" + xhigh  (coder cfg)')
render({"enable_thinking": True, "reasoning_effort": "xhigh"},
       "BOOL true + xhigh")
render({"enable_thinking": "False"}, 'STRING "False" (nonthink cfg)')
render({"enable_thinking": False}, "BOOL false")
render({"enable_thinking": "false"}, 'STRING "false" lowercase')
render({"reasoning_effort": "high"}, "effort=high (invalid value)")
render({"reasoning_effort": "medium"}, "effort=medium")

print("== preserve_thinking variants (multi-turn w/ reasoning) ==")
def render_multi(kw, tag):
    try:
        s = tok.apply_chat_template(multi, tokenize=False,
                                    add_generation_prompt=True, **kw)
        kept = "deep thought about q1" in s
        print(f"{tag:34s} prior_reasoning_kept={kept}")
    except Exception as e:
        print(f"{tag:34s} EXCEPTION: {str(e)[:90]}")

render_multi({}, "no kwargs (undefined)")
render_multi({"preserve_thinking": "True"}, 'STRING "True" (coder cfg)')
render_multi({"preserve_thinking": True}, "BOOL true")
render_multi({"preserve_thinking": "False"}, 'STRING "False" (nonthink)')
render_multi({"preserve_thinking": False}, "BOOL false")
