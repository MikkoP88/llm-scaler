# litellm custom callback: fold mid-conversation system messages so the
# Qwen chat template ("System message must be at the beginning") accepts
# Claude Code's OpenAI-compat request shapes, where system-reminders
# arrive as role=system at index > 0 (typically right after a tool
# result). Folding target is the NEXT user turn — the model sees the
# reminder text just before it responds, which is the sender's intent.
#
# Fail-open: any exception leaves the request untouched.
# Deploy: mounted/copied to /app/custom_callbacks.py in the proxy
# container + general_settings.custom_callback_path in config.yaml.
from litellm.integrations.custom_logger import CustomLogger


def _txt(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" and b.get("text"):
                    parts.append(b["text"])
                elif b.get("text"):
                    parts.append(str(b["text"]))
        return "\n".join(parts)
    return str(content) if content else ""


def _fold_system(messages):
    out = []
    pending = []  # system text awaiting the next user turn
    for m in messages:
        role = m.get("role")
        if role == "system" and out:
            pending.append(_txt(m.get("content")))
            continue
        if role == "user" and pending:
            add = "\n\n".join(p for p in pending if p)
            pending = []
            c = m.get("content")
            if isinstance(c, str):
                m = dict(m, content=(add + "\n\n" + c) if c else add)
            elif isinstance(c, list):
                blocks = ([{"type": "text", "text": add}] if add else []) + list(c)
                m = dict(m, content=blocks)
            elif add:
                pending = [add]
        out.append(m)
    if pending:  # trailing system with no following user turn
        add = "\n\n".join(p for p in pending if p)
        if add and out and out[-1].get("role") in ("user", "assistant"):
            out[-1] = dict(
                out[-1], content=_txt(out[-1].get("content")) + "\n\n" + add)
    return out


class FoldMidSystem(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            msgs = data.get("messages")
            if msgs:
                data["messages"] = _fold_system(msgs)
        except Exception:
            pass
        return data


custom_callback = FoldMidSystem()
custom_logger = FoldMidSystem()
