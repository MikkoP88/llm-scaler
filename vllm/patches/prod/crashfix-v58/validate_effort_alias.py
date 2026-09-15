#!/opt/venv/bin/python3
# validate_effort_alias.py — offline (CPU) chat-template render check
# for the high->xhigh alias in chat_template_qwen38_high.jinja.
import sys
from transformers import AutoTokenizer

PATCHED = open('/root/chat_template_qwen38_high.jinja').read()
tok = AutoTokenizer.from_pretrained('/models/target')
tok.chat_template = PATCHED
msgs = [{'role': 'user', 'content': '2+2?'}]

def render(**kw):
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, **kw)

# 1) high accepted and byte-identical to xhigh
h = render(reasoning_effort='high')
x = render(reasoning_effort='xhigh')
assert h == x, 'MISMATCH high vs xhigh'
print('PASS high accepted; render identical to xhigh (len %d)' % len(h))

# 2) existing tiers unaffected
for e in ('medium', 'low'):
    r = render(reasoning_effort=e)
    print('PASS %s renders (len %d)' % (e, len(r)))
d = render()  # default -> xhigh
assert d == x, 'MISMATCH default vs xhigh'
print('PASS default == xhigh')

# 3) preserve_thinking kwarg (serve default) still works with alias
pt = render(reasoning_effort='high', preserve_thinking=True)
print('PASS preserve_thinking+high renders (len %d)' % len(pt))

# 4) invalid values still rejected (guard intact)
try:
    render(reasoning_effort='ultra')
    print('FAIL: ultra accepted'); sys.exit(1)
except Exception as ex:
    print('PASS invalid still rejected: %s' % str(ex)[:80])

# 5) sanity: instruction sentence present in high render
assert 'Reasoning effort is set to xhigh' in h
print('PASS xhigh instruction present in high render')
print('ALL_VALIDATE_OK')
