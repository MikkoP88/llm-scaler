"""Compare installed grouped-convolution math with generated candidate.

Only the linear parameter container is stubbed; both math functions come
directly from the audited source. This is not a full vLLM graph test.
"""
import ast
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from fp32_cache import fp32_snapshot

device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'

class Linear(nn.Linear):
    def __init__(self, a, b, *, params_dtype, **kwargs):
        super().__init__(a, b, bias=False, dtype=params_dtype)
        self.weight.requires_grad_(False)

def load(path, enabled):
    module = ast.parse(Path(path).read_text())
    module.body = [n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and n.name in ('_grouped_conv', 'DFlashGroupedConv')]
    scope = dict(torch=torch, nn=nn, F=F, ReplicatedLinear=Linear,
                 maybe_prefix=lambda a, b: b, _LCE1_CACHE=enabled,
                 fp32_snapshot=fp32_snapshot)
    exec(compile(module, path, 'exec'), scope)
    return scope['DFlashGroupedConv']

torch.manual_seed(41)
old_cls, new_cls = load(sys.argv[1], False), load(sys.argv[2], True)
cases = 0
for dtype in (torch.float16, torch.bfloat16):
    for block in (5, 8):
        old = old_cls(64, 2, 16, block, dtype, '').to(device)
        with torch.no_grad():
            old.base_kernel.normal_(0, .1)
        new = new_cls(64, 2, 16, block, dtype, '').to(device)
        new.load_state_dict(old.state_dict())
        for clients in (1, 4):
            x = torch.randn(block * clients, 64, dtype=dtype, device=device)
            with torch.inference_mode():
                a, ac = old.prepare(x)
                b, bc = new.prepare(x)
                assert torch.equal(a, b) and torch.equal(ac, bc)
                assert torch.equal(old.finish(a, ac), new.finish(b, bc))
            cases += 1
print(json.dumps(dict(model_contract='pass', cases=cases, device=device)))
