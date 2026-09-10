"""CPU and optional XPU contract tests and copy/GEMM microbenchmark."""
import json
import statistics
import sys
import time
from types import SimpleNamespace
import torch
from fp32_cache import fp32_snapshot, clear_fp32_snapshots

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
owner = SimpleNamespace()
w = torch.nn.Parameter(torch.randn(32, 64, device=device, dtype=torch.bfloat16),
                       requires_grad=False)
first = fp32_snapshot(owner, 'w', w)
assert torch.equal(first, w.float())
assert fp32_snapshot(owner, 'w', w) is first
with torch.no_grad():
    w.add_(1)
second = fp32_snapshot(owner, 'w', w)
assert second is not first and torch.equal(second, w.float())
replacement = torch.nn.Parameter(w.clone(), requires_grad=False)
assert fp32_snapshot(owner, 'w', replacement) is not second
clear_fp32_snapshots(owner)
assert not owner._lce1_fp32_snapshots
with torch.inference_mode():
    unversioned = torch.ones(2, device=device, dtype=torch.bfloat16)
    assert torch.equal(fp32_snapshot(owner, 'unversioned', unversioned), unversioned.float())
print(json.dumps(dict(contract_tests='pass', device=device)), flush=True)

if device.startswith('xpu'):
    torch.manual_seed(17)
    w = torch.nn.Parameter(torch.randn(1280, 5120, device=device, dtype=torch.bfloat16),
                           requires_grad=False)
    owner = SimpleNamespace()
    for rows in (8, 32, 64):
        x = torch.randn(rows, 5120, device=device, dtype=torch.bfloat16)
        def old():
            return torch.nn.functional.linear(x.float(), w.float())
        def new():
            return torch.nn.functional.linear(x.float(), fp32_snapshot(owner, 'w', w))
        with torch.inference_mode():
            reference, repeated, candidate = old(), old(), new()
            comparison = dict(
                baseline_repeat_exact=torch.equal(reference, repeated),
                candidate_exact=torch.equal(reference, candidate),
                baseline_repeat_max_abs=(reference-repeated).abs().max().item(),
                candidate_max_abs=(reference-candidate).abs().max().item(),
                candidate_close=torch.allclose(reference, candidate, rtol=1e-5, atol=1e-4))
            print(json.dumps(dict(rows=rows, comparison=comparison)), flush=True)
            for _ in range(5):
                old(); new()
            measurements = {'old': [], 'cached': []}
            for repeat in range(6):
                order = [('old', old), ('cached', new)]
                if repeat % 2:
                    order.reverse()
                for name, fn in order:
                    torch.xpu.synchronize()
                    start = time.perf_counter()
                    for _ in range(30):
                        fn()
                    torch.xpu.synchronize()
                    measurements[name].append((time.perf_counter()-start)/30*1000)
        medians = {k: statistics.median(v) for k, v in measurements.items()}
        print(json.dumps(dict(rows=rows, ms=medians, samples_ms=measurements,
                              speedup=medians['old']/medians['cached'],
                              persistent_bytes=w.numel()*4,
                              bit_exact=comparison['candidate_exact'])), flush=True)
