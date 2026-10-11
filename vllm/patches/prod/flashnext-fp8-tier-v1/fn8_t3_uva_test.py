#!/usr/bin/env python3
"""fn8-t3 validation part 3: uva.py whitelist default under FN8_TIERED=1."""
import torch

torch.xpu.synchronize = lambda *a, **k: None
torch.xpu.current_device = lambda: "cpu"

import vllm.model_executor.offloader.uva as u  # noqa: E402

assert u._FN8_TIERED is True
# default whitelist fills the empty set
off = u.UVAOffloader(cpu_offload_max_bytes=1 << 30, cpu_offload_params=None)
assert off.cpu_offload_params == {
    "w13_weight", "w2_weight",
    "w13_weight_scale_inv", "w2_weight_scale_inv",
}, off.cpu_offload_params
# explicit CLI set wins untouched
off2 = u.UVAOffloader(1 << 30, {"some_other_param"})
assert off2.cpu_offload_params == {"some_other_param"}
print("UVA_WHITELIST_OK")
