#!/usr/bin/env python3
"""fn8 leg-7 probe: why does ModelConfig.uses_mrope fire on flash-next-fp8?
Dump rope_parameters on top + text configs and run vllm's uses_mrope."""
from vllm.transformers_utils.config import get_config

cfg = get_config("/models/flash-next-fp8", trust_remote_code=True)
tc = cfg.get_text_config()
print("top type:", type(cfg).__name__, "text type:", type(tc).__name__)
print("text rope_parameters:", getattr(tc, "rope_parameters", "ABSENT"))
print("top rope_parameters:", getattr(cfg, "rope_parameters", "ABSENT"))
print("text rope_scaling:", getattr(tc, "rope_scaling", "ABSENT"))

from vllm.transformers_utils.config import uses_mrope  # noqa: E402

print("uses_mrope(cfg):", uses_mrope(cfg))
