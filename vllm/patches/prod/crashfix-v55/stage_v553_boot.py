#!/usr/bin/env python3
"""Stage repro_bootV1212_v553.sh = boot script + v55.3 apply + decode env var."""
import sys

SRC = "/root/build/repro_bootV1212.sh"
DST = "/root/build/repro_bootV1212_v553.sh"

s = open(SRC).read()

old_env = "    -e VLLM_V55_EVENT_TIMEOUT_S=600 \\\n    -e VLLM_F15B=1"
new_env = (
    "    -e VLLM_V55_EVENT_TIMEOUT_S=600 \\\n"
    "    -e VLLM_V55_DECODE_TIMEOUT_S=30 \\\n"
    "    -e VLLM_F15B=1"
)
assert s.count(old_env) == 1, f"env anchor matched {s.count(old_env)}"
s = s.replace(old_env, new_env)

old_blk = (
    "docker cp /root/build/patch_f15b.py lsv-test:/root/patch_f15b.py\n"
    "docker exec lsv-test /opt/venv/bin/python3 /root/patch_f15b.py "
    "| tee /root/build/lce1/f15b_apply_bootV1212.log\n"
)
new_blk = old_blk + (
    "docker cp /root/build/patch_v55_3.py lsv-test:/root/patch_v55_3.py\n"
    "docker exec lsv-test /opt/venv/bin/python3 /root/patch_v55_3.py "
    "| tee /root/build/lce1/v55_3_apply_bootV1212.log\n"
    "docker exec lsv-test grep -c \"llm-scaler v55.3\" "
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
    "gpu_model_runner.py\n"
)
assert s.count(old_blk) == 1, f"block anchor matched {s.count(old_blk)}"
s = s.replace(old_blk, new_blk)

open(DST, "w").write(s)
print("[stage] wrote", DST)
