#!/bin/bash
# harm-extract: pull the 16 modified + 2 added vllm-tree files out of the
# prod image (llm-scaler-vllm-adv:v38) and the pristine wheel copies out of
# adv:v14, then emit per-file unified diffs + an md5 manifest.
# Host-side tool for building patches/harmonized/. Not used at serve time.
set -euo pipefail
SP=/opt/venv/lib/python3.12/site-packages
H=/root/build/harm
IMG_PROD=llm-scaler-vllm-adv:v38
IMG_PRISTINE=llm-scaler-vllm-adv:v14

MOD16="vllm/config/vllm.py
vllm/distributed/device_communicators/xpu_communicator.py
vllm/model_executor/layers/attention/attention.py
vllm/model_executor/models/qwen3_5_mtp.py
vllm/platforms/xpu.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/turboquant_attn.py
vllm/v1/attention/ops/triton_turboquant_decode.py
vllm/v1/attention/ops/triton_unified_attention.py
vllm/v1/core/sched/utils.py
vllm/v1/spec_decode/dflash.py
vllm/v1/spec_decode/llm_base_proposer.py
vllm/v1/worker/gpu/cudagraph_utils.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/mamba_utils.py"
ADD2="vllm/v1/spec_decode/drafter_comm.py
vllm/v1/spec_decode/spec_timing.py"

rm -rf "$H"; mkdir -p "$H/v38" "$H/v14" "$H/diffs" "$H/added"

for f in $MOD16 $ADD2; do
  mkdir -p "$H/v38/$(dirname "$f")"
  docker run --rm --entrypoint cat "$IMG_PROD" "$SP/$f" > "$H/v38/$f"
done
for f in $MOD16; do
  mkdir -p "$H/v14/$(dirname "$f")"
  docker run --rm --entrypoint cat "$IMG_PRISTINE" "$SP/$f" > "$H/v14/$f"
done

i=0
: > "$H/MANIFEST.raw"
cd "$H"
for f in $MOD16; do
  i=$((i+1))
  n=$(printf "%02d" "$i")
  flat=$(echo "$f" | tr / _)
  diff -u "v14/$f" "v38/$f" > "$H/diffs/$n-$flat.patch" || true
  p1=$(md5sum "$H/v14/$f" | cut -d" " -f1)
  p2=$(md5sum "$H/v38/$f" | cut -d" " -f1)
  echo "$f $p1 $p2" >> "$H/MANIFEST.raw"
done
for f in $ADD2; do
  mkdir -p "$H/added/$(dirname "$f")"
  cp "$H/v38/$f" "$H/added/$f"
  flat=$(echo "$f" | tr / _)
  p2=$(md5sum "$H/v38/$f" | cut -d" " -f1)
  echo "$f - $p2" >> "$H/MANIFEST.raw"
done

echo "=== extraction OK; diff line counts ==="
wc -l "$H"/diffs/*.patch | sort -k2
echo "=== added files ==="
wc -l "$H"/added/vllm/v1/spec_decode/*.py
echo "=== manifest ==="
cat "$H/MANIFEST.raw"
