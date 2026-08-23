# Tutorial: Calibrated FP8 KV cache for Qwen3.8-27B

This guide fixes the "repetition loops after ~500 tokens" failure that appears
when serving Qwen3.8-27B with `--kv-cache-dtype fp8_e4m3` from a checkpoint that
has no KV cache scales.

There are two ways to do the calibration, both producing the same fix:

| | Path 1: llm-compressor | Path 2: scale-only on your B70s |
|---|---|---|
| Runs on | A CUDA machine (rented/borrowed) | Your 2× Arc Pro B70 server |
| Extra software | `llmcompressor` | `transformers` + `accelerate` only |
| Also quantizes weights to static FP8 | Yes (W8A8) | No (keeps online `--quantization fp8`) |
| Effort | Run one script on cloud hardware | Run one script locally |

Pick Path 1 if you want the full official recipe and have access to an NVIDIA
GPU for a few hours. Pick Path 2 if you want to stay on your own hardware.
Everything after calibration (deploy, validation) is identical.

## The problem, in plain terms

When you pass `--kv-cache-dtype fp8_e4m3`, vLLM stores the attention cache in
8-bit floats. Storing a number in fp8 requires a **scale factor** that maps real
values into the small fp8 range (max ±448). If the checkpoint does not contain
per-layer `k_scale`/`v_scale` values, vLLM defaults them to 1.0 and warns:

```
WARNING ... Checkpoint does not provide a q scaling factor. Setting it to kScale.
WARNING ... Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended,
           verify that k/v_scale scaling factors are properly set in the checkpoint.
```

A scale of 1.0 wastes almost the whole fp8 range: your keys and values live
around magnitude 1-10, where fp8 has only a few mantissa bits of resolution.
Every cached token therefore carries several percent of quantization noise.
Generation then feeds on itself — slightly wrong token → its noisy K/V gets
cached → later tokens attend over increasingly corrupted context — until output
collapses into repetition loops, typically a few hundred tokens in.

**Calibration fixes this** by measuring the real activation ranges once,
offline: `scale = observed_max / 448` per layer. Values then occupy the
well-resolved part of fp8, and the loops disappear.

What you end up with is a checkpoint copy containing per-layer scale tensors:

```
model.layers.0.self_attn.k_scale   (float32 scalar)
model.layers.0.self_attn.v_scale   (float32 scalar)
...                                 (one pair per attention layer)
```

vLLM automatically prefers checkpoint scales over the 1.0 default when they are
present, and the warning disappears.

---

## Path 1: llm-compressor (on a CUDA machine)

### 1.1 Where to run

The calibration loads the model in bf16 (~55 GB of weights for 27B, plus
activations). Note that llm-compressor is CUDA-first — it does not support or
test Intel XPU, so this path does **not** run on the B70s.

| Setup | Notes |
|---|---|
| 1× A100/H100 80 GB | Simplest; the whole run fits |
| 2× 48-64 GB CUDA GPUs | Pass `device="cuda:0,1"` to `oneshot` for multi-GPU |
| Rented cloud box | A few hours of a single GPU instance is enough |

fp8 compute requires NVIDIA compute capability >= 8.9 (Ada/Hopper or newer).
The output directory (~28-30 GB) is portable — copy it to `/models/` on the B70
server afterwards. Nothing about the result is XPU-specific.

### 1.2 Install

```bash
pip install "llmcompressor>=0.10" datasets transformers
```

### 1.3 Run the calibration

The ready-made script sits next to this document:

```bash
python qwen3.8-27b-fp8-kv-calibration.py
```

What it does:

1. Loads the model (`AutoModelForImageTextToText` for the vision-capable
   `Qwen3_5ForConditionalGeneration` variant, `AutoModelForCausalLM`
   otherwise). Point `MODEL_ID` at a local path, e.g. `/models/qwen3.8-27b`,
   if you are not pulling from Hugging Face.
2. Calibrates on 512 shuffled `ultrachat_200k` chat samples, max length 4096.
   Text-only calibration is fine even for the vision variant — only text
   attention layers get KV scales.
3. Applies the official recipe: static W8A8 FP8 weights and activations plus
   the `kv_cache_scheme` block that triggers per-layer KV scale calibration.
4. Prints a short sample generation as a sanity check and saves everything to
   `Qwen3.8-27B-FP8-KV/`.

Expect a few hours on an 80 GB GPU for a 27B model.

Knobs, if you need them:

- **OOM**: lower `NUM_CALIBRATION_SAMPLES` to 256 or `MAX_SEQUENCE_LENGTH` to
  2048, or spread across more GPUs.
- **Multi-GPU**: add `device="cuda:0,1"` to the `oneshot(...)` call.
- **Vision tower misbehaves when served in fp8**: add `"visual"` to the
  recipe's `ignore` list to keep it in bf16 (KV scales are unaffected).

### 1.4 Verify the output

```bash
python - <<'EOF'
import glob, json
from safetensors import safe_open

cfg = json.load(open("Qwen3.8-27B-FP8-KV/config.json"))
scheme = cfg["quantization_config"]["kv_cache_scheme"]
print("kv_cache_scheme:", scheme)
assert scheme == {"num_bits": 8, "type": "float", "strategy": "tensor",
                  "dynamic": False, "symmetric": True}

kv_keys = []
for path in glob.glob("Qwen3.8-27B-FP8-KV/*.safetensors"):
    with safe_open(path, framework="pt") as f:
        kv_keys += [k for k in f.keys() if k.endswith(("k_scale", "v_scale"))]
print(f"{len(kv_keys)} k/v scale tensors found")
assert len(kv_keys) > 0, "no KV scales in checkpoint"
EOF
```

### 1.5 Deploy to the B70 server

Copy `Qwen3.8-27B-FP8-KV/` to the server (e.g. `/models/qwen3.8-27b-fp8kv`)
and run your usual command **without** `--quantization fp8` and **without**
`--kv-cache-dtype` — both are auto-detected from the checkpoint:

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 VLLM_XPU_ENABLE_XPU_GRAPH=1 \
vllm serve --model /models/qwen3.8-27b-fp8kv \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 --dtype float16 --enable-chunked-prefill \
  --trust-remote-code --gpu-memory-utilization 0.8 --block-size 128 \
  --max-num-batched-tokens 8192 --reasoning-parser deepseek_r1 \
  --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --port 8000 --host 0.0.0.0 --max-model-len=262144
```

The startup log now shows the compressed-tensors fp8 method (instead of
`Fp8OnlineLinearMethod`) and neither of the two KV scale warnings.

---

## Path 2: calibrate the scales directly on the 2× B70 server

The only thing vLLM strictly needs is the per-layer `k_scale`/`v_scale` tensors
in the safetensors — the `kv_cache_scheme` config entry that Path 1 adds is
just a convenience that auto-forces the fp8 dtype. The scales can be measured
on your own hardware with plain transformers, because 27B in bf16 (~55 GB)
fits when split across the two 32 GB GPUs.

### 2.1 Prerequisites

- A Python environment with **XPU-enabled torch** (e.g. the same base used by
  the vLLM container) plus:
  ```bash
  pip install transformers accelerate datasets
  ```
- ~64 GB of **system RAM** (saving gathers the bf16 state dict through host
  memory) and ~60 GB of free disk for the checkpoint copy.
- Internet access to Hugging Face for the calibration dataset
  (`ultrachat_200k`, first 256 samples only). Offline alternative: change
  `DATASET_ID`/the loading code in the script to read a local text file of
  representative prompts.

### 2.2 Run

The ready-made script sits next to this document:

```bash
python qwen3.8-27b-kv-scale-calibration-xpu.py
```

Edit `MODEL_ID` at the top first (e.g. `/models/qwen3.8-27b`). What it does:

1. Loads the model in bf16 with `device_map="auto"`, sharding over `xpu:0`
   and `xpu:1` with ~28 GB per GPU (leaves headroom for activations).
2. Finds every text attention layer (vision tower and GDN layers are skipped —
   only the layers vLLM quantizes get scales; expect 16 on this hybrid model).
3. Measures per-layer maxima of **post-RoPE keys** (by wrapping the model's
   `apply_rotary_pos_emb`) and **values** (hooks on `v_proj`) across 256
   chat prompts at max length 2048.
4. Registers `k_scale = k_amax / 448` and `v_scale = v_amax / 448` float32
   buffers on each layer.
5. Saves the calibrated copy (`save_pretrained` — shards + index + config) to
   `Qwen3.8-27B-KV-Scales/`, plus a `kv_scale_report.json` with every layer's
   amax and scale for inspection.

Expect roughly 1-2 hours end to end (model load, 256 forwards, save).

Knobs: `NUM_CALIBRATION_SAMPLES` (256 → 512 for more stable maxima),
`MAX_SEQUENCE_LENGTH` (2048 → 4096 if you have headroom, at proportionally
slower speed), `MAX_MEMORY_GB_PER_GPU` (lower it if you hit OOM).

### 2.3 Verify and deploy

Check the scales landed in the checkpoint:

```bash
python - <<'EOF'
import glob
from safetensors import safe_open
kv = []
for p in glob.glob("Qwen3.8-27B-KV-Scales/*.safetensors"):
    with safe_open(p, framework="pt") as f:
        kv += [k for k in f.keys() if k.endswith(("k_scale", "v_scale"))]
print(len(kv), "k/v scale tensors")
assert len(kv) > 0
EOF
```

Copy/symlink the directory next to your original model and serve exactly as
you do today, adding `--kv-cache-dtype fp8_e4m3` (keep `--quantization fp8`):

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 VLLM_XPU_ENABLE_XPU_GRAPH=1 \
vllm serve --model /models/qwen3.8-27b-kvscales \
  --served-model-name qwen3.8-27b --tensor-parallel-size 2 \
  --quantization fp8 --kv-cache-dtype fp8_e4m3 --dtype float16 \
  --enable-chunked-prefill --trust-remote-code --gpu-memory-utilization 0.8 \
  --block-size 128 --max-num-batched-tokens 8192 \
  --reasoning-parser deepseek_r1 --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice --port 8000 --host 0.0.0.0 --max-model-len=262144
```

vLLM loads the checkpoint scales automatically
("Using KV cache scaling factor 1.0" warning gone).

The difference from Path 1: weights stay online-quantized fp8 instead of
pre-quantized W8A8, and you pass the two flags explicitly. The KV calibration
result — and the fix for the repetition loops — is the same.

---

## Validate the fix (both paths)

1. **Repetition test**: ask for >1000-token generations (code, summaries,
   long-form answers). The old failure showed loops appearing around token
   ~500; with calibrated scales they should be gone. Keep the old checkpoint
   around and compare side by side.
2. **Optional benchmark** (any machine with the same vLLM):

```bash
lm_eval --model vllm \
  --model_args pretrained=/models/qwen3.8-27b-fp8kv,kv_cache_dtype=fp8,add_bos_token=True,dtype=float16 \
  --tasks gsm8k --num_fewshot 5 --batch_size auto
```

(`add_bos_token=True` — quantized models are sensitive to a missing bos token.)

## Serving notes: DSpark and TurboQuant

- **DSpark speculative decoding**: works unchanged. The calibrated checkpoint
  forces the target's KV dtype to plain `fp8` (not turboquant), the drafter
  inherits it, autoselects FLASH_ATTN, and drafting behaves exactly like your
  existing fp8 DSpark runs.
- **TurboQuant**: with a Path 1 checkpoint, the `kv_cache_scheme` entry takes
  precedence over `--kv-cache-dtype turboquant_*` (the scheme is applied
  unconditionally when present). To switch back to TurboQuant serving, either
  use the original checkpoint or strip `kv_cache_scheme` from the checkpoint's
  `config.json`. Path 2 checkpoints have no scheme entry, so
  `--kv-cache-dtype turboquant_*` keeps working as before (the scale tensors
  are simply ignored by the TurboQuant backend).

## Troubleshooting

- **Still see the 1.0 warning after deploying**: the scales were not found in
  the checkpoint. Check the tensor names end with `self_attn.k_scale` /
  `self_attn.v_scale`, that they appear in `model.safetensors.index.json`,
  and that you are serving the calibrated copy, not the original directory.
- **OOM during calibration**: fewer samples, shorter max length, more GPUs
  (Path 1) or lower max length to 2048 (Path 2).
- **Rollback**: serve the original, unmodified checkpoint — calibration only
  ever writes to a copy.

## References

- [LLM Compressor docs — KV Cache Quantization](https://docs.vllm.ai/projects/llm-compressor/en/0.10.0/examples/quantization_kv_cache/)
- [vLLM docs — Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)
