#!/bin/bash
# p22a_fix3_store_const.sh — ROOT FIX for the e4m3 spec divergence:
# esimd_e4m3_store in utils.h used magic constant 0xF887FFFF; the correct
# value (per its own comment ((7-127)<<23)+0x7FFFF) is 0xC407FFFF.
# Evidence: seq e4m3 T3 pool image decoded to 80/88 (true h ~0.25) on the
# 0xF887FFFF build; loads don't see it in seq (no intra-call reload) but
# spec reloads the corrupted saves each token -> amplification -> 480 cap
# -> NaN. e5m2 store (RNE top-bits) has no magic constant -> immune.
set -eu
U=/root/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/utils.h
if grep -q '0xF887FFFFu' "$U"; then
  sed -i 's/0xF887FFFFu/0xC407FFFFu/g' "$U"
  echo "fix3: constant replaced 0xF887FFFF -> 0xC407FFFF"
elif grep -q '0xC407FFFFu' "$U"; then
  echo "fix3: ALREADY_APPLIED"
else
  echo "fix3: NEITHER CONSTANT FOUND — manual inspect"; exit 1
fi
echo "--- store function now ---"
grep -n -A3 'norm = f_bits' "$U" | head -8
