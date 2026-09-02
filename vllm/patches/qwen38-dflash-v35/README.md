# v35 — k>3 clamp removal (user-selectable spec k4+)

## What

Boot-time patch `v35_k4_unclamp.py` deletes the v31 #12 clamp (and its
`VLLM_XPU_ALLOW_K4_CAPTURE` bypass) from the installed
`platforms/xpu.py` in the container. `num_speculative_tokens` passes
through as requested — k=4 or larger is selected directly in the boot
JSON, no env var, no code protection. The corruption warning lives in
documents and the commit history only (KNOWN_ISSUES #12; qwen38-dflash
README addendum 3).

Applied by `bootp.sh` to EVERY lane immediately after the v32 patches
(no-op for nospec and k<=3); idempotent; exits 12 (`K4UNCLAMP_FAILED`)
on tree drift so a vLLM upgrade cannot silently re-enable the clamp.

## Why (user decision, 2026-09-02)

The clamp protected users from the #12 k4 capture corruption. The user
decided the protection is unnecessary: the warning belongs in docs and
the commit message, and selection of k>=4 should be unrestricted.

## Verification (arm v35k4u, same day)

- `SpeculativeConfig(method='mtp', num_spec_tokens=4)` in the serve
  log — k=4 kept, zero clamp warnings, no env var set.
- f8ref `e899790d3635 / f167d905a10b / d84100508821` — BIT-IDENTICAL
  to the former env-bypass lane: clamp removal ≡ bypass, so every v35
  #12 forensic carries over unchanged (fresh-boot distinct=2 with
  logprob drift; 8/8 mojibake by ~14 requests; P2 high-margin intact;
  no wedge at 65k).
- Perf/caveats unchanged: k4 DOMINATED by k3 everywhere (65k warm 24.1
  vs 29.25 tok/s; conc16 9.6-12.9 vs 13.99; 2k early-stops into
  garbage). k>4 passes through UNTESTED.
