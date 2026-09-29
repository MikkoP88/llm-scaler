#!/opt/venv/bin/python
"""llm-scaler v124 P19.5a — python plumbing for fp8 mamba/GDN SSM cache.

Implements `--mamba-ssm-cache-dtype fp8_e4m3` / `fp8_e5m2` (user directive
2026-09-28; P18-L1 had proven the fork rejected both at argparse).

Three changes:
1. config/cache.py — extend the MambaDType Literal (argparse choices derive
   from it via get_kwargs(CacheConfig), so the CLI accepts both formats).
2. mamba_utils._mamba_state_dtype — resolve the temporal state to the REAL
   torch fp8 dtypes (NOT the shared STR_DTYPE_TO_TORCH_DTYPE uint8 storage
   convention — kernel call sites must be able to branch on the true dtype).
   conv_state stays on the conv dtype (fp16) — only the SSM/temporal state
   is fp8, matching upstream flag semantics.
3. _xpu_ops._gdn_attention_core_xpu_impl — fp8 bridge for the SYCL
   gdn_attention kernel (prefill/mixed/decode-fallback path): gather this
   batch's SSM slots to fp16, remap the slot-index tensors to the compact
   range, run the kernel unchanged, scatter back as fp8. Per-step cast cost
   applies ONLY to prefill/mixed steps in the final design; pure decode
   goes through the ESIMD fused kernel (P19.5b makes it fp8-native — until
   then the _gdn_conv_state_fp16_ok gate falls decode back to this bridge,
   so expect parity-or-worse decode until P19.5b; this stage validates
   numerics/quality E2E).

   CONSTRAINT (verified against the wrapper source): the call site's
   _narrow0 comment proves this op runs under FULL_DECODE_ONLY capture,
   where the index tensors are dim0-padded to static shapes. The bridge's
   torch.unique gather is data-dependent-shape and therefore capture-
   unsafe — fp8 legs MUST boot with cudagraph_mode NONE (the runner does).
   Stage B/C (kernel-native fp8) removes the bridge and restores graphs.

   Gather covers the FULL non-spec index tensor (prefill slots live in it
   — the kernel splits it via num_prefills/num_decodes) plus the narrowed
   spec tensor, mirroring the call site exactly.

Idempotent; asserts every anchor before writing.
"""
import sys

EDITS = [
    # (path, old, new, marker)
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/config/cache.py",
        'MambaDType = Literal["auto", "float32", "float16"]',
        '# llm-scaler v124 P19.5a: fp8 SSM cache formats (GDN temporal state)\n'
        'MambaDType = Literal["auto", "float32", "float16", "fp8_e4m3", "fp8_e5m2"]',
        "v124 P19.5a",
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/mamba_utils.py",
        '''        if mamba_ssm_cache_dtype == "auto":
            temporal_state_dtype = conv_state_dtype
        else:
            temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]

        return (conv_state_dtype, temporal_state_dtype)''',
        '''        if mamba_ssm_cache_dtype == "auto":
            temporal_state_dtype = conv_state_dtype
        elif mamba_ssm_cache_dtype == "fp8_e4m3":
            # llm-scaler v124 P19.5a: REAL fp8 dtype (not the shared STR map's
            # uint8 storage convention) so kernel call sites can branch on it.
            temporal_state_dtype = torch.float8_e4m3fn
        elif mamba_ssm_cache_dtype == "fp8_e5m2":
            temporal_state_dtype = torch.float8_e5m2
        else:
            temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]

        return (conv_state_dtype, temporal_state_dtype)''',
        "v124 P19.5a",
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py",
        "    torch.ops._xpu_C.gdn_attention(",
        '''    # llm-scaler v124 P19.5a: fp8 SSM cache bridge. The SYCL kernel speaks
    # fp16/fp32 state — gather this batch's slots to fp16, remap index
    # tensors to the compact range, run, scatter back. Prefill/mixed (and
    # decode-fallback until the ESIMD kernel is fp8-native) only.
    ssm_pool = self.kv_cache[1]
    _fp8_ssm = ssm_pool.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ssm_run, _remap, _uniq = None, None, None
    if _fp8_ssm:
        # Gather the FULL non-spec index tensor (prefill slots live there too
        # — the kernel splits it via num_prefills/num_decodes, so no slicing)
        # plus the narrowed spec tensor, mirroring the call site below.
        _ns_idx = attn_metadata.non_spec_state_indices_tensor
        _sp_idx = _narrow0(
            getattr(attn_metadata, "spec_state_indices_tensor", None),
            num_spec_decodes,
        )
        _parts = [_ns_idx.reshape(-1)] if _ns_idx is not None else []
        if num_spec_decodes > 0 and _sp_idx is not None:
            _parts.append(_sp_idx.reshape(-1))
        _uniq = torch.unique(torch.cat(_parts).to(torch.int64))
        _remap = torch.zeros(
            ssm_pool.size(0), dtype=torch.int32, device=ssm_pool.device
        )
        _remap[_uniq] = torch.arange(
            _uniq.size(0), dtype=torch.int32, device=ssm_pool.device
        )
        ssm_run = ssm_pool.index_select(0, _uniq).to(torch.float16).contiguous()

    torch.ops._xpu_C.gdn_attention(''',
        "v124 P19.5a: fp8 SSM cache bridge",
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py",
        "        ssm_state=self.kv_cache[1],",
        "        ssm_state=ssm_run if _fp8_ssm else self.kv_cache[1],",
        None,
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py",
        "        non_spec_state_indices_tensor=_contig(attn_metadata.non_spec_state_indices_tensor),  # type: ignore[attr-defined]",
        "        non_spec_state_indices_tensor=_contig(_remap[attn_metadata.non_spec_state_indices_tensor] if _fp8_ssm else attn_metadata.non_spec_state_indices_tensor),  # type: ignore[attr-defined]",
        None,
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py",
        "        spec_state_indices_tensor=_contig(_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes) if num_spec_decodes > 0 else None),  # type: ignore[attr-defined]",
        "        spec_state_indices_tensor=_contig(_remap[_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes)] if _fp8_ssm and num_spec_decodes > 0 else (_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes) if num_spec_decodes > 0 else None)),  # type: ignore[attr-defined]",
        None,
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py",
        '''        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )''',
        '''        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )
    if _fp8_ssm:
        # Scatter via uint8 view: index_copy_xpu is NOT implemented for the
        # fp8 dtypes (stage-A pre-flight), but the byte-identical uint8 view
        # scatters fine. zeros/index_select/dtype casts ARE implemented.
        ssm_pool.view(torch.uint8).index_copy_(
            0, _uniq, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )''',
        None,
    ),
]


def main() -> int:
    applied = 0
    for path, old, new, marker in EDITS:
        src = open(path).read()
        # Idempotency MUST key on the NEW text, not the anchor: some anchors
        # survive patching as the tail of the inserted text (the prelude ends
        # with the very call line it is inserted before), so an anchor-based
        # ALREADY check re-inserts on re-run (found live in stage-A run 2 —
        # duplicate prelude; repaired from the pristine image file).
        if new in src:
            print(f"ALREADY: {path} ({marker or old.strip()[:40]})")
            applied += 1
            continue
        if old not in src:
            print(f"FAIL anchor not found in {path}:\n{old[:120]}...")
            return 1
        if src.count(old) != 1:
            print(f"FAIL anchor not unique ({src.count(old)}x) in {path}")
            return 1
        open(path, "w").write(src.replace(old, new, 1))
        applied += 1
        print(f"OK: {path}")
    print(f"P195A_PATCHED_OK ({applied}/{len(EDITS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
