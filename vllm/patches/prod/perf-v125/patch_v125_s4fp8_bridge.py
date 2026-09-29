#!/usr/bin/env python3
"""patch_v125_s4fp8_bridge.py — v125 P22C5: static fp8 spec bridge.

Root cause of the e4m3s4 boot death (capture 0/64, both TP workers):
RuntimeError: ssm_state dtype must be float32/float16/bfloat16, but got
Float8_e4m3fn — raised by torch.ops._xpu_C.gdn_attention during the
spec-verify dummy run.

Two layers:
  1. WHEEL: the installed vllm-xpu-kernels wheel (0.1.8.3.dev0
     +g3cab97a.d20260925) predates the P22B state-dispatch patch (source
     tree /root/build/vxk patched 2026-09-28, never built+installed).
     _xpu_C.abi3.so carries ONLY the old reject literal — no
     "fp8_e4m3fn/fp8_e5m2" message — so ANY pool-straight fp8 call into
     the SYCL op aborts. The ns fp8 legs never hit it: their decode ran
     entirely on the ESIMD kernels (P22A IS installed) and prefills took
     the fp16 bridge.
  2. ROUTING: _gdn_attention_core_xpu_impl's _native_ok passed the fp8
     pool straight for ANY num_prefills==0 batch — including spec-verify
     (multi-request spec cannot use the ESIMD spec op: it is
     single-request-only), which the wheel cannot host.

Fix (this patch, Python-side only):
  - VLLM_XPU_GDN_FP8_NATIVE=2 becomes the explicit "P22B wheel
    installed" claim; pool-straight fp8 requires it (=1 keeps ESIMD
    decode eligible via the unchanged _gdn_fp8_native_enabled).
  - Under =1 (or unset), EVERY decode-only fp8 SYCL call — spec verify
    or pure decode that ESIMD declined (>128 tokens) — takes the STATIC
    bridge: gather exactly the batch's cache_indices rows to a fp16
    copy, run the wheel kernel on the copy, scatter home via the uint8
    view. Capture-safe by construction: the flat gather order DEFINES
    the remap (remap[r,c] = r*W + c, a constant arange — no value
    lookup, no torch.unique whose output shape is value-dependent and
    poison inside FULL decode graphs). Kernel semantics preserved: the
    xe2 spec kernel loads init state at cache_indices[r, accepted-1]
    and every gathered row scatters back to the slot it came from
    (kernel-untouched rows carry their original bytes — a no-op write).
  - Prefill/mixed batches keep the certified v124 unique bridge
    unchanged (eager-only; prefills are never captured under
    FULL_DECODE_ONLY graphs).

Idempotent via the 'v125 P22C5' marker. Run inside the container:
  docker cp patch_v125_s4fp8_bridge.py lsv-test:/root/
  docker exec lsv-test python3 /root/patch_v125_s4fp8_bridge.py
"""
import py_compile
import sys

P = "/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
MARKER = "v125 P22C5"

src = open(P).read()

if MARKER in src:
    print("V125_C5_ALREADY")
    sys.exit(0)

edits = []


def rep(old, new):
    n = src.count(old)
    assert n == 1, f"anchor not unique (count={n}): {old[:90]!r}"
    edits.append((old, new))


# --- P1: module-level P22B comment block -> accurate =1/=2 semantics -----
rep(
    """# llm-scaler v125 P22B: fp8-native SYCL gdn_attention switch. When the
# installed vllm-xpu-kernels wheel accepts fp8 ssm_state (DISPATCH_STATE_DTYPE
# fp8 branches), VLLM_XPU_GDN_FP8_NATIVE=1 passes the fp8 pool STRAIGHT to
# the op — the v124 P19.5a gather/remap/scatter bridge is skipped entirely.
# Frozen at first call (pre-capture; capture replays must never see a mode
# change). An old wheel under =1 aborts loudly inside the kernel's dtype
# TORCH_CHECK — never a silent fallback.""",
    """# llm-scaler v125 P22B/P22C5: fp8 gdn_attention switch. Two levels:
#   =1  ESIMD-native decode stays eligible (the P22A ESIMD .so accepts fp8
#       pools); every SYCL-op fp8 call takes a bridge (static for
#       decode-only batches, the v124 unique bridge for prefill/mixed).
#   =2  additionally asserts a P22B wheel is installed (DISPATCH_STATE_DTYPE
#       fp8 branches compiled into _xpu_C.abi3.so) and passes the fp8 pool
#       STRAIGHT to the SYCL op for decode-only batches. The v1.2.24 image
#       ships a pre-P22B wheel — verified: only the OLD reject literal is
#       present — so under =2-on-old-wheel the kernel's dtype TORCH_CHECK
#       aborts loudly, never a silent fallback. Frozen at first call
#       (pre-capture; capture replays must never see a mode change).""",
)

# --- P2: _native_ok requires the =2 wheel claim ---------------------------
rep(
    """    _native_ok = (
        _pool_fp8
        and _gdn_fp8_native_enabled()
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )""",
    """    # v125 P22C5: pool-straight fp8 needs the P22B wheel compiled in;
    # =1 alone must NOT pass an fp8 pool to a pre-P22B wheel (the e4m3s4
    # capture crash was exactly that: spec-verify dummy -> old dtype
    # TORCH_CHECK in _xpu_C.abi3.so).
    _native_wheel = os.environ.get("VLLM_XPU_GDN_FP8_NATIVE", "") == "2"
    _native_ok = (
        _pool_fp8
        and _native_wheel
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )""",
)

# --- P3: ENGAGED log text reflects the new claim --------------------------
rep(
    """                "v125 P22B FP8_NATIVE ENGAGED ssm_dtype=%s decode_only=1"
                " (P22B2: prefill batches take the fp16 bridge)" % ssm_pool.dtype,""",
    """                "v125 P22B FP8_NATIVE ENGAGED ssm_dtype=%s decode_only=1"
                " (P22C5: P22B wheel asserted via =2)" % ssm_pool.dtype,""",
)

# --- P4: static-bridge decision after the unique-bridge dispatch ----------
rep(
    """    else:
        _fp8_ssm = _pool_fp8
    ssm_run, _remap, _uniq = None, None, None""",
    """    else:
        _fp8_ssm = _pool_fp8
    # v125 P22C5: decode-only fp8 batches without the P22B wheel take the
    # STATIC bridge (capture-safe). Pre-empts the unique bridge — torch.unique
    # has a value-dependent output shape and must never be captured.
    _fp8_static = (
        _fp8_ssm
        and attn_metadata.num_prefills == 0  # type: ignore[attr-defined]
    )
    if _fp8_static:
        _fp8_ssm = False
        if not globals().get("_GDN_FP8_STATIC_LOGGED"):
            globals()["_GDN_FP8_STATIC_LOGGED"] = True
            print(
                "v125 P22C5 STATIC_FP8_BRIDGE ssm_dtype=%s"
                " (decode-only fp8 via static fp16 copy; pool-straight needs"
                " the P22B wheel + =2)" % ssm_pool.dtype,
                flush=True,
            )
    ssm_run, _remap, _uniq = None, None, None
    _static_flat, _static_remap = None, None""",
)

# --- P5: static gather/remap construction before the kernel call ----------
rep(
    """        ssm_run = ssm_pool.index_select(0, _uniq).to(torch.float16).contiguous()

    torch.ops._xpu_C.gdn_attention(""",
    """        ssm_run = ssm_pool.index_select(0, _uniq).to(torch.float16).contiguous()

    if _fp8_static:
        # Static bridge: gather exactly the batch's cache_indices rows
        # (narrowed — the FULL-graph padded tail is NULL_BLOCK_ID and
        # excluded). The flat gather order DEFINES the remap —
        # remap[r, c] = r*W + c, a CONSTANT arange, no value lookup —
        # so every op is fixed-shape and FULL decode graphs can replay
        # it. The xe2 spec kernel loads init state at
        # cache_indices[r, num_accepted-1] and stores the rolled-forward
        # state back through the same remapped indices; every gathered
        # row scatters home afterwards (kernel-untouched rows carry
        # their original bytes — a no-op write).
        if num_spec_decodes > 0:
            _st = _narrow0(
                getattr(attn_metadata, "spec_state_indices_tensor", None),
                num_spec_decodes,
            )
            _W = _st.size(1)
            _static_flat = _st.reshape(-1).to(torch.int64)
            _static_remap = (
                torch.arange(
                    num_spec_decodes * _W,
                    dtype=torch.int32,
                    device=ssm_pool.device,
                )
                .reshape(num_spec_decodes, _W)
                .contiguous()
            )
        else:
            _nd = attn_metadata.num_decodes  # type: ignore[attr-defined]
            _static_flat = (
                attn_metadata.non_spec_state_indices_tensor.reshape(-1)[:_nd]
                .to(torch.int64)
            )
            _static_remap = torch.arange(
                _nd, dtype=torch.int32, device=ssm_pool.device
            )
        ssm_run = (
            ssm_pool.index_select(0, _static_flat)
            .to(torch.float16)
            .contiguous()
        )

    torch.ops._xpu_C.gdn_attention(""",
)

# --- P6: ssm_state argument ------------------------------------------------
rep(
    "        ssm_state=ssm_run if _fp8_ssm else self.kv_cache[1],",
    "        ssm_state=ssm_run if (_fp8_ssm or _fp8_static) else self.kv_cache[1],",
)

# --- P7: non-spec state indices argument -----------------------------------
rep(
    """        non_spec_state_indices_tensor=_contig(_remap[attn_metadata.non_spec_state_indices_tensor] if _fp8_ssm else attn_metadata.non_spec_state_indices_tensor),  # type: ignore[attr-defined]""",
    """        non_spec_state_indices_tensor=_contig(_static_remap if _fp8_static and num_spec_decodes == 0 else (_remap[attn_metadata.non_spec_state_indices_tensor] if _fp8_ssm else attn_metadata.non_spec_state_indices_tensor)),  # type: ignore[attr-defined]""",
)

# --- P8: spec state indices argument ---------------------------------------
rep(
    """        spec_state_indices_tensor=_contig(_remap[_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes)] if _fp8_ssm and num_spec_decodes > 0 else (_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes) if num_spec_decodes > 0 else None)),  # type: ignore[attr-defined]""",
    """        spec_state_indices_tensor=_contig(_static_remap if _fp8_static and num_spec_decodes > 0 else (_remap[_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes)] if _fp8_ssm and num_spec_decodes > 0 else (_narrow0(getattr(attn_metadata, 'spec_state_indices_tensor', None), num_spec_decodes) if num_spec_decodes > 0 else None))),  # type: ignore[attr-defined]""",
)

# --- P9: scatter-back home for the static rows -----------------------------
rep(
    """        ssm_pool.view(torch.uint8).index_copy_(
            0, _uniq, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )""",
    """        ssm_pool.view(torch.uint8).index_copy_(
            0, _uniq, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )
    elif _fp8_static:
        # v125 P22C5: scatter the static-bridge rows home. index_copy_xpu
        # is not implemented for the fp8 dtypes; the byte-identical uint8
        # view scatters fine (same trick as the unique bridge above).
        ssm_pool.view(torch.uint8).index_copy_(
            0, _static_flat, ssm_run.to(ssm_pool.dtype).view(torch.uint8)
        )""",
)

for old, new in edits:
    src = src.replace(old, new, 1)

bak = P + ".pre_v125_c5"
try:
    open(bak)
except FileNotFoundError:
    with open(bak, "w") as f:
        f.write(open(P).read())
    print(f"backup written: {bak}")

with open(P, "w") as f:
    f.write(src)

py_compile.compile(P, doraise=True)
print("V125_C5_OK")
