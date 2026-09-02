# llm-scaler v34: per-call overhead removal in the v33 fp8-verify varlen
# shim (_v33_mq3d_varlen). py-spy showed the shim on the per-step host
# path; every step it re-created: (a) two torch.ones(1) descale tensors
# when the caller passes None, (b) the fp8 dtype selection branch, and
# (c) two .view(kv8) tensor objects for the (stable) KV cache tensors.
# All three are pure functions of immutable-per-boot state -> cache them
# on the impl. NUMERICS UNCHANGED: same tensors/views, same kernel args,
# bit-identical outputs by construction. Apply AFTER v33_mq3d_triton.py.
import py_compile

FA = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"
fa = open(FA).read()
assert "def _v33_mq3d_varlen" in fa, "v33 mq3d helper not present; apply v33_mq3d_triton.py first"
if "llm-scaler v34: cached descale/ones + kv8 dtype + views" not in fa:

    old1 = """    ntok, Hq, D = q.shape
    D_pad = 1 << (D - 1).bit_length()
    nseg = _V33_MQ3D_SEGS
    bufs = getattr(impl, "_v33_mq3d_bufs", None)
"""
    new1 = """    ntok, Hq, D = q.shape
    D_pad = 1 << (D - 1).bit_length()
    nseg = _V33_MQ3D_SEGS
    bufs = getattr(impl, "_v33_mq3d_bufs", None)
"""
    # (no change at site 1; kept for diff clarity)

    old2 = """    seg_o, seg_m, seg_e = bufs
    kv8 = (
        torch.float8_e5m2
        if impl.kv_cache_dtype == "fp8_e5m2"
        else torch.float8_e4m3fn
    )
    if k_descale is None:
        k_descale = torch.ones(1, dtype=torch.float32, device=q.device)
    if v_descale is None:
        v_descale = torch.ones(1, dtype=torch.float32, device=q.device)
"""
    new2 = """    seg_o, seg_m, seg_e = bufs
    # llm-scaler v34: cached descale/ones + kv8 dtype + views — the KV
    # cache tensors, cache dtype and device are immutable per boot; the
    # per-call ones()/view() churn below was pure host overhead on the
    # every-step verify path. Same objects reused -> bit-identical.
    st = getattr(impl, "_v34_mq3d_state", None)
    if st is None or st[0] is not key_cache or st[1] is not value_cache:
        kv8 = (
            torch.float8_e5m2
            if impl.kv_cache_dtype == "fp8_e5m2"
            else torch.float8_e4m3fn
        )
        ones = torch.ones(1, dtype=torch.float32, device=q.device)
        st = (
            key_cache,
            value_cache,
            kv8,
            key_cache.view(kv8),
            value_cache.view(kv8),
            ones,
        )
        impl._v34_mq3d_state = st
    _, _, kv8, k8_view, v8_view, ones = st
    if k_descale is None:
        k_descale = ones
    if v_descale is None:
        v_descale = ones
"""
    assert fa.count(old2) == 1, f"anchor2={fa.count(old2)}"
    fa = fa.replace(old2, new2)

    old3 = """    unified_attention(
        q=q,
        k=key_cache.view(kv8),
        v=value_cache.view(kv8),
"""
    new3 = """    unified_attention(
        q=q,
        k=k8_view,
        v=v8_view,
"""
    assert fa.count(old3) == 1, f"anchor3={fa.count(old3)}"
    fa = fa.replace(old3, new3)

    open(FA, "w").write(fa)
    py_compile.compile(FA, doraise=True)
    print("V34SHIM OK: mq3d shim per-call allocs removed (bit-identical)")
else:
    print("V34SHIM: already applied")
