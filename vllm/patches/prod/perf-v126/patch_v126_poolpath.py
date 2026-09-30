#!/usr/bin/env python3
"""patch_v126_poolpath.py — v126 P29C: route fp8-SSM decode through the
fp8-native ESIMD kernels (end-to-end fp8 pipeline, no bridge copies on
decode), matching the pool's fp8 type all the way down (e4m3->e4m3,
e5m2->e5m2 — the wrapper dispatch keys on the pool's scalar_type).

Base state: gdn_linear_attn.py gates the fused ESIMD decode path on BOTH
pools being fp16 (_gdn_conv_state_fp16_ok) and on num_spec_decodes == 1,
so every fp8-SSM lane routes ALL batches (prefill AND decode) through the
dual bridge (gather fp8->fp16, run, scatter fp16->fp8 per step) — the
speed overhead this round removes.

Changes (marker "v126 P29C", backup .v126pcbak, idempotent, py_compile):
  E1 flag rename _gdn_conv_state_fp16_ok -> _gdn_conv_state_ok (conv pool
     must be fp16; ssm pool may be fp16/fp8_e4m3/fp8_e5m2 — the v125 P22A
     StateT dispatch + v126 P29A fp8 register chain read it natively).
  E2 docstring + spec gate: with an fp8 SSM pool the spec kernel runs ANY
     number of spec requests (grid dim0 = num_spec_decodes; per-request
     token loop over distinct per-column slots; validated op-level by
     p29b_check.py L1 solo-vs-batch). fp16 keeps the certified
     num_spec_decodes == 1 gate bit-for-bit. Mixed batches (spec decodes +
     plain decodes together) with nsd > 1 stay on the upstream bridge —
     the engine never schedules them, but degrade-not-corrupt if it ever
     did.
  E3 state gate predicate: conv fp16 + ssm in {fp16, fp8_e4m3, fp8_e5m2}.

Non-spec decode (<=128 rows) rides esimd_gdn_conv_fused_seq natively for
fp8 pools too (same relaxation). Prefill/mixed stay on the P28 dual
bridge (SYCL kernels need fp16 pools there — one bridge per forward,
inherent).
"""
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "mamba/gdn_linear_attn.py"
)
MARKER = "v126 P29C"

E1_OLD = """        self._gdn_conv_fp16_ready = False
        # Lazily confirmed once kv_cache exists: the fused conv kernel needs an
        # fp16 conv_state + ssm_state (see _gdn_conv_decode). False until proven.
        self._gdn_conv_state_fp16_ok = False"""

E1_NEW = """        self._gdn_conv_fp16_ready = False
        # Lazily confirmed once kv_cache exists: the fused conv kernels need
        # an fp16 conv_state; the ssm_state may be fp16 OR fp8 (v125 P22A
        # StateT dispatch + v126 P29C/P29A fp8 register chain — see
        # _gdn_conv_decode). False until proven.
        self._gdn_conv_state_ok = False"""

E2_OLD = '''        """Fused conv1d + GDN delta-rule decode core (esimd_gdn_conv_fused).

        Returns True iff it handled the whole batch. Ordinary decode uses the
        existing one-token-per-work-group kernel; single-request speculative
        batches use the rollback-aware sequential variant. Multi-request,
        mixed, and prefill batches stay on the upstream gdn_attention op.
        """'''

E2_NEW = '''        """Fused conv1d + GDN delta-rule decode core (esimd_gdn_conv_fused).

        Returns True iff it handled the whole batch. Ordinary decode uses the
        existing one-token-per-work-group kernel; speculative batches use the
        rollback-aware sequential variant (v126 P29C: ANY number of spec
        requests when the ssm pool is fp8 — grid dim0 = num_spec_decodes,
        distinct per-column slots, fp8 register chain, pool dtype matched
        end-to-end; fp16 keeps the certified solo-spec-only gate
        bit-for-bit). Multi-request, mixed, and prefill batches stay on the
        upstream gdn_attention op.
        """'''

E2B_OLD = """        if (
            is_spec_batch
            and (
                self.gqa_interleaved_layout
                or esimd_gdn_conv_fused_seq_spec is None
                or os.environ.get("DISABLE_ESIMD_GDN_SPEC", "0") == "1"
                or attn_metadata.num_spec_decodes != 1
                or self.num_k_heads // self.tp_size != 8
                or self.num_v_heads // self.tp_size != 24
                or self.head_k_dim != 128
                or self.head_v_dim != 128
            )
        ):
            return False"""

E2B_NEW = """        # v126 P29C: with an fp8 SSM pool the spec kernel runs ANY number
        # of spec requests (grid dim0 = num_spec_decodes; each work-group
        # owns one request's token loop over distinct per-column slots).
        # The fp16 lane keeps the certified num_spec_decodes == 1 gate
        # bit-for-bit. Mixed spec+plain decode batches with nsd > 1 stay
        # on the upstream bridge (degrade, never corrupt).
        ssm_pool_fp8 = (
            self.kv_cache is not None
            and self.kv_cache[1].dtype
            in (torch.float8_e4m3fn, torch.float8_e5m2)
        )
        if (
            is_spec_batch
            and (
                self.gqa_interleaved_layout
                or esimd_gdn_conv_fused_seq_spec is None
                or os.environ.get("DISABLE_ESIMD_GDN_SPEC", "0") == "1"
                or (
                    attn_metadata.num_spec_decodes != 1
                    and not ssm_pool_fp8
                )
                or (
                    ssm_pool_fp8
                    and attn_metadata.num_spec_decodes > 1
                    and attn_metadata.num_decodes > 0
                )
                or self.num_k_heads // self.tp_size != 8
                or self.num_v_heads // self.tp_size != 24
                or self.head_k_dim != 128
                or self.head_v_dim != 128
            )
        ):
            return False"""

E3_OLD = """        # The fused conv kernels read BOTH conv_state and ssm_state as fp16
        # (lsc_load_state_64 over const fp16*). Models that force an fp32 SSM
        # cache (Qwen3.6 sets mamba_ssm_dtype=float32 for accuracy; Qwen3-Next
        # leaves it auto -> fp16) would have their fp32 state bytes
        # reinterpreted as fp16 -> NaN/!!!! state cascade. Gate on the actual
        # cache dtype and fall back to the upstream (unfused) decode otherwise.
        # Decided once: state dtype is fixed for the cache's lifetime.
        if not self._gdn_conv_state_fp16_ok:
            if (
                self.kv_cache[0].dtype == torch.float16
                and self.kv_cache[1].dtype == torch.float16
            ):
                self._gdn_conv_state_fp16_ok = True
            else:
                return False"""

E3_NEW = """        # The fused conv kernels read conv_state as fp16 always; the ssm_state
        # is read natively per dtype since v125 P22A (fp16 via
        # lsc_load_state_64 over const fp16*; fp8_e4m3/fp8_e5m2 via the
        # StateT-templated loads, with the v126 P29A fp8 register chain in
        # the spec kernel). fp32 SSM caches (Qwen3.6) and any non-fp16 conv
        # pool stay on the upstream (unfused) decode. Gate on the actual
        # cache dtype and fall back otherwise. Decided once: state dtype is
        # fixed for the cache's lifetime.
        if not self._gdn_conv_state_ok:
            if (
                self.kv_cache[0].dtype == torch.float16
                and self.kv_cache[1].dtype
                in (torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
            ):
                self._gdn_conv_state_ok = True
            else:
                return False"""

E2C_OLD = """            esimd_gdn_conv_fused_seq_spec(
                projected_states_qkvz,"""

E2C_NEW = """            # v126 P29C: one-shot serve-log proof that an fp8 SSM pool is
            # riding the NATIVE ESIMD spec kernel (no dual-bridge detour on
            # decode; the bridge stays for prefill/mixed only).
            if ssm_pool_fp8 and not getattr(self, "_gdn_p29c_logged", False):
                self._gdn_p29c_logged = True
                print("v126 P29C: fp8 SSM decode NATIVE ESIMD (ssm=%s)"
                      % ssm_state.dtype, flush=True)
            esimd_gdn_conv_fused_seq_spec(
                projected_states_qkvz,"""

EDITS = [
    ("E1 flag rename", E1_OLD, E1_NEW),
    ("E2a docstring", E2_OLD, E2_NEW),
    ("E2b spec gate", E2B_OLD, E2B_NEW),
    ("E2c native marker", E2C_OLD, E2C_NEW),
    ("E3 state gate", E3_OLD, E3_NEW),
]


def main() -> int:
    text = TARGET.read_text()
    if MARKER in text:
        n = text.count(MARKER)
        print("[poolpath] marker already present (%d) — skip" % n)
        return 0
    bak = TARGET.with_suffix(".py.v126pcbak")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
        print("[poolpath] backup %s" % bak)
    for label, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            print("[poolpath] ABORT %s: anchor x%d (need 1)" % (label, n))
            return 1
        text = text.replace(old, new, 1)
    TARGET.write_text(text)
    py_compile.compile(str(TARGET), doraise=True)
    print("[poolpath] %d edits applied; markers=%d; py_compile OK"
          % (len(EDITS), text.count(MARKER)))
    print("[poolpath] P29C_POOLPATH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
