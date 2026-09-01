# llm-scaler v32 kernel patch: MQ verify stage1 -> per-row 2D accumulation
# Diagnosis: k1 spec verify @65k tforward d=72ms vs nospec 42ms while @2k
# parity (25ms) => context-scaled regression. MQ kernel shares KV loads but
# builds [Q_BLOCK, BLOCK_KV, BLOCK_D] temps at three sites (FP8 scores, MSE
# term1, value accumulation). At Q_BLOCK=2 that doubles register footprint
# vs the single-token kernel -> spills -> latency exposure on long scans.
# VLLM_TQ_MQ_STAGE1_WARPS=4 arm REGRESSED both ctx (77.4 @65k, 18.65 @2k),
# confirming occupancy, not ALU, is the binding constraint.
# Fix: static per-row loop (Q_LEN is constexpr) assembling [Q_BLOCK, ...]
# results via where-masks; per-row temps stay [BLOCK_KV, BLOCK_D].
# Numerics: identical reduction axes/order per row -> expect bit-stable
# (coh gate must confirm: distinct=1, Paris ~ -0.451).
import py_compile

P = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py"
src = open(P).read()

# --- site A: KEY_FP8 scores 3D temp -> per-row 2D ---
oldA = """            scores = (
                tl.sum(
                    tl.where(d_mask[None, None, :], q_rot[:, None, :] * k_float[None, :, :], 0.0),
                    axis=2,
                )
                * ATTN_SCALE
            )
"""
newA = """            # llm-scaler v32: per-row 2D score accumulation (register-lean).
            # The [Q_BLOCK, BLOCK_KV, BLOCK_D] broadcast temp spilled at
            # Q_BLOCK >= 2 and cost ~1.7x the single-token kernel on 65k
            # scans; the per-row static loop keeps temps at [BLOCK_KV,
            # BLOCK_D] (identical to the tuned single-token kernel).
            scores = tl.zeros([Q_BLOCK, BLOCK_KV], dtype=tl.float32)
            for _r in tl.static_range(Q_LEN):
                _q_r = tl.sum(
                    tl.where((q_offs == _r)[:, None], q_rot, 0.0), axis=0
                )
                _s_r = (
                    tl.sum(
                        tl.where(d_mask[None, :], _q_r[None, :] * k_float, 0.0),
                        axis=1,
                    )
                    * ATTN_SCALE
                )
                scores = tl.where((q_offs == _r)[:, None], _s_r[None, :], scores)
"""

# --- site B: MSE term1 3D temp -> per-row 2D ---
oldB = """            term1 = tl.sum(
                tl.where(d_mask[None, None, :], q_rot[:, None, :] * c_vals[None, :, :], 0.0),
                axis=2,
            )
"""
newB = """            # llm-scaler v32: per-row 2D term1 (see KEY_FP8 site note).
            term1 = tl.zeros([Q_BLOCK, BLOCK_KV], dtype=tl.float32)
            for _r in tl.static_range(Q_LEN):
                _q_r = tl.sum(
                    tl.where((q_offs == _r)[:, None], q_rot, 0.0), axis=0
                )
                _t_r = tl.sum(
                    tl.where(d_mask[None, :], _q_r[None, :] * c_vals, 0.0),
                    axis=1,
                )
                term1 = tl.where((q_offs == _r)[:, None], _t_r[None, :], term1)
"""

# --- site C: value accumulation 3D temp -> per-row 2D ---
oldC = """        acc = acc * re_scale[:, None] + tl.sum(p[:, :, None] * values[None, :, :], axis=1)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
"""
newC = """        # llm-scaler v32: per-row 2D value accumulation (register-lean;
        # same math: rescale then add each row's [BLOCK_KV, BLOCK_D]
        # contribution assembled via where-mask).
        acc = acc * re_scale[:, None]
        for _r in tl.static_range(Q_LEN):
            _p_r = tl.sum(tl.where((q_offs == _r)[:, None], p, 0.0), axis=0)
            _a_r = tl.sum(_p_r[:, None] * values, axis=0)
            acc = acc + tl.where((q_offs == _r)[:, None], _a_r[None, :], 0.0)
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
"""

for name, old in (("A", oldA), ("B", oldB), ("C", oldC)):
    assert src.count(old) == 1, f"anchor{name}={src.count(old)}"
src = src.replace(oldA, newA).replace(oldB, newB).replace(oldC, newC)
open(P, "w").write(src)
py_compile.compile(P, doraise=True)
print("V32MQ OK: MQ stage1 per-row 2D score+value accumulation")
