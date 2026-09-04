#!/usr/bin/env python3
"""dflash2_winfix_edit.py — DFlash2 read-side sliding-window fix (v41).

Companion to dflash2_edit.py (v40): reads the STOCK fork files from
site-packages of a helper container, applies the v41 window hooks with
strict one-match assertions, writes patched copies under
/w/patched/<relpath>, byte-compiles them and runs an import smoke test.

Defect (v40 §6): TurboQuantAttentionImpl.__init__ accepts sliding_window
but never stores it — the window is silently dropped. A sliding-window
draft (DFlash2: all 5 layers sliding_window=2048) therefore ran a causal
ramp over the FULL prefix in triton_turboquant_mq_decode_attention
(kv_offs < q0 + row over block-table columns from position 0) instead of
the trained trailing window: acceptance 72.3% @2.3k -> 9.3% @74k.

Fix (exact, graph-safe, inert when sliding_window is None):
  * impl stores self.sliding_window;
  * MQ kernel gains HAS_WINDOW constexpr + per-row lower bound
    (kv_offs >= win_start + row) in local coordinates;
  * the causal continuation branch rebases the block table via a device-op
    gather (local position 0 = window's block-aligned start), shifts q0 by
    the same base, and passes the (possibly negative) local window start.
    Short sequences resolve to off=0 / negative win == exact causal
    semantics; all data-dependent math is device ops that re-execute at
    XPU-graph replay.
  * the synthetic-decode fallback gets a block-aligned rebase (no head
    cut: <= block_size-1 stale tokens; debug-env path only).
"""
import shutil
import subprocess
import sys
from pathlib import Path

SP = Path("/opt/venv/lib/python3.12/site-packages")
W = Path("/w")

EDITED = [
    "vllm/v1/attention/backends/turboquant_attn.py",
    "vllm/v1/attention/ops/triton_turboquant_decode.py",
]


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    assert n == 1, f"{what}: expected exactly 1 occurrence, found {n}"
    print(f"  ok: {what}")
    return text.replace(old, new)


def main() -> int:
    # ---- vllm/v1/attention/backends/turboquant_attn.py ----
    p = SP / "vllm/v1/attention/backends/turboquant_attn.py"
    t = p.read_text()

    # B1: store the window (stock drops it on the floor).
    t = edit(
        t,
        "        self.num_kv_groups = num_heads // self.num_kv_heads\n"
        "        self.kv_cache_dtype = kv_cache_dtype\n",
        "        self.num_kv_groups = num_heads // self.num_kv_heads\n"
        "        self.kv_cache_dtype = kv_cache_dtype\n"
        "        # llm-scaler v41: keep the read-side sliding window. Stock\n"
        "        # dropped it here, so a sliding-window draft (DFlash2,\n"
        "        # sliding_window=2048) ramped causal attention over the\n"
        "        # FULL prefix instead of the trained trailing window\n"
        "        # (acceptance 72% @2k -> 9% @74k). None for full-attention\n"
        "        # layers / MTP / DFlash-v1 -> all hooks below stay inert.\n"
        "        self.sliding_window = sliding_window\n",
        "turboquant_attn.py: store sliding_window",
    )

    # B2: windowed causal MQ call.
    t = edit(
        t,
        "                        out = triton_turboquant_mq_decode_attention(\n"
        "                            query=q_seq,\n"
        "                            kv_cache=kv_cache,\n"
        "                            block_table=attn_metadata.block_table"
        "[i : i + 1],\n"
        "                            q0_seq_lens=(\n"
        "                                (\n"
        "                                    attn_metadata.seq_lens"
        "[i : i + 1]\n"
        "                                    - (q_len - 1)\n"
        "                                )\n"
        "                                if causal\n"
        "                                else attn_metadata.seq_lens"
        "[i : i + 1]\n"
        "                            ),\n"
        "                            Pi=Pi,\n",
        "                        mq_bt = attn_metadata.block_table[i : i + 1]\n"
        "                        if causal:\n"
        "                            mq_q0 = (\n"
        "                                attn_metadata.seq_lens[i : i + 1]\n"
        "                                - (q_len - 1)\n"
        "                            )\n"
        "                        else:\n"
        "                            mq_q0 = attn_metadata.seq_lens"
        "[i : i + 1]\n"
        "                        mq_win = None\n"
        "                        if causal and self.sliding_window is not None:\n"
        "                            # llm-scaler v41: enforce the read-side\n"
        "                            # sliding window for sliding drafts\n"
        "                            # (DFlash2). Rebase the block table so\n"
        "                            # local position 0 = the window's\n"
        "                            # block-aligned start, shift q0 by the\n"
        "                            # same base, pass the (possibly\n"
        "                            # negative) local window start for the\n"
        "                            # kernel's per-row lower bound. Device\n"
        "                            # ops only — replay-safe under XPU\n"
        "                            # graphs; short sequences resolve to\n"
        "                            # off=0 / negative win == exact causal\n"
        "                            # semantics.\n"
        "                            if not getattr(\n"
        "                                self, \"_sw_mq_logged\", False\n"
        "                            ):\n"
        "                                self._sw_mq_logged = True\n"
        "                                logger.info(\n"
        "                                    \"TurboQuant sliding-window MQ\"\n"
        "                                    \" draft attention active (v41):\"\n"
        "                                    \" w=%d, q_len=%d.\",\n"
        "                                    self.sliding_window,\n"
        "                                    q_len,\n"
        "                                )\n"
        "                            _bs = kv_cache.shape[1]\n"
        "                            _ws = mq_q0 - self.sliding_window\n"
        "                            _off = torch.clamp_min(_ws, 0) // _bs\n"
        "                            _idx = (\n"
        "                                torch.arange(\n"
        "                                    mq_bt.shape[1],\n"
        "                                    device=mq_bt.device,\n"
        "                                    dtype=torch.int64,\n"
        "                                )\n"
        "                                + _off.to(torch.int64)\n"
        "                            ).clamp_max(mq_bt.shape[1] - 1)"
        ".unsqueeze(0)\n"
        "                            mq_bt = torch.gather(mq_bt, 1, _idx)\n"
        "                            mq_q0 = mq_q0 - _off * _bs\n"
        "                            mq_win = (_ws - _off * _bs).to("
        "torch.int32)\n"
        "                        out = triton_turboquant_mq_decode_attention(\n"
        "                            query=q_seq,\n"
        "                            kv_cache=kv_cache,\n"
        "                            block_table=mq_bt,\n"
        "                            q0_seq_lens=mq_q0,\n"
        "                            Pi=Pi,\n",
        "turboquant_attn.py: windowed MQ call",
    )
    t = edit(
        t,
        "                            non_causal=(not causal),\n"
        "                        )\n"
        "                    else:\n"
        "                        # Synthetic decode: one single-token decode"
        " call\n",
        "                            non_causal=(not causal),\n"
        "                            win_start=mq_win,\n"
        "                        )\n"
        "                    else:\n"
        "                        # Synthetic decode: one single-token decode"
        " call\n",
        "turboquant_attn.py: MQ win_start arg",
    )

    # B3: synthetic-decode fallback (block-aligned rebase, no head cut).
    t = edit(
        t,
        "                        synth_bt = attn_metadata.block_table"
        "[i : i + 1].expand(\n"
        "                            q_len, -1\n"
        "                        )\n"
        "                        if causal:\n"
        "                            synth_dyn = (\n"
        "                                attn_metadata.seq_lens[i : i + 1]\n"
        "                                + torch.arange(\n"
        "                                    q_len,\n"
        "                                    device=attn_metadata.seq_lens"
        ".device,\n"
        "                                    dtype=attn_metadata.seq_lens"
        ".dtype,\n"
        "                                )\n"
        "                                - (q_len - 1)\n"
        "                            )\n"
        "                        else:\n"
        "                            synth_dyn = attn_metadata.seq_lens"
        "[i : i + 1].repeat(\n"
        "                                q_len\n"
        "                            )\n",
        "                        synth_bt_row = attn_metadata.block_table"
        "[i : i + 1]\n"
        "                        if causal:\n"
        "                            synth_dyn = (\n"
        "                                attn_metadata.seq_lens[i : i + 1]\n"
        "                                + torch.arange(\n"
        "                                    q_len,\n"
        "                                    device=attn_metadata.seq_lens"
        ".device,\n"
        "                                    dtype=attn_metadata.seq_lens"
        ".dtype,\n"
        "                                )\n"
        "                                - (q_len - 1)\n"
        "                            )\n"
        "                        else:\n"
        "                            synth_dyn = attn_metadata.seq_lens"
        "[i : i + 1].repeat(\n"
        "                                q_len\n"
        "                            )\n"
        "                        if causal and self.sliding_window is not None:\n"
        "                            # llm-scaler v41: block-aligned window\n"
        "                            # rebase for the synthetic-decode\n"
        "                            # fallback (no per-row head cut in the\n"
        "                            # single-query kernel: up to\n"
        "                            # block_size-1 stale tokens beyond the\n"
        "                            # window start may leak in; debug-env\n"
        "                            # path only — the default MQ path above\n"
        "                            # is exact).\n"
        "                            if not getattr(\n"
        "                                self, \"_sw_synth_logged\", False\n"
        "                            ):\n"
        "                                self._sw_synth_logged = True\n"
        "                                logger.info(\n"
        "                                    \"TurboQuant sliding-window\"\n"
        "                                    \" synthetic decode (v41):\"\n"
        "                                    \" block-aligned rebase,\"\n"
        "                                    \" <= block_size-1 stale head\"\n"
        "                                    \" tokens (q_len=%d, w=%d).\",\n"
        "                                    q_len,\n"
        "                                    self.sliding_window,\n"
        "                                )\n"
        "                            _bs = kv_cache.shape[1]\n"
        "                            _ws = synth_dyn[:1] - self.sliding_window\n"
        "                            _off = torch.clamp_min(_ws, 0) // _bs\n"
        "                            _idx = (\n"
        "                                torch.arange(\n"
        "                                    synth_bt_row.shape[1],\n"
        "                                    device=synth_bt_row.device,\n"
        "                                    dtype=torch.int64,\n"
        "                                )\n"
        "                                + _off.to(torch.int64)\n"
        "                            ).clamp_max(\n"
        "                                synth_bt_row.shape[1] - 1\n"
        "                            ).unsqueeze(0)\n"
        "                            synth_bt_row = torch.gather(\n"
        "                                synth_bt_row, 1, _idx\n"
        "                            )\n"
        "                            synth_dyn = synth_dyn - _off * _bs\n"
        "                        synth_bt = synth_bt_row.expand(q_len, -1)\n",
        "turboquant_attn.py: windowed synthetic-decode fallback",
    )
    (W / "patched/vllm/v1/attention/backends/turboquant_attn.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (W / "patched/vllm/v1/attention/backends/turboquant_attn.py").write_text(t)

    # ---- vllm/v1/attention/ops/triton_turboquant_decode.py ----
    p = SP / "vllm/v1/attention/ops/triton_turboquant_decode.py"
    t = p.read_text()

    # A1: kernel runtime pointer.
    t = edit(
        t,
        "    Q0_lens_ptr,  # [B] int32: tokens visible to query row 0"
        " (cached_len + 1)\n"
        "    Centroids_ptr,  # [n_centroids] float32\n",
        "    Q0_lens_ptr,  # [B] int32: tokens visible to query row 0"
        " (cached_len + 1)\n"
        "    Win_start_ptr,  # [B] int32: local window start for row 0"
        " (v41; may be < 0)\n"
        "    Centroids_ptr,  # [n_centroids] float32\n",
        "triton_turboquant_decode.py: MQ kernel Win_start_ptr",
    )

    # A2: load the window start under constexpr.
    t = edit(
        t,
        "    q0 = tl.load(Q0_lens_ptr + bid)\n"
        "    if NON_CAUSAL:\n",
        "    q0 = tl.load(Q0_lens_ptr + bid)\n"
        "    if HAS_WINDOW:\n"
        "        # llm-scaler v41: sliding-window lower bound in LOCAL\n"
        "        # coordinates (caller rebased the block table); row r's\n"
        "        # start = win_start + r. Negative values (short sequences)\n"
        "        # mask nothing.\n"
        "        win_start = tl.load(Win_start_ptr + bid)\n"
        "    if NON_CAUSAL:\n",
        "triton_turboquant_decode.py: MQ kernel win_start load",
    )

    # A3: causal mask gains the per-row lower bound.
    t = edit(
        t,
        "        else:\n"
        "            att_mask = (kv_offs[None, :] < row_limit[:, None])"
        " & kv_mask[None, :]\n",
        "        else:\n"
        "            if HAS_WINDOW:\n"
        "                att_mask = (\n"
        "                    (kv_offs[None, :] < row_limit[:, None])\n"
        "                    & (\n"
        "                        kv_offs[None, :]\n"
        "                        >= (win_start + q_offs)[:, None]\n"
        "                    )\n"
        "                    & kv_mask[None, :]\n"
        "                )\n"
        "            else:\n"
        "                att_mask = (kv_offs[None, :] < row_limit[:, None])"
        " & kv_mask[None, :]\n",
        "triton_turboquant_decode.py: MQ kernel windowed mask",
    )

    # A4: kernel constexpr.
    t = edit(
        t,
        "    NORM_CORRECTION: tl.constexpr = 0,\n"
        "    FP8_E4B15: tl.constexpr = 0,\n"
        "    NON_CAUSAL: tl.constexpr = 0,\n"
        "):\n"
        "    bid = tl.program_id(0)  # batch index\n"
        "    hid = tl.program_id(1)  # q_head index\n"
        "    sid = tl.program_id(2)  # kv_split index\n"
        "\n"
        "    kv_head = hid // KV_GROUP_SIZE\n"
        "\n"
        "    q_offs = tl.arange(0, Q_BLOCK)\n",
        "    NORM_CORRECTION: tl.constexpr = 0,\n"
        "    FP8_E4B15: tl.constexpr = 0,\n"
        "    NON_CAUSAL: tl.constexpr = 0,\n"
        "    HAS_WINDOW: tl.constexpr = 0,\n"
        "):\n"
        "    bid = tl.program_id(0)  # batch index\n"
        "    hid = tl.program_id(1)  # q_head index\n"
        "    sid = tl.program_id(2)  # kv_split index\n"
        "\n"
        "    kv_head = hid // KV_GROUP_SIZE\n"
        "\n"
        "    q_offs = tl.arange(0, Q_BLOCK)\n",
        "triton_turboquant_decode.py: MQ kernel HAS_WINDOW constexpr",
    )

    # A5: wrapper param.
    t = edit(
        t,
        "    max_num_kv_splits: int = 32,\n"
        "    non_causal: bool = False,\n"
        ") -> torch.Tensor:\n",
        "    max_num_kv_splits: int = 32,\n"
        "    non_causal: bool = False,\n"
        "    win_start: torch.Tensor | None = None,  # [B] int32 local"
        " window start (v41)\n"
        ") -> torch.Tensor:\n",
        "triton_turboquant_decode.py: MQ wrapper win_start param",
    )

    # A6: launcher runtime arg (dummy ptr when unused — never loaded).
    t = edit(
        t,
        "    _tq_mq_decode_stage1[grid](\n"
        "        q_rot,\n"
        "        kv_cache,\n"
        "        block_table,\n"
        "        q0_seq_lens,\n"
        "        centroids,\n"
        "        mid_o,\n",
        "    _tq_mq_decode_stage1[grid](\n"
        "        q_rot,\n"
        "        kv_cache,\n"
        "        block_table,\n"
        "        q0_seq_lens,\n"
        "        win_start if win_start is not None else q0_seq_lens,\n"
        "        centroids,\n"
        "        mid_o,\n",
        "triton_turboquant_decode.py: MQ launcher Win_start_ptr arg",
    )

    # A7: launcher constexpr.
    t = edit(
        t,
        "        NON_CAUSAL=1 if non_causal else 0,\n"
        "        num_warps=_TQ_MQ_STAGE1_WARPS,\n",
        "        NON_CAUSAL=1 if non_causal else 0,\n"
        "        HAS_WINDOW=1 if win_start is not None else 0,\n"
        "        num_warps=_TQ_MQ_STAGE1_WARPS,\n",
        "triton_turboquant_decode.py: MQ launcher HAS_WINDOW",
    )
    (W / "patched/vllm/v1/attention/ops/triton_turboquant_decode.py").parent.mkdir(
        parents=True, exist_ok=True
    )
    (W / "patched/vllm/v1/attention/ops/triton_turboquant_decode.py").write_text(t)

    # ---- byte-compile ----
    import py_compile

    for rel in EDITED:
        py_compile.compile(str(W / "patched" / rel), doraise=True)
    print("  ok: py_compile both modules")

    # ---- import smoke test (disposable helper container) ----
    for rel in EDITED:
        shutil.copy2(W / "patched" / rel, SP / rel)
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import inspect; "
            "from vllm.v1.attention.ops.triton_turboquant_decode "
            "import triton_turboquant_mq_decode_attention as mq; "
            "assert 'win_start' in inspect.signature(mq).parameters; "
            "import vllm.v1.attention.backends.turboquant_attn as ta; "
            "src = inspect.getsource(ta); "
            "assert 'self.sliding_window = sliding_window' in src; "
            "assert 'win_start=mq_win' in src; "
            "ksrc = open(ta.__file__.replace('backends/turboquant_attn', "
            "'ops/triton_turboquant_decode')).read(); "
            "assert ksrc.count(\"HAS_WINDOW\") == 4; "
            "print('IMPORT_OK winfix v41')",
        ],
        capture_output=True,
        text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        print("WINFIX_EDIT_FAIL import smoke test")
        return 4
    print("DFLASH2_WINFIX_EDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
