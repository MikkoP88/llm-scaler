#!/usr/bin/env python3
"""llm-scaler v58 DIAGNOSTIC patcher (crashfix-v58, WEDGE_PLAN D1a'/D2').

Not a fix. Two zero-hot-path observers for the async-event-stall wedge:

1. xpu_communicator.py: fr-wrap `reduce_scatter` and `reduce_scatterv`
   (all_reduce and all_gatherv are already wrapped since v28dbg). At a
   freeze, the LAST fr line then names the exact collective call the
   host thread was inside (or between), completing the freeze-site
   picture for the TP-symmetric stall class.

2. mamba_utils.py: anomaly TRIPWIRE on every mamba copy-row producer
   (deferred postprocess [async lane], sync postprocess, preprocess
   running-state carry). Validates, per row, BEFORE collect_mamba_copy_meta:
   - dest in [0, len(block_ids))
   - src in [0, len(block_ids))
   - accepted cpu count in [1, 8]
   - bias in [0, 8]
   On violation: one logger.warning line with the full row state (goes
   to serve log -> lands in every wedge capture). Healthy rows: a few
   integer compares on the host, no GPU work, no stream change, no
   allocator change, numerics untouched.

Apply in-container BEFORE serve start:
    python3 patch_v58_diag.py [--sp SITE_PACKAGES]
Idempotent: refuses to double-apply (checks for V58 marker).
"""
import sys
import pathlib

MARK = "llm-scaler v58diag"


def patch_file(path: pathlib.Path, replaces: list[tuple[str, str]], tag: str):
    src = path.read_text()
    if MARK in src:
        print(f"[v58diag] {tag}: marker already present, skipping")
        return src, False
    for old, new in replaces:
        if src.count(old) != 1:
            raise SystemExit(
                f"[v58diag] {tag}: anchor not unique ({src.count(old)} hits) for:\n{old[:160]}"
            )
        src = src.replace(old, new)
    path.write_text(src)
    return src, True


def main():
    sp = pathlib.Path(
        sys.argv[sys.argv.index("--sp") + 1] if "--sp" in sys.argv else
        "/opt/venv/lib/python3.12/site-packages"
    )

    # ---- 1. xpu_communicator: wrap reduce_scatter / reduce_scatterv ----
    xc = sp / "vllm/distributed/device_communicators/xpu_communicator.py"
    rs_anchor = (
        "    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):\n"
        "        world_size = self.world_size\n"
    )
    rs_new = (
        "    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):\n"
        "        # llm-scaler v58diag: fr-wrap (freeze-site visibility)\n"
        "        if not torch.compiler.is_compiling():\n"
        "            _fr.log(f\"RS begin numel={input_.numel()}\")\n"
        "        try:\n"
        "            return self._reduce_scatter_v58(input_, dim)\n"
        "        finally:\n"
        "            if not torch.compiler.is_compiling():\n"
        "                _fr.log(\"RS end\")\n"
        "\n"
        "    def _reduce_scatter_v58(self, input_: torch.Tensor, dim: int = -1):\n"
        "        world_size = self.world_size\n"
    )
    rsv_anchor = (
        "    def reduce_scatterv(\n"
        "        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None\n"
        "    ):\n"
        "        world_size = self.world_size\n"
    )
    rsv_new = (
        "    def reduce_scatterv(\n"
        "        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None\n"
        "    ):\n"
        "        # llm-scaler v58diag: fr-wrap (freeze-site visibility)\n"
        "        if not torch.compiler.is_compiling():\n"
        "            _fr.log(f\"RSv begin numel={input_.numel()} sizes={sizes}\")\n"
        "        try:\n"
        "            return self._reduce_scatterv_v58(input_, dim, sizes)\n"
        "        finally:\n"
        "            if not torch.compiler.is_compiling():\n"
        "                _fr.log(\"RSv end\")\n"
        "\n"
        "    def _reduce_scatterv_v58(\n"
        "        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None\n"
        "    ):\n"
        "        world_size = self.world_size\n"
    )

    # ---- 2. mamba_utils: row tripwires ----
    mu = sp / "vllm/v1/worker/mamba_utils.py"

    # 2a. deferred postprocess (THE async-scheduling lane)
    def_anchor = (
        "        if aligned >= running:\n"
        "            bias = aligned - running\n"
        "            dest = aligned // block_size - 1\n"
    )
    def_new = (
        "        if aligned >= running:\n"
        "            bias = aligned - running\n"
        "            dest = aligned // block_size - 1\n"
        "            # llm-scaler v58diag: pre-launch row validation\n"
        "            try:\n"
        "                _nb = len(req_state.block_ids[mamba_group_ids[0]]) if mamba_group_ids else 0\n"
        "                _cpu = int(cpu[i])\n"
        "                if (dest < 0 or dest >= _nb or src < 0 or src >= _nb\n"
        "                        or _cpu < 1 or _cpu > 8 or bias < 0 or bias > 8):\n"
        "                    logger.warning(\n"
        "                        \"V58-TRIPWIRE deferred rid=%s cpu=%d running=%d \"\n"
        "                        \"newc=%d aligned=%d bias=%d dest=%d src=%d nblocks=%d \"\n"
        "                        \"computed=%d sched=%d draft=%d\",\n"
        "                        rid, _cpu, running, newc, aligned, bias, dest, src,\n"
        "                        _nb, computed, sched, draft)\n"
        "            except Exception:\n"
        "                pass\n"
    )

    # 2b. sync postprocess (dead on async lane, instrumented for completeness)
    post_anchor = (
        "            src_block_idx = mamba_state_idx[req_id]\n"
        "            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1\n"
    )
    post_new = (
        "            src_block_idx = mamba_state_idx[req_id]\n"
        "            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1\n"
        "            # llm-scaler v58diag: pre-launch row validation (sync path)\n"
        "            try:\n"
        "                _nb = len(req_state.block_ids[mamba_group_ids[0]]) if mamba_group_ids else 0\n"
        "                if (dest_block_idx < 0 or dest_block_idx >= _nb\n"
        "                        or src_block_idx < 0 or src_block_idx >= _nb\n"
        "                        or num_accepted < 1 or num_accepted > 8\n"
        "                        or accept_token_bias < 0 or accept_token_bias > 8):\n"
        "                    logger.warning(\n"
        "                        \"V58-TRIPWIRE sync rid=%s acc=%d bias=%d src=%d \"\n"
        "                        \"dest=%d nblocks=%d running=%d\",\n"
        "                        req_id, num_accepted, accept_token_bias,\n"
        "                        src_block_idx, dest_block_idx, _nb,\n"
        "                        num_tokens_running_state)\n"
        "            except Exception:\n"
        "                pass\n"
    )

    # 2c. preprocess running-state carry (src = prev_state_idx, bias = acc-1)
    pre_anchor = (
        "        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:\n"
    )
    pre_new = (
        "        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:\n"
        "            # llm-scaler v58diag: carry-row validation\n"
        "            try:\n"
        "                _g0 = mamba_group_ids[0] if mamba_group_ids else None\n"
        "                _nb = len(req_state.block_ids[_g0]) if _g0 is not None else 0\n"
        "                _acc = int(input_batch.num_accepted_tokens_cpu[i])\n"
        "                if (prev_state_idx < 0 or prev_state_idx >= _nb\n"
        "                        or curr_state_idx < 0 or curr_state_idx >= _nb\n"
        "                        or _acc < 1 or _acc > 8):\n"
        "                    logger.warning(\n"
        "                        \"V58-TRIPWIRE carry rid=%s prev=%d curr=%d acc=%d \"\n"
        "                        \"nblocks=%d computed=%d sched=%d\",\n"
        "                        req_id, prev_state_idx, curr_state_idx, _acc, _nb,\n"
        "                        req_state.num_computed_tokens, num_scheduled_tokens)\n"
        "            except Exception:\n"
        "                pass\n"
    )
    # ---- preflight: verify EVERY anchor in EVERY file before any write ----
    jobs = [
        (xc, [(rs_anchor, rs_new), (rsv_anchor, rsv_new)], "xpu_communicator"),
        (mu, [(def_anchor, def_new), (post_anchor, post_new),
              (pre_anchor, pre_new)], "mamba_utils"),
    ]
    for path, replaces, tag in jobs:
        src = path.read_text()
        if MARK in src:
            continue
        for old, _ in replaces:
            n = src.count(old)
            if n != 1:
                raise SystemExit(
                    f"[v58diag] PREFLIGHT FAIL {tag} {path.name}: "
                    f"anchor x{n}:\n{old[:200]}"
                )
    print("[v58diag] preflight: all anchors unique OK")

    for path, replaces, tag in jobs:
        patch_file(path, replaces, tag)
    print("[v58diag] xpu_communicator: RS/RSv wrapped OK")
    print("[v58diag] mamba_utils: 3 tripwires installed OK")
    print("V58_DIAG_OK")


if __name__ == "__main__":
    main()
