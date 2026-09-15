#!/usr/bin/env python3
"""llm-scaler F15b spec-step stall patcher (crashfix-v58, option a).

DIAGNOSTIC ONLY — not a fix. Answers ONE question at the next
262144+mtp4 wedge: which launch region is resident / first-to-stall,
and did both ranks' AR sequences diverge?

Mechanism:
  - _f15b.py module (installed as vllm/_f15b.py): in-memory ring of
    region marks + per-region torch Event completion tracking; daemon
    watchdog fires a DUMP at VLLM_F15B_STALL_S (default 45 s) of zero
    progress — BEFORE the 300 s sample_tokens RPC timeout and the 600 s
    v55 fences destroy the live state. Dump = ring with per-event
    PENDING/DONE, current v55 wait label, py-spy (self + EngineCore
    siblings), xpu-smi dump, dmesg tail. Max 5 dumps, then quiet.
  - gpu_model_runner.py:
      P1 _v55_wait_event wrapped (every freeze site reports its label)
      P2 input_prep sync enter/done marks (Boot H TP0 raw-sync site)
      P3 execute_model entry mark
      P4 sample_tokens entry mark
      P5 inner propose_draft_token_ids wrapped + propose_gpu_done event
  - xpu_communicator.py:
      P6 AR evmark after every all_reduce (kernel-completion ring)
  - qwen3_dflash.py (v4, P7 — drafter forensics for Q8):
      per-layer mark/evmark in DFlashQwen3Model.forward layer loop +
      call-site marks around self.attn(q, k, v) in
      DFlashQwen3Attention.forward. At a mid-propose freeze the dump
      tail then names the exact layer and whether the attention kernel
      completed (last DONE event = hang boundary). §11 leading
      hypothesis: spec-path kernel hangs upstream of the ARs (LR job
      cleanup); Q5 funnel falsification left this as the open thread.

Overhead: ~1 Event record per AR (decode step has many) — acceptable
for a diagnosis boot, NOT for prod. Env kill-switch VLLM_F15B=0.

Apply in-container BEFORE serve start:
    python3 patch_f15b.py            # apply
    python3 patch_f15b.py --revert   # restore .f15bak + remove module
Idempotent; preflight verifies all anchors; backups as <file>.f15bak.
"""
import pathlib
import shutil
import sys

MARK = "llm-scaler f15b"
SP = pathlib.Path(
    sys.argv[sys.argv.index("--sp") + 1] if "--sp" in sys.argv else
    "/opt/venv/lib/python3.12/site-packages"
)

F15B_MODULE = '''
# llm-scaler f15b: spec-step stall watchdog + completion-event ring. (v2: transition-based progress).
# Diagnostic-only (crashfix-v58 F15b). See patch_f15b.py header.
import os
import subprocess
import threading
import time
from collections import deque

DISABLED = os.environ.get("VLLM_F15B", "1") == "0"
STALL = float(os.environ.get("VLLM_F15B_STALL_S", "45"))
OUTDIR = os.environ.get("VLLM_F15B_DIR", "/root")
POLL = 2.0
MAX_DUMPS = 5

_ring = deque(maxlen=4096)  # v3: 256 only spanned ~3 s of prefill-class ARs (boot Q2 miss — the reset victim fell outside the window); 4096 spans minutes
_lock = threading.Lock()
_started = False
_last_progress = time.monotonic()
_wait_what = None
_dump_count = 0


def _start_watchdog():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    t = threading.Thread(target=_watchdog, daemon=True, name="f15b-watchdog")
    t.start()


def _note_progress():
    global _last_progress
    _last_progress = time.monotonic()


def mark(name, detail=""):
    if DISABLED:
        return
    _ring.append((time.monotonic(), name, detail, None))
    _note_progress()
    _start_watchdog()


def evmark(name, detail=""):
    if DISABLED:
        return
    evt = None
    try:
        import torch
        if not torch.compiler.is_compiling():
            evt = torch.Event()
            evt.record()
    except Exception:
        evt = None
    _ring.append((time.monotonic(), name, detail, evt))
    _note_progress()
    _start_watchdog()


def enter_wait(what):
    global _wait_what
    if DISABLED:
        return
    _wait_what = what
    mark("v55wait", what)


def exit_wait():
    global _wait_what
    if DISABLED:
        return
    _wait_what = None


def _engine_pids():
    me = os.getpid()
    out = [me]
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cb = open(f"/proc/{d}/cmdline", "rb").read()
            except Exception:
                continue
            if (b"EngineCore" in cb or b"spawn_main" in cb) and int(d) != me:
                out.append(int(d))
    except Exception:
        pass
    return out[:6]


def _dump(reason):
    global _dump_count
    _dump_count += 1
    path = f"{OUTDIR}/f15b_dump_{os.getpid()}.log"
    now = time.monotonic()
    lines = [
        f"=== F15B DUMP #{_dump_count} {time.strftime('%H:%M:%S')} "
        f"reason={reason} wait={_wait_what} pid={os.getpid()} ==="
    ]
    with _lock:
        items = list(_ring)
    for t, name, detail, evt in items[-96:]:
        st = ""
        if evt is not None:
            try:
                st = "DONE" if evt.query() else "PENDING"
            except Exception as ex:
                st = f"ERR({ex})"
        lines.append(f"  [{now - t:8.1f}s ago] {name} {detail} {st}")
    txt = "\\n".join(lines) + "\\n"
    try:
        open(path, "a").write(txt)
    except Exception:
        pass
    # llm-scaler f15b: arstage decision census into the dump (Q4 —
    # staged-vs-direct split per (numel,dtype,contig,stream) at freeze)
    try:
        from vllm import _arstage as _as
        open(path, "a").write("ARSTAGE_CENSUS " + _as.stats_lines() + "\\n")
    except Exception:
        pass
    for pid in _engine_pids():
        try:
            r = subprocess.run(
                ["/opt/venv/bin/py-spy", "dump", "--pid", str(pid),
                 "--nonblocking"],
                capture_output=True, text=True, timeout=15)
            open(f"{OUTDIR}/f15b_pyspy_{pid}.log", "a").write(
                f"=== dump #{_dump_count} ===\\n{r.stdout}\\n{r.stderr}\\n")
        except Exception as ex:
            open(path, "a").write(f"pyspy {pid} fail: {ex}\\n")
    try:
        r = subprocess.run(["xpu-smi", "dump"], capture_output=True,
                           text=True, timeout=15)
        open(f"{OUTDIR}/f15b_xpusmi.log", "a").write(
            f"=== dump #{_dump_count} ===\\n{r.stdout}\\n")
    except Exception as ex:
        open(path, "a").write(f"xpu-smi fail: {ex}\\n")
    try:
        r = subprocess.run(["dmesg"], capture_output=True, text=True,
                           timeout=15)
        open(f"{OUTDIR}/f15b_dmesg.log", "a").write(
            f"=== dump #{_dump_count} ===\\n" +
            "\\n".join(r.stdout.splitlines()[-30:]) + "\\n")
    except Exception as ex:
        open(path, "a").write(f"dmesg fail: {ex}\\n")


_done_prev: set = set()


def _watchdog():
    # v2: progress requires a DONE TRANSITION — an event newly completed
    # since the last tick — or a fresh mark. The v1 static check ("any
    # event in the window queries DONE") never fired during a stall with
    # completed events parked in the window (boot Q miss). Event objects
    # stay alive via the ring and _done_prev within a tick, so identity
    # across the set diff is stable.
    global _dump_count
    while True:
        time.sleep(POLL)
        if DISABLED:
            continue
        with _lock:
            items = list(_ring)
        progressed = (time.monotonic() - _last_progress) < POLL
        done_now = set()
        for _t, _n, _d, e in items[-24:]:
            if e is None:
                continue
            try:
                if e.query():
                    done_now.add(e)
            except Exception:
                pass
        if done_now - _done_prev:
            progressed = True
        _done_prev.clear()
        _done_prev.update(done_now)
        if progressed:
            _note_progress()
            continue
        stall = time.monotonic() - _last_progress
        if stall > STALL:
            if _dump_count < MAX_DUMPS:
                _dump(f"stall {stall:.0f}s")
            _note_progress()
'''


def patch_file(path: pathlib.Path, replaces: list, tag: str):
    src = path.read_text()
    if MARK in src:
        print(f"[f15b] {tag}: marker present, skipping")
        return
    for old, _ in replaces:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"[f15b] {tag}: anchor x{n} (need 1):\n{old[:200]}")
    bak = path.with_suffix(path.suffix + ".f15bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    for old, new in replaces:
        src = src.replace(old, new)
    path.write_text(src)
    print(f"[f15b] {tag}: patched OK (backup {bak.name})")


def revert_file(path: pathlib.Path, tag: str):
    bak = path.with_suffix(path.suffix + ".f15bak")
    if bak.exists():
        shutil.copy2(bak, path)
        bak.unlink()
        print(f"[f15b] {tag}: reverted")
    else:
        print(f"[f15b] {tag}: no backup, untouched")


def main():
    gmr = SP / "vllm/v1/worker/gpu_model_runner.py"
    xc = SP / "vllm/distributed/device_communicators/xpu_communicator.py"
    df = SP / "vllm/model_executor/models/qwen3_dflash.py"
    mod = SP / "vllm/_f15b.py"

    if "--revert" in sys.argv:
        revert_file(gmr, "gpu_model_runner")
        revert_file(xc, "xpu_communicator")
        revert_file(df, "dflash")
        if mod.exists():
            mod.unlink()
            print("[f15b] module removed")
        print("F15B_REVERT_OK")
        return

    gmr_patches = [
        # P1: v55 wait visibility — every freeze site names itself
        (
            "def _v55_wait_event(evt, what: str) -> None:\n",
            "def _v55_wait_event(evt, what: str) -> None:\n"
            "    # llm-scaler f15b: wait-state visibility (dump names the wait)\n"
            "    from vllm import _f15b as _f15b_mod\n"
            "    _f15b_mod.enter_wait(what)\n"
            "    try:\n"
            "        return _v55_wait_event_inner(evt, what)\n"
            "    finally:\n"
            "        _f15b_mod.exit_wait()\n"
            "\n"
            "\n"
            "def _v55_wait_event_inner(evt, what: str) -> None:\n",
        ),
        # P2: input-prep raw sync (Boot H TP0 freeze site — unfenced)
        (
            "        self.prepare_inputs_event.synchronize()\n"
            "        try:\n",
            "        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n"
            "        _f15b_mod.mark(\"input_prep_sync_enter\")\n"
            "        self.prepare_inputs_event.synchronize()\n"
            "        _f15b_mod.mark(\"input_prep_sync_done\")\n"
            "        try:\n",
        ),
        # P3: execute_model entry
        (
            "        if self.execute_model_state is not None:\n"
            "            raise RuntimeError(\n",
            "        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n"
            "        _f15b_mod.mark(\"execute_model\")\n"
            "        if self.execute_model_state is not None:\n"
            "            raise RuntimeError(\n",
        ),
        # P4: sample_tokens entry
        (
            "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:\n"
            "        if self.execute_model_state is None:\n",
            "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:\n"
            "        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n"
            "        _f15b_mod.mark(\"sample_tokens\")\n"
            "        if self.execute_model_state is None:\n",
        ),
        # P5: propose region + GPU-completion event
        (
            "        def propose_draft_token_ids(sampled_token_ids):\n"
            "            assert spec_decode_common_attn_metadata is not None\n",
            "        def propose_draft_token_ids(sampled_token_ids):\n"
            "            # llm-scaler f15b: propose region + GPU completion\n"
            "            from vllm import _f15b as _f15b_mod\n"
            "            _f15b_mod.mark(\"propose_begin\")\n"
            "            try:\n"
            "                return _propose_draft_inner_f15b(sampled_token_ids)\n"
            "            finally:\n"
            "                _f15b_mod.mark(\"propose_end\")\n"
            "                _f15b_mod.evmark(\"propose_gpu_done\")\n"
            "\n"
            "        def _propose_draft_inner_f15b(sampled_token_ids):\n"
            "            assert spec_decode_common_attn_metadata is not None\n",
        ),
    ]

    xc_patches = [
        # P6: AR kernel-completion tracking
        (
            "        try:\n"
            "            return self._all_reduce_impl(input_)\n"
            "        finally:\n"
            "            if not torch.compiler.is_compiling():\n"
            "                _fr.log(\"AR end\")\n",
            "        try:\n"
            "            return self._all_reduce_impl(input_)\n"
            "        finally:\n"
            "            if not torch.compiler.is_compiling():\n"
            "                _fr.log(\"AR end\")\n"
            "                # llm-scaler f15b: AR kernel-completion ring\n"
            "                try:\n"
            "                    from vllm import _f15b as _f15b_mod\n"
            "                    _f15b_mod.evmark(\"AR\", f\"numel={input_.numel()}\")\n"
            "                except Exception:\n"
            "                    pass\n",
        ),
    ]

    df_patches = [
        # P7a: drafter per-layer marks — the layer loop of the DFlash
        # backbone (DSparkDraftModel -> ... -> DFlashQwen3Model.forward).
        # Anchor = the layer loop body (unique).
        (
            "        residual = None\n"
            "        for layer in self.layers:\n"
            "            hidden_states, residual = layer(\n"
            "                positions=positions,\n"
            "                hidden_states=hidden_states,\n"
            "                residual=residual,\n"
            "            )\n",
            "        residual = None\n"
            "        _f15b_li = -1  # llm-scaler f15b (P7): drafter forensics\n"
            "        for layer in self.layers:\n"
            "            _f15b_li += 1\n"
            "            try:\n"
            "                from vllm import _f15b as _f15b_mod\n"
            "                _f15b_mod.mark(\"draft_layer_in\", str(_f15b_li))\n"
            "            except Exception:\n"
            "                pass\n"
            "            hidden_states, residual = layer(\n"
            "                positions=positions,\n"
            "                hidden_states=hidden_states,\n"
            "                residual=residual,\n"
            "            )\n"
            "            try:\n"
            "                from vllm import _f15b as _f15b_mod\n"
            "                _f15b_mod.evmark(\"draft_layer_out\", str(_f15b_li))\n"
            "            except Exception:\n"
            "                pass\n",
        ),
        # P7b: drafter attention call-site — names the kernel class at
        # the hang boundary (self.attn = paged-decode KV read, the §11
        # e5m2-matched draft-KV suspect).
        (
            "        attn_output = self.attn(q, k, v)\n"
            "        output, _ = self.o_proj(attn_output)\n"
            "        return output\n",
            "        # llm-scaler f15b (P7): attention call-site forensics\n"
            "        try:\n"
            "            from vllm import _f15b as _f15b_mod\n"
            "            _f15b_mod.mark(\n"
            "                \"draft_attn_in\",\n"
            "                \"%s nq=%d\" % (getattr(self, \"layer_name\", \"?\"),\n"
            "                              int(q.shape[0])))\n"
            "        except Exception:\n"
            "            pass\n"
            "        attn_output = self.attn(q, k, v)\n"
            "        try:\n"
            "            from vllm import _f15b as _f15b_mod\n"
            "            _f15b_mod.evmark(\n"
            "                \"draft_attn_out\", getattr(self, \"layer_name\", \"?\"))\n"
            "        except Exception:\n"
            "            pass\n"
            "        output, _ = self.o_proj(attn_output)\n"
            "        return output\n",
        ),
    ]

    # preflight
    for path, replaces, tag in ((gmr, gmr_patches, "gmr"),
                                (xc, xc_patches, "xpu_comm"),
                                (df, df_patches, "dflash")):
        src = path.read_text()
        if MARK in src:
            print(f"[f15b] {tag}: already patched")
            continue
        for old, _ in replaces:
            n = src.count(old)
            if n != 1:
                raise SystemExit(
                    f"[f15b] PREFLIGHT FAIL {tag}: anchor x{n}:\n{old[:200]}")
    print("[f15b] preflight: all anchors unique OK")

    mod.write_text(F15B_MODULE)
    print("[f15b] module vllm/_f15b.py installed")
    patch_file(gmr, gmr_patches, "gmr")
    patch_file(xc, xc_patches, "xpu_comm")
    patch_file(df, df_patches, "dflash")
    print("F15B_APPLY_OK")


if __name__ == "__main__":
    main()
