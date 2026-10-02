#!/usr/bin/env python3
"""perf-v134 P75: strip left-behind diagnostic logging to the
default fork surface (user directive 2026-10-02).

Removes — surgically, by exact-anchor match — every site of the two
always-on file-writing instrument generations baked into the image:

  1. v28dbg/v29 flight recorder (`fr` at site-packages root):
     /tmp/fr_<pid>.log writers in xpu_communicator (AR/AG),
     cudagraph_utils (replay), llm_base_proposer (propose).
  2. crashfix-v58 f15b stall-watchdog (vllm/_f15b.py): P1-P5 hooks
     in gpu_model_runner, P6 in xpu_communicator, P7a/P7b in
     qwen3_dflash, and the arstage census mark hook in _arstage.

KEEPS (asserted after strip): the v55 fences (_v55_wait_event),
Fix L staging (_arstage / _all_reduce_impl_direct), the v27 drafter
comm and v20 spec_timing hooks (env-gated OFF, no files), and every
functional fork feature. The patcher's own --revert (backup restore)
is FORBIDDEN here: the .f15bak files predate Fix L and later wedge
fixes (xpu_communicator diff proved ordering) — only hook text is
removed, never whole-file restore.

Python-only: no .so is touched; SHIP-MEASUREMENT LAW sha must hold.
Two-phase: verify every anchor in every file BEFORE writing any
file; any drift aborts with zero changes.
"""
import pathlib
import py_compile
import shutil
import subprocess
import sys

SP = pathlib.Path("/opt/venv/lib/python3.12/site-packages")
V = SP / "vllm"

# (path, [(hook_text, pristine_text), ...]) — hook -> pristine.
SITES = {
    V / "v1/worker/gpu_model_runner.py": [
        # P1: wait-state wrapper -> plain fence def (v55.3 signature)
        (
            'def _v55_wait_event(evt, what: str, step_tokens=None) -> None:\n'
            '    # llm-scaler f15b: wait-state visibility (dump names the wait)\n'
            '    from vllm import _f15b as _f15b_mod\n'
            '    _f15b_mod.enter_wait(what)\n'
            '    try:\n'
            '        return _v55_wait_event_inner(evt, what, step_tokens)\n'
            '    finally:\n'
            '        _f15b_mod.exit_wait()\n'
            '\n'
            '\n'
            'def _v55_wait_event_inner(evt, what: str, step_tokens=None) -> None:\n',
            'def _v55_wait_event(evt, what: str, step_tokens=None) -> None:\n',
        ),
        # P2: input-prep sync marks
        (
            '        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n'
            '        _f15b_mod.mark("input_prep_sync_enter")\n'
            '        self.prepare_inputs_event.synchronize()\n'
            '        _f15b_mod.mark("input_prep_sync_done")\n'
            '        try:\n',
            '        self.prepare_inputs_event.synchronize()\n'
            '        try:\n',
        ),
        # P3: execute_model entry mark (signature drifted: | None)
        (
            '        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n'
            '        _f15b_mod.mark("execute_model")\n'
            '        if self.execute_model_state is not None:\n'
            '            raise RuntimeError(\n'
            '                "State error: sample_tokens() must be called "\n'
            '                "after execute_model() returns None."\n'
            '            )\n',
            '        if self.execute_model_state is not None:\n'
            '            raise RuntimeError(\n'
            '                "State error: sample_tokens() must be called "\n'
            '                "after execute_model() returns None."\n'
            '            )\n',
        ),
        # P4: sample_tokens entry mark
        (
            '        from vllm import _f15b as _f15b_mod  # llm-scaler f15b\n'
            '        _f15b_mod.mark("sample_tokens")\n'
            '        if self.execute_model_state is None:\n',
            '        if self.execute_model_state is None:\n'
        ),
        # P5: propose region wrapper
        (
            '        def propose_draft_token_ids(sampled_token_ids):\n'
            '            # llm-scaler f15b: propose region + GPU completion\n'
            '            from vllm import _f15b as _f15b_mod\n'
            '            _f15b_mod.mark("propose_begin")\n'
            '            try:\n'
            '                return _propose_draft_inner_f15b(sampled_token_ids)\n'
            '            finally:\n'
            '                _f15b_mod.mark("propose_end")\n'
            '                _f15b_mod.evmark("propose_gpu_done")\n'
            '\n'
            '        def _propose_draft_inner_f15b(sampled_token_ids):\n'
            '            assert spec_decode_common_attn_metadata is not None\n',
            '        def propose_draft_token_ids(sampled_token_ids):\n'
            '            assert spec_decode_common_attn_metadata is not None\n',
        ),
        # P8: v55.3 crash-path ring dump (writes /root/v55_3_crash.log;
        # functional SIGKILL below stays)
        (
            '    # workers. Write the crash context (stderr is broken in-container),\n'
            '    # then SIGKILL the serving process group so the lane watchdog can\n'
            '    # relaunch within one cycle.\n'
            '    try:\n'
            '        import time as _t\n'
            '        with open("/root/v55_3_crash.log", "a") as f:\n'
            '            f.write(\n'
            '                f"[{_t.strftime(\'%Y-%m-%d %H:%M:%S\')}] v55.3 KILL "\n'
            '                f"pid={os.getpid()} wait={what!r} bound={bound:.0f}s "\n'
            '                f"rec_step_tokens={step_tokens!r}\\n")\n'
            '            try:\n'
            '                from vllm import _f15b as _f\n'
            '                with _f._lock:\n'
            '                    _items = list(_f._ring)\n'
            '                _now = _t.monotonic()\n'
            '                f.write("  f15b ring tail:\\n")\n'
            '                for _tt, _n, _d, _e in _items[-24:]:\n'
            '                    _st = ""\n'
            '                    if _e is not None:\n'
            '                        try:\n'
            '                            _st = "DONE" if _e.query() else "PENDING"\n'
            '                        except Exception:\n'
            '                            _st = "ERR"\n'
            '                    f.write(\n'
            '                        f"  [{_now - _tt:8.1f}s ago] {_n} {_d} {_st}\\n")\n'
            '            except Exception as _ex:\n'
            '                f.write(f"  f15b ring unavailable: {_ex}\\n")\n'
            '    except Exception:\n'
            '        pass\n'
            '    import signal\n',
            '    # workers. SIGKILL the serving process group so the lane watchdog\n'
            '    # can relaunch within one cycle.\n'
            '    import signal\n',
        ),
    ],
    V / "distributed/device_communicators/xpu_communicator.py": [
        # v28dbg import
        (
            'import fr as _fr  # llm-scaler v28dbg flight recorder (site-packages root)\n',
            '',
        ),
        # AR site: fr begin/end + f15b P6 -> plain impl call
        (
            '        # llm-scaler v28dbg flight recorder: entry/exit around the whole\n'
            '        # collective so a hang localizes to it. The is_compiling() guard is\n'
            '        # load-bearing: dynamo traces through here during profile_run and\n'
            '        # faults on the f-string tensor method (verified boot crash).\n'
            '        if not torch.compiler.is_compiling():\n'
            '            _fr.log(f"AR begin numel={input_.numel()} '
            "via_env={_env_enabled('VLLM_XPU_ALLREDUCE_VIA_ALLGATHER', False)}\")\n"
            '        try:\n'
            '            return self._all_reduce_impl(input_)\n'
            '        finally:\n'
            '            if not torch.compiler.is_compiling():\n'
            '                _fr.log("AR end")\n'
            '                # llm-scaler f15b: AR kernel-completion ring\n'
            '                try:\n'
            '                    from vllm import _f15b as _f15b_mod\n'
            '                    _f15b_mod.evmark("AR", f"numel={input_.numel()}")\n'
            '                except Exception:\n'
            '                    pass\n',
            '        return self._all_reduce_impl(input_)\n',
        ),
        # AG site: fr begin/end -> plain impl call
        (
            '        if not torch.compiler.is_compiling():\n'
            '            _fr.log(f"AG begin dim={dim} sizes={sizes}")  # llm-scaler v28dbg\n'
            '        try:\n'
            '            return self._all_gatherv_impl(input_, dim, sizes)\n'
            '        finally:\n'
            '            if not torch.compiler.is_compiling():\n'
            '                _fr.log("AG end")\n',
            '        return self._all_gatherv_impl(input_, dim, sizes)\n',
        ),
    ],
    V / "v1/worker/gpu/cudagraph_utils.py": [
        (
            'import fr as _fr  # llm-scaler v28dbg flight recorder (site-packages root)\n',
            '',
        ),
        (
            '        _fr.log(f"replay begin desc={desc}")  # llm-scaler v28dbg\n'
            '        self.graphs[desc].replay()\n'
            '        _fr.log("replay end")  # llm-scaler v28dbg\n',
            '        self.graphs[desc].replay()\n',
        ),
    ],
    V / "v1/spec_decode/llm_base_proposer.py": [
        (
            'import fr as _fr  # llm-scaler v29 flight recorder (site-packages root)\n',
            '',
        ),
        (
            '        _fr.log("propose begin")\n'
            '        try:\n'
            '            return self._propose_impl(*args, **kwargs)\n'
            '        finally:\n'
            '            _fr.log("propose end")\n',
            '        return self._propose_impl(*args, **kwargs)\n',
        ),
    ],
    V / "model_executor/models/qwen3_dflash.py": [
        # P7a: layer-loop marks -> plain loop
        (
            '        residual = None\n'
            '        _f15b_li = -1  # llm-scaler f15b (P7): drafter forensics\n'
            '        for layer in self.layers:\n'
            '            _f15b_li += 1\n'
            '            try:\n'
            '                from vllm import _f15b as _f15b_mod\n'
            '                _f15b_mod.mark("draft_layer_in", str(_f15b_li))\n'
            '            except Exception:\n'
            '                pass\n'
            '            hidden_states, residual = layer(\n'
            '                positions=positions,\n'
            '                hidden_states=hidden_states,\n'
            '                residual=residual,\n'
            '            )\n'
            '            try:\n'
            '                from vllm import _f15b as _f15b_mod\n'
            '                _f15b_mod.evmark("draft_layer_out", str(_f15b_li))\n'
            '            except Exception:\n'
            '                pass\n',
            '        residual = None\n'
            '        for layer in self.layers:\n'
            '            hidden_states, residual = layer(\n'
            '                positions=positions,\n'
            '                hidden_states=hidden_states,\n'
            '                residual=residual,\n'
            '            )\n',
        ),
        # P7b: attention call-site marks -> plain call
        (
            '        # llm-scaler f15b (P7): attention call-site forensics\n'
            '        try:\n'
            '            from vllm import _f15b as _f15b_mod\n'
            '            _f15b_mod.mark(\n'
            '                "draft_attn_in",\n'
            '                "%s nq=%d" % (getattr(self, "layer_name", "?"),\n'
            '                              int(q.shape[0])))\n'
            '        except Exception:\n'
            '            pass\n'
            '        attn_output = self.attn(q, k, v)\n'
            '        try:\n'
            '            from vllm import _f15b as _f15b_mod\n'
            '            _f15b_mod.evmark(\n'
            '                "draft_attn_out", getattr(self, "layer_name", "?"))\n'
            '        except Exception:\n'
            '            pass\n'
            '        output, _ = self.o_proj(attn_output)\n'
            '        return output\n',
            '        attn_output = self.attn(q, k, v)\n'
            '        output, _ = self.o_proj(attn_output)\n'
            '        return output\n',
        ),
    ],
    V / "_arstage.py": [
        # census mark hook (keep counters, drop the f15b feed)
        (
            '        _CALLS += 1\n'
            '        do_mark = (_CALLS % _MARK_EVERY) == 0\n'
            '        summ = _summary() if do_mark else ""\n'
            '    if do_mark:\n'
            '        try:\n'
            '            from vllm import _f15b as _f\n'
            '            _f.mark("arstage", summ)\n'
            '        except Exception:\n'
            '            pass\n',
            '        _CALLS += 1\n',
        ),
    ],
}

DELETE_FILES = [
    V / "_f15b.py",
    SP / "fr.py",
    V / "v1/worker/gpu_model_runner.py.f15bak",
    V / "model_executor/models/qwen3_dflash.py.f15bak",
    V / "distributed/device_communicators/xpu_communicator.py.f15bak",
]

KEEP_ASSERTS = [
    (V / "v1/worker/gpu_model_runner.py", "_v55_wait_event"),
    (V / "distributed/device_communicators/xpu_communicator.py", "_arstage"),
    (V / "distributed/device_communicators/xpu_communicator.py", "_all_reduce_impl_direct"),
    (V / "v1/spec_decode/llm_base_proposer.py", "spec_seg"),
    (V / "v1/spec_decode/llm_base_proposer.py", "with_drafter_communicator"),
]

FORBIDDEN = ("_f15b", "import fr as _fr", "_fr.log", "VLLM_XPU_FR")


def main() -> None:
    # Phase 1: verify every anchor (hook present, or already pristine).
    plans = []
    for path, pairs in SITES.items():
        src = path.read_text()
        for hook, pristine in pairs:
            n_hook, n_pris = src.count(hook), src.count(pristine)
            if n_hook == 1:
                plans.append((path, hook, pristine))
            elif n_hook == 0 and n_pris >= 1:
                print(f"[p75] {path.name}: already clean")
            else:
                sys.exit(
                    f"[p75] ANCHOR DRIFT {path}: hook x{n_hook}, pristine "
                    f"x{n_pris}:\n--- hook head ---\n{hook[:300]}")
    if not plans:
        print("[p75] nothing to strip (already clean) — asserts only")
    # Phase 2: apply all.
    by_file = {}
    for path, hook, pristine in plans:
        by_file.setdefault(path, []).append((hook, pristine))
    for path, reps in by_file.items():
        src = path.read_text()
        for hook, pristine in reps:
            src = src.replace(hook, pristine)
        path.write_text(src)
        print(f"[p75] {path.relative_to(SP)}: {len(reps)} site(s) stripped")

    # Deletions.
    for f in DELETE_FILES:
        if f.exists():
            f.unlink()
            print(f"[p75] deleted {f.relative_to(SP.parent)}")
    for pyc in list(SP.glob("__pycache__/fr*.pyc")) + list(
            V.glob("__pycache__/_f15b*.pyc")):
        pyc.unlink()
        print(f"[p75] deleted {pyc.name}")
    for junk in list(pathlib.Path("/tmp").glob("fr_*.log")) + list(
            pathlib.Path("/tmp").glob("f15b_dump_*.log")):
        junk.unlink()
        print(f"[p75] deleted /tmp/{junk.name}")

    # Asserts.
    for path in SITES:
        py_compile.compile(str(path), doraise=True)
    print("[p75] py_compile OK on all touched files")
    for path, marker in KEEP_ASSERTS:
        if marker not in path.read_text():
            sys.exit(f"[p75] KEEP-ASSERT FAILED: {marker} missing from {path.name}")
    print("[p75] keep-asserts OK (fences, arstage/Fix-L, v20/v27 hooks)")
    hits = subprocess.run(
        ["grep", "-rl", "-e", "_f15b", "-e", "import fr as _fr", "-e",
         "_fr.log", "-e", "VLLM_XPU_FR", str(V), "--include=*.py"],
        capture_output=True, text=True).stdout.split()
    if hits:
        sys.exit(f"[p75] FORBIDDEN residue: {hits}")
    print("[p75] forbidden-signature grep clean")
    print("P75_STRIP_OK")


if __name__ == "__main__":
    main()
