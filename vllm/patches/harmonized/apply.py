#!/usr/bin/env python3
"""harmonized/apply.py — apply the full era-1/2 accumulated tree delta
(pristine vllm-0.21.1.dev0+gad7125a43.d20260826.xpu -> prod v38 tree) to a
site-packages tree, md5-gated on BOTH sides. Fail-loud, idempotent.

  python3 apply.py [--sp /opt/venv/lib/python3.12/site-packages]

Per modified file:
  md5 == patched  -> already applied, skip
  md5 == pristine -> `patch -p1` applies the unified diff, then md5 must
                     equal the patched value or we abort
  anything else   -> ABORT (unknown tree state; rebase the patch set)
Per added file:
  absent          -> copied in from added/<sp-rel-path>
  md5 == patched  -> skip
  anything else   -> ABORT

Exit codes: 0 = applied/verified, 3 = patch binary missing,
4 = unknown tree state, 5 = patch failed / post-md5 mismatch.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SP = "/opt/venv/lib/python3.12/site-packages"

# (patch file, site-packages-relative target)
PATCHES = [
    ("01-config-tq-spec-enable.patch",
     "vllm/config/vllm.py"),
    ("02-comm-via-allgather-stablebuf.patch",
     "vllm/distributed/device_communicators/xpu_communicator.py"),
    ("03-attn-e5m2-guard.patch",
     "vllm/model_executor/layers/attention/attention.py"),
    ("04-mtp-eager-head-compile-guard.patch",
     "vllm/model_executor/models/qwen3_5_mtp.py"),
    ("05-xpu-unsafe-spec-graphs-gate.patch",
     "vllm/platforms/xpu.py"),
    ("06-flash-attn-f8-scale-fixes.patch",
     "vllm/v1/attention/backends/flash_attn.py"),
    ("07-tq-mq-verify-backend.patch",
     "vllm/v1/attention/backends/turboquant_attn.py"),
    ("08-tq-mq-verify-kernel.patch",
     "vllm/v1/attention/ops/triton_turboquant_decode.py"),
    ("09-unified-attn-scalefold.patch",
     "vllm/v1/attention/ops/triton_unified_attention.py"),
    ("10-sched-ignore-eos-guard.patch",
     "vllm/v1/core/sched/utils.py"),
    ("11-dflash-parity-stall-guard.patch",
     "vllm/v1/spec_decode/dflash.py"),
    ("12-proposer-drafter-comm.patch",
     "vllm/v1/spec_decode/llm_base_proposer.py"),
    ("13-cudagraph-replay-fr-logs.patch",
     "vllm/v1/worker/gpu/cudagraph_utils.py"),
    ("14-runner-async-counts-barrier-off.patch",
     "vllm/v1/worker/gpu_model_runner.py"),
    ("15-worker-capture-stabilizer.patch",
     "vllm/v1/worker/gpu_worker.py"),
    ("16-mamba-deferred-postprocess.patch",
     "vllm/v1/worker/mamba_utils.py"),
]

# site-packages-relative path -> md5 (pristine, patched); pristine "-" = added
MANIFEST = {
    "vllm/config/vllm.py":
        ("9284db37c44066d334b2ff2216276eec", "79555dc5dd5e2e3c7a0e096eec62be07"),
    "vllm/distributed/device_communicators/xpu_communicator.py":
        ("8116bb3c34ec0d114d8bae3e971e001e", "c2a842bbe8eefcafe6162bf6d0f05fb8"),
    "vllm/model_executor/layers/attention/attention.py":
        ("98ad1d8ece8c6cae41ce817264cc04cd", "d3321f582d82d4585ac7dfa829ceda80"),
    "vllm/model_executor/models/qwen3_5_mtp.py":
        ("57d9ea1b07b5eef0487accf5fe407a0d", "2d03d2c3dabcc8899b76e1780aa5b84d"),
    "vllm/platforms/xpu.py":
        ("6177cd5f04294f3188ca54d12b711e59", "123c5dee9ce0920b58f2699fcb2bb40d"),
    "vllm/v1/attention/backends/flash_attn.py":
        ("f14b441440974f66904c0560b264f18d", "1000effd71318a29b7e3279114084e4d"),
    "vllm/v1/attention/backends/turboquant_attn.py":
        ("b3ba14385f6a97eecb45ae3422d6c1cd", "9ac3aa1027db95f208b39179ae37cbe1"),
    "vllm/v1/attention/ops/triton_turboquant_decode.py":
        ("ea40287ec647014a1d3bffab700f54f7", "e0408c1536a030e42590dd2fbcf57cbd"),
    "vllm/v1/attention/ops/triton_unified_attention.py":
        ("ffd3b71923f7ff0b0a446b487419821b", "acc28df0f3887f0b2c98e4d348e05a4a"),
    "vllm/v1/core/sched/utils.py":
        ("3fb6e1d37e9173fda630b1f5be0a56e3", "c396f0e17167bd942d4e2428e37dfa17"),
    "vllm/v1/spec_decode/dflash.py":
        ("734f0cc46e97abff6440f4807e2921ed", "0ed3a93384a9f487be0c234c8ad83e14"),
    "vllm/v1/spec_decode/llm_base_proposer.py":
        ("b26d023c26e251614494e208a3cecab3", "0371f785cfadb51a34adcbbe3b0e8e19"),
    "vllm/v1/worker/gpu/cudagraph_utils.py":
        ("010d0f260b63d956faa7edaa6b251f53", "ee6efac4f114ec8ba355c3e47899f50e"),
    "vllm/v1/worker/gpu_model_runner.py":
        ("cb931a3b3288c0de7838ddc621d7f079", "a6a4937a7233f6ae921fd8301f36eed8"),
    "vllm/v1/worker/gpu_worker.py":
        ("6ad6fe978714116d710533398841d0c9", "d84f343061fd22204e7906c185fd91be"),
    "vllm/v1/worker/mamba_utils.py":
        ("c13b37a9b063f58385f9d62d0bfeb393", "a8b598acdbc3a6931383563bb9ef6030"),
    "fr.py":
        ("-", "6ffe7cc368f1ef431c2a74f81162b680"),
    "vllm/v1/spec_decode/drafter_comm.py":
        ("-", "73fbda6192b27a4c6ed1d533804505de"),
    "vllm/v1/spec_decode/spec_timing.py":
        ("-", "09e932bd5b08735bf20ca2674923315d"),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", default=DEFAULT_SP,
                    help="site-packages root to patch (default %(default)s)")
    args = ap.parse_args()
    sp = os.path.abspath(args.sp)

    if shutil.which("patch") is None:
        print("FATAL: 'patch' binary not found on PATH", file=sys.stderr)
        sys.exit(3)

    applied, skipped = 0, 0
    for patch_name, rel in PATCHES:
        tgt = os.path.join(sp, rel)
        pristine, patched = MANIFEST[rel]
        if not os.path.exists(tgt):
            print(f"FATAL: target missing: {tgt}", file=sys.stderr)
            sys.exit(4)
        cur = md5(tgt)
        if cur == patched:
            skipped += 1
            continue
        if cur != pristine:
            print(f"FATAL: unknown tree state for {rel}: md5 {cur} "
                  f"matches neither pristine ({pristine}) nor patched "
                  f"({patched}); rebase the patch set", file=sys.stderr)
            sys.exit(4)
        r = subprocess.run(
            ["patch", "-p1", "--forward", "--batch", "-i",
             os.path.join(HERE, "patches", patch_name), "-d", sp],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FATAL: patch {patch_name} failed:\n{r.stdout}\n{r.stderr}",
                  file=sys.stderr)
            sys.exit(5)
        if md5(tgt) != patched:
            print(f"FATAL: {rel} md5 mismatch after patch "
                  f"(got {md5(tgt)}, want {patched})", file=sys.stderr)
            sys.exit(5)
        applied += 1

    for rel in ("fr.py", "vllm/v1/spec_decode/drafter_comm.py",
                "vllm/v1/spec_decode/spec_timing.py"):
        tgt = os.path.join(sp, rel)
        want = MANIFEST[rel][1]
        if os.path.exists(tgt):
            if md5(tgt) == want:
                skipped += 1
                continue
            print(f"FATAL: {rel} exists with unknown md5 {md5(tgt)}",
                  file=sys.stderr)
            sys.exit(4)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        shutil.copyfile(os.path.join(HERE, "added", rel), tgt)
        if md5(tgt) != want:
            print(f"FATAL: added file {rel} md5 mismatch", file=sys.stderr)
            sys.exit(5)
        applied += 1

    print(f"harmonized apply OK: {applied} applied, {skipped} already-ok "
          f"(19 files total)")


if __name__ == "__main__":
    main()
