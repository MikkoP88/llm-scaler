#!/usr/bin/env python3
"""llm-scaler v58 P1 patcher: uniform-decode first-chunk-prefill guard.

ROOT CAUSE (P1, §16 of crashfix-v58 WEDGE_PLAN): _is_uniform_decode
classifies a batch as uniform decode purely by shape:
    max_num_scheduled_tokens == uniform_decode_query_len (= 1 + k)
    and num_tokens == max_num_scheduled_tokens * num_reqs.
A SOLO FIRST PREFILL of exactly k+1 tokens passes both tests. The step
is then built as a decode/verify batch: decode attention routine,
decode cudagraphs, and — for hybrid GDN/mamba models — resume-from-
state-slot handling. The request's mamba state slot has never been
written, so the GDN layers scan forward from stale pool memory:
nondeterministic garbage first tokens (zeros -> clean, dirty -> garbage
/ echo-of-other-text; capture dummy-runs and slot recycling dirty the
pool). Length-keyed: k=4 breaks 5-token prompts, k=3 breaks 4-token
(boot Q16 config flip). The classic P1 probe "The capital of France
is" is exactly 5 tokens on the prod k=4 lane.

FIX: uniform_decode is forced False while any scheduled row is a
first-chunk prefill (num_computed_tokens == 0). Real decode/verify rows
always have num_computed_tokens >= 1 (prompt length >= 1), so the uniform
decode optimization is untouched for every legitimate decode batch.
Chunked-prefill continuations (num_computed > 0) also keep it — their
state slot was written by the previous chunk, so the shape was already
correct there. Capture/dummy-run callers pass force_uniform_decode !=
None and are unaffected.
"""
import sys

PATH = ("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/"
        "gpu_model_runner.py")
MARKER = "llm-scaler v58 P1"

ANCHOR = """\
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
"""

GUARD = """\
        # llm-scaler v58 P1: a first-chunk prefill of exactly udql
        # tokens passes the uniform-decode SHAPE test but its GDN state
        # slot was never written; the decode/verify path resumes from
        # that slot and reads stale pool memory -> nondeterministic
        # garbage first tokens (k+1-length-keyed, P1 §16). Force the
        # extend path while any scheduled row is a first-chunk prefill.
        if uniform_decode and force_uniform_decode is None and num_reqs > 0:
            try:
                _v58_p1_prefill = bool(
                    (self.input_batch.num_computed_tokens_cpu[:num_reqs]
                     == 0).any()
                )
            except Exception:
                _v58_p1_prefill = False
            if _v58_p1_prefill:
                uniform_decode = False
                self._v58_p1_fires = getattr(self, "_v58_p1_fires", 0) + 1
                if self._v58_p1_fires == 1 or self._v58_p1_fires % 1000 == 0:
                    logger.info(
                        "llm-scaler v58 P1 uniform-decode prefill guard "
                        "fired (#%d): reqs=%d udql=%d",
                        self._v58_p1_fires, num_reqs,
                        self.uniform_decode_query_len,
                    )
"""


def main() -> int:
    with open(PATH, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(f"v58p1: ALREADY APPLIED ({PATH})")
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print(f"v58p1: FATAL anchor count={n} (need 1) in {PATH}")
        return 7

    src = src.replace(ANCHOR, ANCHOR + GUARD, 1)

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src)

    with open(PATH, "r", encoding="utf-8") as f:
        chk = f.read()
    ok = MARKER in chk and chk.count(ANCHOR) == 1
    print(f"v58p1: {'APPLIED+VERIFIED' if ok else 'VERIFY FAILED'} "
          f"guard-lines={chk.count(MARKER)}")
    return 0 if ok else 8


if __name__ == "__main__":
    sys.exit(main())
