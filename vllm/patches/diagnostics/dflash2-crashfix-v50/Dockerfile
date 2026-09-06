# llm-scaler v50 (crash-fix era) — bake v1.2.2 -> v1.2.3
#
# v1.2.3 adds ONE overlay to the baked v1.2.2 set:
#   async_sched_v50.py    -> vllm/v1/core/sched/async_scheduler.py
#                            (F8 runaway-request guard: force-finish a
#                             request whose num_output_placeholders exceeds
#                             VLLM_V50_PLACEHOLDER_LIMIT (default 64) or
#                             whose real count exceeds max_tokens+spec+8 —
#                             closes the crash-3 death chain: usage-less
#                             silent stream end -> no abort -> placeholder
#                             inflation -> worker gather OOB / fatal SYCL
#                             assert. Validated live (F8a, 2026-09-06):
#                             class recurred, guard fired 72>64, request
#                             reaped FINISHED_ABORTED, engine survived,
#                             0 asserts; inert on healthy traffic.)
#
# v1.2.2 (base image, baked 2026-09-05 from v1.2.1):
#   core_v50.py          -> vllm/v1/engine/core.py          (hasher resync + F6 fp8-spec refusal)
#   block_pool_v50.py    -> vllm/v1/core/block_pool.py      (starvation clamp)
#   dflash2_v50.py       -> vllm/v1/spec_decode/dflash2.py  (TP1 default OFF)
#   drafter_comm_v50.py  -> vllm/v1/spec_decode/drafter_comm.py (TP1 default OFF)
#   dt_warmup_v50.py     -> /root/dt_warmup.py              (warmup phases 4-5)
#   Validated live before baking: R4 (exact R2 crash cell, TP1=1) ALL-CLEAN,
#   resync fired 2048->1024, clamp 0 events; R5 (certified lane TP1=0) clean.
FROM llm-scaler-exp:v1.2.2

COPY async_sched_v50.py /tmp/v503/

RUN set -eux; \
    P=/opt/venv/bin/python; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    md5sum /tmp/v503/async_sched_v50.py | grep -q a096c5a846409e83f83445fe97620af3; \
    cp /tmp/v503/async_sched_v50.py $SP/vllm/v1/core/sched/async_scheduler.py; \
    grep -q 'llm-scaler v50 F8: runaway-request guard armed' $SP/vllm/v1/core/sched/async_scheduler.py; \
    grep -q 'VLLM_V50_PLACEHOLDER_LIMIT' $SP/vllm/v1/core/sched/async_scheduler.py; \
    grep -q 'FINISHED_ABORTED' $SP/vllm/v1/core/sched/async_scheduler.py; \
    $P -m py_compile $SP/vllm/v1/core/sched/async_scheduler.py; \
    md5sum $SP/vllm/v1/core/sched/async_scheduler.py | grep -q a096c5a846409e83f83445fe97620af3; \
    rm -rf /tmp/v503; \
    find $SP/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true; \
    touch /root/.v503_baked; \
    echo "v50 bake OK (v1.2.3 = v1.2.2 + F8 runaway-request guard)"
