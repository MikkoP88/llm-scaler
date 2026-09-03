FROM llm-scaler-vllm-adv:v26
# llm-scaler-vllm-adv:v27 - dedicated oneCCL communicator for the MTP
# drafter's collectives: the host-side fix for the >=32k spec x TP=2 serve
# wedge (KNOWN_ISSUES #11) that survives the v26 kernel fix.
#
# Mechanism (oneCCL debug trace, v26 + CCL_LOG_LEVEL=debug, 2026-08-30):
# per spec decode step TWO independent host-side sites enqueue oneCCL
# collectives onto the SAME ProcessGroupXCCL communicator:
#   (a) the target verify/decode step, and
#   (b) the eager MTP drafter (~6 colls/step: head all-reduces 5120 fp16 x4,
#       allgather 2560 fp16, get_top_tokens allgather 64 fp32).
# Under --async-scheduling the drafter of step N+1 overlaps the target of
# step N, so per-rank issue ORDER into the shared communicator can diverge
# between ranks (and replayed graphs bake capture-time comm state); oneCCL
# matching inverts and both ranks spin in never-retiring collective
# kernels (py-spy: victim rank stuck at the align-mode D2H in
# _update_states_after_model_execute, peer a step ahead in the eager MTP
# head; engines 100% at ~22% EU; no xe reset). Single-site configs are
# clean (nospec+graphs 0/10; enforce-eager spec 0/13); all-eager on one
# comm (VLLM_XPU_ALLOW_COMM_IN_GRAPH=0) only moves the wedge to >=133k.
#
# Fix: a SECOND GroupCoordinator ("drafter_tp", backend=xccl, same TP
# ranks -> a fresh oneCCL communicator via new_group) whose
# device_communicator/device_group are swapped onto the TP group for the
# duration of LLMBBaseProposer.propose()/dummy_run(). The drafter's op
# sequence then matches on a communicator only it uses, in identical order
# on both ranks, regardless of how target collectives interleave; the
# target keeps the stock TP group. Eager-only by construction (skipped
# while a graph is capturing). VLLM_XPU_DRAFTER_PG=0 restores stock.
#
# Contents: v26 (kernel fix + v25 host arms) + drafter_comm.py (new) +
# llm_base_proposer.py (decorator wiring, 3 hunks).

COPY drafter_comm.py /tmp/v27/drafter_comm.py
COPY llm_base_proposer.py /tmp/v27/llm_base_proposer.py

RUN SP=/opt/venv/lib/python3.12/site-packages && \
    SDIR=$SP/vllm/v1/spec_decode && \
    cp /tmp/v27/drafter_comm.py $SDIR/drafter_comm.py && \
    cp /tmp/v27/llm_base_proposer.py $SDIR/llm_base_proposer.py && \
    grep -q 'llm-scaler v27 (KNOWN_ISSUES #11)' $SDIR/drafter_comm.py && \
    grep -q 'VLLM_XPU_DRAFTER_PG' $SDIR/drafter_comm.py && \
    grep -q 'with_drafter_communicator' $SDIR/llm_base_proposer.py && \
    test "$(grep -c '@with_drafter_communicator' $SDIR/llm_base_proposer.py)" = "2" && \
    test "$(grep -c 'def propose' $SDIR/llm_base_proposer.py)" = "1" && \
    test "$(grep -c 'def dummy_run' $SDIR/llm_base_proposer.py)" = "1" && \
    python -m py_compile $SDIR/drafter_comm.py $SDIR/llm_base_proposer.py && \
    python -c "import ast, sys; \
        tree = ast.parse(open('$SDIR/llm_base_proposer.py').read()); \
        fns = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]; \
        assert 'propose' in fns and 'dummy_run' in fns and 'wrapper' in fns; \
        print('v27 overlay ast OK')" && \
    find $SDIR -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    rm -rf /tmp/v27 && \
    echo "v27 overlay done"
