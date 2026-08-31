# Ticket A (draft) — vLLM: XPU piecewise graph capture + MTP k=4 corrupts
greedy outputs at bs=1 (deterministic multi-path logits on identical
temp-0 requests)

Status: DRAFT — not posted. Target: new issue on vllm-project/vllm,
referencing #26963 (XPU speculative decoding tracker). The oneCCL livelock
(#11) is a SEPARATE defect tracked in upstream_oneccl_ticket.md.

---

## Title

[Bug][XPU] FULL_DECODE_ONLY cudagraph pieces + MTP num_speculative_tokens=4
produce non-deterministic, wrong logits at temperature=0 (k<=3 clean, eager
clean, bs>=2 eager fallback clean)

## Environment

- vLLM 0.21.1.dev0+gad7125a43 (XPU build) + vllm-xpu-kernels
  0.1.8.3.dev0+g3cab97a
- torch 2.11.0+xpu, triton-xpu 3.7.0, oneAPI 2025.3.2 / oneCCL 2021.15
- 2x Intel Arc Pro B70 (Battlemage BMG G31, 8086:e223), NEO 37833,
  kernel 6.17.0-1010-intel
- Model: Qwen3.8-27B fp8 (GDN hybrid attention + MTP drafter heads),
  TP=2, `--kv-cache-dtype turboquant_4bit_nc`, `--block-size 512`,
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`,
  default XPU capture sizes (<=128), `--speculative-config
  '{"method":"mtp","num_speculative_tokens":4}'`

## Minimal repro (3 curls after one boot)

Boot the server (flags above), then send the SAME request 8x at temperature 0:

```bash
for i in $(seq 1 8); do
  curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"MODEL","prompt":"The capital of France is","max_tokens":6,"temperature":0,"logprobs":5}' \
    | python3 -c 'import sys,json; ch=json.load(sys.stdin)["choices"][0];
       top=sorted(((float(v),k) for k,v in (ch["logprobs"]["top_logprobs"][0] or {}).items()), reverse=True)[:2];
       print(repr(ch["text"]), ["%s:%.3f"%(k,v) for v,k in top])'
done
```

With k=4: outputs cycle between 2-3 variants (distinct >= 2); the top-1
logprob for the same position swings up to ~1.5 nats (-0.099 / -0.294 /
-1.598 observed) and the top token itself can flip (' Paris' -> 'The').
After ~40 requests on the same boot, outputs degenerate
(' France France France...').

Restart the SAME server with `num_speculative_tokens` = 3 (or 1, or 2):
distinct = 1, top-1 logprob -0.451 bit-stable, all runs. That contrast is
the whole bug report.

## What we established (each point from a dedicated boot/arm)

1. **The boundary is exactly k=4.** k=1/2/3 + graphs: distinct=1, top-1
   -0.451 bit-stable, identical to the compile-no-capture reference
   (-0.453). k=4: corrupt. Not a gradient — a cliff.
2. **Capture is necessary.** `VLLM_XPU_ENABLE_XPU_GRAPH=0` + k=4 =
   deterministic (that is our reference number).
3. **Batch-size fallback is clean.** 2 concurrent requests on a
   capture-sizes-[1,2,4,8] boot (rows exceed the list -> eager fallback)
   both return the reference continuation. Corruption lives in the captured
   bs=1 pieces.
4. **Padding is NOT the trigger.** k=1 bs=3 (6 rows -> padded 8-piece) and
   k=2 bs=1 (3 rows -> padded 4-piece) are clean. The boundary tracks the
   draft-token count, not padded shape.
5. **Byte-reproducible.** Same responses in the same order across boots,
   across two image generations, and across capture lists [1..128] vs
   [1,2,4,8] — a deterministic function of request ordinal. The scheduler
   alternates among 2-3 execution paths and at least one computes wrong
   logits.
6. **temperature=0 IS honored**: a high-margin 40-token prompt returns 6/6
   identical text even on a corrupt boot — the logits differ
   prompt-sensitively; this is not a sampler fault.
7. **Not a recent regression**: reproduces byte-for-byte on builds dating
   back ~4 image generations (>= v24-era, vLLM 0.21.1.dev line). Earlier
   "bare prompts are stable" observations were small-sample artifacts of
   the per-request-ordinal path cycling (a same-path run of 4 is
   unremarkable).
8. **Captured collectives are not required**: we do NOT run
   `VLLM_XPU_ALLOW_COMM_IN_GRAPH` (that mode returns deterministic garbage
   outright and is disabled); the corruption occurs with the default
   comm-outside-graph placement.

## Suspect area

Multi-draft-position spec machinery under piecewise graph replay — the same
machinery as the GDN spec-kernel ragged-batch OOB we previously found and
fixed locally (wheel rebuild): GDN speculative state buffers and/or the
verify mask for 4 draft positions, as replayed inside FULL_DECODE_ONLY
pieces. k<=3 clean / k=4 corrupt / bs>=2-eager clean / padding-refuted
together point at state that is sized or advanced per draft position and
not correctly reset or re-bound between piece replays.

## Impact

Any XPU user running MTP k=4 with graphs (the shipped default combination
for our config) gets silently wrong greedy outputs. Throughput/stability
testing does not catch it — every historical arm of ours measured wedges
and token rates only, and all of those k=4 arms almost certainly served
corrupt text.

## Related

- #26963 (speculative decoding support on XPU — tracking issue)
- Separate defect, same ingredients, filed to oneCCL: piecewise-replay x
  spec collective livelock at >=32k ctx (k-agnostic) — see our #212/#215
  comments.
