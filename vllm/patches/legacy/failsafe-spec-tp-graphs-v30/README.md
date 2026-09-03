# qwen38-dflash v30 - fail-safe for graphs x speculative decode x TP

Image `llm-scaler-vllm-adv:v30` fixes KNOWN_ISSUES #11 operationally by
rejecting its unsafe execution path. When speculative decoding and TP>1 are
both configured on XPU, platform configuration now:

1. sets the effective XPU-graph environment state to disabled, including for
   kernels that consult the environment directly;
2. sets `TORCH_COMPILE_DISABLE=1` and compilation mode to `NONE` before worker
   creation, covering secondary configuration paths;
3. changes `cudagraph_mode` to `NONE` and sets `model_config.enforce_eager`
   while retaining speculative decoding;
4. emits a warning that names the defect and the selected fallback.

No-spec TP graph serving and TP=1 speculative graph serving are unchanged.
`VLLM_XPU_ALLOW_UNSAFE_SPEC_TP_GRAPH=1` bypasses the gate solely for defect
reproduction.

## Why the safety gate is the fix

The live v30 investigation tested two narrower mechanisms before settling on
the gate:

- persistent communicator-owned all-reduce staging still reproduced the
  canonical livelock on the first 65k request;
- staging plus a device-retirement fence after every eager collective also
  reproduced it. Both ranks recorded every all-reduce and fence as complete;
  the next captured piece stopped forever with the canonical 100% compute +
  100% copy engine signature.

That refutes stale caller-buffer IPC lifetime and incomplete collective
retirement as sufficient causes for this residual. Piece capture itself is
the necessary surface for the permanent device spin. Extended live testing
also found one recoverable 90-second request stall on the second
compile-without-capture pass (after 563 streamed chunks); cancellation drained
the devices and post-stall output stayed coherent, but the request was still
lost. The final gate therefore selects fully eager MTP, the report's clean
speculative TP posture, rather than stopping at compile-without-capture. A
true graph-on repair requires the offending compiled piece or the XPU graph
runtime to be fixed upstream.

## Validation gates

- image patch, marker, and bytecode checks;
- startup warning and effective configuration (`enforce_eager`,
  `cudagraph_mode=NONE`, graph environment disabled) under MTP k=3 + TP=2;
- temp-0 coherence (k=3 avoids separate defect #12);
- two standard large-context provocation passes on one boot;
- no watcher capture, engine reset, or post-recovery degenerate output;
- short-context throughput measurement documenting the fallback cost.
