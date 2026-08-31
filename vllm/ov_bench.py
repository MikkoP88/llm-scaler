#!/usr/bin/env python3
"""ov_bench.py -- OpenVINO GenAI benchmark for Qwen3.8-27B (int4-ov export) on Intel Arc Pro B70.

Stack requirements (hard-won):
  - export is a VLM artifact (pipeline_tag=image-text-to-text, LM port = inputs_embeds)
    -> must use VLMPipeline; LLMPipeline ctor works but generate() fails 'input_ids not found'
  - openvino-genai >= 2026.5.0 nightly: stable 2026.2.1/2026.3.1 segfault in
    ov::genai::MakePaddingSatateful -> ov::Node::get_output_shape on this export's
    shipped tokenizer IR (gdb-verified); export README itself demands 2026.4.0 nightly+
  - GPU userspace must come from the v27 vLLM image (NEO 37833/igc 2.32.7/gmmlib 12.10);
    the official OV runtime image ships pre-Battlemage NEO + shadowed igc/gmmlib

Comparison target: vLLM v27 stack (TP=2, same host, same model family):
  nospec+graphs decode 32.1 tok/s @2k ctx | eager+MTP-k4 48.4 tok/s short ctx
  prefill 1.5-2.0k tok/s | wedge (#11) at >=32k ctx on oneCCL

Arms:
  devices   enumerate OpenVINO devices (sanity, GPU plugin must list both B70s)
  core      warmup, decode@~512ctx, decode@~2k ctx (TTFT + steady decode), prefill@~8k,
            greedy coherence probe x3 (distinct==1 check, given genai issue #3870 history)
  longctx   decode@~16k and ~32k ctx (VRAM-bound on 32GB card; catches OOM gracefully)

Usage: python3 ov_bench.py {devices|core|longctx} [--model DIR] [--device GPU.0]
"""
import argparse
import json
import sys
import time

try:
    import openvino_genai as ovgen
    from openvino import Core
except ImportError as e:
    print("FATAL: openvino/openvino-genai not importable:", e)
    sys.exit(3)

PARA = (
    "The history of computing machinery is in large part a history of layered "
    "abstraction. Each generation of engineers wraps the irregularities of the "
    "hardware below in a cleaner interface, and then builds new irregularities of "
    "its own for the next generation to bury. Compilers buried instruction sets, "
    "operating systems buried interrupt controllers, virtual machines buried "
    "memory layouts, and container runtimes buried whole operating systems. "
)

RESULTS = {}


def now():
    return time.perf_counter()


def text_of(res):
    """VLMDecodedResults -> str (LLMPipeline .text also works)."""
    v = getattr(res, "text", None)
    if isinstance(v, str):
        return v
    t = getattr(res, "texts", None)
    if t:
        return t[0]
    return str(res)


class Bench:
    def __init__(self, model_dir, device):
        t0 = now()
        # export is pipeline_tag=image-text-to-text (VLM): LM port is inputs_embeds,
        # so LLMPipeline builds but generate() fails on 'input_ids not found' -> VLMPipeline.
        # Nightly (>=2026.5.0) required: stable 2026.3.1 genai segfaults in
        # MakePaddingSatateful on this export's shipped tokenizer IR.
        self.pipe = ovgen.VLMPipeline(model_dir, device)
        self.load_s = now() - t0
        try:
            self.tok = self.pipe.get_tokenizer()
        except Exception as e:
            self.tok = None
            print(f"[load] get_tokenizer failed ({str(e)[:120]}); token counts fall back to estimate", flush=True)
        print(f"[load] VLMPipeline '{model_dir}' on '{device}' in {self.load_s:.1f}s", flush=True)

    def n_tokens(self, text):
        if self.tok is None:
            return max(1, int(len(text.split()) * 1.35))
        try:
            enc = self.tok.encode(text)
            ids = enc.input_ids if hasattr(enc, "input_ids") else enc
            try:
                return int(ids.shape[-1])
            except Exception:
                return len(ids)
        except Exception:
            return max(1, int(len(text.split()) * 1.35))  # rough fallback

    def timed_generate(self, prompt, n_new, label):
        """One timed generation. Returns dict with TTFT + steady decode rate."""
        times = []

        def cb(_chunk):
            times.append(now())
            return None

        cfg = ovgen.GenerationConfig(max_new_tokens=n_new, apply_chat_template=False)
        try:
            cfg.ignore_eos = True  # force exactly n_new tokens (vLLM probe parity)
        except Exception:
            pass

        t_start = now()
        try:  # VLMPipeline overload 1: (prompt, images, videos, config, streamer)
            self.pipe.generate(prompt, [], [], cfg, cb)
        except TypeError:
            self.pipe.generate(prompt, [], [], cfg)
        t_end = now()

        dur = t_end - t_start
        n_cb = len(times)
        ptok = self.n_tokens(prompt)
        r = {
            "label": label,
            "prompt_tokens": ptok,
            "wall_s": round(dur, 3),
            "callbacks": n_cb,
        }
        if n_cb >= 2:
            ttft = times[0] - t_start
            dec_dur = times[-1] - times[0]
            r["ttft_s"] = round(ttft, 3)
            r["decode_tok_s"] = round((n_cb - 1) / dec_dur, 2) if dec_dur > 0 else None
            r["prefill_tok_s"] = round(ptok / ttft, 1) if ttft > 0 else None
        else:
            # fallback: no callback granularity -> whole-run rate
            r["decode_tok_s"] = round(n_new / dur, 2) if dur > 0 else None
        print(json.dumps(r), flush=True)
        return r

    def build_prompt(self, target_tokens, tail):
        """Repeat PARA until ~target_tokens, then append tail."""
        para_tok = self.n_tokens(PARA)
        reps = max(1, target_tokens // max(1, para_tok))
        return PARA * reps + "\n\n" + tail

    def chat(self, user_msg):
        """Chat-template generation; returns generated text."""
        if self.tok is not None:
            try:
                prompt = self.tok.apply_chat_template(
                    [{"role": "user", "content": user_msg}], add_generation_prompt=True
                )
            except Exception:
                prompt = user_msg
        else:
            prompt = user_msg
        cfg = ovgen.GenerationConfig(max_new_tokens=48, apply_chat_template=False)
        out = self.pipe.generate(prompt, [], [], cfg)
        return text_of(out)


def arm_devices(args):
    core = Core()
    devs = core.get_available_devices()
    print(json.dumps({"devices": devs}, indent=1))
    for d in devs:
        if d.startswith("GPU"):
            print(d, core.get_property(d, "FULL_DEVICE_NAME"))


def arm_core(args, b=None):
    b = b or Bench(args.model, args.device)
    RESULTS["load_s"] = round(b.load_s, 1)

    # warmup (kernel JIT / lazy init) - untimed
    t0 = now()
    b.pipe.generate(
        b.build_prompt(64, "Summarize the passage in one sentence."),
        [],
        [],
        ovgen.GenerationConfig(max_new_tokens=8, apply_chat_template=False),
    )
    print(f"[warmup] done in {now()-t0:.1f}s", flush=True)

    cont = "Continue the passage in the same style without repeating it."
    RESULTS["decode_512"] = b.timed_generate(b.build_prompt(450, cont), 256, "decode@~512ctx")
    RESULTS["decode_2k"] = b.timed_generate(b.build_prompt(1800, cont), 256, "decode@~2kctx")

    # prefill arm: 8k-token prompt, generate 8 -> TTFT ~= prefill time
    RESULTS["prefill_8k"] = b.timed_generate(
        b.build_prompt(8000, "Summarize the passage in one sentence."), 8, "prefill@~8kctx"
    )

    # coherence probe (genai #3870 history): greedy x3, distinct==1 + sane answer
    outs = []
    for i in range(3):
        outs.append(b.chat("What is the capital of France? Answer with the city name only."))
    distinct = len(set(outs))
    sane = any("Paris" in o for o in outs)
    RESULTS["coherence"] = {
        "distinct_outputs": distinct,
        "mentions_paris": sane,
        "sample": outs[0][:120],
    }
    print(json.dumps(RESULTS["coherence"], indent=1), flush=True)
    print("=== CORE RESULTS ===")
    print(json.dumps(RESULTS, indent=1), flush=True)


def arm_longctx(args):
    b = Bench(args.model, args.device)
    cont = "Continue the passage in the same style without repeating it."
    for target, n_new in ((16000, 128), (32000, 128)):
        try:
            RESULTS[f"decode_{target//1000}k"] = b.timed_generate(
                b.build_prompt(target, cont), n_new, f"decode@~{target//1000}kctx"
            )
        except Exception as e:
            RESULTS[f"decode_{target//1000}k"] = {"label": f"decode@~{target//1000}kctx", "error": str(e)[:300]}
            print(json.dumps(RESULTS[f"decode_{target//1000}k"]), flush=True)
    print("=== LONGCTX RESULTS ===")
    print(json.dumps(RESULTS, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["devices", "core", "longctx"])
    ap.add_argument("--model", default="/models/qwen38-27b-int4-ov")
    ap.add_argument("--device", default="GPU.0")
    args = ap.parse_args()
    if args.arm == "devices":
        arm_devices(args)
    elif args.arm == "core":
        arm_core(args)
    else:
        arm_longctx(args)


if __name__ == "__main__":
    main()
