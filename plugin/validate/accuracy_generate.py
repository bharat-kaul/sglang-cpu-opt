"""Accuracy check: greedy-generate from real prompts and print the text.

Validates the MXFP4 W4A16 MoE path end-to-end (coherent output => the fp4 weights + e8m0 scale +
w13 SwiGLU ordering are all correct). Run once with INTEL_CPU_DSV4_MXFP4_MOE=1 (MXFP4) and once
without (fp8 Plan-A reference) and compare the completions; they should match closely (lossless:
same fp4 weights, bf16 compute).
"""
import os

import sglang as sgl

PROMPTS = [
    "The capital of France is",
    "Q: What is 17 + 26? A:",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    ",
]


def main():
    if os.environ.get("ACC_PROMPT_IDX") is not None:
        prompts = [PROMPTS[int(os.environ["ACC_PROMPT_IDX"])]]
    elif os.environ.get("ACC_QUICK") == "1":
        prompts = PROMPTS[:1]
    else:
        prompts = PROMPTS
    max_new = int(os.environ.get("ACC_MAX_NEW", "32"))
    engine = sgl.Engine(
        model_path="/scratch/bkaul/models/DeepSeek-V4-Flash",
        device="cpu",
        tp_size=1,
        trust_remote_code=True,
        disable_cuda_graph=True,
        mem_fraction_static=0.85,
        max_total_tokens=8192,
        swa_full_tokens_ratio=0.8,
        watchdog_timeout=100000,  # fp4-raw ref is ~100s/tok -> don't let the watchdog abort it
    )
    sampling = {"temperature": 0.0, "max_new_tokens": max_new}
    if os.environ.get("INTEL_CPU_DSV4_MXFP4_MOE") == "1":
        tag = "MXFP4"
    elif os.environ.get("INTEL_CPU_DSV4_FP4_MOE_BF16") == "1":
        tag = "FP4RAW-ref"
    else:
        tag = "FP8-ref"
    for p in prompts:
        out = engine.generate(p, sampling)
        txt = out["text"] if isinstance(out, dict) else out[0]["text"]
        print(f"=== [{tag}] PROMPT: {p!r}\n=== [{tag}] COMPLETION: {txt!r}\n", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
