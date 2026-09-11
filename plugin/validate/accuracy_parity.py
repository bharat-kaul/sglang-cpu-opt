#!/usr/bin/env python3
"""Generic accuracy-oracle (layer 1): greedy next-token parity, SGLang CPU vs HF.

Model-agnostic. Loads the checkpoint under (a) the SGLang CPU engine (with the
plugin override) and (b) HuggingFace Transformers on CPU, and compares the greedy
next token for a fixed prompt set. Top-1 agreement is the primary CPU-vs-HF
correctness signal (a mis-wired kernel diverges immediately). Task accuracy
(gsm8k/mmlu) is driven separately by the in-repo harnesses.

Run as a FILE (SGLang spawns subprocesses that re-import __main__):

  SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \\
  PYTHONPATH=<plugin> srun ... python accuracy_parity.py --model <path> --tp 1
"""

import argparse
import json
import os
import sys

PROMPTS = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "The first president of the United States was",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "def add(a, b):\n    return",
    "The largest planet in the solar system is",
    "Roses are red, violets are",
    "The speed of light is approximately 3 x 10^",
    "To be or not to be, that is the",
    "The square root of 64 is",
    "Machine learning is a subfield of",
]


def hf_next_tokens(model, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    out = []
    with torch.no_grad():
        for pr in prompts:
            ids = tok(pr, return_tensors="pt").input_ids
            nxt = int(m(ids).logits[0, -1].float().argmax())
            out.append((nxt, tok.decode([nxt])))
    return out, tok


def sglang_next_tokens(model, tp, prompts, tok, dtype="bfloat16", quantization=None):
    import sglang as sgl

    e = sgl.Engine(
        model_path=model, device="cpu", tp_size=tp, dtype=dtype,
        quantization=quantization,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.5")),
        log_level="warning",
    )
    out = []
    for pr in prompts:
        r = e.generate(pr, {"temperature": 0.0, "max_new_tokens": 1})
        txt = r["text"] if isinstance(r, dict) else r
        ids = tok(txt, add_special_tokens=False).input_ids
        out.append((ids[0] if ids else None, txt))
    e.shutdown()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", help="CPU AMX compute dtype")
    p.add_argument("--quantization", default=None, help="e.g. w8a8_int8")
    p.add_argument("--ref-model", default=None,
                   help="HF reference (defaults to --model); for quantized runs point "
                        "to the ORIGINAL unquantized checkpoint (HF can't load w8a8 int8)")
    p.add_argument("--agree-min", type=float, default=0.9)
    p.add_argument("--out", default="")
    args = p.parse_args()

    hf, tok = hf_next_tokens(args.ref_model or args.model, PROMPTS)
    sg = sglang_next_tokens(args.model, args.tp, PROMPTS, tok, args.dtype, args.quantization)

    rows, agree = [], 0
    for pr, (hid, htxt), (sid, stxt) in zip(PROMPTS, hf, sg):
        ok = (htxt.strip() == stxt.strip()) or (hid == sid)
        agree += ok
        rows.append({"prompt": pr[:36], "hf": htxt.strip()[:16],
                     "sglang": stxt.strip()[:16], "agree": ok})
    rate = agree / len(PROMPTS)
    verdict = "PASS" if rate >= args.agree_min else "FAIL"
    result = {"model": args.model, "prompts": len(PROMPTS),
              "top1_agreement": round(rate, 3), "agree_min": args.agree_min,
              "rows": rows, "overall_verdict": verdict}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
