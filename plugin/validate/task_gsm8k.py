#!/usr/bin/env python3
"""Generic gsm8k task-accuracy runner for SGLang CPU (accuracy-oracle layer 2).

Offline (no server): builds the standard 8-shot gsm8k prompt, generates greedily
through the SGLang CPU engine (with the plugin override), extracts the final
numeric answer, and reports exact-match accuracy. Mirrors the prompt + parsing of
sglang/benchmark/gsm8k/bench_sglang.py so results are comparable.

Run as a FILE on-node (SGLang spawns subprocesses):
  SGLANG_USE_CPU_ENGINE=1 SGLANG_EXTERNAL_MODEL_PACKAGE=intel_cpu_models \\
  PYTHONPATH=<plugin> srun ... python task_gsm8k.py --model <path> \\
      --data /scratch/$USER/gsm8k_test.jsonl --num-questions 100
"""

import argparse
import ast
import json
import re
import time

INVALID = -9999999


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def one_example(x, include_answer):
    s = "Question: " + x["question"] + "\nAnswer:"
    if include_answer:
        s += " " + x["answer"]
    return s


def answer_value(s):
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return INVALID
    try:
        return ast.literal_eval(nums[-1])
    except (SyntaxError, ValueError):
        return INVALID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--acc-min", type=float, default=0.0, help="pass threshold (else report only)")
    p.add_argument("--mem-fraction", type=float, default=0.5)
    p.add_argument("--dtype", default="bfloat16", help="CPU AMX compute dtype")
    p.add_argument("--quantization", default=None, help="e.g. w8a8_int8")
    p.add_argument("--out", default="")
    args = p.parse_args()

    lines = read_jsonl(args.data)
    shots = "".join(one_example(lines[i], True) + "\n\n" for i in range(args.num_shots))
    # evaluate on the questions AFTER the few-shot block to avoid overlap
    eval_lines = lines[args.num_shots:args.num_shots + args.num_questions]
    prompts = [shots + one_example(x, False) for x in eval_lines]
    labels = [answer_value(x["answer"]) for x in eval_lines]

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=args.tp, dtype=args.dtype,
        quantization=args.quantization,
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning",
    )
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens,
          "stop": ["Question", "Assistant:", "\n\n"]}
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sp)
    dt = time.perf_counter() - t0
    engine.shutdown()

    preds = [answer_value(o["text"] if isinstance(o, dict) else o) for o in outs]
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    invalid = sum(int(p == INVALID) for p in preds)
    acc = correct / len(labels)
    verdict = "PASS" if acc >= args.acc_min else "REPORT"

    result = {
        "model": args.model, "num_questions": len(labels), "num_shots": args.num_shots,
        "accuracy": round(acc, 4), "correct": correct, "invalid": invalid,
        "latency_s": round(dt, 1),
        "gen_tok_s_est": None, "acc_min": args.acc_min, "verdict": verdict,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
