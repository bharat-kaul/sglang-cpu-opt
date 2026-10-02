#!/usr/bin/env python3
"""Diagnostic: dump real gsm8k generations to explain a low accuracy number.

Runs a few questions under TWO settings on the same engine and prints, per question,
the gold answer, what our parser extracts, match/no-match, and the full generation —
so we can see whether a low score is TRUNCATION (256 / "\\n\\n" stop cutting CoT),
a PARSE error (wrong number extracted), or genuinely WRONG reasoning (-> needs GPU
parity). Settings:
  A = current harness (max_new 256, stop ["Question","Assistant:","\\n\\n"])
  B = CoT-matched   (max_new 512, stop ["Question"])
"""
import argparse
import json

from task_gsm8k import answer_value, extract_final_answer, one_example, read_jsonl


def run(engine, prompts, labels, sp, tag, dumps):
    outs = engine.generate(prompts, sp)
    old = new = 0
    for i, (o, gold) in enumerate(zip(outs, labels)):
        txt = o["text"] if isinstance(o, dict) else o
        p_old = answer_value(txt)
        p_new = extract_final_answer(txt)
        old += int(p_old == gold)
        new += int(p_new == gold)
        dumps.append({"tag": tag, "q": i, "gold": gold, "pred_old": p_old,
                      "pred_new": p_new, "text": txt})
        if i < 4:
            print(f"\n[{tag} Q{i}] gold={gold} old={p_old} new={p_new} gen_len~{len(txt)}")
            print("  GEN:", repr(txt[:600]))
    n = len(labels)
    print(f"\n=== {tag}: old(last-num)={old}/{n}={old/n:.3f}  "
          f"new(first-answer)={new}/{n}={new/n:.3f} ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--mem-fraction", type=float, default=0.5)
    p.add_argument("--watchdog-timeout", type=float, default=7000.0)
    p.add_argument("--chunked-prefill-size", type=int, default=512)
    p.add_argument("--gens-out", default="", help="save all generations here for offline extractor tuning")
    args = p.parse_args()
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # flush to the sbatch log immediately

    lines = read_jsonl(args.data)
    shots = "".join(one_example(lines[i], True) + "\n\n" for i in range(args.num_shots))
    eval_lines = lines[args.num_shots:args.num_shots + args.n]
    prompts = [shots + one_example(x, False) for x in eval_lines]
    labels = [answer_value(x["answer"]) for x in eval_lines]

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
        disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=args.mem_fraction, log_level="warning",
        watchdog_timeout=args.watchdog_timeout,
        chunked_prefill_size=args.chunked_prefill_size,
    )
    base = {"temperature": 0.0}
    dumps = []

    def save():
        if args.gens_out:
            with open(args.gens_out, "w") as f:
                json.dump(dumps, f)
            print(f"saved {len(dumps)} generations to {args.gens_out}", flush=True)

    run(engine, prompts, labels,
        {**base, "max_new_tokens": 256, "stop": ["Question", "Assistant:", "\n\n"]},
        "A current(256,\\n\\n)", dumps)
    save()
    run(engine, prompts, labels,
        {**base, "max_new_tokens": 512, "stop": ["Question"]},
        "B matched(512,Question)", dumps)
    save()
    engine.shutdown()
    if args.gens_out:
        with open(args.gens_out, "w") as f:
            json.dump(dumps, f)
        print("saved", len(dumps), "generations to", args.gens_out)


if __name__ == "__main__":
    main()
