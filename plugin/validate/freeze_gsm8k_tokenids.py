#!/usr/bin/env python3
"""P1 (GSM8K review): FREEZE a fixed diagnostic set of gsm8k questions as exact TOKEN IDs, so the
GPU / CPU-eager / CPU-C legs prefill BYTE-IDENTICAL sequences (the review requires 'exact input token
IDs', not re-tokenized text, since tokenizers can differ across backends).

Builds the same 8-shot 'Question:/Answer:' prompt as task_gsm8k, tokenizes each with the model's own
tokenizer, and saves {q_id (global, post-shots), gold, token_ids, prompt_chars}. Downstream legs call
engine.generate(input_ids=token_ids) so every backend sees the identical prefill.

  python freeze_gsm8k_tokenids.py --model <path> --data <jsonl> --num-shots 8 \
      --q-ids 0,1,2,... (or --num-questions N) --out frozen.json
"""
import argparse
import json
import os

_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
if _pkg:
    __import__(_pkg)

from task_gsm8k import one_example, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-shots", type=int, default=8)
    ap.add_argument("--num-questions", type=int, default=80)
    ap.add_argument("--q-ids", default="", help="explicit comma list of GLOBAL post-shots question ids; "
                                                "overrides --num-questions (e.g. the review's hard cases)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = read_jsonl(args.data)
    shots = "".join(one_example(lines[i], True) + "\n\n" for i in range(args.num_shots))
    post = lines[args.num_shots:]

    if args.q_ids.strip():
        qids = [int(x) for x in args.q_ids.split(",") if x.strip()]
    else:
        qids = list(range(args.num_questions))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    frozen = {"model": args.model, "num_shots": args.num_shots, "tokenizer": tok.name_or_path,
              "data": args.data, "items": []}
    for qid in qids:
        x = post[qid]
        prompt = shots + one_example(x, False)
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        gold_line = x["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gold = int(gold_line)
        except ValueError:
            gold = gold_line
        frozen["items"].append({"q": qid, "gold": gold, "n_tokens": len(ids), "token_ids": ids})

    with open(args.out, "w") as f:
        json.dump(frozen, f)
    ntok = [it["n_tokens"] for it in frozen["items"]]
    print(f"[freeze] wrote {len(frozen['items'])} items -> {args.out}  "
          f"tokens min/med/max = {min(ntok)}/{sorted(ntok)[len(ntok)//2]}/{max(ntok)}", flush=True)


if __name__ == "__main__":
    main()
