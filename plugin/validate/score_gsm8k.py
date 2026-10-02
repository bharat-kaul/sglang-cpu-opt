#!/usr/bin/env python3
"""Offline gsm8k RE-SCORER / protocol grid-search — NO model, milliseconds.

Generation (the model) is the expensive part; stop-truncation + answer-extraction are
pure string ops. So generate RAW ONCE (long max_new, minimal stop) and SAVE the texts
(diag_gsm8k_dump.py --gens-out, or task_gsm8k_chunked.py --gens-out), then tune the
eval PROTOCOL here for free instead of re-running the engine per idea.

  python score_gsm8k.py --gens /scratch/$USER/dsv4_gsm8k_diag_gens_<JID>.json [--tag B]

Prints an accuracy grid over {stop-set} x {extractor} so you can LOCK the best protocol,
then run the full/sharded eval once with it.
"""
import argparse
import json

from task_gsm8k import answer_value, extract_final_answer

EXTRACTORS = {"last_num": answer_value, "first_answer": extract_final_answer}
STOP_SETS = {
    "none": [],
    "nn": ["\n\n"],
    "Question": ["Question"],
    "nl+Question": ["\nQuestion", "\n\nQuestion"],
    "Question+nn": ["Question", "\n\n"],
}


def apply_stop(text, stops):
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="json list of {text, gold[, tag]}")
    ap.add_argument("--tag", default="", help="only score rows whose tag startswith this")
    args = ap.parse_args()

    rows = json.load(open(args.gens))
    if args.tag:
        rows = [r for r in rows if str(r.get("tag", "")).startswith(args.tag)]
    n = len(rows)
    print(f"scoring {n} saved generations OFFLINE (no model)\n")
    best = (None, -1.0)
    for sname, stops in STOP_SETS.items():
        for ename, ex in EXTRACTORS.items():
            c = sum(int(ex(apply_stop(r["text"], stops)) == r["gold"]) for r in rows)
            acc = c / n if n else 0.0
            print(f"  stop={sname:14s} extract={ename:12s} -> {c}/{n} = {acc:.3f}")
            if acc > best[1]:
                best = (f"stop={sname} extract={ename}", acc)
    print(f"\nBEST: {best[0]} = {best[1]:.3f}")


if __name__ == "__main__":
    main()
