#!/usr/bin/env python3
"""Combine sharded gsm8k json outputs (task_gsm8k_chunked.py --num-shards N) into one
overall exact-match accuracy. Sums correct/done across shards, so PARTIAL shards (killed
at the wall) still contribute their completed questions.

  python combine_gsm8k_shards.py --glob "/scratch/$USER/dsv4_gsm8k_sh_<ARRAYJOBID>_*.json"
"""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="shell glob of the shard json files")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    correct = done = invalid = 0
    shards = []
    for f in files:
        d = json.load(open(f))
        correct += d.get("correct", 0)
        done += d.get("done", 0)
        invalid += d.get("invalid", 0)
        shards.append({"file": f.split("/")[-1], "shard": d.get("shard_index"),
                       "done": d.get("done"), "total": d.get("total"),
                       "correct": d.get("correct"), "partial": d.get("partial")})
    acc = correct / done if done else 0.0
    out = {
        "shards_found": len(files), "questions_scored": done, "correct": correct,
        "invalid": invalid, "accuracy": round(acc, 4),
        "any_partial": any(s["partial"] for s in shards), "detail": shards,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
