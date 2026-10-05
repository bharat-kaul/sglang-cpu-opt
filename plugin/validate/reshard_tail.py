#!/usr/bin/env python3
"""Re-shard a straggler TAIL across many nodes (finish-faster for a sharded accuracy run).

Reads the gens checkpoints of the SLOW shards of an array run, computes their REMAINING global
question indices (slice-range minus completed-q), and writes N balanced id-files so a phase-2 array
(run_glm5_gsm8k.sbatch with IDS_DIR) can finish the tail wide-and-parallel instead of a few nodes
grinding ~50 Q each. The still-running fast shards are untouched (they finish free + warm).

Usage:
  python reshard_tail.py --tag 382474 --straggler-shards 8,11,12,15 \
      --orig-num-shards 16 --num-questions 1319 --num-shots 8 \
      --data /scratch/bkaul/gsm8k_test.jsonl \
      --phase2-shards 16 --out-dir /scratch/bkaul/reshard_382474
Then: sbatch --array=0-(P-1) --export=ALL,NQ=<nq>,CHUNK=..,MAX_RUNNING=..,MEM_FRAC=..,MAXNEW=..,
      STAGGER=45,RUN_TAG=<tag>r,IDS_DIR=<out-dir> run_glm5_gsm8k.sbatch
Combine at the end sums the ORIGINAL shard jsons + the phase-2 jsons (disjoint by construction).
"""
import argparse
import json
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="RUN_TAG of the original array (file key)")
    p.add_argument("--straggler-shards", required=True, help="comma-sep shard indices to re-shard")
    p.add_argument("--orig-num-shards", type=int, required=True)
    p.add_argument("--num-questions", type=int, default=1319)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--data", required=True)
    p.add_argument("--gens-prefix", default="/scratch/bkaul/glm5_gsm8k_gens_")
    p.add_argument("--phase2-shards", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    with open(args.data) as f:
        n_lines = sum(1 for _ in f)
    n = min(args.num_questions, n_lines - args.num_shots)  # global eval indices 0..n-1
    per = (n + args.orig_num_shards - 1) // args.orig_num_shards

    stragglers = [int(x) for x in args.straggler_shards.split(",")]
    remaining: list[int] = []
    for i in stragglers:
        lo, hi = i * per, min((i + 1) * per, n)
        gens_path = f"{args.gens_prefix}{args.tag}_{i}.json"
        done_q = set()
        if os.path.exists(gens_path):
            with open(gens_path) as f:
                done_q = {int(g["q"]) for g in json.load(f)}
        rem_i = [q for q in range(lo, hi) if q not in done_q]
        remaining.extend(rem_i)
        print(f"shard {i}: slice [{lo},{hi}) size {hi-lo}, done {len(done_q & set(range(lo,hi)))}, remaining {len(rem_i)}")
    remaining.sort()
    print(f"TOTAL remaining = {len(remaining)} across {len(stragglers)} stragglers")

    os.makedirs(args.out_dir, exist_ok=True)
    P = args.phase2_shards
    chunk = (len(remaining) + P - 1) // P
    written = 0
    for k in range(P):
        sub = remaining[k * chunk:(k + 1) * chunk]
        with open(os.path.join(args.out_dir, f"ids_{k}.json"), "w") as f:
            json.dump(sub, f)
        if sub:
            written += 1
        print(f"  ids_{k}.json: {len(sub)} questions")
    print(f"wrote {P} id-files ({written} non-empty) to {args.out_dir}; ~{chunk} Q/node")


if __name__ == "__main__":
    main()
