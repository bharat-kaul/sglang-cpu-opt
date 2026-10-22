#!/usr/bin/env python3
"""Combine sharded gsm8k json outputs (task_gsm8k_chunked.py --num-shards N) into one overall
exact-match accuracy — FAIL-CLOSED (adversarial-self-audit cluster 3: a consumer that aggregates
evidence must VALIDATE the inputs it combines, reject invalid, and write NO output on reject).

Validates, before writing anything:
  * non-empty input set (a glob that matches nothing is a FAILURE, not 0 shards);
  * each shard file parses and carries the required fields (shard_index, done, total, correct);
  * counts are coordinate-consistent (0 <= correct <= done <= total; invalid <= done);
  * shard identity is UNIQUE (two files claiming the same shard_index -> reject);
  * run contracts are COMPATIBLE across shards (same model, num_shots, num_shards);
  * if --require-complete, EXACTLY the shards [0, num_shards) are present (no missing/extra),
    and every shard is non-partial (done == total).
On any failure: print the reason, write NOTHING, exit nonzero. Only a fully valid set prints +
writes the combined result.

  python combine_gsm8k_shards.py --glob "<dir>/dsv4_d2acc_<ARRAYJOBID>_sh*.json" \
      [--require-complete] [--expect-shards 8] [--out combined.json]
"""
import argparse
import glob
import json
import sys


def _fail(msg):
    print(f"[combine] REJECT: {msg}  (no output written)", file=sys.stderr)
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="shell glob of the shard json files")
    ap.add_argument("--out", default="")
    ap.add_argument("--require-complete", action="store_true",
                    help="require EXACTLY shards [0,num_shards) present + every shard done==total")
    ap.add_argument("--expect-shards", type=int, default=0,
                    help="assert this many shards (0 = infer from the shards' own num_shards field)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        return _fail(f"glob matched NO files: {args.glob!r}")

    shards = {}            # shard_index -> record
    contracts = set()      # (model, num_shots, num_shards)
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception as e:  # noqa: BLE001
            return _fail(f"{f}: unreadable/invalid JSON ({e})")
        for k in ("shard_index", "done", "total", "correct"):
            if k not in d:
                return _fail(f"{f}: missing required field {k!r}")
        si = int(d["shard_index"])
        done, total, corr = int(d["done"]), int(d["total"]), int(d["correct"])
        inval = int(d.get("invalid", 0))
        if not (0 <= corr <= done <= total) or not (0 <= inval <= done):
            return _fail(f"{f}: inconsistent counts correct={corr} invalid={inval} done={done} total={total}")
        if si in shards:
            return _fail(f"duplicate shard_index {si}: {shards[si]['_file']} and {f}")
        d["_file"] = f.split("/")[-1]
        shards[si] = d
        contracts.add((d.get("model"), d.get("num_shots"), d.get("num_shards")))

    if len(contracts) != 1:
        return _fail(f"incompatible run contracts across shards (model/num_shots/num_shards): {sorted(contracts)}")
    (_model, _shots, _nshards) = contracts.pop()

    expect = args.expect_shards or (int(_nshards) if _nshards else 0)
    if args.require_complete:
        if not expect:
            return _fail("--require-complete needs a known shard count (--expect-shards or a num_shards field)")
        missing = sorted(set(range(expect)) - set(shards))
        extra = sorted(set(shards) - set(range(expect)))
        if missing or extra:
            return _fail(f"shard coverage incomplete: missing={missing} extra={extra} (expected [0,{expect}))")
        partials = sorted(si for si, d in shards.items() if int(d["done"]) != int(d["total"]))
        if partials:
            return _fail(f"partial shards present (done!=total): {partials}")
    elif expect and set(shards) != set(range(expect)):
        print(f"[combine] WARNING: not --require-complete; have shards {sorted(shards)} of expected [0,{expect})",
              file=sys.stderr)

    correct = sum(int(d["correct"]) for d in shards.values())
    done = sum(int(d["done"]) for d in shards.values())
    invalid = sum(int(d.get("invalid", 0)) for d in shards.values())
    if done == 0:
        return _fail("zero questions scored across all shards")
    acc = correct / done
    out = {
        "model": _model, "num_shots": _shots, "num_shards_expected": expect,
        "shards_found": len(shards), "questions_scored": done, "correct": correct,
        "invalid": invalid, "accuracy": round(acc, 4),
        "require_complete": args.require_complete,
        "any_partial": any(int(d["done"]) != int(d["total"]) for d in shards.values()),
        "status_label": "historical custom-protocol accuracy (NOT a published-score reproduction, "
                        "NOT a CPU-numerical-correctness certificate)",
        "detail": [{"file": d["_file"], "shard": si, "done": d["done"], "total": d["total"],
                    "correct": d["correct"], "partial": int(d["done"]) != int(d["total"])}
                   for si, d in sorted(shards.items())],
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
