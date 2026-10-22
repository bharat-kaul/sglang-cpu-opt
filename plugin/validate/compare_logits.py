#!/usr/bin/env python3
"""Option 2 comparator: TEACHER-FORCED per-position top-1 agreement between two legs (CPU vs GPU, or
CPU-C vs CPU-eager) on the SAME frozen sequences. Because each position is conditioned on the identical
fixed prefix, this is a clean numerical-fidelity signal (no greedy amplification):

  * per-sequence top-1 agreement fraction (both argmax the same next token given the same prefix);
  * FIRST-DIVERGENCE position per sequence (where the two backends first disagree on argmax) -> localizes;
  * overall top-1 agreement + a verdict against --agree-min.

Top-1 teacher-forced agreement is robust to benign build/ULP differences (they only flip argmax on a
near-tie); a systematic early divergence indicates a real implementation difference, not noise.

  python compare_logits.py --a gpu=logits_gpu.json --b cpu=logits_cpu.json --agree-min 0.98
"""
import argparse
import json
import sys


def _load(spec):
    name, path = spec.split("=", 1)
    d = json.load(open(path))
    return name, d, {r["q"]: r for r in d["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="nameA=pathA.json")
    ap.add_argument("--b", required=True, help="nameB=pathB.json")
    ap.add_argument("--agree-min", type=float, default=0.98)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    na, da, ra = _load(args.a)
    nb, db, rb = _load(args.b)
    qs = sorted(set(ra) & set(rb))
    if not qs:
        print("[compare] no common sequences", file=sys.stderr)
        return 2

    per_seq = []
    tot_pos = tot_agree = 0
    for q in qs:
        am_a = ra[q]["argmax"]
        am_b = rb[q]["argmax"]
        n = min(len(am_a), len(am_b))
        # position 0 has no prediction target convention differences; compare where both defined
        agree = 0
        first_div = None
        compared = 0
        for i in range(n):
            if am_a[i] is None or am_b[i] is None:
                continue
            compared += 1
            if am_a[i] == am_b[i]:
                agree += 1
            elif first_div is None:
                first_div = i
        per_seq.append({"q": q, "positions": compared, "agree": agree,
                        "agreement": round(agree / max(1, compared), 4),
                        "first_divergence_pos": first_div, "seq_len": n})
        tot_pos += compared
        tot_agree += agree

    overall = tot_agree / max(1, tot_pos)
    verdict = "FAITHFUL" if overall >= args.agree_min else "DIVERGENT"
    report = {"a": {na: da.get("git_sha"), "device": da.get("device")},
              "b": {nb: db.get("git_sha"), "device": db.get("device")},
              "n_seqs": len(qs), "total_positions": tot_pos,
              "overall_top1_agreement": round(overall, 4), "agree_min": args.agree_min,
              "VERDICT": verdict, "per_seq": per_seq}
    print(f"[compare] {na} vs {nb}: top-1 teacher-forced agreement = {overall:.4f} "
          f"over {tot_pos} positions ({len(qs)} seqs) -> {verdict}")
    for s in per_seq:
        print(f"  q{s['q']}: {s['agreement']:.4f} ({s['agree']}/{s['positions']}) "
              f"first_div@{s['first_divergence_pos']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[compare] wrote {args.out}")
    return 0 if verdict == "FAITHFUL" else 1


if __name__ == "__main__":
    sys.exit(main())
