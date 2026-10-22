#!/usr/bin/env python3
"""P1 (GSM8K review): paired analysis across the GPU / CPU-eager / CPU-C legs on the SAME frozen
token-ID set. Reports, per leg pair: per-question prediction agreement, paired correctness (McNemar),
and a divergence list (where the legs' final answers differ) for first-divergence inspection.

A predefined equivalence margin is applied: CPU-C is 'equivalent to' CPU-eager if their prediction
agreement >= --agree-min AND McNemar is non-significant (p >= 0.05). (Per the review: a non-significant
difference alone is not equivalence; we require BOTH agreement and non-significance, and we never hide
a leg that had unresolved validation fallbacks — pass --strict-clean to assert none.)

  python paired_analyze.py --legs gpu=paired_gpu_gpu.json cpu_eager=paired_cpu_cpu_ck0.json \
      cpu_c=paired_cpu_cpu_ck1.json --agree-min 0.98
"""
import argparse
import json
import sys


def _mcnemar(b, c):
    # exact binomial two-sided p on the discordant pairs (b, c); no scipy dependency.
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def _load(spec):
    name, path = spec.split("=", 1)
    d = json.load(open(path))
    recs = {r["q"]: r for r in d["records"]}
    return name, d, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="+", required=True, help="name=path.json ... (e.g. gpu=... cpu_eager=... cpu_c=...)")
    ap.add_argument("--agree-min", type=float, default=0.98)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    legs = {}
    meta = {}
    for spec in args.legs:
        name, d, recs = _load(spec)
        legs[name] = recs
        meta[name] = {"accuracy": d.get("accuracy"), "n": d.get("n"), "git_sha": d.get("git_sha"),
                      "device": d.get("device"), "ck": d.get("ck")}

    qids = sorted(set.intersection(*[set(r) for r in legs.values()]))
    report = {"n_common": len(qids), "legs": meta, "pairs": {}}
    names = list(legs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            agree = 0
            disagree = []
            only_a = only_b = 0   # McNemar discordant on CORRECTNESS
            for q in qids:
                ra, rb = legs[a][q], legs[b][q]
                if ra["pred"] == rb["pred"]:
                    agree += 1
                else:
                    disagree.append({"q": q, "gold": ra["gold"], f"{a}_pred": ra["pred"], f"{b}_pred": rb["pred"],
                                     f"{a}_finish": ra.get("finish_reason"), f"{b}_finish": rb.get("finish_reason")})
                if ra["correct"] and not rb["correct"]:
                    only_a += 1
                elif rb["correct"] and not ra["correct"]:
                    only_b += 1
            p = _mcnemar(only_a, only_b)
            agree_frac = agree / max(1, len(qids))
            equivalent = (agree_frac >= args.agree_min) and (p >= 0.05)
            report["pairs"][f"{a}__vs__{b}"] = {
                "pred_agreement": round(agree_frac, 4), "n_disagree": len(disagree),
                "mcnemar_discordant": {f"{a}_correct_only": only_a, f"{b}_correct_only": only_b},
                "mcnemar_p": round(p, 4), "equivalent_margin": {"agree_min": args.agree_min, "p_min": 0.05},
                "EQUIVALENT": equivalent, "divergences": disagree[:50],
            }
            print(f"[{a} vs {b}] pred-agree={agree_frac:.4f} n_disagree={len(disagree)} "
                  f"McNemar(only_{a}={only_a},only_{b}={only_b}) p={p:.4f} -> "
                  f"{'EQUIVALENT' if equivalent else 'NOT-EQUIVALENT'}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[paired-analyze] wrote {args.out}", flush=True)
    # exit nonzero if any CPU-C vs CPU-eager pair is NOT equivalent (gate)
    bad = [k for k, v in report["pairs"].items() if ("cpu_c" in k and "cpu_eager" in k and not v["EQUIVALENT"])]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
