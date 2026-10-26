#!/usr/bin/env python3
"""Compare per-layer MoE selected expert ids between two legs (e.g. CPU vs GPU).

Inputs: two '<file>.moetopk' JSONs produced by dsv4_layer_capture (keyed by layer_id, each
layer has topk_ids [num_tokens, top_k] of GLOBAL expert ids + is_hash flag).

For each MoE layer, reports the fraction of tokens whose selected expert-id SET matches exactly
(discrete set-match, order-independent), and the mean Jaccard overlap. Prints the FIRST layer whose
exact-match fraction drops below a threshold -> localizes where routing divergence enters.

Usage: compare_moe_topk.py CPU.moetopk GPU.moetopk [--thresh 0.99]
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("leg", "?"), {int(k): v for k, v in d["layers"].items()}, d.get("input_ids"), d.get("seq_idx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--thresh", type=float, default=0.99)
    args = ap.parse_args()

    leg_a, la, ids_a, si_a = load(args.a)
    leg_b, lb, ids_b, si_b = load(args.b)
    common = sorted(set(la) & set(lb))
    if not common:
        print("NO COMMON LAYERS", file=sys.stderr)
        sys.exit(2)

    print(f"leg A={leg_a} ({args.a})  seq_idx={si_a} ntok={len(ids_a) if ids_a else '?'}")
    print(f"leg B={leg_b} ({args.b})  seq_idx={si_b} ntok={len(ids_b) if ids_b else '?'}")
    if ids_a is not None and ids_b is not None:
        if ids_a == ids_b:
            print("ALIGNMENT OK: input_ids identical across legs.")
        else:
            print("!!! ALIGNMENT MISMATCH: input_ids differ across legs — comparison is INVALID. !!!")
            print(f"    A first 8={ids_a[:8]} ... B first 8={ids_b[:8]}")
    else:
        print("WARN: input_ids missing in one/both captures — cannot verify alignment.")
    print(f"{'layer':>5} {'hash':>4} {'ntok':>6} {'exact%':>8} {'jaccard':>8}  verdict")
    first_bad = None
    for lid in common:
        ta, tb = la[lid]["topk_ids"], lb[lid]["topk_ids"]
        is_hash = la[lid].get("is_hash", False)
        n = min(len(ta), len(tb))
        if n == 0:
            print(f"{lid:>5} {str(is_hash):>4} {0:>6}   (empty)")
            continue
        exact = 0
        jacc = 0.0
        for i in range(n):
            sa = set(x for x in ta[i] if x >= 0)
            sb = set(x for x in tb[i] if x >= 0)
            if sa == sb:
                exact += 1
            u = sa | sb
            jacc += (len(sa & sb) / len(u)) if u else 1.0
        exfrac = exact / n
        jfrac = jacc / n
        bad = exfrac < args.thresh
        if bad and first_bad is None:
            first_bad = lid
        print(f"{lid:>5} {str(is_hash):>4} {n:>6} {exfrac*100:>7.2f}% {jfrac:>8.4f}  {'<<< DIVERGES' if bad else 'ok'}")

    print()
    if first_bad is None:
        print(f"ALL MoE layers match within thresh {args.thresh}: routing is IDENTICAL across legs.")
    else:
        print(f"FIRST diverging MoE layer: {first_bad} (hash={la[first_bad].get('is_hash')}).")
        print("  -> routing differs here. If upstream layers' expert sets match but this one doesn't,")
        print("     the divergence enters at/after this layer's router input (check attention/MHC feeding it).")


if __name__ == "__main__":
    main()
