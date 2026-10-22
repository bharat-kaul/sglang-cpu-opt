#!/usr/bin/env python3
"""Per-layer divergence localizer: compare two legs' per-layer last-token hidden vectors (from
dsv4_layer_capture) and report, layer-by-layer, the cosine similarity + where the divergence ENTERS.

The output-logit comparison says CPU and GPU disagree on a token; this says WHICH LAYER the hidden
state first diverges -> localizes the culprit op (that layer's attention / MoE / MHC / norm). For a
CPU-C vs CPU-eager pair (same build/quant) a drop localizes a C-KERNEL wiring bug cleanly; for a
CPU vs GPU pair, a drop at a MoE layer may be the expected quant-impl difference, not a bug.

  python compare_layers.py --a cpu_c=layers_cpu.json --b gpu=layers_gpu.json [--cos-min 0.999]
"""
import argparse
import json
import math
import sys


def _cos(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na < 1e-12 and nb < 1e-12:
        return 1.0
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _load(spec):
    name, path = spec.split("=", 1)
    return name, json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="nameA=layers_a.json")
    ap.add_argument("--b", required=True, help="nameB=layers_b.json")
    ap.add_argument("--cos-min", type=float, default=0.999,
                    help="per-layer cosine below this = divergence entered at this layer")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    na, da = _load(args.a)
    nb, db = _load(args.b)
    la, lb = da.get("layers", {}), db.get("layers", {})
    common = sorted(set(la) & set(lb), key=lambda s: int(s))
    if not common:
        print("[compare-layers] no common layers", file=sys.stderr)
        return 2

    rows = []
    first_div = None
    for lid in common:
        c = _cos(la[lid], lb[lid])
        rows.append({"layer": int(lid), "cos": round(c, 6)})
        if first_div is None and (c < args.cos_min or math.isnan(c)):
            first_div = int(lid)

    # final logits top-1 agreement
    fa = (da.get("final_logits_top5") or [[None]])[0][0]
    fb = (db.get("final_logits_top5") or [[None]])[0][0]
    report = {"a": na, "b": nb, "n_layers": len(common), "cos_min": args.cos_min,
              "first_divergent_layer": first_div,
              "final_top1": {na: fa, nb: fb, "agree": fa == fb},
              "per_layer": rows}
    print(f"[compare-layers] {na} vs {nb}: {len(common)} layers; "
          f"first divergence @ layer {first_div} (cos_min={args.cos_min}); "
          f"final top1 {na}={fa} {nb}={fb} {'AGREE' if fa == fb else 'DISAGREE'}")
    for r in rows:
        mark = "  <-- DIVERGES" if (first_div is not None and r["layer"] == first_div) else ""
        print(f"  layer {r['layer']:3d}: cos={r['cos']:.6f}{mark}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[compare-layers] wrote {args.out}")
    return 0 if first_div is None else 1


if __name__ == "__main__":
    sys.exit(main())
