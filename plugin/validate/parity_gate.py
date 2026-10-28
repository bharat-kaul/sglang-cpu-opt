#!/usr/bin/env python3
"""FAIL-CLOSED per-(pass,op,layer,step,rank) parity gate for two FULLCAP .pt dumps.

Fixes the review finding that diff_fullcap.py is a diagnostic printer (never exits nonzero, passes on
NaN / opposite tensors, strips rank suffixes without collision checks). This gate EXITS NONZERO on ANY:
  - missing expected keys / empty capture / zero shared keys
  - duplicate key after normalization (collision)
  - shape or dtype mismatch
  - non-finite values on EITHER side (NaN/Inf)
  - cosine < --cos-thresh  OR  rel-max-abs-err > --rel-thresh
  - (optional) exact-equality required (--exact) for a pure control-flow port
Prints the first failure and the full table. Usage:
    python parity_gate.py <ref.pt> <test.pt> [--cos-thresh 0.9995] [--rel-thresh 2e-3] [--exact] \
        [--expect-min-keys N] [--require-decode]
Exit 0 only if every shared key passes AND coverage requirements are met.
"""
import argparse
import sys

import torch


def _norm(k):
    # strip trailing .rN rank suffix but REMEMBER it collided (caller checks)
    import re
    return re.sub(r"\.r-?\d+$", "", k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--cos-thresh", type=float, default=0.9995)
    ap.add_argument("--rel-thresh", type=float, default=2e-3)
    ap.add_argument("--exact", action="store_true", help="require bit/near-exact equality (control-flow port)")
    ap.add_argument("--expect-min-keys", type=int, default=1)
    ap.add_argument("--require-decode", action="store_true", help="fail if no decode (dc*) keys captured")
    args = ap.parse_args()

    def fail(msg):
        print(f"PARITY GATE: FAIL — {msg}", file=sys.stderr)
        sys.exit(2)

    try:
        A = torch.load(args.ref, map_location="cpu"); B = torch.load(args.test, map_location="cpu")
    except Exception as e:
        fail(f"could not load captures: {e!r}")
    if not isinstance(A, dict) or not isinstance(B, dict) or not A or not B:
        fail(f"empty or non-dict capture (ref={len(A) if hasattr(A,'__len__') else '?'}, "
             f"test={len(B) if hasattr(B,'__len__') else '?'})")

    # collision check: two raw keys mapping to the same normalized key on the SAME side
    for name, D in (("ref", A), ("test", B)):
        seen = {}
        for k in D:
            n = _norm(k)
            if n in seen:
                fail(f"{name} duplicate normalized key '{n}' from '{k}' and '{seen[n]}' (ambiguous identity)")
            seen[n] = k
    An = {_norm(k): v for k, v in A.items()}
    Bn = {_norm(k): v for k, v in B.items()}
    shared = sorted(set(An) & set(Bn))
    if len(shared) < args.expect_min_keys:
        fail(f"only {len(shared)} shared keys (< expect-min-keys {args.expect_min_keys}); "
             f"ref-only={sorted(set(An)-set(Bn))[:4]} test-only={sorted(set(Bn)-set(An))[:4]}")
    if args.require_decode and not any(k.startswith("dc") for k in shared):
        fail("no decode (dc*) keys captured — the B=32 decode target was not measured")

    print(f"{'key':<26}{'cos':>10}{'rel_maxerr':>12}{'shape':>16}  verdict")
    worst_cos = 1.0
    for k in shared:
        a = An[k].float(); b = Bn[k].float()
        if a.shape != b.shape:
            fail(f"{k}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        if An[k].dtype != Bn[k].dtype:
            fail(f"{k}: dtype {An[k].dtype} != {Bn[k].dtype}")
        if not torch.isfinite(a).all():
            fail(f"{k}: ref has NaN/Inf")
        if not torch.isfinite(b).all():
            fail(f"{k}: test has NaN/Inf")
        af, bf = a.reshape(-1), b.reshape(-1)
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=0, eps=1e-12).item()
        denom = af.abs().max().item() or 1.0
        rel = (af - bf).abs().max().item() / denom
        worst_cos = min(worst_cos, cos)
        ok = (cos >= args.cos_thresh) and (rel <= args.rel_thresh)
        if args.exact:
            ok = ok and torch.equal(a, b)
        print(f"{k:<26}{cos:10.6f}{rel:12.3e}{str(tuple(a.shape)):>16}  {'OK' if ok else 'FAIL <<<'}")
        if not ok:
            fail(f"{k}: cos={cos:.6f} (<{args.cos_thresh}) or rel={rel:.3e} (>{args.rel_thresh})"
                 + (" or not bit-exact" if args.exact else ""))
    print(f"PARITY GATE: PASS — {len(shared)} shared keys, worst cos={worst_cos:.6f}, "
          f"all finite, shapes/dtypes matched{' , bit-exact' if args.exact else ''}")
    sys.exit(0)


if __name__ == "__main__":
    main()
