#!/usr/bin/env python3
"""Offline cosine+magnitude per-(pass,op,layer) diff of two FULLCAP .pt dumps.

Keys look like ``pf.L12.r3`` / ``pf.hcpost.L7.r3`` / ``logits.0.r3``. The rank suffix
(``.rN``) is stripped so a CPU rank-0 dump lines up with a TP-sharded GPU dump on the
UNSHARDED signals (decoder-layer hidden, hc_post, logits are all rank-replicated here).

Prints, in layer order, cosine + relative max-abs-err + magnitude ratio, and flags the
FIRST op whose cosine drops below --cos-thresh (default 0.999) as the divergence point.

    python diff_fullcap.py <gpu.pt> <cpu.pt> [--cos-thresh 0.999]
"""
import argparse
import re

import torch


def _strip_rank(k: str) -> str:
    return re.sub(r"\.r-?\d+$", "", k)


def _load(path: str) -> dict:
    d = torch.load(path, map_location="cpu")
    out = {}
    for k, v in d.items():
        out[_strip_rank(k)] = v.float()
    return out


def _sortkey(k: str):
    # Order by pass (pf before dc), then layer id, then op (hidden before hcpost), logits last.
    m = re.match(r"(pf|dc\d+)\.(?:(hcpost)\.)?L(-?\d+)", k)
    if m:
        ps = 0 if m.group(1) == "pf" else 1 + int(m.group(1)[2:])
        op = 1 if m.group(2) == "hcpost" else 0
        return (0, ps, int(m.group(3)), op)
    m = re.match(r"logits\.(\d+)", k)
    if m:
        return (1, 0, int(m.group(1)), 0)
    return (2, 0, 0, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("cpu")
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    args = ap.parse_args()

    g = _load(args.gpu)
    c = _load(args.cpu)
    shared = sorted(set(g) & set(c), key=_sortkey)
    only_g = sorted(set(g) - set(c), key=_sortkey)
    only_c = sorted(set(c) - set(g), key=_sortkey)

    print(f"GPU keys={len(g)} CPU keys={len(c)} shared={len(shared)}")
    if only_g:
        print(f"ONLY in GPU ({len(only_g)}): {only_g[:12]}{' ...' if len(only_g) > 12 else ''}")
    if only_c:
        print(f"ONLY in CPU ({len(only_c)}): {only_c[:12]}{' ...' if len(only_c) > 12 else ''}")
    print(f"{'key':<24} {'cos':>9} {'rel_maxerr':>11} {'mag_ratio':>10} {'shape':>18}")

    first_div = None
    for k in shared:
        a, b = g[k], c[k]
        if a.shape != b.shape:
            print(f"{k:<24} {'SHAPE':>9} {str(tuple(a.shape)):>11} {str(tuple(b.shape)):>10}  MISMATCH")
            if first_div is None:
                first_div = (k, "shape")
            continue
        af, bf = a.reshape(-1), b.reshape(-1)
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=0, eps=1e-12).item()
        denom = af.abs().max().clamp_min(1e-12)
        rel = ((af - bf).abs().max() / denom).item()
        mag = (bf.abs().max().clamp_min(1e-12) / denom).item()
        flag = "" if cos >= args.cos_thresh else "  <-- DIVERGE"
        if cos < args.cos_thresh and first_div is None:
            first_div = (k, cos)
        print(f"{k:<24} {cos:9.6f} {rel:11.3e} {mag:10.4f} {str(tuple(a.shape)):>18}{flag}")

    print()
    if first_div is None:
        print(f"PARITY OK: all {len(shared)} shared ops cos >= {args.cos_thresh}")
    else:
        print(f"FIRST DIVERGENCE: {first_div[0]} (cos={first_div[1]})")


if __name__ == "__main__":
    main()
