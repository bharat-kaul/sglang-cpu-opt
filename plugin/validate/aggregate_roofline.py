#!/usr/bin/env python3
"""Aggregate the per-process roofline sweep JSONs (bench_roofline_sweep.py output) into the published,
self-identifying results/roofline_sweep.json. Committed so the aggregate is reproducible/auditable (R6-F2):
embeds provenance, the 3 PROCESS samples + [min,max] spread, UNROUNDED floors recomputed from the byte/FLOP
contracts, and achieved_pct computed BEFORE display rounding (fraction of the NOMINAL reference, not a ceiling).

Usage: python aggregate_roofline.py <p1.json> <p2.json> <p3.json> [--out results/roofline_sweep.json]
"""
import argparse
import json
import statistics as st

BW = 358.4e9; AMX = 124.5184e12; FP32 = 7.7824e12; B4 = 4.0; I8 = 8.0
H, S, D = 64, 1024, 128
# (flops(M), bytes(M), peak) -- identical contracts to bench_roofline_sweep.py (simplified FLOPs; compressor omits exp)
CONTRACTS = {
    "indexer_logits/tiled": (lambda M: 2 * M * H * S * D + 3 * M * H * S, lambda M: M * H * D * B4 + M * S * D * B4 + M * H * B4 + M * S * B4, AMX),
    "indexer_topk": (lambda M: 0.0, lambda M: M * S * B4 + M * 512 * I8, FP32),
    "compressor/r128d512": (lambda M: 3.0 * M * 128 * 512, lambda M: M * 128 * 512 * B4 * 2 + M * 512 * B4 + 128 * 512 * B4, FP32),
    "compressor/r8d512": (lambda M: 3.0 * M * 8 * 512, lambda M: M * 8 * 512 * B4 * 2 + M * 512 * B4 + 8 * 512 * B4, FP32),
    "compressor/r8d128": (lambda M: 3.0 * M * 8 * 128, lambda M: M * 8 * 128 * B4 * 2 + M * 128 * B4 + 8 * 128 * B4, FP32),
    "sparse/bestof": (lambda M: 4.0 * M * 64 * 512 * 512, lambda M: M * 512 * 512 * B4 + 2 * M * 64 * 512 * B4, FP32),
    "sinkhorn": (lambda M: float(M * 4 * 4 * 20 * 5), lambda M: M * 24 * B4 + 3 * B4 + 24 * B4 + M * 24 * B4, FP32),
    "combine": (lambda M: float(M * 4096 * 7), lambda M: M * 4 * 4096 * B4 + M * 4 * B4 + M * 4096 * B4, FP32),
}
MS = [1, 8, 16, 32, 64]


def _kernels(blob):
    return blob["kernels"] if isinstance(blob, dict) and "kernels" in blob else blob


def aggregate(paths):
    procs = [json.load(open(p)) for p in paths]
    provs = [p.get("_provenance") for p in procs if isinstance(p, dict) and "_provenance" in p]
    ks = [_kernels(p) for p in procs]
    out = {"_schema": "Performance-vs-roofline M-sweep. Per (kernel,M): 3 process medians + [min,max] spread, "
                      "UNROUNDED floor = max(bytes/BW, flops/peak) vs NOMINAL reference peak (1.9GHz base, NOT a "
                      "measured ceiling), achieved_pct = floor/median of the NOMINAL reference (pre-rounding). "
                      "regime = larger theoretical term (diagnostic, not a measured bottleneck; cache rows not "
                      "DRAM-saturated). Simplified FLOPs (compressor omits exp). Timing-only; FP32-KV indexer "
                      "contract (does NOT isolate I1's incremental effect).",
           "provenance": {"process_provenance": provs, "aggregator": "aggregate_roofline.py",
                          "inputs": list(paths)},
           "kernels": {}}
    for lb, (flf, byf, pk) in CONTRACTS.items():
        out["kernels"][lb] = {}
        for M in MS:
            vals = [k[lb][str(M)]["meas_us"] for k in ks]
            med = st.median(vals)
            fl = max(byf(M) / BW, flf(M) / pk) * 1e6
            reg = "BW" if byf(M) / BW >= flf(M) / pk else "compute"
            out["kernels"][lb][str(M)] = {
                "process_meas_us": vals, "median_meas_us": round(med, 1),
                "spread_us": [round(min(vals), 1), round(max(vals), 1)],
                "floor_us": round(fl, 4), "achieved_pct": round(100.0 * fl / med, 2), "regime": reg}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("procs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    agg = aggregate(a.procs)
    if a.out:
        json.dump(agg, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    else:
        print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
