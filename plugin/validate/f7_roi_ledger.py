#!/usr/bin/env python3
"""F7 — provisional Amdahl ROI ledger (authored kernels). PICKS the optimization target; it is NOT an
in-engine measurement. Per authored op: per-token contribution = standalone-microbench median latency x the
roofline model's per-token CALL COUNT, the off-roofline ratio (measured/ideal), and RoI = share*(1-1/off).
CAVEATS (read): (1) medians are STANDALONE microbench calls, NOT in-engine time (each in-engine call also
pays dispatch/barrier/cache context) -> absolute shares are provisional; (2) only AUTHORED kernels are
ranked (projections/dense GEMMs/MoE/norm excluded) -> this is a RELATIVE ranking among authored ops, not a
model-wide fraction; (3) compressor R=8 pools (the bulk of its call weight) are UNMEASURED -> flagged. Use it
to ORDER the optimization passes, then confirm with real in-engine profiling before any exhaustion claim.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dsv4_roofline_vs_measured import OPS as JOPS, BW, CPEAK  # ideal floor per authored op

_PS = json.load(open(os.path.join(HERE, "results", "perf_sweep.json")))

# authored op (join name) -> per-token CALL COUNT from the roofline cost model (.layers), + coverage note
CALLS = {
    "indexer logits (q.ck+reduce)": (21.0, "measured @ S=1024"),
    "indexer top-k (512 of 1024)": (21.0, "measured"),
    "compressor softmax-pool": (0.15625, "ONLY R=128/D=512 measured; R=8 pools (calls~10.5) UNMEASURED"),
    "sparse attend (MQA+sink)": (1.0, "row aggregates window+compressed 43L; K=512 pre-gathered"),
    "MHC sinkhorn (hc=4,20it)": (86.0, "measured"),
    "MHC combine (reduce)": (86.0, "measured"),
}
IDEAL = {n: (flf, byf, cdt) for (n, flf, byf, cdt, _rec, _pl) in JOPS}


def ideal_us(name, M):
    flf, byf, cdt = IDEAL[name]
    fl, by = flf(M), byf(M)
    return max(by / BW, (fl / CPEAK[cdt] if fl else 0.0)) * 1e6


def ledger(M):
    rows = []
    for name, (calls, note) in CALLS.items():
        med = _PS["ops"].get(name, {}).get("median_ms")
        if not med:
            continue
        Mi = _PS["ms"].index(M)
        meas_us = med[Mi] * 1e3
        contrib_us = meas_us * calls                       # per-token (provisional)
        ideal = ideal_us(name, M)
        off = meas_us / ideal if ideal else float("inf")
        rows.append({"op": name, "calls": calls, "meas_us_per_call": round(meas_us, 3),
                     "contrib_us_per_token": round(contrib_us, 3), "off_roofline": round(off, 1),
                     "note": note})
    tot = sum(r["contrib_us_per_token"] for r in rows)
    for r in rows:
        r["share"] = round(r["contrib_us_per_token"] / tot, 4) if tot else 0.0
        r["roi"] = round(r["share"] * (1 - 1 / r["off_roofline"]), 4)   # heavy AND far-from-roof -> high RoI
    rows.sort(key=lambda r: -r["roi"])
    return rows, tot


def main():
    out = {"_schema": "F7 provisional Amdahl ROI ledger (authored kernels; standalone-microbench x call-count, "
                      "NOT in-engine). Ranks the optimization order; confirm with real profiling before any "
                      "ROI-exhaustion claim. Compressor R=8 pools UNMEASURED.",
           "raw_record": _PS["raw_record"], "levels": {}}
    for M in (1, 32):
        rows, tot = ledger(M)
        out["levels"][f"M={M}"] = {"total_authored_us_per_token": round(tot, 2), "ranked": rows}
        print(f"\n=== F7 authored-kernel ROI ledger @ M={M}  (total authored {tot:.1f} us/token, PROVISIONAL) ===")
        print(f"  {'op':30s} {'calls':>7} {'us/call':>9} {'us/tok':>9} {'share':>6} {'off_roof':>9} {'RoI':>6}")
        for r in rows:
            print(f"  {r['op'][:30]:30s} {r['calls']:>7} {r['meas_us_per_call']:>9.2f} "
                  f"{r['contrib_us_per_token']:>9.1f} {r['share']*100:>5.1f}% {r['off_roofline']:>8.1f}x {r['roi']:>6.3f}")
    dest = os.path.join(HERE, "results", "f7_roi_ledger.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n  wrote {dest}")
    print("  CAVEAT: provisional (standalone x call-count, not in-engine); authored ops only; compressor R=8 "
          "UNMEASURED. Confirm the top target with in-engine profiling before locking the pass order.")


if __name__ == "__main__":
    main()
