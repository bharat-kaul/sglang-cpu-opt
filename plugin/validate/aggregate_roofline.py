#!/usr/bin/env python3
"""Aggregate the per-process roofline sweep JSONs (bench_roofline_sweep.py output) into the published,
self-identifying results/roofline_sweep.json. Committed so the aggregate is reproducible/auditable (R6-F2).

VALIDATES its inputs before combining or writing (R7-F1) -- metadata presence is not enough:
  - the declared replicate count (default 3) must match the number of inputs;
  - NEW-format inputs must each carry `_provenance` with a DISTINCT `process_index` and COMPATIBLE
    run/source/runtime contracts (slurm_job_id, head, threads, omp, bind); mixed runs are rejected;
  - every input must cover all required (kernel, M) coordinates with FINITE POSITIVE latencies;
  - fail-closed: on any violation it raises SystemExit and writes NO output.
Historical UNSTAMPED scratch files are only accepted under an EXPLICIT, separately-qualified `--legacy`
path (status = LEGACY_RECONSTRUCTED, provenance NOT producer-stamped); a stamped input under --legacy, or an
unstamped input WITHOUT --legacy, is rejected.

Usage: python aggregate_roofline.py <p1.json> ... [--replicates N] [--legacy] [--out <path>]
       python aggregate_roofline.py --selftest
"""
import argparse
import json
import math
import statistics as st
import sys

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
_CONTRACT_KEYS = ("slurm_job_id", "head", "threads", "omp", "bind")


def _kernels(blob):
    return blob["kernels"] if isinstance(blob, dict) and "kernels" in blob else blob


def _finite_pos(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def aggregate(procs, expect_replicates=3, legacy=False):
    """Validate + aggregate a list of loaded per-process blobs. Raises SystemExit (writes nothing) on any
    violation. Returns the published aggregate dict."""
    if len(procs) != expect_replicates:
        raise SystemExit(f"FATAL: expected {expect_replicates} process replicates, got {len(procs)}")
    provs = [p.get("_provenance") if isinstance(p, dict) else None for p in procs]
    if legacy:
        if any(provs):
            raise SystemExit("FATAL: --legacy inputs must be UNSTAMPED (no _provenance); got a stamped input")
        status = "LEGACY_RECONSTRUCTED: unstamped scratch inputs; top-level provenance reconstructed from the raw log, NOT producer-stamped"
        process_ids = None
    else:
        if not all(provs):
            raise SystemExit("FATAL: non-legacy inputs must each carry _provenance (use --legacy for historical unstamped files)")
        base = {k: provs[0].get(k) for k in _CONTRACT_KEYS}
        for i, pr in enumerate(provs):
            bad = {k: pr.get(k) for k in _CONTRACT_KEYS if pr.get(k) != base[k]}
            if bad:
                raise SystemExit(f"FATAL: process {i} has an INCOMPATIBLE run contract {bad} != base {base}")
        process_ids = [pr.get("process_index") for pr in provs]
        if any(x is None for x in process_ids) or len(set(process_ids)) != len(process_ids):
            raise SystemExit(f"FATAL: process replicates must have DISTINCT process_index; got {process_ids}")
        status = "VALIDATED"
    ks = [_kernels(p) for p in procs]
    for i, k in enumerate(ks):                                 # complete coordinates + finite positive latencies
        if not isinstance(k, dict):
            raise SystemExit(f"FATAL: process {i} has no kernels map")
        for lb in CONTRACTS:
            if lb not in k:
                raise SystemExit(f"FATAL: process {i} missing required kernel {lb}")
            for M in MS:
                cell = k[lb].get(str(M))
                if not isinstance(cell, dict) or "meas_us" not in cell:
                    raise SystemExit(f"FATAL: process {i} {lb} missing M={M}")
                if not _finite_pos(cell["meas_us"]):
                    raise SystemExit(f"FATAL: process {i} {lb} M={M} meas_us {cell['meas_us']!r} not finite positive")
    out = {"_schema": "Performance-vs-roofline M-sweep. Per (kernel,M): process medians + [min,max] spread, "
                      "UNROUNDED floor = max(bytes/BW, flops/peak) vs NOMINAL reference peak (1.9GHz base, NOT a "
                      "measured ceiling), achieved_pct = floor/median of the NOMINAL reference (pre-rounding). "
                      "regime = larger theoretical term (diagnostic, not a measured bottleneck; cache rows not "
                      "DRAM-saturated). Simplified FLOPs (compressor omits exp). Timing-only; FP32-KV indexer "
                      "contract (does NOT isolate I1's incremental effect).",
           "provenance": {"status": status, "replicates": len(procs), "process_ids": process_ids,
                          "process_provenance": provs, "aggregator": "aggregate_roofline.py"},
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


def _mk_proc(job="384677", head="aaccef6", threads=64, pidx=0, nan_topk=False):
    k = {lb: {str(M): {"meas_us": (float("nan") if (nan_topk and lb == "indexer_topk" and M == 1) else 10.0 + M)} for M in MS}
         for lb in CONTRACTS}
    return {"_provenance": {"slurm_job_id": job, "head": head, "threads": threads, "omp": "64", "bind": "close",
                            "process_index": pidx}, "kernels": k}


def selftest():
    ok = True

    def _acc(procs, **kw):
        try:
            aggregate(procs, **kw); return True
        except SystemExit:
            return False

    def chk(c, m):
        nonlocal ok; ok = ok and bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    good = [_mk_proc(pidx=i) for i in range(3)]
    chk(_acc(good), "ACCEPTS 3 matching stamped process replicates (positive)")
    chk(not _acc(good[:1]), "REJECTS fewer than the declared replicate count (R7-F1)")
    conflict = [_mk_proc(pidx=0), _mk_proc(job="different-run", head="53f1e57", threads=1, pidx=1), _mk_proc(pidx=2)]
    chk(not _acc(conflict), "REJECTS incompatible run/source/thread contracts (R7-F1)")
    dup = [_mk_proc(pidx=0), _mk_proc(pidx=0), _mk_proc(pidx=0)]
    chk(not _acc(dup), "REJECTS duplicate (non-distinct) process_index (R7-F1)")
    nan = [_mk_proc(pidx=i, nan_topk=True) for i in range(3)]
    chk(not _acc(nan), "REJECTS non-finite (NaN) latencies (R7-F1)")
    legacy = [{"kernels": _mk_proc(pidx=i)["kernels"]} for i in range(3)]   # unstamped
    chk(_acc(legacy, legacy=True), "ACCEPTS 3 unstamped inputs under --legacy (positive)")
    chk(not _acc(legacy), "REJECTS unstamped inputs WITHOUT --legacy (R7-F1)")
    chk(not _acc(good, legacy=True), "REJECTS stamped inputs UNDER --legacy (R7-F1)")
    incomplete = [_mk_proc(pidx=i) for i in range(3)]
    del incomplete[1]["kernels"]["sparse/bestof"]
    chk(not _acc(incomplete), "REJECTS an input missing a required kernel coordinate (R7-F1)")
    print(f"  AGGREGATE-SELFTEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("procs", nargs="*")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.procs:
        ap.error("no process JSONs given")
    procs = [json.load(open(p)) for p in a.procs]
    agg = aggregate(procs, expect_replicates=a.replicates, legacy=a.legacy)   # raises + writes NOTHING on reject
    if a.out:
        json.dump(agg, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    else:
        print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
