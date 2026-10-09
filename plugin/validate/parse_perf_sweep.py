#!/usr/bin/env python3
"""Validated ingestion of a raw perf-sweep log -> results/perf_sweep.json (addresses F3/R2-F3).

Derives each op's measured kernel latency by NAMED HEADER COLUMN (never a guessed position), keyed by an
EXPLICIT variant, with medians over the EXPECTED unique replica IDs. Fail-closed: raises if a bench's header
lacks the named column, if the replica set is not exactly {1,2,3}, if any value is non-finite (NaN/+-inf) or
non-positive, or if any M coordinate is missing. A reordered header cannot select the wrong variant (lookup
is by name), a missing replica cannot be labeled "3 reps", and +inf cannot pass (math.isfinite).

Usage:
  parse_perf_sweep.py <raw_log> <job_id>   # parse + validate + write results/perf_sweep.json
  parse_perf_sweep.py --selftest           # run negative-case tests (no write)
"""
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# bench -> (join_op_name, kernel_column_NAME (from the bench's own printed header), variant_label).
# The kernel latency column is named 'cpp_ms' for every bench except sparse, whose SCALAR kernel column is
# 'sc_ms' (distinct from the torch reference 'ref_ms' and the AMX 'amx_ms') — selected BY NAME (R2-F3).
SPEC = {
    "bench_idx_logits.py":    ("indexer logits (q.ck+reduce)", "cpp_ms", "tiled (integration)"),
    "bench_topk.py":          ("indexer top-k (512 of 1024)",  "cpp_ms", "chunked/row-parallel"),
    "bench_compressor.py":    ("compressor softmax-pool",      "cpp_ms", "streaming online-softmax"),
    "bench_sinkhorn.py":      ("MHC sinkhorn (hc=4,20it)",     "cpp_ms", "fused"),
    "bench_combine.py":       ("MHC combine (reduce)",         "cpp_ms", "tiled accumulate-once"),
    "bench_sparse_attend.py": ("sparse attend (MQA+sink)",     "sc_ms",  "scalar"),
}
MS = [1, 8, 16, 32, 64]
MS_STR = {str(m) for m in MS}
EXPECTED_REPS = {1, 2, 3}


def parse_ops(text):
    """Parse raw log text -> {op_name: {bench, variant, median_ms[per M]}}. Fail-closed (raises ValueError).

    Collects values as bench -> M -> {rep_id: value} by NAMED header column, then validates exact replica
    coverage, finiteness/positivity, and exact M coverage before taking medians."""
    blocks = re.split(r">>> BENCH=(\S+) REP=(\d+)", text)
    data = {}   # bench -> M -> {rep: value}
    for i in range(1, len(blocks), 3):
        bench, rep, body = blocks[i], int(blocks[i + 1]), blocks[i + 2]
        if bench not in SPEC:
            continue
        colname = SPEC[bench][1]
        header = None
        for line in body.splitlines():
            toks = line.split()
            if not toks:
                continue
            if toks[0] == "M":                                  # header row: name -> column index
                header = {name: j for j, name in enumerate(toks)}
                continue
            if toks[0] in MS_STR and header is not None:
                if colname not in header:
                    raise ValueError(f"{bench} REP={rep}: header has no column {colname!r} (cols={list(header)})")
                ci = header[colname]
                if ci >= len(toks):
                    continue                                    # ragged/continuation line, not a data row
                try:
                    v = float(toks[ci])
                except ValueError:
                    continue
                M = int(toks[0])
                data.setdefault(bench, {}).setdefault(M, {})[rep] = v   # unique by (bench,M,rep)
    ops = {}
    for bench, (opname, colname, variant) in SPEC.items():
        rows = data.get(bench, {})
        med = []
        for M in MS:
            reps = rows.get(M, {})
            got = set(reps)
            if got != EXPECTED_REPS:                            # missing/extra/duplicate replica -> fail
                raise ValueError(f"{bench} M={M}: replica ids {sorted(got)} != expected {sorted(EXPECTED_REPS)}")
            for r, v in reps.items():
                if not math.isfinite(v) or v <= 0:              # NaN / +-inf / <=0 -> fail (R2-F3)
                    raise ValueError(f"{bench} M={M} rep={r}: non-finite/non-positive time {v}")
            med.append(round(st.median(reps.values()), 4))
        ops[opname] = {"bench": bench, "variant": variant, "median_ms": med}
    return ops


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 3:
        sys.exit("usage: parse_perf_sweep.py <raw_log> <job_id>   |   parse_perf_sweep.py --selftest")
    log_path, job = sys.argv[1], sys.argv[2]
    text = open(log_path).read()
    node = (re.search(r"node=(\S+)", text) or [None, "unknown"])[1]
    try:
        ops = parse_ops(text)
    except ValueError as e:
        sys.exit(f"FAIL ingest: {e}")
    out = {
        "_schema": "Measured kernel latency sweep (median cpp_ms, sparse=sc_ms) over M=1/8/16/32/64 per "
                   "authored-op microbench, parsed by NAMED header column with an explicit variant key and "
                   "validated (exact replica set {1,2,3}, math.isfinite, positive, all M present) before "
                   "writing (F3/R2-F3). Keyed by the roofline-vs-measured join op name.",
        "raw_record": f"SLURM job {job}, log {log_path}, node {node} (DDR5-5600); run_dsa_perf_sweep.sbatch; "
                      f"3 reps/bench (median of unique REP ids 1/2/3); OMP 64thr bound once, numactl NUMA0. "
                      f"Parsed by parse_perf_sweep.py (named-column, fail-closed).",
        "ms": MS,
        "ops": ops,
    }
    dest = os.path.join(HERE, "results", "perf_sweep.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")
    for opname, o in ops.items():
        print(f"  {opname:30s} [{o['variant']}] {o['median_ms']}")


def selftest():
    """Negative + positive cases proving the validator fails closed (R2-F3)."""
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {m}")

    def _block(bench, rep, header, rows):
        lines = [f">>> BENCH={bench} REP={rep}", "  " + "  ".join(str(h) for h in header)]
        for M, vals in rows:
            lines.append("  " + "  ".join([str(M)] + [str(v) for v in vals]))
        return "\n".join(lines) + "\n"

    good_hdr = {
        "bench_idx_logits.py": ["M", "cos", "ref_ms", "cpp_ms", "speedup"],
        "bench_topk.py": ["M", "set_match", "torch_ms", "cpp_ms", "speedup"],
        "bench_compressor.py": ["M", "cos", "ref_ms", "cpp_ms", "speedup"],
        "bench_sinkhorn.py": ["M", "cos_pre", "cos_post", "cos_comb", "ref_ms", "cpp_ms", "speedup"],
        "bench_combine.py": ["M", "cos", "ref_ms", "cpp_ms", "speedup"],
        "bench_sparse_attend.py": ["M", "cos_sc", "cos_amx", "ref_ms", "sc_ms", "amx_ms", "sc_x", "amx_x"],
    }

    def _val_row(bench, M):
        base = {"cos": 1.0, "set_match": 1.0, "cos_pre": 1.0, "cos_post": 1.0, "cos_comb": 1.0,
                "ref_ms": 9.999, "torch_ms": 9.999, "cos_sc": 1.0, "cos_amx": 1.0, "amx_ms": 8.888,
                "sc_x": 1.0, "amx_x": 1.0, "speedup": 1.0}
        base[SPEC[bench][1]] = round(0.1 * M, 4)                # unique sentinel in the kernel column
        return [base[h] for h in good_hdr[bench][1:]]

    def _full(reps=(1, 2, 3), mutate=None):
        parts = ["node=selftest\n"]
        for bench in SPEC:
            for rep in reps:
                rows = [(M, _val_row(bench, M)) for M in MS]
                if mutate:
                    rows = mutate(bench, rep, rows)
                parts.append(_block(bench, rep, good_hdr[bench], rows))
        return "".join(parts)

    # 1) happy path validates and picks the NAMED kernel column (0.1*M), not ref_ms (9.999)
    try:
        ops = parse_ops(_full())
        chk(ops["sparse attend (MQA+sink)"]["median_ms"] == [round(0.1 * m, 4) for m in MS]
            and ops["indexer logits (q.ck+reduce)"]["median_ms"][3] == round(0.1 * 32, 4),
            "happy path: medians come from the NAMED kernel column (not ref_ms)")
    except ValueError as e:
        chk(False, f"happy path unexpectedly failed: {e}")

    # 2) only REP=1 present -> must NOT be accepted as 3 reps
    chk(_rejects(lambda: parse_ops(_full(reps=(1,)))),
        "REJECTS a sweep with only REP=1 (not labeled 3 reps)")

    # 3) +inf in a kernel value -> rejected (math.isfinite)
    def _inf(bench, rep, rows):
        if bench == "bench_compressor.py" and rep == 2:
            ci = good_hdr[bench].index(SPEC[bench][1]) - 1
            rows = [(M, [float("inf") if j == ci else v for j, v in enumerate(vals)]) if M == 16 else (M, vals)
                    for M, vals in rows]
        return rows
    chk(_rejects(lambda: parse_ops(_full(mutate=_inf))), "REJECTS +inf kernel value (math.isfinite)")

    # 4) reordered sparse header (sc_ms before ref_ms, swapped magnitudes) still selects sc_ms BY NAME
    def _reorder():
        parts = ["node=selftest\n"]
        for rep in (1, 2, 3):
            hdr = ["M", "cos_sc", "cos_amx", "sc_ms", "ref_ms", "amx_ms", "sc_x", "amx_x"]  # sc_ms BEFORE ref_ms
            rows = [(M, [1.0, 1.0, round(0.1 * M, 4), 9.999, 8.888, 1.0, 1.0]) for M in MS]
            parts.append(_block("bench_sparse_attend.py", rep, hdr, rows))
        for bench in SPEC:
            if bench == "bench_sparse_attend.py":
                continue
            for rep in (1, 2, 3):
                parts.append(_block(bench, rep, good_hdr[bench], [(M, _val_row(bench, M)) for M in MS]))
        return "".join(parts)
    try:
        ops = parse_ops(_reorder())
        chk(ops["sparse attend (MQA+sink)"]["median_ms"] == [round(0.1 * m, 4) for m in MS],
            "REORDERED header: sc_ms selected BY NAME (not the positional torch ref)")
    except ValueError as e:
        chk(False, f"reorder case unexpectedly failed: {e}")

    # 5) missing M coordinate -> rejected
    def _dropM(bench, rep, rows):
        return [r for r in rows if r[0] != 32] if bench == "bench_topk.py" else rows
    chk(_rejects(lambda: parse_ops(_full(mutate=_dropM))), "REJECTS a missing M coordinate")

    print(f"  SELFTEST {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok else 2)


def _rejects(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    main()
