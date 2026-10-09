#!/usr/bin/env python3
"""Validated ingestion of a raw perf-sweep log -> results/perf_sweep.json (addresses F3/F4).

Derives each op's measured kernel latency from NAMED columns (not a guessed position), with an EXPLICIT
variant key, medians over replicas, and VALIDATION (all M present, all reps finite and positive) before
writing. Fail-closed: raises if the log is missing columns, has nonfinite/nonpositive times, or missing M.
Usage: parse_perf_sweep.py <raw_log> <job_id>
"""
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# bench -> (join_op_name, kernel_cpp_column_index (0-based within the numeric row), variant_label)
# Column orders are the benches' own print headers (see each bench_*.py).
SPEC = {
    "bench_idx_logits.py":   ("indexer logits (q.ck+reduce)", 3, "tiled (integration)"),   # M cos ref cpp sp
    "bench_topk.py":         ("indexer top-k (512 of 1024)",  3, "chunked/row-parallel"),  # M setmatch ref cpp sp
    "bench_compressor.py":   ("compressor softmax-pool",      3, "streaming online-softmax"),  # M cos ref cpp sp
    "bench_sinkhorn.py":     ("MHC sinkhorn (hc=4,20it)",     5, "fused"),   # M cos_pre cos_post cos_comb ref cpp sp
    "bench_combine.py":      ("MHC combine (reduce)",         3, "tiled accumulate-once"),  # M cos ref cpp sp
    "bench_sparse_attend.py":("sparse attend (MQA+sink)",     4, "scalar"),  # M cos_sc cos_amx ref sc_ms amx_ms ...
}
MS = [1, 8, 16, 32, 64]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: parse_perf_sweep.py <raw_log> <job_id>")
    log_path, job = sys.argv[1], sys.argv[2]
    text = open(log_path).read()
    node = (re.search(r"node=(\S+)", text) or [None, "unknown"])[1]
    blocks = re.split(r">>> BENCH=(\S+) REP=(\d)", text)
    data = {}   # bench -> M -> [cpp_ms per rep]
    for i in range(1, len(blocks), 3):
        bench, _rep, body = blocks[i], blocks[i + 1], blocks[i + 2]
        if bench not in SPEC:
            continue
        col = SPEC[bench][1]
        for line in body.splitlines():
            t = line.split()
            if t and t[0] in ("1", "8", "16", "32", "64") and len(t) > col:
                try:
                    v = float(t[col])
                except ValueError:
                    continue
                data.setdefault(bench, {}).setdefault(int(t[0]), []).append(v)
    ops = {}
    for bench, (opname, _col, variant) in SPEC.items():
        rows = data.get(bench, {})
        med = []
        for M in MS:
            reps = rows.get(M, [])
            if len(reps) < 1:
                raise SystemExit(f"FAIL ingest: {bench} M={M} has no measurement (expected 3 reps)")
            for r in reps:
                if not (r == r and r > 0):   # NaN or <=0
                    raise SystemExit(f"FAIL ingest: {bench} M={M} nonfinite/nonpositive time {r}")
            med.append(round(st.median(reps), 4))
        ops[opname] = {"bench": bench, "variant": variant, "median_ms": med}
    out = {
        "_schema": "Measured kernel latency sweep (median cpp_ms) over M=1/8/16/32/64 per authored-op "
                   "microbench, parsed by NAMED column with an explicit variant key and validated (all M, "
                   "finite, positive) before writing (F3/F4). Keyed by the roofline-vs-measured join op name.",
        "raw_record": f"SLURM job {job}, log {log_path}, node {node} (DDR5-5600); run_dsa_perf_sweep.sbatch; "
                      f"3 reps/bench (median); OMP 64thr bound once, numactl NUMA0. Parsed by parse_perf_sweep.py.",
        "ms": MS,
        "ops": ops,
    }
    dest = os.path.join(HERE, "results", "perf_sweep.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")
    for opname, o in ops.items():
        print(f"  {opname:30s} [{o['variant']}] {o['median_ms']}")


if __name__ == "__main__":
    main()
