#!/usr/bin/env python3
"""Combine N operating-point replicate JSONs into a cross-run MEDIAN operating_point.json.
Medians the timing fields per (kernel, M); recomputes speedup_stream and pct_roof_stream from the medians.
Usage: python combine_op_reps.py out.json rep0.json rep1.json rep2.json ..."""
import json
import statistics as st
import sys

out_path, rep_paths = sys.argv[1], sys.argv[2:]
reps = [json.load(open(p)) for p in rep_paths]
base = reps[0]
base["_schema"] = base.get("_schema", "") + f" [cross-run MEDIAN of {len(reps)} replicates]"
base["replicates"] = len(reps)
TIMING = ["cpp_warm_us", "cpp_stream_us", "torch_warm_us", "torch_stream_us"]
for kname, kobj in base["kernels"].items():
    for m, cell in kobj["M"].items():
        for f in TIMING:
            vals = [r["kernels"][kname]["M"][m][f] for r in reps if f in r["kernels"][kname]["M"][m]]
            cell[f] = round(st.median(vals), 1)
        cs, ts = cell["cpp_stream_us"], cell["torch_stream_us"]
        cell["speedup_stream"] = round(ts / cs, 2) if cs else 0
        cell["pct_roof_stream"] = round(100 * cell["floor_us"] / cs, 1) if cs else 0
json.dump(base, open(out_path, "w"), indent=1)
print(f"wrote {out_path} (median of {len(reps)} reps)")
