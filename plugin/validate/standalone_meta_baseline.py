#!/usr/bin/env python3
"""Run the pinned published benchmark, retaining raw samples and build identity."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

import bench_operating_point as bench


BASELINE = "76cbcff6e0bcd81d0320262e5e9f39e723144089"
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(os.environ["OP_OUT"])
    if output.exists() or output.with_suffix(".identity.json").exists():
        raise FileExistsError(f"refusing to overwrite baseline artifacts: {output}")
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != BASELINE:
        raise RuntimeError(f"expected pinned baseline {BASELINE}, got {revision}")
    source_paths = sorted((ROOT / "plugin/kernels/dsa_pilot").glob("*.cpp"))
    source_paths += sorted((ROOT / "plugin/kernels/dsa_pilot").glob("*.h"))
    source_paths += [
        Path(bench.__file__),
        ROOT / "plugin/validate/bench_replicated.py",
        ROOT / "plugin/validate/sparse_ref.py",
        Path(__file__),
        Path(__file__).with_name("run_standalone_meta_baseline.sbatch"),
    ]
    identity = {
        "baseline_revision": revision,
        "job": os.environ["SLURM_JOB_ID"],
        "process": os.getpid(),
        "replicate": int(os.environ["META_REPLICATE"]),
        "node": platform.node(),
        "started_ns": time.time_ns(),
        "torch": torch.__version__,
        "python": sys.version,
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "seed": 0,
        "flags": bench.CF,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
            "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME",
            "TORCH_EXTENSIONS_DIR", "MAX_JOBS",
        )},
        "scope": "Historical protocol reproduction, not paired candidate acceptance",
        "sample_groups": [],
    }
    original_call = bench._call
    original_warm = bench.twarm
    original_stream = bench.tstream
    current = None

    def measured_call(fn, args):
        elapsed = original_call(fn, args)
        if current is None:
            raise RuntimeError("timing sample outside a declared measurement group")
        current["samples_seconds"].append(elapsed)
        return elapsed

    def measure(original, cache):
        def wrapped(fn, args, **kwargs):
            nonlocal current
            ordinal = len(identity["sample_groups"])
            kernels = tuple(bench.KC)
            current = {
                "ordinal": ordinal,
                "kernel": kernels[ordinal // (len(bench.MS) * 4)],
                "M": bench.MS[(ordinal // 4) % len(bench.MS)],
                "arm": "cpp" if ordinal % 4 < 2 else "torch",
                "cache": cache,
                "shapes": [list(a.shape) if isinstance(a, torch.Tensor) else a for a in args],
                "samples_seconds": [],
            }
            identity["sample_groups"].append(current)
            try:
                return original(fn, args, **kwargs)
            finally:
                current = None
        return wrapped

    bench._call = measured_call
    bench.twarm = measure(original_warm, "warm")
    bench.tstream = measure(original_stream, "LLC-flushed")
    torch.manual_seed(0)
    bench.main()
    if len(identity["sample_groups"]) != len(bench.KC) * len(bench.MS) * 4:
        raise RuntimeError("incomplete benchmark sample coverage")
    for group in identity["sample_groups"]:
        samples = group["samples_seconds"]
        expected = 40 if group["cache"] == "warm" else 25
        if len(samples) != expected or not all(math.isfinite(x) and x > 0 for x in samples):
            raise RuntimeError(f"invalid timing samples: group {group['ordinal']}")
    identity["binaries"] = {
        str(path): digest(path)
        for path in sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("*.so"))
    }
    if not identity["binaries"]:
        raise RuntimeError("no private compiled kernels found")
    identity["completed_ns"] = time.time_ns()
    output.with_suffix(".identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
