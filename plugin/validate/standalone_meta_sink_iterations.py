#!/usr/bin/env python3
"""Attribute serial Sinkhorn recurrence cost without changing its implementation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

import torch

from standalone_meta_dispatch_screen import (
    BASELINE, ROOT, bench, call, check_outputs, digest, select_arm,
)


def run(output, replicate):
    if output.exists():
        raise FileExistsError(output)
    job = os.environ["SLURM_JOB_ID"]
    source = ROOT / "plugin/kernels/dsa_pilot/sinkhorn.cpp"
    original = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASELINE}:{source.relative_to(ROOT)}"]
    )
    if hashlib.sha256(original).hexdigest() != digest(source):
        raise RuntimeError("native baseline changed")
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    select_arm({"SINK_SERIAL": "1"})
    fn = bench._mod("meta_sink_iterations", "sinkhorn.cpp").mhc_sinkhorn
    paths = [source, Path(__file__), Path(__file__).with_name("standalone_meta_dispatch_screen.py"),
             Path(__file__).with_name("run_standalone_meta_sink_iterations.sbatch")]
    result = {
        "scope": "Iteration-count intervention on unchanged serial kernel; NOT an optimization or correctness comparison between iteration counts",
        "baseline": BASELINE, "job": job, "pid": os.getpid(), "replicate": replicate,
        "node": os.uname().nodename, "torch": torch.__version__, "threads": 64,
        "flags": bench.CF, "seed": 0, "sources": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "timing": "Balanced randomized iteration order; fresh outputs allocated/released inside timer; 768MiB add_ before each cold call",
        "environment": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                         "OMP_WAIT_POLICY", "SINK_SERIAL", "TORCH_EXTENSIONS_DIR")},
        "cases": [],
    }
    flush = torch.zeros(192 * 1024 * 1024)
    rng = random.Random(replicate)
    iterations = [1, 2, 5, 10, 20]
    for m in bench.MS:
        mixes, scale, base = torch.randn(1, m, 24), torch.rand(3) + 0.5, torch.randn(24)
        arguments = {it: (mixes, scale, base, 4, it, 1e-6) for it in iterations}
        for args in arguments.values():
            check_outputs(fn(*args), fn(*args))
        record = {"M": m, "hc": 4, "eps": 1e-6, "input_sha256": {
            k: hashlib.sha256(x.numpy().tobytes()).hexdigest()
            for k, x in (("mixes", mixes), ("scale", scale), ("base", base))
        }}
        for mode in ("cold", "warm"):
            samples = {it: [] for it in iterations}
            orders = []
            for args in arguments.values():
                for _ in range(5):
                    fn(*args)
            for _ in range(5):
                rng.shuffle(iterations)
                shifts = list(range(5))
                rng.shuffle(shifts)
                for shift in shifts:
                    order = iterations[shift:] + iterations[:shift]
                    orders.append(order)
                    for it in order:
                        if mode == "cold":
                            flush.add_(1)
                        samples[it].append(call(fn, arguments[it]))
            medians = {it: statistics.median(v) for it, v in samples.items()}
            record[mode] = {"samples_us": samples, "orders": orders, "medians_us": medians}
            print(m, mode, medians, flush=True)
        result["cases"].append(record)
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()
    })
    if result["worker_affinity"] != [(i,) for i in range(64)]:
        raise RuntimeError("unexpected worker affinity")
    binaries = sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("*.so"))
    if len(binaries) != 1:
        raise RuntimeError("expected one private native module")
    result["binaries"] = {str(p): digest(p) for p in binaries}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    run(args.output, args.replicate)
