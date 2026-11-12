#!/usr/bin/env python3
"""Screen existing dispatch knobs; do not change kernels or serving defaults."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

import bench_operating_point as bench


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "76cbcff6e0bcd81d0320262e5e9f39e723144089"
KNOBS = ("TOPK_SERIAL", "TOPK_CHUNKS", "TOPK_SKIP_SELECT",
         "SINK_SERIAL", "SINK_PARALLEL", "SINK_GRAIN")
ARMS = {
    "topk": {"default": {}, "serial": {"TOPK_SERIAL": "1"}},
    "sinkhorn": {"default": {}, "grain1": {"SINK_GRAIN": "1"},
                 "grain4": {"SINK_GRAIN": "4"}, "grain16": {"SINK_GRAIN": "16"}},
}


def select_arm(settings):
    for name in KNOBS:
        os.environ.pop(name, None)
    os.environ.update(settings)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_outputs(expected, actual):
    if isinstance(expected, torch.Tensor):
        expected, actual = [expected], [actual]
    if len(expected) != len(actual):
        raise AssertionError("output arity changed")
    for left, right in zip(expected, actual):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError("output contract changed")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError("nonfinite output")
        if not torch.equal(left, right):
            raise AssertionError("existing dispatch paths differ")


def call(fn, inputs):
    start = time.perf_counter_ns()
    value = fn(*inputs)
    del value
    return (time.perf_counter_ns() - start) / 1000


def validate_case(name, fn, inputs):
    select_arm({})
    expected = fn(*inputs)
    for settings in ARMS[name].values():
        select_arm(settings)
        check_outputs(expected, fn(*inputs))
    select_arm({})


def run(output, replicate):
    if output.exists():
        raise FileExistsError(output)
    job = os.environ["SLURM_JOB_ID"]
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip() != BASELINE:
        raise RuntimeError("baseline revision changed")
    paths = [
        ROOT / "plugin/kernels/dsa_pilot/indexer_topk.cpp",
        ROOT / "plugin/kernels/dsa_pilot/sinkhorn.cpp",
        Path(bench.__file__),
        ROOT / "plugin/validate/bench_replicated.py",
        ROOT / "plugin/validate/sparse_ref.py",
    ]
    for path in paths:
        original = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BASELINE}:{path.relative_to(ROOT)}"]
        )
        if hashlib.sha256(original).hexdigest() != digest(path):
            raise RuntimeError(f"baseline source changed: {path}")
    paths += [Path(__file__), Path(__file__).with_name("run_standalone_meta_dispatch.sbatch")]
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    topk = bench._mod("meta_dispatch_topk", "indexer_topk.cpp").indexer_topk
    sink = bench._mod("meta_dispatch_sink", "sinkhorn.cpp").mhc_sinkhorn
    functions = {"topk": topk, "sinkhorn": sink}
    cases = []
    controls = []
    for m in bench.MS:
        cases.extend([
            ("topk", m, (torch.randn(m, 1024), 512)),
            ("sinkhorn", m, (torch.randn(1, m, 24), torch.rand(3) + 0.5,
                            torch.randn(24), 4, 20, 1e-6)),
        ])
    for m, s, k in ((2, 128, 64), (8, 4096, 512), (2, 257, 17),
                    (2, 128, 0), (2, 128, 128)):
        for distribution in ("signed", "ties"):
            logits = torch.randn(m, s) if distribution == "signed" else torch.zeros(m, s)
            validate_case("topk", topk, (logits, k))
            controls.append({"kernel": "topk", "M": m, "S": s, "K": k,
                             "distribution": distribution})
    for rows in (511, 512, 513):
        args = (torch.randn(1, rows, 24), torch.rand(3) + 0.5,
                torch.randn(24), 4, 20, 1e-6)
        validate_case("sinkhorn", sink, args)
        controls.append({"kernel": "sinkhorn", "rows": rows, "hc": 4})
    for name, _, inputs in cases:
        validate_case(name, functions[name], inputs)
    result = {
        "scope": "Existing-knob feasibility screen; equality to current native baseline, not source-oracle certification",
        "baseline": BASELINE, "job": job, "pid": os.getpid(), "replicate": replicate,
        "node": os.uname().nodename, "torch": torch.__version__, "threads": 64,
        "flags": bench.CF, "seed": 0, "arms": ARMS, "controls": controls,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "timing": "Fresh output allocation/release timed; 768MiB add_ eviction before each cold call, outside timer; balanced randomized arm order",
        "environment": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY",
                         "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "TORCH_EXTENSIONS_DIR")},
        "cases": [],
    }
    flush = torch.zeros(192 * 1024 * 1024)
    rng = random.Random(replicate)
    for name, m, inputs in cases:
        fn = functions[name]
        record = {"kernel": name, "M": m,
                  "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype),
                              "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest()}
                             if isinstance(x, torch.Tensor) else x for x in inputs]}
        for mode in ("cold", "warm"):
            arms = list(ARMS[name])
            samples = {arm: [] for arm in arms}
            orders = []
            for arm in arms:
                select_arm(ARMS[name][arm])
                for _ in range(5):
                    fn(*inputs)
            # Every arm occupies each order position equally, with shuffled blocks.
            for _ in range(8):
                rng.shuffle(arms)
                shifts = list(range(len(arms)))
                rng.shuffle(shifts)
                for shift in shifts:
                    order = arms[shift:] + arms[:shift]
                    orders.append(order)
                    for arm in order:
                        select_arm(ARMS[name][arm])
                        if mode == "cold":
                            flush.add_(1)
                        samples[arm].append(call(fn, inputs))
            if not all(math.isfinite(x) and x > 0 for v in samples.values() for x in v):
                raise RuntimeError("invalid sample")
            record[mode] = {"orders": orders, "samples_us": samples,
                            "medians_us": {a: statistics.median(v) for a, v in samples.items()}}
            print(name, m, mode, record[mode]["medians_us"], flush=True)
        result["cases"].append(record)
    select_arm({})
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()
    })
    if result["worker_affinity"] != [(i,) for i in range(64)]:
        raise RuntimeError("unexpected worker affinity")
    binaries = sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("*.so"))
    if len(binaries) != 2:
        raise RuntimeError("expected exactly two private native modules")
    result["binaries"] = {str(p): digest(p) for p in binaries}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    run(args.output, args.replicate)
