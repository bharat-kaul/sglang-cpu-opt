#!/usr/bin/env python3
"""Compare the opt-in hc=4 recurrence with the immutable serial/parallel native donor."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics

import torch

from standalone_meta_dispatch_screen import BASELINE, ROOT, bench, call, digest, select_arm


ARMS = {
    "old_serial": {"SINK_SERIAL": "1"},
    "new_gate_off": {"SINK_SERIAL": "1"},
    "new_simd1": {"SINK_SERIAL": "1"},
    "new_simd4": {"SINK_SERIAL": "1"},
    "new_simd8": {"SINK_SERIAL": "1"},
    "old_grain1": {"SINK_GRAIN": "1"},
    "old_grain4": {"SINK_GRAIN": "4"},
}


def select(name):
    os.environ.pop("SINK_SIMD4", None)
    select_arm(ARMS[name])


def equal_bits(expected, actual):
    if len(expected) != len(actual):
        raise AssertionError("output arity changed")
    for left, right in zip(expected, actual):
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise AssertionError("output contract changed")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError("nonfinite native output")
        if left.numel() and not torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        ):
            raise AssertionError("native output bits changed")


def make_inputs(rows, hc=4, iters=20, eps=1e-6, distribution="normal", dtype=torch.float32):
    width = (2 + hc) * hc
    mixes = torch.randn(1, rows, width)
    scale, base = torch.randn(3), torch.randn(width)
    if distribution == "zeros":
        mixes.zero_()
        scale.zero_()
        base.zero_()
    elif distribution == "sharp":
        mixes.mul_(40)
    elif distribution == "strided":
        mixes = torch.randn(1, rows, 2 * width)[..., ::2]
        scale, base = torch.randn(6)[::2], torch.randn(2 * width)[::2]
    return mixes.to(dtype), scale, base, hc, iters, eps


def run(output, replicate):
    if output.exists():
        raise FileExistsError(output)
    baseline_dir = Path("/scratch/bkaul/dsv4-standalone-meta-opt/sink-iterations-job-398645")
    baseline = json.loads((baseline_dir / "replicate-1.json").read_text())
    if baseline["job"] != "398645" or baseline["baseline"] != BASELINE or len(baseline["binaries"]) != 1:
        raise RuntimeError("unexpected baseline identity")
    binary_name, binary_hash = next(iter(baseline["binaries"].items()))
    binary = Path(binary_name)
    if digest(binary) != binary_hash:
        raise RuntimeError("baseline binary changed")
    spec = importlib.util.spec_from_file_location("meta_sink_iterations", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load baseline")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    bench.CF = bench.CF + ["-DSINK_SIMD_EXPERIMENT"]
    new = bench._mod("meta_sink_simd4", "sinkhorn.cpp")
    from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch as reference

    functions = {a: (new if a.startswith("new_") else old).mhc_sinkhorn for a in ARMS}
    for streams in (1, 4, 8):
        functions[f"new_simd{streams}"] = (
            lambda *inputs, streams=streams: new.mhc_sinkhorn_simd4(*inputs, streams)
        )
    checks = []

    def check(inputs, label):
        select("old_serial")
        expected = functions["old_serial"](*inputs)
        for arm, fn in functions.items():
            select(arm)
            try:
                equal_bits(expected, fn(*inputs))
            except AssertionError as exc:
                raise AssertionError(f"{label}, arm={arm}: {exc}") from exc
        checks.append(label)
        return expected

    for seed in (0, 7):
        torch.manual_seed(seed)
        for rows in (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 511, 512, 513):
            for distribution in ("normal", "zeros", "sharp", "strided"):
                check(make_inputs(rows, distribution=distribution),
                      {"seed": seed, "rows": rows, "distribution": distribution, "hc": 4, "iters": 20})
        for iters in (0, 1, 2, 5, 40):
            for eps in (0.0, 1e-6, 1e-3):
                check(make_inputs(17, iters=iters, eps=eps),
                      {"seed": seed, "rows": 17, "hc": 4, "iters": iters, "eps": eps})
        for hc in (2, 3, 8):
            check(make_inputs(17, hc=hc), {"seed": seed, "rows": 17, "hc": hc, "iters": 20})
        for dtype in (torch.bfloat16, torch.float16):
            check(make_inputs(17, dtype=dtype), {"seed": seed, "rows": 17, "dtype": str(dtype)})
        inputs = list(make_inputs(32))
        inputs[0] = inputs[0].reshape(2, 16, 24)
        check(tuple(inputs), {"seed": seed, "shape": [2, 16, 24]})

    paths = [ROOT / "plugin/kernels/dsa_pilot/sinkhorn.cpp",
             ROOT / "plugin/kernels/dsa_pilot/sinkhorn_simd4.h", Path(__file__),
             Path(__file__).with_name("standalone_meta_dispatch_screen.py"),
             Path(__file__).with_name("run_standalone_meta_sink_simd.sbatch")]
    result = {
        "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(), "replicate": replicate,
        "node": os.uname().nodename, "torch": torch.__version__, "threads": 64,
        "baseline_identity": baseline, "arms": ARMS, "flags": bench.CF,
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "reference": {"path": inspect.getfile(reference), "sha256": digest(Path(inspect.getfile(reference))),
                      "scope": "Observed error only; no new tolerance or model-accuracy approval"},
        "scope": "Opt-in recurrence prototype; same native output bits required, no serving promotion",
        "timing": "Balanced randomized order, 56 samples/arm/cache; fresh output allocation/release timed; 768MiB add_ eviction before each cold call",
        "cases": [],
    }
    flush = torch.zeros(192 * 1024 * 1024)
    rng = random.Random(replicate)
    torch.manual_seed(19)
    for m in bench.MS:
        inputs = make_inputs(m)
        expected = check(inputs, {"timed_M": m, "hc": 4, "iters": 20})
        oracle = reference(*inputs)
        if len(oracle) != len(expected):
            raise AssertionError("reference arity changed")
        differences = []
        for left, right in zip(expected, oracle):
            if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
                raise AssertionError("reference boundary mismatch")
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise AssertionError("invalid reference comparison")
            differences.append(float((left - right).abs().max()))
        record = {"M": m, "hc": 4, "iters": 20, "reference_max_abs_by_output": differences,
                  "input_sha256": {name: hashlib.sha256(x.numpy().tobytes()).hexdigest()
                                   for name, x in zip(("mixes", "scale", "base"), inputs[:3])}}
        for mode in ("cold", "warm"):
            arms = list(ARMS)
            samples = {a: [] for a in arms}
            orders = []
            for arm in arms:
                select(arm)
                for _ in range(5):
                    functions[arm](*inputs)
            for _ in range(8):
                rng.shuffle(arms)
                shifts = list(range(len(arms)))
                rng.shuffle(shifts)
                for shift in shifts:
                    order = arms[shift:] + arms[:shift]
                    orders.append(order)
                    for arm in order:
                        select(arm)
                        if mode == "cold":
                            flush.add_(1)
                        samples[arm].append(call(functions[arm], inputs))
            if not all(math.isfinite(x) and x > 0 for v in samples.values() for x in v):
                raise RuntimeError("invalid timing sample")
            record[mode] = {"samples_us": samples, "orders": orders,
                            "medians_us": {a: statistics.median(v) for a, v in samples.items()}}
            print(m, mode, record[mode]["medians_us"], flush=True)
        result["cases"].append(record)
    result["bitwise_native_checks"] = checks
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()
    })
    if result["worker_affinity"] != [(i,) for i in range(64)]:
        raise RuntimeError("unexpected worker placement")
    binaries = sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("*.so"))
    if len(binaries) != 1:
        raise RuntimeError("expected one private candidate binary")
    result["candidate_binaries"] = {str(p): digest(p) for p in binaries}
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    run(args.output, args.replicate)
