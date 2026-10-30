#!/usr/bin/env python3
"""Match the original FP32 input preparation, output allocation, and timing policy."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch

from bench import BASELINE, FLAGS, HERE, Kernel, block_reference, errors, inputs

MODULE = "emr_sparse_matched_allocations"
SOURCES = ("sparse_attn.cpp", "sparse_attn.h", "matched_allocations.cpp",
           "bench_matched_allocations.py", "bench.py", "run_matched_allocations.sh")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(directory):
    from torch.utils.cpp_extension import load

    directory.mkdir(parents=True, exist_ok=True)
    original = subprocess.check_output(
        ["git", "show", f"{BASELINE}:plugin/kernels/dsa_pilot/sparse_attend.cpp"],
        cwd=HERE, text=True,
    )
    start = original.index("torch::Tensor sparse_attend_fp32bmm_hx(")
    end = original.index("\n}\n", start) + 3
    donor = directory / "fp32_original.cpp"
    donor.write_text(
        '#include <torch/extension.h>\n#include <ATen/Parallel.h>\n'
        '#include <cmath>\n#include <cstdlib>\n#include "avx512_exp.h"\n'
        + original[start:end]
    )
    (directory / "avx512_exp.h").write_bytes(subprocess.check_output(
        ["git", "show", f"{BASELINE}:plugin/kernels/dsa_pilot/avx512_exp.h"], cwd=HERE
    ))
    core_flags = FLAGS + ["-std=c++17", "-fno-math-errno", "-ffp-contract=off"]
    subprocess.run(
        ["g++", *core_flags, "-shared", "-fPIC", str(HERE / "sparse_attn.cpp"),
         "-o", str(directory / "libsparse_attn_emr.so")], check=True,
    )
    load(
        name=MODULE, sources=[str(donor), str(HERE / "matched_allocations.cpp")],
        build_directory=str(directory), extra_cflags=FLAGS + ["-DCPU_CAPABILITY_AVX512"],
        extra_ldflags=["-fopenmp", f"-L{directory}", "-lsparse_attn_emr", f"-Wl,-rpath,{directory}"],
        verbose=True,
    )
    manifest = {
        "baseline_commit": BASELINE, "flags": FLAGS + ["-DCPU_CAPABILITY_AVX512"],
        "core_flags": core_flags, "torch": torch.__version__,
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "sources": {name: digest(HERE / name) for name in SOURCES},
        "donor_sha256": digest(donor), "donor_header_sha256": digest(directory / "avx512_exp.h"),
        "binary_sha256": digest(directory / f"{MODULE}.so"),
        "core_binary_sha256": digest(directory / "libsparse_attn_emr.so"),
    }
    (directory / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_module(directory):
    manifest = json.loads((directory / "build.json").read_text())
    for name in SOURCES:
        if digest(HERE / name) != manifest["sources"][name]:
            raise RuntimeError(f"Rebuild required: {name}")
    for name, key in ((f"{MODULE}.so", "binary_sha256"),
                      ("libsparse_attn_emr.so", "core_binary_sha256")):
        if digest(directory / name) != manifest[key]:
            raise RuntimeError(f"Binary identity mismatch: {name}")
    spec = importlib.util.spec_from_file_location(MODULE, directory / f"{MODULE}.so")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize()
    return module, manifest


def prepare(data):
    q, kv, sink, selected, _ = data
    b, m, h, d = q.shape
    if (selected < 0).any():
        raise ValueError("The original pre-gathered FP32 contract has no masked indices")
    gathered = kv[torch.arange(b)[:, None, None], selected.long()].reshape(b * m, -1, d).contiguous()
    qb = q.reshape(b * m, h, d).contiguous()
    identity = torch.arange(gathered.shape[1], dtype=torch.int32).repeat(b * m, 1).unsqueeze(1)
    return qb, gathered, sink, identity, qb.float(), gathered.float()


def check_case(module, core, data, label):
    qb, kb, sink, identity, qf, kf = prepare(data)
    original = module.fp32(qf, kf, sink, 512**-0.5)
    actual = module.bf16(qb, kb, sink, identity, 512**-0.5)
    direct = core.call(*data).reshape_as(actual)
    assert torch.equal(actual, direct), label
    mixed = block_reference(data).reshape_as(actual)
    comparisons = {"fp32": errors(actual, original), "block_reference": errors(actual, mixed)}
    for key, l2, linf in (("fp32", 0.006, 0.012), ("block_reference", 0.003, 0.008)):
        e = comparisons[key]
        assert e["finite"] and e["relative_l2"] < l2 and e["relative_linf"] < linf, (label, key, e)
    for fn, args in ((module.fp32, (qf, kf, sink, 512**-0.5)),
                     (module.bf16, (qb, kb, sink, identity, 512**-0.5))):
        a, b = fn(*args), fn(*args)
        assert a.data_ptr() != b.data_ptr() and torch.equal(a, b), label
    return {"case": label, "matches_indexed_core_bitwise": True,
            "both_outputs_fresh_and_repeatable": True, **comparisons}


def timed_call(fn):
    start = time.perf_counter_ns()
    value = fn()
    del value
    return (time.perf_counter_ns() - start) / 1000


def run(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run only on an EMR Slurm allocation")
    for flag in ("SPARSE_SKIP_SOFTMAX", "SPARSE_SKIP_VALUE_GEMM"):
        if flag in os.environ:
            raise RuntimeError(f"Unset diagnostic shortcut {flag}")
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    module, manifest = load_module(args.build)
    core = Kernel(args.build / "libsparse_attn_emr.so")
    cases = [
        ("normal", {}), ("seed7", {"seed": 7}), ("sharp", {"sharp": 4}),
        ("tail", {"batch": 2, "queries": 3, "heads": 32, "topk": 73}),
        ("three_groups", {"batch": 2, "queries": 2, "heads": 96, "topk": 129}),
        ("one_key", {"batch": 1, "source": 1, "topk": 1}),
    ]
    accuracy = [check_case(module, core, inputs(**kw), label) for label, kw in cases]
    data = inputs()
    qb, kb, sink, identity, qf, kf = prepare(data)
    functions = {
        "fp32": lambda: module.fp32(qf, kf, sink, 512**-0.5),
        "bf16": lambda: module.bf16(qb, kb, sink, identity, 512**-0.5),
    }
    result = {
        "node": os.uname().nodename, "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
        "replicate": args.replicate, "build": manifest, "threads": 64,
        "shape": {"B": 32, "M": 1, "H": 64, "D": 512, "source_S": 1024, "selected_K": 512},
        "contract": {
            "inputs": "Allocated, gathered, made contiguous, and converted to each native dtype BEFORE timing",
            "outputs": "Both allocate a fresh tensor through PyTorch INSIDE every timed call; both released before timer stops",
            "input_normalization": "Both call to(native_dtype).contiguous() inside the function; no copy for prepared inputs",
            "intermediates": "Natural algorithm workspace remains inside each function; no artificial buffer equalization",
            "bf16_indices": "Identity indices allocated BEFORE timing for existing C ABI; no source-cache gather timed",
            "cold": "768 MiB eviction BEFORE EACH call, outside timer",
            "allocator": "Same warmed PyTorch CPU allocator; allocations may reuse pages",
        },
        "input_sha256": {name: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
                         for name, t in (("q_bf16", qb), ("kv_bf16", kb), ("sink", sink), ("identity", identity))},
        "accuracy": accuracy,
    }
    # Untimed allocation audit: dtype normalization must not allocate/copy prepared inputs.
    result["allocation_audit"] = {}
    for name, fn in functions.items():
        fn()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True, record_shapes=True) as prof:
            output = fn()
        events = prof.events()
        assert not any(e.name == "aten::_to_copy" for e in events), name
        result["allocation_audit"][name] = {
            "output_bytes": output.numel() * output.element_size(),
            "dtype_conversion_copies": 0,
            "positive_self_allocations": [
                {"op": e.name, "bytes": e.self_cpu_memory_usage}
                for e in events if e.self_cpu_memory_usage > 0
            ],
        }
        del output
    flush = torch.zeros(192 * 1024 * 1024)
    rng = random.Random(args.replicate)
    for mode in ("cold", "hot"):
        for fn in functions.values():
            for _ in range(10):
                fn()
        samples = {name: [] for name in functions}
        orders = []
        for _ in range(51):
            names = list(functions)
            rng.shuffle(names)
            orders.append(names)
            for name in names:
                if mode == "cold":
                    flush.add_(1)
                samples[name].append(timed_call(functions[name]))
        result[mode] = {
            "order": orders,
            "timings": {name: {"median_us": statistics.median(v), "samples_us": v}
                        for name, v in samples.items()},
        }
        print(mode, {k: statistics.median(v) for k, v in samples.items()}, flush=True)
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()
    })
    assert result["worker_affinity"] == [(i,) for i in range(64)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.build_only:
        build(args.build.resolve())
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args)
