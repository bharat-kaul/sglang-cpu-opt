#!/usr/bin/env python3
"""Build on login; execute only via Slurm on one 64-core EMR socket."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import torch

HERE = Path(__file__).resolve().parent
BASELINE = "7305559f0eb6ac9c2be127b8efb2e5bff34712ca"
FLAGS = ["-O3", "-fopenmp", "-march=sapphirerapids", "-mprefer-vector-width=512"]


def build(directory):
    from torch.utils.cpp_extension import load

    directory.mkdir(parents=True, exist_ok=True)
    for name in ("sparse_attend.cpp", "avx512_exp.h"):
        source = subprocess.check_output(
            ["git", "show", f"{BASELINE}:plugin/kernels/dsa_pilot/{name}"], cwd=HERE
        )
        (directory / name).write_bytes(source)
    load(
        name="emr_sparse_fp32_7305559",
        sources=[str(directory / "sparse_attend.cpp")],
        build_directory=str(directory),
        extra_cflags=FLAGS + ["-DCPU_CAPABILITY_AVX512"],
        extra_ldflags=["-fopenmp"],
        verbose=True,
    )
    subprocess.run(
        ["g++", *FLAGS, "-std=c++17", "-fno-math-errno", "-ffp-contract=off",
         "-Wall", "-Wextra", "-shared", "-fPIC", str(HERE / "sparse_attn.cpp"),
         "-o", str(directory / "libsparse_attn_emr.so")],
        check=True,
    )
    sanitized = directory / "ubsan"
    sanitized.mkdir(exist_ok=True)
    shutil.copy2(directory / "emr_sparse_fp32_7305559.so", sanitized)
    subprocess.run(
        ["g++", *FLAGS, "-O2", "-g", "-std=c++17", "-ffp-contract=off",
         "-fsanitize=undefined", "-fsanitize-undefined-trap-on-error",
         "-shared", "-fPIC", str(HERE / "sparse_attn.cpp"),
         "-o", str(sanitized / "libsparse_attn_emr.so")],
        check=True,
    )
    subprocess.run(
        ["g++", *FLAGS, "-std=c++17", "-fno-math-errno", "-ffp-contract=off",
         "-Wall", "-Wextra", str(HERE / "sparse_attn.cpp"), str(HERE / "bench.cpp"),
         "-o", str(directory / "bench")],
        check=True,
    )


class Kernel:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.sparse_attn_emr_init.restype = ctypes.c_int
        self.lib.sparse_attn_emr.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
            + [ctypes.c_float, ctypes.POINTER(ctypes.c_double)]
        )
        self.lib.sparse_attn_emr.restype = ctypes.c_int
        status = self.lib.sparse_attn_emr_init()
        if status:
            raise RuntimeError(f"AMX initialization failed: status={status}")

    def call(self, q, kv, sink, indices, out, phases=None):
        b, m, h, d = q.shape
        assert d == 512 and q.dtype == kv.dtype == out.dtype == torch.bfloat16
        assert sink.dtype == torch.float32 and indices.dtype == torch.int32
        assert all(t.is_contiguous() for t in (q, kv, sink, indices, out))
        status = self.lib.sparse_attn_emr(
            *(t.data_ptr() for t in (q, kv, sink, indices, out)),
            b, m, h, kv.shape[1], indices.shape[-1], d**-0.5, phases,
        )
        if status:
            raise ValueError(f"sparse_attn_emr rejected input: status={status}")
        return out


def inputs(seed=42, batch=32, queries=1, heads=64, source=1024, topk=512, sharp=1):
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(batch, queries, heads, 512, generator=gen) * sharp).bfloat16()
    kv = (torch.randn(batch, source, 512, generator=gen) * sharp).bfloat16()
    sink = torch.randn(heads, generator=gen)
    # Independent random permutations: no accidental duplicate-selection shortcut.
    indices = torch.stack([
        torch.randperm(source, generator=gen)[:topk]
        for _ in range(batch * queries)
    ]).view(batch, queries, topk).int()
    return q, kv, sink, indices, torch.empty_like(q)


def gather(data):
    q, kv, _, indices, _ = data
    b, m, h, d = q.shape
    selected = kv[torch.arange(b)[:, None, None], indices.clamp_min(0).long()]
    selected = selected.masked_fill((indices < 0).unsqueeze(-1), 0)
    return q.float().reshape(b * m, h, d), selected.float().reshape(b * m, indices.shape[-1], d)


def block_reference(data):
    q, kv, sink, indices, _ = data
    qf, gathered = gather(data)
    rows, h, d = qf.shape
    maximum = torch.full((rows, h, 1), -torch.inf)
    total = torch.zeros_like(maximum)
    acc = torch.zeros_like(qf)
    valid = (indices >= 0).reshape(rows, 1, -1)
    for start in range(0, indices.shape[-1], 64):
        keys = gathered[:, start:start + 64]
        score = torch.bmm(qf, keys.transpose(1, 2)) * (d**-0.5)
        score.masked_fill_(~valid[:, :, start:start + 64], -torch.inf)
        new_max = torch.maximum(maximum, score.amax(-1, keepdim=True))
        safe = new_max.masked_fill(torch.isneginf(new_max), 0)
        correction = (maximum - safe).exp()
        p = (score - safe).exp()
        total = total * correction + p.sum(-1, keepdim=True)
        acc = acc * correction + torch.bmm(p.bfloat16().float(), keys)
        maximum = new_max
    delta = sink.view(1, h, 1) - maximum
    correction = (-delta.clamp_min(0)).exp()
    denominator = total * correction + delta.clamp_max(0).exp()
    return (acc * (correction / denominator)).view_as(q).bfloat16()


def errors(actual, reference):
    a, r = actual.float().flatten().double(), reference.float().flatten().double()
    diff = a - r
    peak, norm = r.abs().max().item(), r.norm().item()
    cosine = (torch.dot(a, r) / (a.norm() * r.norm())).item() if norm else float(not a.any())
    return {
        "finite": bool(torch.isfinite(a).all()),
        "fp32_bit_mismatches": int((actual.float().view(torch.int32)
                                  != reference.float().view(torch.int32)).sum()),
        "bf16_bit_mismatches": int((actual.bfloat16().view(torch.int16)
                                  != reference.bfloat16().view(torch.int16)).sum()),
        "elements": a.numel(), "cosine": cosine,
        "max_abs": diff.abs().max().item(),
        "relative_l2": diff.norm().item() / max(norm, 1e-30),
        "relative_linf": diff.abs().max().item() / max(peak, 1e-30),
    }


def verify(kernel, baseline):
    results = []
    cases = [
        ("normal", dict()), ("seed7", dict(seed=7)), ("sharp", dict(sharp=4)),
        ("tail", dict(batch=2, queries=3, heads=32, topk=73)),
        ("masked_first_block", dict(batch=2, topk=128)),
        ("all_masked", dict(batch=1, topk=65)),
        ("duplicate", dict(batch=2, topk=65)),
        ("empty", dict(batch=1, topk=0)),
        ("large_sink", dict(batch=1)),
        ("one_key", dict(batch=1, source=1, topk=1)),
        ("three_groups", dict(batch=2, queries=2, heads=96, topk=129)),
    ]
    for name, options in cases:
        data = inputs(**options)
        q, kv, sink, idx, out = data
        if name == "masked_first_block":
            idx[:, :, :64] = -1
        if name == "all_masked":
            idx.fill_(-1)
        if name == "duplicate":
            idx.fill_(17)
        if name == "large_sink":
            sink.fill_(10000)
        actual = kernel.call(*data).clone()
        # Repeatability is exact, unlike cross-algorithm mixed-precision parity.
        assert torch.equal(actual, kernel.call(*data)), name
        block = block_reference(data)
        detail = {"case": name, "block_reference": errors(actual, block)}
        if name == "normal":
            detail["output_sha256"] = hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()
        e = detail["block_reference"]
        assert e["finite"] and e["relative_l2"] < 0.003 and e["relative_linf"] < 0.008, detail
        if bool((idx >= 0).all()) and idx.numel() and name != "large_sink":
            qf, kf = gather(data)
            fp32 = baseline.sparse_attend_fp32bmm_hx(qf, kf, sink, 512**-0.5).view_as(q)
            detail["fp32_baseline"] = errors(actual, fp32)
            e = detail["fp32_baseline"]
            assert e["finite"] and e["relative_l2"] < 0.006 and e["relative_linf"] < 0.012, detail
        if name in ("all_masked", "empty", "large_sink"):
            assert not actual.any(), name
        results.append(detail)
        print(json.dumps(detail), flush=True)
    data = inputs(batch=1, topk=1)
    for index in (-2, 1024):
        data[3].fill_(index)
        try:
            kernel.call(*data)
        except ValueError as exc:
            assert "status=4" in str(exc)
        else:
            raise AssertionError("invalid index was accepted")
    invalid_shape = inputs(batch=1, heads=16, topk=1)
    try:
        kernel.call(*invalid_shape)
    except ValueError as exc:
        assert "status=3" in str(exc)
    else:
        raise AssertionError("invalid head count was accepted")
    return results


def measure(fn, repetitions, flush=None):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(repetitions):
        if flush is not None:
            flush.add_(1)
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return {"median_us": statistics.median(samples), "min_us": min(samples),
            "max_us": max(samples), "samples_us": samples}


def run(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Run benchmarks on an EMR allocation, not on the login node")
    spec = importlib.util.spec_from_file_location(
        "emr_sparse_fp32_7305559", args.build / "emr_sparse_fp32_7305559.so"
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    kernel = Kernel(args.build / "libsparse_attn_emr.so")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    result = {
        "node": os.uname().nodename, "slurm_job": os.environ["SLURM_JOB_ID"],
        "threads": args.threads, "torch": torch.__version__, "baseline_commit": BASELINE,
        "kernel_sha256": hashlib.sha256((HERE / "sparse_attn.cpp").read_bytes()).hexdigest(),
        "shape": {"B": 32, "M": 1, "H": 64, "D": 512, "S": 1024, "K": 512},
    }
    if not args.perf_only:
        result["accuracy"] = verify(kernel, baseline)
    if args.accuracy_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    data = inputs()
    qf, kf = gather(data)
    sink = data[2]
    fast = lambda: kernel.call(*data)
    ref = lambda: baseline.sparse_attend_fp32bmm_hx(qf, kf, sink, 512**-0.5)
    # Includes indexing and FP32 materialization, unlike the pre-gathered reference.
    def gathered_ref():
        q, k = gather(data)
        return baseline.sparse_attend_fp32bmm_hx(q, k, sink, 512**-0.5)

    flush = torch.zeros(192 * 1024 * 1024)
    for name, fn in (("amx", fast), ("fp32_pregathered", ref), ("fp32_with_gather", gathered_ref)):
        for mode, buffer in (("hot", None), ("cold", flush)):
            result[f"{name}_{mode}"] = measure(fn, args.repetitions, buffer)
            print(name, mode, result[f"{name}_{mode}"]["median_us"], flush=True)
    for mode, buffer in (("hot", None), ("cold", flush)):
        traces = []
        for _ in range(10):
            if buffer is not None:
                buffer.add_(1)
            phases = (ctypes.c_double * 8)()
            start = time.perf_counter_ns()
            kernel.call(*data, phases=phases)
            traces.append([(time.perf_counter_ns() - start) / 1000,
                           *[phases[p] * 1e6 / args.threads for p in range(4)],
                           *[phases[p] * 1e6 for p in range(4, 8)]])
        result[f"profile_{mode}"] = {
            "wall_us": statistics.median(t[0] for t in traces),
            "phases_us": [statistics.median(t[i] for t in traces) for i in range(1, 5)],
            "critical_phases_us": [statistics.median(t[i] for t in traces) for i in range(5, 9)],
        }
    result["phase_names"] = ["gather_pack_init", "score_amx", "softmax_rescale_output", "value_amx"]
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(path.name))) for path in Path("/proc/self/task").iterdir()
    })
    result["process_threads"] = len(list(Path("/proc/self/task").iterdir()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--threads", type=int, choices=(1, 16, 32, 64), default=64)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.perf_only and args.accuracy_only:
        parser.error("--perf-only and --accuracy-only are mutually exclusive")
    if args.build_only:
        build(args.build.resolve())
    else:
        run(args)
