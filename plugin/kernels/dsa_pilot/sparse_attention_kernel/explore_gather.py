#!/usr/bin/env python3
"""Reproducible AVX-512 gather/packing experiments; never modifies the shipped core."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

import torch

from bench import HERE, FLAGS, Kernel, inputs
from bench_matched_allocations import prepare

BASE_HASH = "013ac2b16a665ec2eb510bf71ba14864dd8cef2e225ccfc85aabb854dcbe0b39"
VARIANTS = ("baseline", "vector_pointers", "unpack_shuffle", "hardware_gather", "no_prefetch")

POINTERS = """
        for (int j = 0; j < Block; j += 8) {
            const int count = std::clamp(topk - start - j, 0, 8);
            const __mmask8 present = (1u << count) - 1;
            __m256i idx = _mm256_set1_epi32(-1);
            if (present)
                idx = _mm256_mask_loadu_epi32(idx, present, indices + start + j);
            const __mmask8 live = _mm256_cmp_epi32_mask(idx, _mm256_setzero_si256(), _MM_CMPINT_GE);
            const __m512i offset = _mm512_slli_epi64(_mm512_cvtepi32_epi64(idx), 10);
            const __m512i address = _mm512_add_epi64(offset, _mm512_set1_epi64(reinterpret_cast<int64_t>(kv)));
            const __m512i safe = _mm512_mask_blend_epi64(
                live, _mm512_set1_epi64(reinterpret_cast<int64_t>(zeros)), address);
            _mm512_storeu_si512(selected + j, safe);
            for (int lane = 0; lane < 8; ++lane) valid[j + lane] = (live >> lane) & 1;
        }
"""

SHUFFLE = """
                const __m512i low = _mm512_unpacklo_epi16(a, b);
                const __m512i high = _mm512_unpackhi_epi16(a, b);
                const __m512i first = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
                const __m512i second = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
                _mm512_store_si512(s.values + j * D + d * 2, _mm512_permutex2var_epi64(low, first, high));
                _mm512_store_si512(s.values + j * D + d * 2 + 32, _mm512_permutex2var_epi64(low, second, high));
"""

GATHER = """
        // Experimental S <= 1024 contract: dword indices cannot overflow here.
        for (int group = 0; group < Block; group += 16) {
            alignas(64) int32_t offsets[16];
            __mmask16 live = 0;
            for (int lane = 0; lane < 16; ++lane) {
                const int j = group + lane;
                const int index = start + j < topk ? indices[start + j] : -1;
                offsets[lane] = index >= 0 ? index * (D / 2) : 0;
                if (index >= 0) live |= 1u << lane;
            }
            const __m512i offset = _mm512_load_si512(offsets);
            for (int d = 0; d < D; d += 32) {
                __m512i rows[16];
                for (int pair = 0; pair < 16; ++pair)
                    rows[pair] = _mm512_mask_i32gather_epi32(
                        _mm512_setzero_si512(), live, offset, kv + d + pair * 2, 4);
                transpose16(rows);
                for (int lane = 0; lane < 16; ++lane)
                    _mm512_store_si512(s.gathered + (group + lane) * D + d, rows[lane]);
            }
        }
        for (int j = 0; j < Block; j += 2)
            for (int d = 0; d < D; d += 32) {
                const __m512i a = _mm512_load_si512(s.gathered + j * D + d);
                const __m512i b = _mm512_load_si512(s.gathered + (j + 1) * D + d);
                _mm512_store_si512(s.values + j * D + d * 2, _mm512_permutex2var_epi16(a, lo, b));
                _mm512_store_si512(s.values + j * D + d * 2 + 32, _mm512_permutex2var_epi16(a, hi, b));
            }
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Pinned source no longer matches the experiment")
    return source.replace(old, new, 1)


def variant_source(source, name):
    pointer_start = source.index("        for (int j = 0; j < Block; ++j) {")
    pack_start = source.index("        // Keep several rows in flight", pointer_start)
    pack_end = source.index("        if (phases)", pack_start)
    pointer = source[pointer_start:pack_start]
    pack = source[pack_start:pack_end]
    if name == "vector_pointers":
        source = replace_once(source, pointer, POINTERS)
    elif name == "unpack_shuffle":
        first = pack.index("                _mm512_store_si512(s.values")
        end = pack.index("\n            }", first)
        source = replace_once(source, pack[first:end], SHUFFLE.rstrip())
    elif name == "hardware_gather":
        source = replace_once(source, pack, GATHER)
    elif name == "no_prefetch":
        body = pack.rindex("            for (int d = 0; d < D; d += 32) {")
        source = replace_once(source, pack, "        for (int j = 0; j < Block; j += 2) {\n" + pack[body:])
        if "_mm_prefetch" in source:
            raise ValueError("No-prefetch variant still contains a prefetch")
    elif name != "baseline":
        raise ValueError(name)
    # Export the exact staging code too, for bit-pattern/tail/mask tests independent of attention.
    constants = source[source.index("    alignas(64) static constexpr uint16_t even_odd_lo"):
                       source.index("    for (int start = 0; start < topk;")]
    stage_start = source.index("        bool valid[Block];")
    stage_end = source.index("        if (phases)", stage_start)
    stage = source[stage_start:stage_end]
    source += """
extern "C" void test_stage(const uint16_t* kv, const int32_t* indices,
                           int topk, int start, uint16_t* gathered, uint16_t* packed) {
    Scratch s;
""" + constants + stage + """
    std::memcpy(gathered, s.gathered, sizeof(s.gathered));
    std::memcpy(packed, s.values, sizeof(s.values));
}
"""
    return source


def build(directory):
    source = (HERE / "sparse_attn.cpp").read_text()
    if digest(HERE / "sparse_attn.cpp") != BASE_HASH:
        raise ValueError("Experiment expects the original 810a8f6 core")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "sparse_attn.h", directory)
    flags = FLAGS + ["-std=c++17", "-fno-math-errno", "-ffp-contract=off",
                     "-Werror=aggressive-loop-optimizations"]
    manifest = {
        "flags": flags, "baseline_sha256": BASE_HASH,
        "experiment_sha256": digest(Path(__file__).resolve()),
        "helpers": {n: digest(HERE / n) for n in (
            "bench.py", "bench_matched_allocations.py", "run_gather_exploration.sh", "sparse_attn.h")},
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "variants": {},
    }
    for name in VARIANTS:
        path = directory / f"{name}.cpp"
        path.write_text(variant_source(source, name))
        library = directory / f"{name}.so"
        subprocess.run(["g++", *flags, "-shared", "-fPIC", str(path), "-o", str(library)], check=True)
        disassembly = subprocess.check_output(["objdump", "-d", str(library)], text=True)
        (directory / f"{name}.asm").write_text(disassembly)
        manifest["variants"][name] = {
            "source_sha256": digest(path), "binary_sha256": digest(library),
            "instructions": {op: disassembly.count(op) for op in (
                "vpgatherdd", "vpermt2w", "vpermi2w", "vpunpcklwd", "vpermt2q",
                "vpmovsxdq", "prefetcht0", "tdpbf16ps")},
        }
    (directory / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


def stage_tests(kernels):
    generator = torch.Generator().manual_seed(913)
    # Arbitrary BF16 bits (including NaNs/subnormals) test movement, not arithmetic.
    source = torch.randint(-32768, 32768, (1024, 512), dtype=torch.int16, generator=generator)
    records = []
    for count in (0, 1, 7, 8, 15, 16, 63, 64, 65, 73):
        idx = torch.randint(0, 1024, (count,), dtype=torch.int32, generator=generator)
        for kind in ("random", "masked", "duplicate"):
            if kind == "masked":
                idx[::3] = -1
            if kind == "duplicate":
                idx.fill_(1023)
            for start in range(0, max(count, 1), 64):
                expected = torch.zeros(64, 512, dtype=torch.int16)
                for j in range(min(64, count - start)):
                    if idx[start + j] >= 0:
                        expected[j] = source[idx[start + j]]
                packed = expected.reshape(32, 2, 512).permute(0, 2, 1).contiguous()
                for name, kernel in kernels.items():
                    fn = kernel.lib.test_stage
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p]
                    fn.restype = None
                    a = torch.empty_like(expected)
                    b = torch.empty_like(packed)
                    fn(source.data_ptr(), idx.data_ptr(), count, start, a.data_ptr(), b.data_ptr())
                    assert torch.equal(a, expected) and torch.equal(b, packed), (name, count, kind, start)
                records.append([count, kind, start])
    return records


def parity_tests(kernels):
    records = []
    for label, config in (
        ("normal", {}), ("seed7", {"seed": 7}), ("sharp", {"sharp": 4}),
        ("tail", {"batch": 2, "queries": 3, "heads": 32, "topk": 73}),
        ("masked", {"batch": 2, "topk": 128}), ("all_masked", {"batch": 1, "topk": 65}),
        ("one", {"batch": 1, "source": 1, "topk": 1}), ("empty", {"batch": 1, "topk": 0}),
    ):
        data = inputs(**config)
        if label == "masked":
            data[3][:, :, :64] = -1
        if label == "all_masked":
            data[3].fill_(-1)
        expected = kernels["baseline"].call(*data).clone()
        assert torch.isfinite(expected).all()
        for name, kernel in kernels.items():
            out = kernel.call(*data)
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), (label, name)
        records.append(label)
    return records


def measure(functions, flush, repetitions, rng):
    for fn in functions.values():
        for _ in range(5):
            fn()
    samples = {name: [] for name in functions}
    orders = []
    for _ in range(repetitions):
        names = list(functions)
        rng.shuffle(names)
        orders.append(names)
        for name in names:
            if flush is not None:
                flush.add_(1)
            start = time.perf_counter_ns()
            out = functions[name]()
            del out
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    return {"order": orders, "timings": {
        name: {"median_us": statistics.median(v), "samples_us": v} for name, v in samples.items()}}


def run(args):
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Execute on EMR through Slurm, never the login node")
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    manifest = json.loads((args.build / "build.json").read_text())
    assert manifest["experiment_sha256"] == digest(Path(__file__).resolve())
    assert all(manifest["helpers"][n] == digest(HERE / n) for n in manifest["helpers"])
    kernels = {}
    for name, item in manifest["variants"].items():
        assert digest(args.build / f"{name}.so") == item["binary_sha256"]
        kernels[name] = Kernel(args.build / f"{name}.so")
    result = {
        "node": os.uname().nodename, "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
        "replicate": args.replicate, "build": manifest, "torch": torch.__version__,
        "shape": {"B": 32, "M": 1, "H": 64, "D": 512, "S": 1024, "K": 512},
        "stage_bitwise_cases": stage_tests(kernels), "attention_bitwise_cases": parity_tests(kernels),
        "contract": "All inputs prepared before timing; each variant allocates and releases output inside same Python/ctypes call. No FP32 timing comparison.",
        "cold": "768 MiB eviction before every candidate call, outside timer",
    }
    data = inputs()
    qb, kb, sink, identity, _, _ = prepare(data)
    prepared = qb.unsqueeze(1), kb, sink, identity, torch.empty_like(qb).unsqueeze(1)
    rng = random.Random(args.replicate)
    flush = torch.zeros(192 * 1024 * 1024)
    for layout, tensors in (("indexed", data), ("pregathered", prepared)):
        q, kv, sink, indices, _ = tensors
        result[f"{layout}_input_sha256"] = {
            n: hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
            for n, t in (("q", q), ("kv", kv), ("indices", indices), ("sink", sink))}
        def allocate(kernel):
            return kernel.call(q, kv, sink, indices, torch.empty_like(q))
        functions = {name: lambda k=k: allocate(k) for name, k in kernels.items()}
        # Pre-gathered identity input is the matched-allocation contract used previously.
        # Indexed mode separately tests the actual random source-cache accesses.
        for mode, eviction in (("cold", flush), ("hot", None)):
            key = f"{layout}_{mode}"
            result[key] = measure(functions, eviction, args.repetitions, rng)
            print(key, {n: d["median_us"] for n, d in result[key]["timings"].items()}, flush=True)
        profiles = {}
        for name, kernel in kernels.items():
            traces = []
            for _ in range(9):
                flush.add_(1)
                phases = (ctypes.c_double * 8)()
                out = torch.empty_like(q)
                start = time.perf_counter_ns()
                kernel.call(q, kv, sink, indices, out, phases)
                elapsed = (time.perf_counter_ns() - start) / 1000
                traces.append({"wall_us": elapsed, "worker_mean_us": [phases[i] * 1e6 / 64 for i in range(4)],
                               "critical_us": [phases[i] * 1e6 for i in range(4, 8)]})
            profiles[name] = traces
        result[f"{layout}_cold_profile"] = profiles
    result["affinity"] = sorted({tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()})
    assert result["affinity"] == [(i,) for i in range(64)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=31)
    args = parser.parse_args()
    if args.repetitions < 1 or args.replicate not in (1, 2, 3):
        parser.error("Require positive repetitions and replicate 1, 2, or 3")
    if args.build_only:
        build(args.build.resolve())
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args)
