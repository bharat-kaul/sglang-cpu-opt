#!/usr/bin/env python3
"""Summarize the gather experiments without promoting a candidate to production."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
NAMES = ("baseline", "vector_pointers", "unpack_shuffle", "hardware_gather", "no_prefetch")
MODES = ("indexed_cold", "indexed_hot", "pregathered_cold", "pregathered_hot")
COUNTS = (0, 1, 7, 8, 15, 16, 63, 64, 65, 73)
STAGE_CASES = [[count, kind, start] for count in COUNTS
               for kind in ("random", "masked", "duplicate")
               for start in range(0, max(count, 1), 64)]
ATTENTION_CASES = ["normal", "seed7", "sharp", "tail", "masked", "all_masked", "one", "empty"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(runs):
    if len(runs) != 3 or {r["replicate"] for r in runs} != {1, 2, 3}:
        raise ValueError("Require three numbered process replicates")
    first = runs[0]
    if len({(r["job"], r["node"], r["pid"]) for r in runs}) != 3:
        raise ValueError("Duplicate process identity")
    expected_helpers = {n: digest(HERE / n) for n in (
        "bench.py", "bench_matched_allocations.py", "run_gather_exploration.sh", "sparse_attn.h")}
    for r in runs:
        for key in ("job", "node", "build", "contract", "shape",
                    "indexed_input_sha256", "pregathered_input_sha256"):
            if r[key] != first[key]:
                raise ValueError(f"Incompatible replicate: {key}")
        if r["build"]["baseline_sha256"] != digest(HERE / "sparse_attn.cpp"):
            raise ValueError("Baseline source mismatch")
        if r["build"]["experiment_sha256"] != digest(HERE / "explore_gather.py") or r["build"]["helpers"] != expected_helpers:
            raise ValueError("Experiment source mismatch")
        if r["shape"] != {"B": 32, "M": 1, "H": 64, "D": 512, "S": 1024, "K": 512}:
            raise ValueError("Wrong workload shape")
        if r["affinity"] != [[i] for i in range(64)]:
            raise ValueError("Wrong core affinity")
        if r["stage_bitwise_cases"] != STAGE_CASES or r["attention_bitwise_cases"] != ATTENTION_CASES:
            raise ValueError("Incomplete bitwise test coverage")
        if set(r["build"]["variants"]) != set(NAMES):
            raise ValueError("Missing built variant")
        for mode in MODES:
            m = r[mode]
            if len(m["order"]) != 41 or any(sorted(o) != sorted(NAMES) for o in m["order"]):
                raise ValueError("Incomplete randomized comparison")
            if set(m["timings"]) != set(NAMES):
                raise ValueError("Missing timing variant")
            for t in m["timings"].values():
                values = t["samples_us"]
                if len(values) != 41 or not all(math.isfinite(v) and v > 0 for v in values):
                    raise ValueError("Invalid timing samples")
                if t["median_us"] != statistics.median(values):
                    raise ValueError("Median does not match samples")
    summary = {
        "node": first["node"], "job": first["job"], "build": first["build"], "shape": first["shape"],
        "contract": first["contract"], "processes": [r["pid"] for r in runs],
        "estimator": "median of three process medians, each over 41 randomized paired calls",
        "stage_cases_per_variant_per_process": len(STAGE_CASES),
        "attention_cases_per_variant_per_process": len(ATTENTION_CASES),
        "decision": "Keep production unchanged; small pointer/packing changes do not establish a substantial win.",
    }
    for mode in MODES:
        medians = {n: [r[mode]["timings"][n]["median_us"] for r in runs] for n in NAMES}
        baseline = statistics.median(medians["baseline"])
        summary[mode] = {
            n: {"median_us": statistics.median(v), "process_medians_us": v,
                "latency_change_pct": (statistics.median(v) / baseline - 1) * 100,
                "per_process_change_pct": [(a / b - 1) * 100 for a, b in zip(v, medians["baseline"])]}
            for n, v in medians.items()
        }
    return summary


def report(s):
    lines = [
        "# AVX-512 gather and packing exploration", "",
        f"Job **{s['job']}**, **{s['node']}**, EMR Xeon 8592+, one socket,",
        "**64 physical cores / CPUs 0-63 / NUMA memory 0 / no SMT**.", "",
        "## Conclusion", "",
        "**Keep the current kernel.** More explicit pointer SIMD yields a sub-1%",
        "change here; alternative unpack/shuffle packing is effectively tied.",
        "Hardware gather plus transpose is slower, and removing prefetch hurts cold inputs.",
        "These are results for this workload and the tested alternatives, not a claim",
        "that every possible permutation/gather strategy has been exhausted.", "",
        "## What the current code actually does", "",
        "Lines 144-148 select pointers, not BF16 payloads. Per 64-key block they read",
        "256 bytes of indices and select 64 row addresses. Each selected row then",
        "supplies 1024 contiguous bytes (512 BF16 values), for 64 KiB of KV payload.",
        "The following loop already uses 512-bit loads and two",
        "`_mm512_permutex2var_epi16` operations per pair of row segments to produce VNNI.",
        "Permutes rearrange values already in registers; they cannot fetch arbitrary",
        "source rows without loads or gathers.", "",
        "Disassembly of the baseline staging function confirms that GCC already",
        "auto-vectorizes the pointer setup: masked `vmovdqu32`, `vpmovsxdq`,",
        "`vpsllq $0xa`, masked `vpaddq`, and vector stores. Source-level scalar",
        "syntax does not imply scalar machine execution.", "",
        "## Variants", "",
        "- **baseline:** unchanged production source, plus a test-only staging export.",
        "- **vector_pointers:** explicitly build eight 64-bit row addresses using",
        "  AVX-512 masked loads, sign-extension, shifts, adds and mask blending.",
        "- **unpack_shuffle:** replace two word permutes with two lane-local",
        "  word unpacks and two qword permutes. Loads and prefetch remain unchanged.",
        "- **hardware_gather:** gather BF16 pairs across 16 selected rows using",
        "  `vpgatherdd`, transpose 16x16 dwords, then VNNI-pack. No prefetch.",
        "  This is an experiment restricted to the tested S<=1024 domain.",
        "- **no_prefetch:** original loads/packing with software prefetch removed",
        "  as a control to distinguish gathering from prefetch effects.", "",
        "Hardware gather must fetch the same data but scatters a vector's lanes across",
        "rows and then reconstructs row-major AMX operands. That extra work is not",
        "needed when contiguous row loads already feed the current packer.",
        "The measurements support rejecting this particular design; no PMU claim",
        "about a unique bottleneck is made.", "",
        "## Timing results", "",
        "B=32, M=1, H=64, D=512, S=1024, K=512. Three independent processes,",
        "41 samples per variant/layout/cache condition, randomized order.",
        "Inputs are prepared outside timing. Every variant allocates and releases",
        "its output inside the same Python/ctypes call. No allocation advantage",
        "is given to a candidate. These are BF16-versus-BF16 experiments, not a new",
        "comparison with the FP32 baseline.", "",
        "**Indexed** uses random selected rows of the source KV cache. **Pre-gathered**",
        "uses already-selected contiguous rows plus identity indices, preserving the",
        "allocation boundary of the previous matched-allocation experiment.",
        "The wrapper here is identical across variants but differs from the previous",
        "pybind benchmark; do not compare absolute times across those experiments.",
        "Cold tests evict 768 MiB before each call; hot tests reuse cached buffers.",
        "Both eviction and input preparation are outside the timer. Frequency is not locked.", "",
    ]
    for mode in MODES:
        lines += [f"### {mode.replace('_', ' ')}", "",
                  "| Variant | Median (us) | Latency change vs baseline | Process medians (us) |",
                  "|---|---:|---:|---|"]
        for name in NAMES:
            d = s[mode][name]
            spread = ", ".join(f"{v:.3f}" for v in d["process_medians_us"])
            lines.append(f"| {name} | {d['median_us']:.3f} | {d['latency_change_pct']:+.2f}% | {spread} |")
        lines.append("")
    lines += [
        "Negative latency change means faster. Sub-1% differences are not presented",
        "as a proven production gain: process variation and code layout remain",
        "confounders. The shuffle variant also changes sign across some cases.",
        "The no-prefetch variant improves hot-cache timing slightly but regresses",
        "the cold-cache path, so it is not a general replacement.", "",
        "## Correctness and audit trail", "",
        "Every variant passed **36 raw-bit staging cases** per process, comparing",
        "both gathered and VNNI-packed buffers with an independent indexing/layout",
        "oracle. Inputs include arbitrary 16-bit patterns (including BF16 NaNs and",
        "subnormals), duplicate indices, masks and K=0/1/7/8/15/16/63/64/65/73.",
        "All variants also matched the baseline output **bit-for-bit** on eight",
        "attention cases: two normal seeds, sharper inputs, a multi-query tail,",
        "masked first block, all-masked, one key and empty selections.",
        "UBSan trap-mode results are recorded separately in [ubsan.json](ubsan.json).",
        "This preserves existing BF16 behavior; it is not FP32/GPU bitwise parity.", "",
        "Raw samples, paired order, source/binary hashes, instruction counts,",
        "CPU affinity and diagnostic phase traces are saved in",
        "[replicate-1.json](replicate-1.json), [replicate-2.json](replicate-2.json),",
        "and [replicate-3.json](replicate-3.json). Phase traces are instrumented",
        "preallocated-output diagnostics, not the allocating headline times.",
        "No new bandwidth roofline or full-model claim is made.", "",
        "## Reproduce", "",
        "Compile on the login node; run only in an EMR Slurm allocation:",
        "```bash",
        "PYTHON=/scratch/savancha/emr-roofline-venv-20261008/bin/python",
        "KERNEL=plugin/kernels/dsa_pilot/sparse_attention_kernel",
        'BUILD="$HOME/.cache/sparse-gather-exploration"',
        'PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$KERNEL/explore_gather.py" --build "$BUILD" --build-only',
        "srun -p emr --constraint=ddr5600 -N 1 -n 1 -c 64 \\",
        "  --hint=nomultithread --exclusive -t 00:10:00 \\",
        '  bash "$KERNEL/run_gather_exploration.sh" "$BUILD" \\',
        '  "$PWD/$KERNEL/results/gather_exploration" "$PYTHON"',
        'python "$KERNEL/report_gather_exploration.py"',
        "```", "",
        "The build emits separate generated C++ files, shared libraries and disassemblies",
        "in BUILD. Production `sparse_attn.cpp` is never edited. Compiler flags are",
        "`-O3 -fopenmp -march=sapphirerapids -mprefer-vector-width=512 -std=c++17",
        "-fno-math-errno -ffp-contract=off -Werror=aggressive-loop-optimizations`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/gather_exploration")
    args = parser.parse_args()
    runs = [json.loads((args.results / f"replicate-{i}.json").read_text()) for i in range(1, 4)]
    s = summarize(runs)
    (args.results / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    (args.results / "README.md").write_text(report(s))
    print(json.dumps({mode: s[mode] for mode in MODES}, indent=2))


if __name__ == "__main__":
    main()
