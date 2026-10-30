#!/usr/bin/env python3
"""Validate and summarize the original-policy, matched-allocation comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SHAPE = {"B": 32, "M": 1, "H": 64, "D": 512, "source_S": 1024, "selected_K": 512}
CASES = {"normal", "seed7", "sharp", "tail", "three_groups", "one_key"}


def summarize(runs):
    if len(runs) != 3 or {r["replicate"] for r in runs} != {1, 2, 3}:
        raise ValueError("Require three numbered process replicates")
    first = runs[0]
    if len({(r["job"], r["node"], r["pid"]) for r in runs}) != 3:
        raise ValueError("Duplicate process identity")
    for key in ("job", "node"):
        if not isinstance(first[key], str) or not first[key]:
            raise ValueError(f"Missing {key}")
    names = ("sparse_attn.cpp", "sparse_attn.h", "matched_allocations.cpp",
             "bench_matched_allocations.py", "bench.py", "run_matched_allocations.sh")
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    for r in runs:
        if r["shape"] != SHAPE or r["threads"] != 64 or r["worker_affinity"] != [[i] for i in range(64)]:
            raise ValueError("Shape or affinity mismatch")
        for key in ("job", "node", "build", "contract", "input_sha256"):
            if r[key] != first[key]:
                raise ValueError(f"Incompatible replicates: {key}")
        if r["build"]["sources"] != hashes:
            raise ValueError("Results do not match current source")
        if len(r["accuracy"]) != len(CASES) or {a["case"] for a in r["accuracy"]} != CASES:
            raise ValueError("Missing accuracy coverage")
        for a in r["accuracy"]:
            if a["matches_indexed_core_bitwise"] is not True or a["both_outputs_fresh_and_repeatable"] is not True:
                raise ValueError("Output ownership or core parity failed")
            for name, l2, linf in (("fp32", 0.006, 0.012), ("block_reference", 0.003, 0.008)):
                e = a[name]
                if e["finite"] is not True or not (0 <= e["relative_l2"] < l2 and 0 <= e["relative_linf"] < linf):
                    raise ValueError("Numerical accuracy failed")
        for name, output_bytes in (("fp32", 4194304), ("bf16", 2097152)):
            audit = r["allocation_audit"][name]
            if audit["dtype_conversion_copies"] != 0 or audit["output_bytes"] != output_bytes:
                raise ValueError("Input/output allocation contract failed")
            if not any(e["bytes"] == output_bytes for e in audit["positive_self_allocations"]):
                raise ValueError("Output allocation missing from profiler")
        for mode in ("cold", "hot"):
            if len(r[mode]["order"]) != 51 or any(sorted(o) != ["bf16", "fp32"] for o in r[mode]["order"]):
                raise ValueError("Incomplete paired measurement order")
            for name in ("fp32", "bf16"):
                timing = r[mode]["timings"][name]
                samples = timing["samples_us"]
                if len(samples) != 51 or not all(math.isfinite(v) and v > 0 for v in samples):
                    raise ValueError("Invalid timing samples")
                if timing["median_us"] != statistics.median(samples):
                    raise ValueError("Stored median differs from raw samples")
    summary = {
        "node": first["node"], "job": first["job"], "shape": SHAPE,
        "build": first["build"], "contract": first["contract"],
        "processes": [r["pid"] for r in runs], "accuracy": first["accuracy"],
        "allocation_audit": first["allocation_audit"],
        "estimator": "median of three process medians, each over 51 calls",
    }
    for mode in ("cold", "hot"):
        timings = {name: [r[mode]["timings"][name]["median_us"] for r in runs] for name in ("fp32", "bf16")}
        summary[mode] = {
            name: {"median_us": statistics.median(v), "process_medians_us": v,
                   "range_us": [min(v), max(v)]} for name, v in timings.items()
        }
        summary[mode]["speedup"] = statistics.median(timings["fp32"]) / statistics.median(timings["bf16"])
        summary[mode]["per_process_speedups"] = [a / b for a, b in zip(timings["fp32"], timings["bf16"])]
    return summary


def report(s):
    lines = [
        "# Sparse attention with matched allocation and timing policy", "",
        f"Slurm job **{s['job']}**, node **{s['node']}**, one EMR 8592+ socket,",
        "**64 physical cores (CPU 0-63), NUMA memory 0, no SMT**.", "",
        "## What is identical", "",
        "- Both allocate, gather and prepare their inputs **before** timing.",
        "- Both receive contiguous Q[N,H,512], already-selected KV[N,K,512], and FP32 sink.",
        "- Each calls `to(native_dtype).contiguous()` internally, matching the original",
        "  FP32 input-normalization method. These are no-ops for the prepared inputs.",
        "- Both allocate a fresh output tensor using the **same PyTorch CPU allocator**",
        "  inside every timed function call, and release that output before the timer stops.",
        "- Both are measured by the same Python timing helper and pybind interface,",
        "  with randomized paired order and identical CPU/memory binding.",
        "- Cold-cache samples each evict 768 MiB outside the timer.", "",
        "The FP32 function is copied verbatim from pinned commit",
        "`7305559f0eb6ac9c2be127b8efb2e5bff34712ca`: no `bmm_out`, no preallocated output,",
        "and no modified math. BF16 uses a new allocating wrapper around the unchanged core.", "",
        "## What remains different by design", "",
        "Native dtype determines bytes, not allocation policy: FP32 output is 4 MiB;",
        "BF16 output is 2 MiB. FP32 allocates a 4 MiB score tensor inside its first",
        "batched GEMM. The BF16 algorithm does not materialize that full tensor; it",
        "uses worker-local packing and online softmax scratch. No dummy allocation",
        "is added to either algorithm to conceal that difference.",
        "BF16 receives an identity-index tensor prepared outside timing solely for",
        "the existing indexed C ABI; actual source-cache selection is already done.",
        "Its sequential staging/packing remains timed, as does FP32's library workspace.",
        "A warmed allocator may reuse freed pages; fresh allocation does not mean",
        "a fresh OS mapping. Internal BLAS workspace behavior is library-controlled.", "",
        "## Results", "",
        "B=32, M=1, H=64, D=512, source S=1024, selected K=512.",
        "Median of three process medians, 51 samples per path/cache condition.", "",
        "| Condition | Original FP32 (us) | BF16, matched policy (us) | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("cold", "hot"):
        d = s[mode]
        lines.append(f"| {mode} | {d['fp32']['median_us']:.3f} | {d['bf16']['median_us']:.3f} | {d['speedup']:.2f}x |")
    lines += ["", "| Condition | FP32 process medians (us) | BF16 process medians (us) |",
              "|---|---|---|"]
    for mode in ("cold", "hot"):
        a = ", ".join(f"{v:.3f}" for v in s[mode]["fp32"]["process_medians_us"])
        b = ", ".join(f"{v:.3f}" for v in s[mode]["bf16"]["process_medians_us"])
        lines.append(f"| {mode} | {a} | {b} |")
    e = next(a["fp32"] for a in s["accuracy"] if a["case"] == "normal")
    lines += [
        "", "## Allocation audit and accuracy", "",
        "Untimed PyTorch memory profiling observed:", "",
        "| Path | Input dtype-conversion copies | Recorded positive allocations |",
        "|---|---:|---|",
    ]
    for name in ("fp32", "bf16"):
        audit = s["allocation_audit"][name]
        allocations = ", ".join(f"{e['op']}: {e['bytes']:,} bytes" for e in audit["positive_self_allocations"])
        lines.append(f"| {name} | {audit['dtype_conversion_copies']} | {allocations} |")
    lines += [
        "", "All three processes test two normal seeds, sharper inputs, partial blocks,",
        "multiple queries/96 heads, and one key. Both outputs are freshly allocated",
        "and repeatable; the BF16 wrapper is bit-identical to the existing indexed",
        "BF16 core on the same selected values.",
        f"Normal-input FP32-reference cosine: **{e['cosine']:.10f}**, relative L2:",
        f"**{e['relative_l2']:.8f}**, maximum absolute error: **{e['max_abs']:.8f}**.",
        f"FP32 bit mismatches: **{e['fp32_bit_mismatches']:,} / {e['elements']:,}**.",
        "Accuracy follows the agreed mixed-precision tolerance, not FP32 bit equality.", "",
        "**Conclusion:** the implementation-level speedup survives matching the",
        "original FP32 allocation/timing policy. This is not an attribution of the",
        "gain to precision alone or AMX alone. No new bandwidth-saturation or",
        "whole-model throughput claim is made.", "",
        "Evidence: [summary.json](summary.json), [build.json](build.json),",
        "[original FP32 source](fp32_original.cpp),",
        "[replicate 1](replicate-1.json), [replicate 2](replicate-2.json),",
        "[replicate 3](replicate-3.json).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results/matched_allocations")
    args = parser.parse_args()
    runs = [json.loads((args.results / f"replicate-{i}.json").read_text()) for i in range(1, 4)]
    summary = summarize(runs)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "README.md").write_text(report(summary))
    print(json.dumps({mode: summary[mode] for mode in ("cold", "hot")}, indent=2))


if __name__ == "__main__":
    main()
