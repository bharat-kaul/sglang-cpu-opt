#!/usr/bin/env python3
"""Summarize measured results and feed the repository's two report generators."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


paired = [read(f"paired-{i}.json") for i in range(1, 4)]
native = [read(f"native-{i}.json") for i in range(1, 4)]
calibration = read("calibration.json")
fingerprint = hashlib.sha256((HERE / "sparse_attn.cpp").read_bytes()).hexdigest()
assert all(p["kernel_sha256"] == fingerprint for p in paired)
assert all(n["cpus"] == list(range(64)) for n in native)
golden = paired[0]["accuracy"][0]["output_sha256"]
assert read("one-thread.json")["accuracy"][0]["output_sha256"] == golden
assert read("ubsan.json")["accuracy"][0]["output_sha256"] == golden

paired_us = {
    key: statistics.median(p[key]["median_us"] for p in paired)
    for key in ("amx_hot", "amx_cold", "fp32_pregathered_hot",
                "fp32_pregathered_cold", "fp32_with_gather_hot", "fp32_with_gather_cold")
}
native_us = {
    key: statistics.median(p[key]["median_us"] for p in native)
    for key in ("hot", "cold", "stream")
}
flops = 4 * 32 * 64 * 512 * 512
useful_bytes = 2 * (32 * 512 * 512 + 2 * 32 * 64 * 512) + 32 * 512 * 4 + 64 * 4
memory_floor_us = useful_bytes / (calibration["dram_triad_gbs"] * 1e3)
compute_floor_us = flops / (calibration["resident_amx_tflops"] * 1e6)
floor_us = max(memory_floor_us, compute_floor_us)
counter_run = read("imc-native.json")
with (RESULTS / "imc.csv").open() as f:
    counters = [row for row in csv.reader(f) if row and row[0] and not row[0].startswith("#")]
assert len(counters) == 16 and all(row[1] == "MiB" for row in counters)
traffic_bytes = sum(float(row[0]) for row in counters) * 2**20
bandwidth = traffic_bytes / counter_run["stream"]["loop_seconds"] / 1e9
summary = {
    "node": paired[0]["node"], "job": paired[0]["slurm_job"], "kernel_sha256": fingerprint,
    "paired_median_us": paired_us, "native_median_us": native_us,
    "paired_speedup_hot": paired_us["fp32_pregathered_hot"] / paired_us["amx_hot"],
    "paired_speedup_cold": paired_us["fp32_pregathered_cold"] / paired_us["amx_cold"],
    "useful_flops": flops, "useful_bytes": useful_bytes,
    "ideal_memory_floor_us": memory_floor_us, "resident_compute_floor_us": compute_floor_us,
    "streamed_useful_byte_roofline_efficiency": floor_us / native_us["stream"],
    "streamed_imc_gbs": bandwidth,
    "streamed_imc_bytes_per_call": traffic_bytes / counter_run["stream"]["calls"],
    "accuracy": paired[0]["accuracy"], "thread_count_bitwise_determinism": True,
    "ubsan_trap_mode": "PASS",
    "verdict": {
        "faster_than_465us": all(v < 465 for v in native_us.values()),
        "mixed_precision_accuracy": "PASS (not bit-identical to FP32)",
        "bandwidth_saturation": "NOT ACHIEVED",
        "scope": "standalone kernel only; no end-to-end model validation",
    },
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

common = {
    "model": "DSv4 sparse_attn ONLY (standalone kernel)",
    "node": paired[0]["node"] + "; one socket, CPU 0-63, NUMA 0, no SMT",
    "precision": "BF16 operands/output, FP32 accumulation/softmax",
    "batch": 32,
}
roofline = {
    **common, "phase": "decode / rotating inputs larger than LLC", "unit": "TF/s",
    "params": "M=1 H=64 D=512 S=1024 K=512; optimistic useful-byte lower bound",
    "model_roofline_target": flops / floor_us / 1e6,
    "model_measured": flops / native_us["stream"] / 1e6,
    "ops": [{
        "op": "sparse_attn (gather + packing + QK + softmax + PV)",
        "roofline_target": flops / floor_us / 1e6,
        "measured": flops / native_us["stream"] / 1e6,
        "share_pct": 100, "regime": "memory",
    }],
}
# Pick one actual profile (median wall), rather than combining unrelated phase minima.
profile = sorted((p["profile_stream"] for p in native), key=lambda p: p["wall_us"])[1]
names = ["gather / packing / initialization", "AMX QK", "softmax / rescale / output", "AMX PV"]
assert sum(profile["critical_phases_us"]) <= profile["wall_us"]
pivot = {
    **common, "phases": {"decode": {
        "wall_s": profile["wall_us"] * 1e-6,
        "ops": [{"op": name, "s": t * 1e-6, "bucket": "kernel"}
                for name, t in zip(names, profile["critical_phases_us"])],
    }},
}
for name, data, tool in (
    ("roofline", roofline, "roofline_vs_measured.py"),
    ("pivot", pivot, "time_attribution_pivot.py"),
):
    source = RESULTS / f"{name}-input.json"
    source.write_text(json.dumps(data, indent=2) + "\n")
    subprocess.run(
        [sys.executable, str(HERE.parents[2] / "validate" / tool),
         "--in", str(source), "--out-prefix", str(RESULTS / name)],
        check=True,
    )
print(json.dumps({k: v for k, v in summary.items() if k != "accuracy"}, indent=2))
