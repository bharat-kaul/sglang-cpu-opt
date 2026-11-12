#!/usr/bin/env python3
"""Numerical and timing screen of query/key geometry and mask density, not batch alone."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

import bench_long_prefill_sparse as donor
from check_c4_numerics import differences, gathered_fp64, load_capture
from dsv4_sparse_trial import shared_chunked
from screen_serving_opportunities import write_json


QUERIES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
KEYS = (128, 512, 1024, 1536, 3072)
DIVISORS = (1, 2, 4, 8, 16)
PATTERNS = ("block", "strided")


def coordinates():
    return [(n, k, d, p) for n in QUERIES for k in KEYS for d in DIVISORS
            for p in PATTERNS if p == "block" or d != 1]


def summarize(root, job, manifest_sha):
    """Reconcile complete replicas before emitting a descriptive crossover table."""
    root = Path(root)
    if donor.sha(root / "source-sha256.txt") != manifest_sha:
        raise RuntimeError("Unexpected frozen source manifest")
    results = root / f"job-{job}"
    if (results / "exit_code").read_text().strip() != "0":
        raise RuntimeError("Job did not exit successfully")
    names = {f"n{n}-k{k}-v{v}-{p}.json": (n, k, v, p) for n, k, v, p in coordinates()}
    identities, packets, calibrations = [], [], []
    def unique(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise RuntimeError("Duplicate JSON fields")
        return value
    def read(path):
        return json.loads(path.read_text(), object_pairs_hook=unique)
    for replica in range(3):
        folder = results / f"replicate{replica}"
        if {p.name for p in folder.glob("*.json")} != set(names) | {
                "captured.json", "calibration.json", "identity.json", "complete.json"}:
            raise RuntimeError("Missing or unexpected crossover coordinates")
        identity = read(folder / "identity.json")
        if (set(identity) != {"job", "pid", "node", "replicate", "torch", "threads", "affinity",
                              "llc_size", "flush_bytes", "capture_sha256", "native_sha256",
                              "sources", "environment", "scope"}
                or identity["torch"] != torch.__version__
                or identity["environment"] != {"OMP_NUM_THREADS": "32", "OMP_WAIT_POLICY": "active",
                                               "GOMP_SPINCOUNT": "1000000"}
                or not isinstance(identity["llc_size"], str)
                or not identity["llc_size"].endswith("K")
                or not identity["llc_size"][:-1].isdigit()
                or int(identity["llc_size"][:-1]) <= 0
                or identity["flush_bytes"] != 2 * int(identity["llc_size"][:-1]) * 1024 + 32 * 2**20):
            raise RuntimeError("Missing or invalid runtime/cache contract")
        if (identity["job"] != job or identity["replicate"] != replica
                or type(identity["pid"]) is not int or identity["pid"] <= 0
                or any(type(identity[k]) is not int for k in ("replicate", "threads", "flush_bytes"))
                or any(type(cpu) is not int for cpu in identity["affinity"])
                or identity["threads"] != 32 or identity["affinity"] != list(range(32))
                or identity["node"] != (results / "hostname.txt").read_text().strip()
                or identity["capture_sha256"] != donor.CAPTURE_HASHES[2]
                or identity["native_sha256"] != donor.NATIVE_HASH):
            raise RuntimeError("Unexpected process/input identity")
        expected_sources = {
            root / "source/plugin/validate" / name for name in (
                "screen_c4_crossover.py", "bench_long_prefill_sparse.py",
                "check_c4_numerics.py", "replay_serving_sparse.py")}
        expected_sources.update({
            root / "source/plugin/dsv4_sparse_trial.py", donor.NATIVE,
            donor.NATIVE.with_name("build.ninja"),
            donor.FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp",
            Path(torch.__file__).parent / "lib/libtorch_cpu.so",
        })
        if set(identity["sources"]) != {str(p) for p in expected_sources}:
            raise RuntimeError("Unexpected source contract")
        if any(donor.sha(p) != value for p, value in identity["sources"].items()):
            raise RuntimeError("Source/runtime digest mismatch")
        if identities and {k: v for k, v in identity.items() if k not in ("pid", "replicate")} != {
                k: v for k, v in identities[0].items() if k not in ("pid", "replicate")}:
            raise RuntimeError("Incompatible replica contracts")
        if identity["pid"] in {i["pid"] for i in identities}:
            raise RuntimeError("Duplicate process identity")
        identities.append(identity)
        if read(folder / "complete.json") != {"identity": identity, "cases": len(names)}:
            raise RuntimeError("Invalid completion record")
        cal = read(folder / "calibration.json")
        if (cal["identity"] != identity or cal["gemm_flops"] != 2 * 4096**3
                or cal["streaming_rmw_logical_bytes"] != 2 * identity["flush_bytes"]
                or set(cal["samples_ns"]) != {"fp32_gemm", "streaming_rmw"}
                or any(len(samples) != 3 or any(type(v) is not int or v <= 0 for v in samples)
                       for samples in cal["samples_ns"].values())):
            raise RuntimeError("Invalid calibration evidence")
        calibrations.append(cal)
        records = {name: read(folder / name) for name in (*names, "captured.json")}
        for name, record in records.items():
            if record["identity"] != identity:
                raise RuntimeError("Packet identity mismatch")
            n = 512
            if name in names:
                n, k, divisor, pattern = names[name]
                if ((record["n"], record["k"], record["divisor"], record["pattern"]) != names[name]
                        or any(type(record[f]) is not int for f in ("n", "k", "divisor", "h", "d",
                                                                   "visible_min", "visible_max"))
                        or type(record["density"]) not in (int, float)
                        or record["h"] != 64 or record["d"] != 512
                        or record["visible_min"] != k // divisor
                        or record["visible_max"] != k // divisor
                        or record["visible_mean"] != k / divisor
                        or record["density"] != 1 / divisor):
                    raise RuntimeError("Coordinate mismatch")
            if record["numerical_sampled_rows"] != sorted({0, n // 2, n - 1}):
                raise RuntimeError("Numerical coverage mismatch")
            if set(record["timings"]) != {"warm", "llc_evicted"}:
                raise RuntimeError("Cache mode coverage mismatch")
            for mode in record["timings"].values():
                if set(mode["samples_ns"]) != {"online", "python_gemm"}:
                    raise RuntimeError("Arm coverage mismatch")
                for arm, samples in mode["samples_ns"].items():
                    if (len(samples) != 3 or any(type(v) is not int or v <= 0 for v in samples)
                            or mode["median_ns"][arm] != statistics.median(samples)):
                        raise RuntimeError("Invalid timing evidence")
                    error = record["numerical"][arm]["relative_l2"]
                    if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1e-4:
                        raise RuntimeError("Invalid numerical screen")
                if mode["speedup"] != mode["median_ns"]["online"] / mode["median_ns"]["python_gemm"]:
                    raise RuntimeError("Invalid derived speedup")
        packets.append(records)
    rows = {}
    for name in (*names, "captured.json"):
        rows[name] = {}
        for mode in ("warm", "llc_evicted"):
            ratios = [p[name]["timings"][mode]["speedup"] for p in packets]
            rows[name][mode] = {"median_speedup": statistics.median(ratios),
                               "min_speedup": min(ratios), "max_speedup": max(ratios),
                               "process_speedups": ratios}
    return {
        "job": job, "source_manifest_sha256": manifest_sha, "identities": identities,
        "aggregation_source": str(Path(__file__)), "aggregation_source_sha256": donor.sha(__file__),
        "calibrations": calibrations,
        "estimator": "Median/min/max of three independent-process ratios of arm medians; "
                     "each arm median uses three order-varied trials.",
        "scope": "Descriptive Python-candidate crossover, not a native dispatch acceptance gate.",
        "rows": rows,
    }


def calibration(flush):
    generator = torch.Generator().manual_seed(20261111)
    a = torch.randn(4096, 4096, generator=generator)
    b = torch.randn(4096, 4096, generator=generator)
    result = a @ b
    timings = {"fp32_gemm": [], "streaming_rmw": []}
    for _ in range(3):
        start = time.perf_counter_ns()
        result = a @ b
        timings["fp32_gemm"].append(time.perf_counter_ns() - start)
        if not torch.isfinite(result).all():
            raise RuntimeError("Invalid calibration output")
        start = time.perf_counter_ns()
        flush.add_(1)
        timings["streaming_rmw"].append(time.perf_counter_ns() - start)
    return {
        "samples_ns": timings, "gemm_flops": 2 * 4096**3,
        "streaming_rmw_logical_bytes": 2 * flush.numel(),
        "note": "Shape-specific achievable FP32 GEMM and streaming read-modify-write proxies; "
                "not ISA peak or a universal DRAM ceiling. No host frequency controls changed.",
    }


def geometry(sample, n, k, divisor, pattern):
    q = sample["q"].index_select(0, torch.arange(n) % sample["q"].shape[0]).contiguous()
    kv = sample["kv"].index_select(0, torch.arange(k) % sample["kv"].shape[0]).contiguous()
    visible = max(1, k // divisor)
    keys, rows = torch.arange(k)[None, :], torch.arange(n)[:, None]
    if pattern == "block":
        mask = ((keys - (rows % divisor) * visible) % k) < visible
    elif pattern == "strided":
        mask = (keys % divisor) == (rows % divisor)
    else:
        raise ValueError("Unknown mask geometry")
    return q, kv, mask.contiguous(), sample["sink"], sample["scale"]


def timed(args, native, flush):
    arms = {"online": native.sparse_attend_masked, "python_gemm": shared_chunked}
    numerical = {}
    first = {}
    # A bounded, independent FP64 reference checks matched rows, not GPU precision semantics.
    selected = torch.tensor(sorted({0, args[0].shape[0] // 2, args[0].shape[0] - 1}))
    reference = gathered_fp64(args[0][selected], args[1], args[2][selected], args[3], args[4])
    for arm, function in arms.items():
        output = function(*args)
        if output.shape != args[0].shape or output.dtype != torch.float32 or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid {arm} output")
        first[arm] = output
        numerical[arm] = differences(output[selected], reference)
        if numerical[arm]["relative_l2"] is None or numerical[arm]["relative_l2"] > 1e-4:
            raise RuntimeError(f"{arm} exceeds screening tolerance; not an acceptance threshold")
    timings = {}
    for mode in ("warm", "llc_evicted"):
        samples = {arm: [] for arm in arms}
        for rep in range(3):
            names = tuple(arms) if rep % 2 == 0 else tuple(reversed(arms))
            for arm in names:
                if mode == "llc_evicted":
                    flush.add_(1)
                start = time.perf_counter_ns()
                output = arms[arm](*args)
                elapsed = time.perf_counter_ns() - start
                if elapsed <= 0 or elapsed >= 30_000_000_000:
                    raise RuntimeError("Invalid or over-budget timing")
                if not donor.same_bytes(output, first[arm]):
                    raise RuntimeError("Repeated output bytes differ")
                samples[arm].append(elapsed)
        medians = {arm: statistics.median(v) for arm, v in samples.items()}
        timings[mode] = {
            "samples_ns": samples, "median_ns": medians,
            "speedup": medians["online"] / medians["python_gemm"],
        }
    return {"timings": timings, "numerical_sampled_rows": selected.tolist(), "numerical": numerical}


def run(output, replicate):
    output.mkdir()
    if torch.get_num_threads() != 32 or sorted(os.sched_getaffinity(0)) != list(range(32)):
        raise RuntimeError("Requires fixed32 CPU0-31")
    if donor.sha(donor.NATIVE) != donor.NATIVE_HASH:
        raise RuntimeError("Baseline binary changed")
    expected_cpp = donor.FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp"
    if str(expected_cpp) not in donor.NATIVE.with_name("build.ninja").read_text():
        raise RuntimeError("Baseline build source mismatch")
    spec = importlib.util.spec_from_file_location("dsv4ck_sa", donor.NATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sample = load_capture(2)
    original = native.sparse_attend_masked(*(sample[k] for k in ("q", "kv", "mask", "sink", "scale")))
    if not donor.same_bytes(original, sample["output"]):
        raise RuntimeError("Baseline does not reproduce original captured bytes")
    cache_size = Path("/sys/devices/system/cpu/cpu0/cache/index3/size").read_text().strip()
    if not cache_size.endswith("K") or not cache_size[:-1].isdigit():
        raise RuntimeError("Unrecognized LLC capacity")
    flush = torch.zeros(2 * int(cache_size[:-1]) * 1024 + 32 * 2**20, dtype=torch.uint8)
    identity = {
        "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(), "node": os.uname().nodename,
        "replicate": replicate, "torch": torch.__version__, "threads": 32,
        "affinity": sorted(os.sched_getaffinity(0)), "llc_size": cache_size,
        "flush_bytes": flush.numel(), "capture_sha256": donor.CAPTURE_HASHES[2],
        "native_sha256": donor.NATIVE_HASH,
        "sources": {str(p): donor.sha(p) for p in (
            Path(__file__), Path(donor.__file__), Path(__file__).parents[1] / "dsv4_sparse_trial.py",
            Path(__file__).with_name("check_c4_numerics.py"),
            Path(__file__).with_name("replay_serving_sparse.py"),
            donor.NATIVE, donor.NATIVE.with_name("build.ninja"), expected_cpp,
            Path(torch.__file__).parent / "lib/libtorch_cpu.so",
        )},
        "environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")},
        "scope": "Full real capture plus synthetic geometry using captured q/KV values; "
                 "not a request-batch serving sweep. FP64 checks three rows per case; "
                 "1e-4 relativeL2 is a corruption screen, not production arithmetic acceptance.",
    }
    write_json(output / "identity.json", identity)
    write_json(output / "calibration.json", {"identity": identity, **calibration(flush)})
    captured_args = tuple(sample[k] for k in ("q", "kv", "mask", "sink", "scale"))
    write_json(output / "captured.json", {"identity": identity, **timed(captured_args, native, flush)})
    completed = 0
    points = coordinates()
    if replicate % 2:
        points.reverse()
    for n, k, divisor, pattern in points:
        args = geometry(sample, n, k, divisor, pattern)
        if (args[0].shape != (n, sample["q"].shape[1], sample["q"].shape[2])
                or args[1].shape != (k, sample["kv"].shape[1])
                or args[2].shape != (n, k) or args[2].dtype != torch.bool):
            raise RuntimeError("Evaluated geometry differs from declared coordinate")
        visible = args[2].sum(1)
        if not torch.equal(visible, torch.full((n,), k // divisor)):
            raise RuntimeError("Evaluated mask density differs from declared coordinate")
        record = {
            "identity": identity, "n": n, "k": k, "h": args[0].shape[1], "d": args[0].shape[2],
            "divisor": divisor, "pattern": pattern, "visible_min": int(visible.min()),
            "visible_max": int(visible.max()), "visible_mean": float(visible.double().mean()),
            "density": float(args[2].double().mean()), **timed(args, native, flush),
        }
        write_json(output / f"n{n}-k{k}-v{divisor}-{pattern}.json", record)
        completed += 1
        if completed % 25 == 0:
            print(f"C4_CROSSOVER_PROGRESS replicate={replicate} cases={completed}/{len(points)}", flush=True)
    if any(donor.sha(p) != value for p, value in identity["sources"].items()):
        raise RuntimeError("Source/runtime changed")
    write_json(output / "complete.json", {"identity": identity, "cases": completed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    run(args.output, args.replicate)
