#!/usr/bin/env python3
"""Matched socket calibration inputs, not a whole-model achievable roofline."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


LAYOUTS = {"emr": ((0, 0), (0, 32)), "gnr": ((0, 0), (1, 42), (2, 85))}
CASES = ("read", "copy", "triad", "bf16_m32", "bf16_m512", "bf16_m4096")
ROUNDS = 3
ITERATIONS = 7
ARRAY_BYTES = 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def wait(path, children=(), seconds=180):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if any(child.poll() is not None for child in children):
            raise RuntimeError(f"Calibration child exited while waiting for {path}")
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)


def aggregate(rows, platform, case):
    if case not in CASES:
        raise ValueError("Unknown calibration case")
    count = len(LAYOUTS[platform])
    if len(rows) != count or {r["rank"] for r in rows} != set(range(count)):
        raise ValueError("Incomplete replica coverage")
    for row in rows:
        if type(row["rank"]) is not int or type(row["round"]) is not int or row["round"] not in range(ROUNDS):
            raise ValueError("Invalid process coordinate")
        node, first = LAYOUTS[platform][row["rank"]]
        if (row["platform"] != platform or row["case"] != case or row["numa"] != node
                or row["cpus"] != list(range(first, first + 32))
                or row["threads"] != 32 or len(row["samples_ns"]) != ITERATIONS
                or any(type(t) is not int or t <= 0 for t in row["samples_ns"])):
            raise ValueError("Calibration identity or timing mismatch")
        if (type(row["start_ns"]) is not int or type(row["end_ns"]) is not int
                or row["end_ns"] - row["start_ns"] < sum(row["samples_ns"])):
            raise ValueError("Invalid clock interval")
    for key in ("job", "host", "round", "script_sha256", "torch", "work_per_iteration", "unit"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"Mixed calibration {key}")
    if len({r["pid"] for r in rows}) != count:
        raise ValueError("Duplicate calibration process")
    if max(r["start_ns"] for r in rows) >= min(r["end_ns"] for r in rows):
        raise ValueError("Calibration replicas did not overlap")
    work = rows[0]["work_per_iteration"]
    expected_work = ({"read": 1, "copy": 2, "triad": 3}[case] * ARRAY_BYTES
                     if case in ("read", "copy", "triad")
                     else 2 * int(case.removeprefix("bf16_m")) * 4096 * 4096)
    expected_unit = "flops" if case.startswith("bf16_") else "bytes"
    if type(work) is not int or work != expected_work or rows[0]["unit"] != expected_unit:
        raise ValueError("Invalid work model")
    span_s = (max(r["end_ns"] for r in rows) - min(r["start_ns"] for r in rows)) / 1e9
    rate = work * ITERATIONS * count / span_s
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Invalid calibration rate")
    return {
        "case": case, "round": rows[0]["round"], "unit": rows[0]["unit"] + "/s",
        "socket_rate": rate, "common_span_s": span_s,
        "replicas": rows, "estimator": "sum(logical work) / common wall span",
    }


def worker(root, platform, rank, repeat):
    import torch

    node, first = LAYOUTS[platform][rank]
    cpus = list(range(first, first + 32))
    if sorted(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError("Unexpected calibration CPU allocation")
    torch.set_num_threads(32)
    if torch.get_num_threads() != 32:
        raise RuntimeError("Calibration must use 32 threads")
    folder = root / f"round{repeat}" / f"rank{rank}"
    folder.mkdir()
    identity = {
        "platform": platform, "rank": rank, "round": repeat, "numa": node, "cpus": cpus,
        "job": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename, "pid": os.getpid(),
        "threads": 32, "torch": torch.__version__, "script_sha256": digest(__file__),
        "numa_policy": subprocess.check_output(["numactl", "--show"], text=True),
    }
    write(folder / "identity.json", identity)
    n = ARRAY_BYTES // 4
    a = torch.empty(n)
    b = torch.ones(n)
    c = torch.full((n,), 2.0)
    a.fill_(0)
    for case in CASES:
        if case == "read":
            operation = lambda: b.sum()
            work, unit = ARRAY_BYTES, "bytes"
        elif case == "copy":
            operation = lambda: a.copy_(b)
            work, unit = 2 * ARRAY_BYTES, "bytes"
        elif case == "triad":
            operation = lambda: torch.add(b, c, alpha=3.0, out=a)
            work, unit = 3 * ARRAY_BYTES, "bytes"
        else:
            m = int(case.removeprefix("bf16_m"))
            left = torch.full((m, 4096), 0.125, dtype=torch.bfloat16)
            right = torch.full((4096, 4096), 0.125, dtype=torch.bfloat16)
            result = torch.empty((m, 4096), dtype=torch.bfloat16)
            operation = lambda: torch.mm(left, right, out=result)
            work, unit = 2 * m * 4096 * 4096, "flops"
        for _ in range(3):
            operation()
        write(folder / f"{case}.ready.json", identity)
        go = folder.parent / f"{case}.go"
        wait(go)
        deadline = json.loads(go.read_text())["start_ns"]
        time.sleep(max(0, (deadline - time.perf_counter_ns()) / 1e9 - 0.002))
        while time.perf_counter_ns() < deadline:
            pass
        times = []
        start = time.perf_counter_ns()
        for _ in range(ITERATIONS):
            before = time.perf_counter_ns()
            output = operation()
            times.append(time.perf_counter_ns() - before)
        end = time.perf_counter_ns()
        observed = float(output.reshape(-1)[0])
        expected = float(n) if case == "read" else 1.0 if case == "copy" else 7.0 if case == "triad" else 64.0
        if observed != expected:
            raise RuntimeError(f"Calibration output invalid: {case}: {observed}")
        write(folder / f"{case}.json", {
            **identity, "case": case, "samples_ns": times, "start_ns": start, "end_ns": end,
            "work_per_iteration": work, "unit": unit, "output_check": observed,
            "working_set_bytes": (3 * ARRAY_BYTES if unit == "bytes" else
                                  (m * 4096 * 2 + 4096 * 4096) * 2),
        })
        wait(folder.parent / f"{case}.accepted")
    write(folder / "complete.json", identity)


def coordinate(root, platform):
    root.mkdir()
    topology = subprocess.check_output(["lscpu", "-p=CPU,NODE,SOCKET,CORE"], text=True)
    cpus = {int(parts[0]): tuple(map(int, parts[1:]))
            for line in topology.splitlines() if not line.startswith("#")
            for parts in [line.split(",")]}
    for node, first in LAYOUTS[platform]:
        group = [cpus[c] for c in range(first, first + 32)]
        if any(n != node or socket != 0 for n, socket, _ in group) or len({c[2] for c in group}) != 32:
            raise RuntimeError("Physical CPU topology differs from measured model layout")
    (root / "topology.txt").write_text(topology)
    (root / "numa.txt").write_text(subprocess.check_output(["numactl", "--hardware"], text=True))
    (root / "cpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (root / "meminfo.txt").write_text(Path("/proc/meminfo").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from uarch_perf_probe.hygiene import env_report
    write(root / "uarch-environment.json", env_report())
    write(root / "identity.json", {
        "job": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
        "platform": platform, "script_sha256": digest(__file__), "created_ns": time.time_ns(),
        "scope": "calibration inputs only; warm-library GEMM is not a packed-MoE or model ceiling",
        "memory_note": "logical read/copy/triad bytes; write-allocate/controller traffic not inferred",
    })
    results = []
    for repeat in range(ROUNDS):
        folder = root / f"round{repeat}"
        folder.mkdir()
        children, streams = [], []
        try:
            for rank, (node, first) in enumerate(LAYOUTS[platform]):
                stream = (folder / f"rank{rank}.log").open("x")
                streams.append(stream)
                env = {**os.environ, "OMP_NUM_THREADS": "32", "OMP_WAIT_POLICY": "active",
                       "GOMP_SPINCOUNT": "1000000", "MKL_NUM_THREADS": "32",
                       "PYTHONDONTWRITEBYTECODE": "1"}
                children.append(subprocess.Popen([
                    "numactl", f"--physcpubind={first}-{first+31}", f"--membind={node}",
                    sys.executable, "-u", __file__, "worker", "--root", str(root),
                    "--platform", platform, "--rank", str(rank), "--round", str(repeat),
                ], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
            for case in CASES:
                for rank in range(len(children)):
                    wait(folder / f"rank{rank}" / f"{case}.ready.json", children)
                write(folder / f"{case}.go", {"start_ns": time.perf_counter_ns() + 500_000_000})
                rows = []
                for rank in range(len(children)):
                    path = folder / f"rank{rank}" / f"{case}.json"
                    wait(path, children)
                    row = json.loads(path.read_text())
                    if (row["pid"] != children[rank].pid or row["job"] != os.environ["SLURM_JOB_ID"]
                            or row["host"] != os.uname().nodename or row["round"] != repeat):
                        raise RuntimeError("Calibration result is not from the expected process")
                    rows.append(row)
                result = aggregate(rows, platform, case)
                if result["replicas"][0]["script_sha256"] != digest(__file__):
                    raise RuntimeError("Calibration source changed")
                write(folder / f"{case}.aggregate.json", result)
                results.append(result)
                (folder / f"{case}.accepted").touch(exist_ok=False)
                print(f"CALIBRATION {platform} round={repeat} case={case} rate={result['socket_rate']:.6g}", flush=True)
            for child in children:
                if child.wait(timeout=60) != 0:
                    raise RuntimeError("Calibration worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
            for stream in streams:
                stream.close()
    summary = {}
    for case in CASES:
        samples = [r["socket_rate"] for r in results if r["case"] == case]
        summary[case] = {"samples": samples, "median": statistics.median(samples),
                         "min": min(samples), "max": max(samples)}
    write(root / "complete.json", {
        "job": os.environ["SLURM_JOB_ID"], "script_sha256": digest(__file__),
        "platform": platform, "summary": summary,
        "estimator": "median of three independent concurrent-process rounds",
        "status": "CALIBRATION_ONLY",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("coordinate", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--platform", choices=tuple(LAYOUTS), required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    if args.action == "coordinate":
        coordinate(args.root, args.platform)
    else:
        worker(args.root, args.platform, args.rank, args.round)
