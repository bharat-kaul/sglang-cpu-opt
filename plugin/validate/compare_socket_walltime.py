#!/usr/bin/env python3
"""Matched real-weight EMR/GNR domain and concurrent-socket wall-time probes."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from c4_accuracy_wave import (
    ENV, MODEL, PERF_POINTS, PERF_SAMPLING, case_events, performance_sample,
)
from screen_serving_opportunities import BASE_COMMIT, digest, write_json


LAYOUTS = {
    "emr": ((0, 0), (0, 32)),
    "gnr": ((0, 0), (1, 42), (2, 85)),
}
ACCURACY_ROOT = Path("/scratch/bkaul/dsv4-c4-accuracy/wave-20261110-wJzAC6")
ROUNDS = 3


def cases(platform):
    count = len(LAYOUTS[platform])
    groups = [(i,) for i in range(count)] + [tuple(range(count))]
    for ranks in groups:
        group = "single" + str(ranks[0]) if len(ranks) == 1 else "socket"
        for batch, length in PERF_POINTS:
            for repeat in range(-1, ROUNDS if group == "socket" else 1):
                stage = "warm" if repeat == -1 else "measured"
                arms = ("baseline", "candidate") if (ranks[0] + repeat) % 2 == 0 else ("candidate", "baseline")
                for arm in arms:
                    name = f"{group}-b{batch}-s{length}-r{repeat}-{arm}"
                    yield {
                        "case": name, "ranks": list(ranks), "batch": batch, "arm": arm,
                        "length": length, "stage": stage, "repeat": repeat, "group": group,
                    }


def plan(root):
    p = json.loads((root / "plan.json").read_text())
    if (p["base_commit"] != BASE_COMMIT or p["rounds"] != ROUNDS
            or p["layouts"] != {k: [list(v) for v in rows] for k, rows in LAYOUTS.items()}
            or p["cases"] != {k: list(cases(k)) for k in LAYOUTS}
            or p["sampling"] != PERF_SAMPLING or p["environment"] != ENV
            or p["driver_sha256"] != digest(__file__)
            or p["source_manifest_sha256"] != digest(root / "source-sha256.txt")
            or p["inputs_sha256"] != digest(root / "inputs.json")
            or p["model_config_sha256"] != digest(MODEL / "config.json")):
        raise RuntimeError("Socket comparison plan mismatch")
    for name, value in p["installed_sources"].items():
        if digest(name) != value:
            raise RuntimeError("Installed runtime changed")
    for name, value in p["compute_sources"].items():
        if digest(root / "source" / name) != value:
            raise RuntimeError("Compute source differs from EMR accuracy wave")
    return p


def prepare(root):
    if (root / "plan.json").exists():
        raise RuntimeError("Refusing to overwrite plan")
    if (root / "source-base.txt").read_text().strip() != BASE_COMMIT:
        raise RuntimeError("Unexpected baseline")
    previous = json.loads((ACCURACY_ROOT / "plan.json").read_text())
    inputs = ACCURACY_ROOT / "performance-inputs.json"
    if digest(inputs) != previous["performance_inputs_sha256"]:
        raise RuntimeError("Accuracy-wave performance inputs changed")
    compute = {}
    for line in (ACCURACY_ROOT / "source-sha256.txt").read_text().splitlines():
        value, name = line.split("  ", 1)
        name = name.removeprefix("./")
        if name.startswith("plugin/intel_cpu_models/") or name == "plugin/dsv4_sparse_trial.py":
            if digest(root / "source" / name) != value:
                raise RuntimeError(f"Compute source differs from EMR: {name}")
            compute[name] = value
    if not compute:
        raise RuntimeError("Missing EMR compute-source identity")
    write_json(root / "inputs.json", json.loads(inputs.read_text()))
    write_json(root / "plan.json", {
        "base_commit": BASE_COMMIT, "rounds": ROUNDS,
        "layouts": LAYOUTS, "cases": {k: list(cases(k)) for k in LAYOUTS},
        "sampling": PERF_SAMPLING, "environment": ENV,
        "driver_sha256": digest(__file__),
        "source_manifest_sha256": digest(root / "source-sha256.txt"),
        "inputs_sha256": digest(root / "inputs.json"),
        "model_config_sha256": digest(MODEL / "config.json"),
        "installed_sources": previous["installed_sources"],
        "compute_sources": compute,
        "emr_wave_root": str(ACCURACY_ROOT), "emr_plan_sha256": digest(ACCURACY_ROOT / "plan.json"),
        "scope": "Native baseline and unpromoted C4 candidate separately. Fixed32 per replica. "
                 "One warm and one measured pair per point per isolated process; three measured "
                 "socket pairs after warmup, not independent reloads. Full43 real weights. Four decode steps: "
                 "not long steady-state decode. One socket per host, one host per platform.",
    })


def memory(pid):
    status = Path(f"/proc/{pid}/status").read_text()
    fields = {}
    for name in ("VmRSS", "VmHWM", "VmSwap"):
        match = re.search(rf"^{name}:\s+(\d+) kB$", status, re.M)
        if match is None:
            raise RuntimeError(f"Missing process memory field {name}")
        fields[name + "_kib"] = int(match[1])
    raw = Path(f"/proc/{pid}/numa_maps").read_text()
    nodes = {}
    for line in raw.splitlines():
        pages = re.findall(r"\bN(\d+)=(\d+)\b", line)
        if not pages:
            continue
        size = re.search(r"\bkernelpagesize_kB=(\d+)\b", line)
        if size is None:
            raise RuntimeError("Unstamped NUMA page size")
        for node, count in pages:
            nodes[node] = nodes.get(node, 0) + int(count) * int(size[1])
    return {"pid": pid, **fields, "numa_kib": nodes}


def validate_locality(record, node):
    total = sum(record["numa_kib"].values())
    if (total <= 0 or record["VmSwap_kib"] != 0
            or record["numa_kib"].get(str(node), 0) / total < 0.99):
        raise RuntimeError(
            f"Replica memory rejected: pid={record['pid']} swap_kib={record['VmSwap_kib']} "
            f"resident_numa_kib={record['numa_kib']} required_node={node}; "
            "requires no swap and at least99% NUMA-local residency")


def wait_file(path, children=(), timeout=3600):
    deadline = time.monotonic() + timeout
    while not path.exists():
        for child in children:
            if child.poll() is not None:
                raise RuntimeError(f"Replica exited early: PID{child.pid}, code{child.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.05)


def topology(platform, out):
    raw = subprocess.check_output(["lscpu", "-p=CPU,NODE,SOCKET,CORE"], text=True)
    (out / "cpu-topology.txt").write_text(raw)
    (out / "numa-topology.txt").write_text(
        subprocess.check_output(["numactl", "--hardware"], text=True))
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    cpus = {int(row[0]): tuple(map(int, row[1:])) for line in raw.splitlines()
            if not line.startswith("#") and (row := line.split(","))}
    for node, first in LAYOUTS[platform]:
        group = [cpus[c] for c in range(first, first + 32)]
        if any(n != node or socket != 0 for n, socket, _ in group) or len({c[2] for c in group}) != 32:
            raise RuntimeError("Unexpected physical CPU/socket/NUMA layout")


def worker(root, platform, rank):
    p = plan(root)
    node, first = LAYOUTS[platform][rank]
    out = root / platform / f"rank{rank}"
    out.mkdir()
    expected_env = {**ENV, "SGLANG_CPU_OMP_THREADS_BIND": f"{first}-{first + 31}"}
    if any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise RuntimeError("Replica environment mismatch")
    assigned = [s for s in cases(platform) if rank in s["ranks"]]
    write_json(out / "control-plan.json", {s["case"]: s["arm"] for s in assigned})
    write_json(out / "control.json", {"case": "initializing", "arm": "baseline"})
    (out / "events").mkdir()
    os.environ.update({
        "DSV4_C4_ACCURACY_CONTROL": str(out / "control.json"),
        "DSV4_C4_ACCURACY_PLAN": str(out / "control-plan.json"),
        "DSV4_C4_ACCURACY_OUT": str(out / "events"),
    })
    import intel_cpu_models  # noqa: F401
    import sglang as sgl
    import sgl_kernel
    import torch

    identity = {
        "platform": platform, "rank": rank, "node": os.uname().nodename,
        "numa": node, "cpus": list(range(first, first + 32)),
        "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
        "plan_sha256": digest(root / "plan.json"), "environment": expected_env,
        "common_ops_sha256": digest(sgl_kernel.common_ops.__file__),
        "torch": torch.__version__, "sglang": sgl.__version__,
    }
    write_json(out / "identity.json", identity)
    engine = sgl.Engine(
        model_path=str(MODEL), device="cpu", tp_size=1, dtype="bfloat16",
        disable_overlap_schedule=True, trust_remote_code=True, mem_fraction_static=0.5,
        log_level="warning", watchdog_timeout=7000, chunked_prefill_size=512,
        disable_radix_cache=True, context_length=4096, max_total_tokens=131072,
        swa_full_tokens_ratio=0.5, max_running_requests=256, enable_mixed_chunk=False, random_seed=0,
    )
    try:
        runtime = engine.get_server_info()
        if runtime["enable_mixed_chunk"] is not False:
            raise RuntimeError("Mixed prefill/decode invalidates candidate gating")
        write_json(out / "runtime.json", runtime)
        event_files = list((out / "events").glob("events.*.jsonl"))
        if len(event_files) != 1:
            raise RuntimeError("Cannot identify the loaded inference process")
        inference_pid = int(event_files[0].stem.split(".")[-1])
        load_memory = memory(inference_pid)
        write_json(out / "load-memory.json", {
            "scheduler": load_memory, "parent": memory(os.getpid()),
            "system": Path("/proc/meminfo").read_text(),
            "domain": Path(f"/sys/devices/system/node/node{node}/meminfo").read_text(),
        })
        validate_locality(load_memory, node)
        write_json(out / "ready.json", identity)
        prompts = json.loads((root / "inputs.json").read_text())
        for spec in assigned:
            case = spec["case"]
            command = root / platform / "commands" / case
            wait_file(command.with_suffix(".prepare"))
            if engine.flush_cache().success is not True:
                raise RuntimeError("Scheduler reset failed")
            write_json(out / "control.json", {"case": case, "arm": spec["arm"]})
            write_json(out / f"{case}.ready", {"case": case})
            wait_file(command.with_suffix(".go"))
            started = time.perf_counter_ns()
            responses = engine.generate(input_ids=prompts[str(spec["length"])][:spec["batch"]],
                                        sampling_params=PERF_SAMPLING)
            ended = time.perf_counter_ns()
            events = case_events(out, case)
            packet = {
                **spec, "identity": identity,
                "elapsed_s": (ended - started) / 1e9, "responses": responses,
                "started_ns": started, "ended_ns": ended,
            }
            sample = performance_sample(
                packet, events, identity,
                (case, spec["batch"], spec["length"], spec["stage"], spec["arm"]),
                affinity=(first,))
            packet["memory"] = memory(sample["inference_pid"])
            packet["parent_memory"] = memory(os.getpid())
            ep = out / f"{case}.events.json"
            write_json(ep, events)
            packet["events_sha256"] = digest(ep)
            write_json(out / f"{case}.raw.json", packet)
            validate_locality(packet["memory"], node)
            write_json(out / f"{case}.json", packet)
            print(f"SOCKET_CASE_DONE {platform} rank={rank} case={case} client_s={packet['elapsed_s']:.3f}",
                  flush=True)
        wait_file(root / platform / "shutdown.json")
        libraries = sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("*.so"))
        if not libraries:
            raise RuntimeError("No native extension build evidence")
        write_json(out / "native-builds.json", {str(path): digest(path) for path in libraries})
        if digest(sgl_kernel.common_ops.__file__) != identity["common_ops_sha256"]:
            raise RuntimeError("Native dependency changed")
    finally:
        engine.shutdown()
    plan(root)
    write_json(out / "complete.json", identity)


def aggregate(packets, events, spec):
    if (len(events) != len(packets) or len(packets) != len(spec["ranks"])
            or {p["identity"]["rank"] for p in packets} != set(spec["ranks"])):
        raise RuntimeError("Incomplete or duplicate replica coverage")
    starts, ends, pids, samples = [], [], set(), []
    phase_spans = {"prefill": [], "decode": []}
    for packet, ev in zip(packets, events):
        identity = packet["identity"]
        node, first = LAYOUTS[identity["platform"]][identity["rank"]]
        if any(packet[k] != v for k, v in spec.items()):
            raise RuntimeError("Replica case mismatch")
        sample = performance_sample(
            packet, ev, identity,
            (spec["case"], spec["batch"], spec["length"], spec["stage"], spec["arm"]),
            affinity=(first,))
        if sample["inference_pid"] in pids:
            raise RuntimeError("Duplicate inference process")
        pids.add(sample["inference_pid"])
        validate_locality(packet["memory"], node)
        if packet["memory"]["pid"] != sample["inference_pid"]:
            raise RuntimeError("Memory evidence belongs to a different process")
        start, end = packet["started_ns"], packet["ended_ns"]
        if (type(start) is not int or type(end) is not int or end <= start
                or abs((end - start) / 1e9 - packet["elapsed_s"]) > 1e-9):
            raise RuntimeError("Invalid request clock interval")
        starts.append(start)
        ends.append(end)
        for event in ev:
            at = event["start_ns"]
            until = at + event["forward_ns"]
            if type(at) is not int or not start <= at < until <= end:
                raise RuntimeError("Forward clock outside request")
            phase_spans[event["phase"]].append((at, until))
        samples.append(sample)
    if max(starts) >= min(ends):
        raise RuntimeError("Concurrent replicas did not overlap")
    if len({(p["identity"]["node"], p["identity"]["job"]) for p in packets}) != 1:
        raise RuntimeError("Cannot combine different hosts/jobs")
    for key in ("anchors", "schedule"):
        if any(s[key] != samples[0][key] for s in samples):
            raise RuntimeError("Replica inputs or forward schedules differ")
    n = len(packets)
    wall_s = (max(ends) - min(starts)) / 1e9
    phase_s = {key: (max(b for _, b in spans) - min(a for a, _ in spans)) / 1e9
               for key, spans in phase_spans.items()}
    return {
        **spec, "samples": samples, "request_wall_s": wall_s,
        "generated_tokens_s": n * spec["batch"] * 5 / wall_s,
        "phase_wall_s": phase_s,
        "prefill_tokens_s": n * spec["batch"] * spec["length"] / phase_s["prefill"],
        "decode_tokens_s": n * spec["batch"] * 4 / phase_s["decode"],
        "launch_skew_s": (max(starts) - min(starts)) / 1e9,
    }


def run_case(root, platform, spec, children, identities, paired):
    out = root / platform
    case = spec["case"]
    command = out / "commands" / case
    write_json(command.with_suffix(".prepare"), spec)
    for rank in spec["ranks"]:
        wait_file(out / f"rank{rank}" / f"{case}.ready", children)
    write_json(command.with_suffix(".go"), spec)
    packets, events = [], []
    for rank in spec["ranks"]:
        path = out / f"rank{rank}" / f"{case}.json"
        wait_file(path, children)
        packet = json.loads(path.read_text())
        if packet["identity"] != identities[rank]:
            raise RuntimeError("Replica identity changed")
        ep = path.with_name(f"{case}.events.json")
        if digest(ep) != packet["events_sha256"]:
            raise RuntimeError("Replica event hash mismatch")
        packets.append(packet)
        events.append(json.loads(ep.read_text()))
    result = aggregate(packets, events, spec)
    coordinate = spec["group"], spec["batch"], spec["length"], spec["repeat"]
    if coordinate in paired:
        other = paired.pop(coordinate)
        for a, b in zip(result["samples"], other["samples"]):
            if any(a[k] != b[k] for k in ("anchors", "schedule", "inference_pid")):
                raise RuntimeError("Baseline/candidate inputs, schedules or processes differ")
    else:
        paired[coordinate] = result
    write_json(out / f"{case}.aggregate.json", result)
    print(f"SOCKET_GROUP_DONE platform={platform} case={case} wall_s={result['request_wall_s']:.3f}",
          flush=True)


def coordinate(root, platform):
    plan(root)
    out = root / platform
    out.mkdir()
    (out / "commands").mkdir()
    topology(platform, out)
    children, streams = [], []
    paired, identities = {}, []
    try:
        for rank, (node, first) in enumerate(LAYOUTS[platform]):
            env = dict(os.environ)
            env.update(ENV)
            env.update({
                "SGLANG_CPU_OMP_THREADS_BIND": f"{first}-{first + 31}",
                "TORCH_EXTENSIONS_DIR": str(out / f"extensions{rank}"),
                "MAX_JOBS": "2",
            })
            stream = (out / f"rank{rank}.log").open("x")
            streams.append(stream)
            child = subprocess.Popen(
                ["numactl", f"--physcpubind={first}-{first + 31}", f"--membind={node}",
                 sys.executable, "-u", __file__, "worker", "--root", str(root),
                 "--platform", platform, "--rank", str(rank)],
                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            wait_file(out / f"rank{rank}" / "ready.json", children)
            identities.append(json.loads((out / f"rank{rank}" / "identity.json").read_text()))
            print(f"SOCKET_REPLICA_READY platform={platform} rank={rank}", flush=True)
            for spec in cases(platform):
                if spec["group"] == f"single{rank}":
                    run_case(root, platform, spec, children, identities, paired)
        for spec in cases(platform):
            if spec["group"] == "socket":
                run_case(root, platform, spec, children, identities, paired)
        if paired:
            raise RuntimeError("Incomplete baseline/candidate pairs")
        write_json(out / "shutdown.json", {"shutdown": True})
        for child in children:
            if child.wait(timeout=120) != 0:
                raise RuntimeError("Replica failed on shutdown")
        plan(root)
        write_json(out / "complete.json", {"job": os.environ["SLURM_JOB_ID"],
                                          "plan_sha256": digest(root / "plan.json")})
    finally:
        for child in children:
            # Each worker owns a separate session, including its engine children.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "coordinate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--platform", choices=tuple(LAYOUTS))
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "worker":
        worker(args.root, args.platform, args.rank)
    else:
        coordinate(args.root, args.platform)
