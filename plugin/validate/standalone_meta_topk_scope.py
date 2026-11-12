#!/usr/bin/env python3
"""Screen the existing serial top-k knob across M/S; require unchanged native order."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics

import torch

from standalone_meta_dispatch_screen import ROOT, call, digest, select_arm


def valid_selection(logits, indices, k):
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise AssertionError("invalid logits")
    if indices.shape != (logits.shape[0], k) or indices.dtype != torch.int64:
        raise AssertionError("selection shape/dtype mismatch")
    if (indices < 0).any() or (indices >= logits.shape[1]).any():
        raise AssertionError("selection index out of bounds")
    ordered = indices.sort(dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise AssertionError("duplicate selected index")
    values = logits.gather(1, indices).sort(dim=1).values
    expected = torch.topk(logits, k, dim=1).values.sort(dim=1).values
    if not torch.equal(values, expected):
        raise AssertionError("selected values are not exact top-k")


def run(output, replicate):
    if output.exists():
        raise FileExistsError(output)
    baseline = json.loads(Path(
        "/scratch/bkaul/dsv4-standalone-meta-opt/dispatch-job-398566/replicate-1.json"
    ).read_text())
    if baseline["job"] != "398566" or baseline["torch"] != torch.__version__:
        raise RuntimeError("baseline identity/runtime mismatch")
    binaries = [(Path(p), h) for p, h in baseline["binaries"].items()
                if Path(p).name == "meta_dispatch_topk.so"]
    if len(binaries) != 1 or digest(binaries[0][0]) != binaries[0][1]:
        raise RuntimeError("native baseline binary mismatch")
    source = ROOT / "plugin/kernels/dsa_pilot/indexer_topk.cpp"
    if digest(source) != baseline["sources"][str(source.relative_to(ROOT))]:
        raise RuntimeError("top-k source changed")
    spec = importlib.util.spec_from_file_location("meta_dispatch_topk", binaries[0][0])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load top-k baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    functions = {
        "default": module.indexer_topk, "serial": module.indexer_topk,
        "torch": lambda x, k: torch.topk(x, k, dim=1, sorted=False).indices,
    }

    def select(arm):
        select_arm({"TOPK_SERIAL": "1"} if arm == "serial" else {})

    result = {
        "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(), "replicate": replicate,
        "node": os.uname().nodename, "torch": torch.__version__, "threads": 64,
        "baseline_identity": baseline, "binary": str(binaries[0][0]),
        "binary_sha256": binaries[0][1],
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in
                    (source, Path(__file__), Path(__file__).with_name("standalone_meta_dispatch_screen.py"),
                     Path(__file__).with_name("run_standalone_meta_topk_scope.sbatch"))},
        "scope": "Existing-knob feasibility; blocked native-order differences are explicit, not promoted as parity",
        "timing": "24 balanced samples/arm/cache; native output release included; 768MiB add_ eviction outside timer",
        "cases": [],
    }
    rng = random.Random(replicate)
    flush = torch.zeros(192 * 1024 * 1024)
    for m in (1, 2, 4, 8, 16, 32, 64):
        for s in (128, 512, 1024, 4096, 16384, 65536):
            k = min(512, s)
            logits = torch.randn(m, s)
            ties = torch.randint(-2, 3, (m, s)).float()
            masked = logits.clone()
            masked[:, s // 2:] = -torch.inf
            checks = {}
            for label, x in (("normal", logits), ("ties", ties), ("masked", masked)):
                select("default")
                old = module.indexer_topk(x, k)
                select("serial")
                serial = module.indexer_topk(x, k)
                valid_selection(x, old, k)
                valid_selection(x, serial, k)
                checks[label] = torch.equal(old, serial)
            record = {"M": m, "S": s, "K": k, "native_order_equal": checks,
                      "input_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest()}
            if not all(checks.values()):
                record["status"] = "blocked-native-order-change"
                result["cases"].append(record)
                print("BLOCKED native-order change", m, s, k, checks, flush=True)
                continue
            record["status"] = "timed"
            for mode in ("cold", "warm"):
                arms = list(functions)
                samples = {a: [] for a in arms}
                orders = []
                for arm in arms:
                    select(arm)
                    for _ in range(5):
                        functions[arm](logits, k)
                for _ in range(8):
                    rng.shuffle(arms)
                    shifts = list(range(3))
                    rng.shuffle(shifts)
                    for shift in shifts:
                        order = arms[shift:] + arms[:shift]
                        orders.append(order)
                        for arm in order:
                            select(arm)
                            if mode == "cold":
                                flush.add_(1)
                            samples[arm].append(call(functions[arm], (logits, k)))
                if not all(math.isfinite(x) and x > 0 for v in samples.values() for x in v):
                    raise RuntimeError("invalid timing sample")
                record[mode] = {"samples_us": samples, "orders": orders,
                                "medians_us": {a: statistics.median(v) for a, v in samples.items()}}
                print(m, s, k, mode, record[mode]["medians_us"], flush=True)
            result["cases"].append(record)
    result["worker_affinity"] = sorted({
        tuple(os.sched_getaffinity(int(p.name))) for p in Path("/proc/self/task").iterdir()
    })
    if result["worker_affinity"] != [(i,) for i in range(64)]:
        raise RuntimeError("unexpected worker placement")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    run(args.output, args.replicate)
