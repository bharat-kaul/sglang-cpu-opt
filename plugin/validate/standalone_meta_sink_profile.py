#!/usr/bin/env python3
"""Warm serial baseline loop for sampled instruction attribution, not speedup timing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from standalone_meta_dispatch_screen import check_outputs, digest, select_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = json.loads((args.baseline_dir / "replicate-1.json").read_text())
    if identity["job"] != "398645" or len(identity["binaries"]) != 1:
        raise RuntimeError("unexpected baseline identity")
    binary_name, expected_hash = next(iter(identity["binaries"].items()))
    binary = Path(binary_name)
    if digest(binary) != expected_hash:
        raise RuntimeError("baseline binary changed")
    torch.set_num_threads(64)
    torch.set_num_interop_threads(1)
    select_arm({"SINK_SERIAL": "1"})
    spec = importlib.util.spec_from_file_location("meta_sink_iterations", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    inputs = (torch.randn(1, 32, 24), torch.rand(3) + 0.5, torch.randn(24),
              4, 20, 1e-6)
    fn = module.mhc_sinkhorn
    expected = fn(*inputs)
    print("PROFILE_REGION_START M=32 hc=4 iters=20 calls=200000", flush=True)
    for _ in range(200000):
        fn(*inputs)
    print("PROFILE_REGION_END", flush=True)
    check_outputs(expected, fn(*inputs))
    result = {
        "job": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
        "node": os.uname().nodename, "baseline_job": identity["job"],
        "baseline_binary": str(binary), "binary_sha256": expected_hash,
        "source_sha256": digest(Path(__file__)), "torch": torch.__version__,
        "scope": "perf samples include startup and Python dispatch; warm native calls dominate loop, not a pure native timing result",
        "shape": [1, 32, 24], "hc": 4, "iters": 20, "calls": 200000,
        "main_affinity": sorted(os.sched_getaffinity(0)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
