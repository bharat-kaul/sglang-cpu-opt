#!/usr/bin/env python3
"""Platform characterization scan — records target-node specs as the variable set
that the roofline analysis consumes. No hardcoded machine constants live downstream;
they all come from the emitted spec JSON.

Auto-detected (run ON the target node, e.g. via srun -w <node>):
  sockets / cores / threads / logical CPUs, NUMA topology + per-domain RAM, SNC on/off,
  ISA (AMX bf16/int8, AVX512 bf16/fp16/f), L2/L3 cache.
Measured separately (micro-benchmarks; fields left null until filled):
  mem_bw_gbps (per-domain stream BW), amx_bf16_tflops / amx_int8_tops (GEMM peak).

Usage:
  srun -p <part> -w <node> --time=00:02:00 python platform_scan.py --name emr --out platforms/emr.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str)).stdout
    except Exception:
        return ""


def scan():
    spec = {"name": None, "host": os.uname().nodename,
            "scanned_at": datetime.datetime.now().isoformat(timespec="seconds")}

    # ---- CPU topology / ISA via lscpu ----
    lscpu = _run(["lscpu"])
    def g(pat, cast=str, default=None):
        m = re.search(pat, lscpu, re.MULTILINE)
        return cast(m.group(1)) if m else default
    spec["sockets"] = g(r"Socket\(s\):\s+(\d+)", int)
    spec["cores_per_socket"] = g(r"Core\(s\) per socket:\s+(\d+)", int)
    spec["threads_per_core"] = g(r"Thread\(s\) per core:\s+(\d+)", int)
    spec["logical_cpus"] = g(r"^CPU\(s\):\s+(\d+)", int)
    if spec["sockets"] and spec["cores_per_socket"]:
        spec["physical_cores"] = spec["sockets"] * spec["cores_per_socket"]
    spec["model_name"] = g(r"Model name:\s+(.+)")
    spec["l2_cache"] = g(r"L2 cache:\s+(.+)")
    spec["l3_cache"] = g(r"L3 cache:\s+(.+)")

    flags = _run(["bash", "-lc", "grep -m1 flags /proc/cpuinfo"]) or lscpu
    def has(f):
        return bool(re.search(r"\b" + re.escape(f) + r"\b", flags))
    spec["isa"] = {
        "amx_tile": has("amx_tile"), "amx_bf16": has("amx_bf16"), "amx_int8": has("amx_int8"),
        "avx512f": has("avx512f"), "avx512_bf16": has("avx512_bf16"), "avx512_fp16": has("avx512_fp16"),
        "avx512_vnni": has("avx512_vnni") or has("avx_vnni"),
    }
    # AMX compute tiles are bf16/fp16/int8 ONLY; fp8/fp4 must be dequant->bf16.
    spec["amx_compute_dtypes"] = [d for d, ok in
                                  (("bf16", spec["isa"]["amx_bf16"]), ("int8", spec["isa"]["amx_int8"]))
                                  if ok] or (["bf16", "int8"] if spec["isa"]["amx_tile"] else [])

    # ---- NUMA / SNC / per-domain RAM via numactl ----
    nh = _run(["numactl", "--hardware"])
    nodes = {int(a): int(b) for a, b in re.findall(r"node (\d+) size: (\d+) MB", nh)}
    spec["numa_nodes"] = len(nodes) or None
    spec["per_numa_gb"] = {str(k): round(v / 1024, 1) for k, v in sorted(nodes.items())}
    if nodes:
        spec["domain_ram_gb"] = round(min(nodes.values()) / 1024, 1)  # the capacity FLOOR
        spec["total_ram_gb"] = round(sum(nodes.values()) / 1024, 1)
    # SNC on when NUMA domains exceed sockets (socket sub-clustered)
    if spec["numa_nodes"] and spec["sockets"]:
        spec["snc"] = spec["numa_nodes"] > spec["sockets"]
        spec["domains_per_socket"] = spec["numa_nodes"] // spec["sockets"]

    # ---- measured constants: left null, filled by micro-benchmarks ----
    spec.setdefault("mem_bw_gbps", None)       # per-domain stream BW (GB/s)
    spec.setdefault("amx_bf16_tflops", None)   # measured bf16 AMX GEMM peak (TFLOP/s)
    spec.setdefault("amx_int8_tops", None)
    spec["measured_how"] = ("mem_bw_gbps: stream-triad pinned to one domain; "
                            "amx_bf16_tflops: large square bf16 GEMM sweep (bench_dense_gemm_threads.py) "
                            "-> take the plateau. Fill these before running the roofline.")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="platform label, e.g. emr / gnr")
    ap.add_argument("--out", required=True, help="output spec JSON path")
    ap.add_argument("--bw", type=float, default=None, help="override measured per-domain mem BW (GB/s)")
    ap.add_argument("--amx-bf16", type=float, default=None, help="override measured bf16 AMX peak (TFLOP/s)")
    a = ap.parse_args()
    spec = scan()
    spec["name"] = a.name
    if a.bw is not None:
        spec["mem_bw_gbps"] = a.bw
    if a.amx_bf16 is not None:
        spec["amx_bf16_tflops"] = a.amx_bf16
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(spec, f, indent=2)
    print(json.dumps(spec, indent=2))
    print(f"\nSAVED: {a.out}", file=sys.stderr)
    if spec["mem_bw_gbps"] is None or spec["amx_bf16_tflops"] is None:
        print("WARNING: mem_bw_gbps / amx_bf16_tflops still null -> measure before roofline.", file=sys.stderr)


if __name__ == "__main__":
    main()
