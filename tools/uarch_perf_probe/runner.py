"""uArch Performance Probe runner: orchestrate probes -> machine_constants.{json,md}.

Usage (from anywhere):
    python tools/uarch_perf_probe/runner.py --out machine_constants.json
    python tools/uarch_perf_probe/runner.py --quick           # skip subprocess NUMA matrix
    python tools/uarch_perf_probe/runner.py --probes compute_peak,memory,threading

Emits a machine_constants.json (schema below) + a human-readable .md, and prints a
one-screen summary with the derived kernel knobs. Any workflow points its kernel-authoring
/ roofline steps at the JSON; on a new uarch it is the sole source of the constants.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make the package importable whether run as a script or a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uarch_perf_probe import __version__, hygiene, probes  # noqa: E402

ALL_PROBES = ("compute_peak", "memory", "roofline_ridge", "gather_crossover", "threading", "numa")


def derive_knobs(c: dict) -> dict:
    """Map measured constants -> the kernel-authoring knobs they inform."""
    knobs = {}
    numa = c.get("numa", {})
    if numa.get("status") == "ok":
        knobs["one_tp_rank_per_domain"] = True
        knobs["n_tp_ranks_hint"] = numa.get("n_domains")
        lo, ro = numa.get("local_bw_gbps"), numa.get("remote_bw_gbps")
        if lo and ro:
            knobs["remote_bw_penalty"] = round(ro / lo, 3)
            knobs["per_domain_bw_gbps"] = round(lo, 1)
    gx = c.get("gather_crossover", {})
    if gx.get("status") == "ok":
        knobs["prefer_amx_stage_M_ge"] = gx.get("prefer_stage_M_ge")
    thr = c.get("threading", {})
    if thr.get("status") == "ok":
        knobs["cores_to_saturate_bw"] = thr.get("cores_to_saturate_bw")
    rr = c.get("roofline_ridge", {})
    if rr.get("status") == "ok":
        knobs["ridge_flops_per_byte"] = rr.get("ridge_flops_per_byte")
    return knobs


def _fmt(v, u=""):
    return f"{v:.1f}{u}" if isinstance(v, (int, float)) else "n/a"


def render_md(c: dict) -> str:
    m = c["meta"]
    L = [
        f"# uArch Performance Probe report — {m['node']}",
        "",
        f"- probe_version: {c['probe_version']}  ·  torch: {m.get('torch')}  ·  "
        f"NUMA nodes: {m.get('numa_nodes')}  ·  cores: {m.get('cpu_count')}",
        f"- ISA: " + ", ".join(k for k, v in m["isa"].items() if v),
        f"- freq hygiene: governor={m['freq_hygiene'].get('governor')} "
        f"turbo_disabled={m['freq_hygiene'].get('turbo_disabled')} "
        f"clean={m['freq_hygiene'].get('clean')}",
    ]
    for w in m["freq_hygiene"].get("warnings", []):
        L.append(f"  - ⚠ {w}")
    cp = c.get("compute_peak", {}).get("peak", {})
    L += ["", "## Compute peak (achieved GEMM)"]
    for dt, v in cp.items():
        g = v.get("gflops") or v.get("gops")
        L.append(f"- {dt}: {_fmt(g)} GF/s" if g else f"- {dt}: {v.get('error','n/a')}")
    mem = c.get("memory", {})
    L += ["", "## Memory", f"- DRAM triad BW: {_fmt(mem.get('dram_triad_bw_gbps'))} GB/s"]
    for r in mem.get("cache_ladder", []):
        if "bw_gbps" in r:
            L.append(f"  - {r['mb']:>7} MB: {_fmt(r['bw_gbps'])} GB/s")
    numa = c.get("numa", {})
    if numa.get("status") == "ok":
        L += [
            "",
            "## NUMA / SNC",
            f"- domains: {numa['n_domains']}  ·  local BW: {_fmt(numa.get('local_bw_gbps'))} GB/s"
            f"  ·  remote BW: {_fmt(numa.get('remote_bw_gbps'))} GB/s",
        ]
    L += ["", "## Derived kernel knobs", "```json", json.dumps(c["derived_kernel_knobs"], indent=2), "```"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="uArch Performance Probe")
    ap.add_argument("--out", default="machine_constants.json")
    ap.add_argument("--quick", action="store_true", help="skip the subprocess NUMA matrix")
    ap.add_argument("--probes", default="all", help="comma list or 'all'")
    args = ap.parse_args()

    want = set(ALL_PROBES) if args.probes == "all" else set(args.probes.split(","))
    if args.quick:
        want.discard("numa")

    c: dict = {"probe_version": __version__, "meta": hygiene.env_report()}
    if not c["meta"]["freq_hygiene"].get("clean"):
        print("[uPP] WARNING: frequency hygiene not clean — see report warnings.", file=sys.stderr)

    compute = probes.probe_compute_peak() if "compute_peak" in want else {"status": "skipped"}
    c["compute_peak"] = compute
    memory = probes.probe_memory_bandwidth() if "memory" in want else {"status": "skipped"}
    c["memory"] = memory
    c["roofline_ridge"] = (
        probes.probe_roofline_ridge(compute, memory) if "roofline_ridge" in want else {"status": "skipped"}
    )
    c["gather_crossover"] = (
        probes.probe_gather_crossover() if "gather_crossover" in want else {"status": "skipped"}
    )
    c["threading"] = probes.probe_threading() if "threading" in want else {"status": "skipped"}
    c["numa"] = probes.probe_numa_snc() if "numa" in want else {"status": "skipped"}
    c["derived_kernel_knobs"] = derive_knobs(c)

    with open(args.out, "w") as f:
        json.dump(c, f, indent=2)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(render_md(c))

    print(f"[uPP] wrote {args.out} and {md_path}")
    print(json.dumps(c["derived_kernel_knobs"], indent=2))


if __name__ == "__main__":
    main()
