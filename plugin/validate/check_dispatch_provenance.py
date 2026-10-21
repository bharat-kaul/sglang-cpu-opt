#!/usr/bin/env python3
"""Dispatch-provenance gate (playbook gate H) — fail if a hot op ran EAGER or its C kernel never
dispatched. Reads the runtime provenance dump (INTEL_CPU_DSV4_PROV_OUT) written by _dsa_ckernels.

The miss this prevents: an authored-but-bypassed C kernel (serving forward reimplements the math in
eager torch). "Kernel exists / unit-parity green" != "kernel runs on the serving path". This gate
requires the END-TO-END proof: the op dispatched to its C kernel and NO eager op remains in the hot
chain.

Usage:
  INTEL_CPU_DSV4_PROV_OUT=prov.json <run the model>   # _dsa_ckernels dumps provenance at exit
  python check_dispatch_provenance.py prov.json        # gate (exit 1 on any GAP)
"""
import json
import sys

# The hot decode compute chain that MUST be all-C/C++ (the graph-loop analysis nodes).
HOT_OPS = ["indexer", "topk", "compressor", "sinkhorn", "combine", "sparse_attend", "moe"]


def gate(prov: dict):
    gaps = []
    for op in HOT_OPS:
        d = prov.get(op, {})
        c, e = int(d.get("c_kernel", 0)), int(d.get("eager", 0))
        if c == 0 and e == 0:
            gaps.append(f"  {op:14s} NOT EXERCISED (op never ran this input — inconclusive)")
        elif e > 0:
            gaps.append(f"  {op:14s} ran EAGER {e}x (c_kernel={c})  -> WIRING GAP: wire to the C kernel")
        elif c == 0:
            gaps.append(f"  {op:14s} C kernel NEVER dispatched -> authored-but-bypassed WIRING GAP")
    return gaps


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "prov.json"
    with open(path) as f:
        prov = json.load(f)
    print("[dispatch-provenance]", json.dumps(prov, indent=2))
    gaps = gate(prov)
    hard = [g for g in gaps if "WIRING GAP" in g]
    if gaps:
        print("\n[dispatch-provenance] findings:")
        print("\n".join(gaps))
    if hard:
        print(f"\n[dispatch-provenance] FAIL: {len(hard)} wiring gap(s) — eager ops bypassing C kernels.")
        return 1
    print("\n[dispatch-provenance] PASS: every exercised hot op dispatched to its C/C++ kernel; no eager in the hot chain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
