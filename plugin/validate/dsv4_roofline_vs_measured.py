#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-measured (authored ops) — the join the verdict asked for.

For each AUTHORED Phase-A kernel, compute the IDEAL floor (max(bytes/BW, flops/PEAK))
at its GROUNDED microbench shape, and compare to the MEASURED microbench time. The
distance-from-roof is a DIAGNOSTIC (verdict guidance), not an achievability claim.
Measured numbers are the pilot microbenches at priority M=32 (DDR5-5600 node where re-run).
"""
import json
import os

_SPEC = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
P = json.load(open(_SPEC))
MP = P["machine_peak"]
BW = MP["mem_bw_gbps"] * 1e9       # nominal target
PEAK = MP["amx_bf16_tflops"] * 1e12
BW_MEAS = P["mem_bw_gbps"] * 1e9   # measured reference
B2, B4 = 2.0, 4.0                  # bf16 / fp32 bytes

M = 32  # priority operating point

# op: (name, flops, bytes, measured_ms, note)   shapes GROUNDED in published model.py
OPS = [
    ("indexer logits (q.ck+reduce)", 2*M*64*1024*128 + 3*M*64*1024,
     M*1024*128*B2 + M*64*128*B2 + M*1024*B4, 0.536,
     "ctx=1024 compressed; bf16 compute, fp32 out"),
    ("indexer top-k (512 of 1024)", 0, M*1024*B4, 0.039, "selection; reads logits[M,1024]"),
    ("compressor softmax-pool", 3*M*128*512, (M*128*512*2)*B4 + 128*512*B4, 0.479,
     "R=128,D=512; kv+score read, out write (fp32)"),
    ("sparse attend (MQA+sink)", 4*M*64*512*512, M*512*512*B4 + 2*M*64*512*B4, 1.781,
     "H=64,K=512,D=512; scalar flash (loses to BLAS)"),
    ("MHC sinkhorn (hc=4,20it)", M*4*4*20*5, M*24*B4, 0.015, "tiny 4x4 per row"),
    ("MHC combine (reduce)", M*4*4096, (M*4*4096 + M*4096)*B4, 0.013, "x[M,4,4096]->y[M,4096]"),
]

print(f"DSv4 roofline-VS-measured (authored ops, M={M})  nominal BW={BW/1e9:.0f} GB/s PEAK={PEAK/1e12:.0f} TF")
print(f"  measured-reference BW (node {P.get('measurement_node','?')}) = {BW_MEAS/1e9:.0f} GB/s")
print("-" * 104)
print(f"{'op':34s} {'ideal_us':>9s} {'bind':>5s} {'meas_us':>9s} {'off_roof':>9s} {'note'}")
print("-" * 104)
for name, fl, by, meas_ms, note in OPS:
    t_bw, t_cc = by / BW, (fl / PEAK if fl else 0.0)
    ideal = max(t_bw, t_cc)
    bind = "C" if t_cc > t_bw else "B"
    meas = meas_ms * 1e-3
    off = meas / ideal if ideal else float("inf")
    print(f"{name:34s} {ideal*1e6:9.1f} {bind:>5s} {meas*1e6:9.1f} {off:8.1f}x  {note}")
print("-" * 104)
print("off_roof = measured/ideal (distance from the nominal roof; diagnostic). Small ops are overhead-\n"
      "bound (far from roof); sparse-attend's gap confirms routing to the donor MLA flash. Track latency\n"
      "and useful throughput as primary; use off-roof only to explain plateaus (verdict guidance).")
