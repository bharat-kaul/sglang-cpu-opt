#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-measured (authored ops) — the join the verdict asked for.

For each AUTHORED Phase-A kernel, compute the IDEAL floor (max(bytes/BW, flops/PEAK))
at its GROUNDED microbench shape, and compare to the MEASURED microbench time. The
distance-from-roof is a DIAGNOSTIC (verdict guidance), not an achievability claim.

PROVENANCE (R5): every measured value is a SINGLE operating point, not a batch sweep.
  node      = pcl-sprh02 (DDR5-5600, 64 threads NUMA0)
  revision  = DeepSeek-V4-Flash HF rev 60d8d770770c6776ff598c94bb586a859a38244f1
  batch     = M=32 (priority decode point)
  dtype     = bf16 compute / fp32 reductions-&-outputs (as noted per op)
  ctx       = compressed index/attention context (not full 4096) where applicable
Byte accounting (R5) now includes head-weight reads, selection-output traffic, pool
output per-request (not per-window), sinkhorn/combine output writes, and comb weights.
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
REV = "60d8d707"

# op: (name, flops, bytes, measured_ms, note)   shapes GROUNDED in published model.py @ REV
OPS = [
    ("indexer logits (q.ck+reduce)",
     2*M*64*1024*128 + 3*M*64*1024,
     M*1024*128*B2            # index-kv read (ctx=1024, HD=128)
     + M*64*128*B2            # per-request query
     + 64*128*B2              # wq_b head query weights (bf16, read once)
     + 64*B4                  # fp32 head-weights (IDX_NH, read once)
     + M*1024*B4,             # fp32 logits out
     0.536, "ctx=1024 compressed; bf16 compute, fp32 out; +head-weights"),
    ("indexer top-k (512 of 1024)",
     0,
     M*1024*B4                # read logits[M,1024]
     + M*512*4,               # write 512 selected indices (int32) per request
     0.039, "selection; reads logits[M,1024], writes [M,512] idx"),
    ("compressor softmax-pool",
     3*M*128*512,
     M*128*512*2*B4           # read kv window (win=128, D=512), 2 streams
     + M*512*B4,              # write 1 compressed token (D) PER REQUEST (not per-window)
     0.479, "R=128,D=512; per-request out (fp32)"),
    ("sparse attend (MQA+sink)",
     4*M*64*512*512,
     M*512*512*B4             # latent KV read (K=512, D=512)
     + 2*M*64*512*B4,         # q + out (NH=64, D=512)
     1.781, "H=64,K=512,D=512; scalar flash (loses to BLAS)"),
    ("MHC sinkhorn (hc=4,20it)",
     M*4*4*20*5,
     M*24*B4                  # read 24-wide logits
     + M*16*B4,               # write 4x4 doubly-stochastic plan out
     0.015, "tiny 4x4 per row; +plan out"),
    ("MHC combine (reduce)",
     M*4*4096,
     (M*4*4096                # read x[M,4,4096]
      + M*4096                # write y[M,4096]
      + 4*4)*B4,              # comb weights (4x4, read once)
     0.013, "x[M,4,4096]->y[M,4096]; +comb weights"),
]

print(f"DSv4 roofline-VS-measured (authored ops, M={M}, rev {REV})  nominal BW={BW/1e9:.0f} GB/s PEAK={PEAK/1e12:.0f} TF")
print(f"  measured-reference node={P.get('measurement_node','?')} ({P.get('mem_type','?')}) BW={BW_MEAS/1e9:.0f} GB/s; "
      f"each meas_us = SINGLE point (not a sweep)")
print("-" * 112)
print(f"{'op':34s} {'ideal_us':>9s} {'bind':>5s} {'meas_us':>9s} {'off_roof':>9s} {'note'}")
print("-" * 112)
for name, fl, by, meas_ms, note in OPS:
    t_bw, t_cc = by / BW, (fl / PEAK if fl else 0.0)
    ideal = max(t_bw, t_cc)
    bind = "C" if t_cc > t_bw else "B"
    meas = meas_ms * 1e-3
    off = meas / ideal if ideal else float("inf")
    print(f"{name:34s} {ideal*1e6:9.1f} {bind:>5s} {meas*1e6:9.1f} {off:8.1f}x  {note}")
print("-" * 112)
print("off_roof = measured/ideal (distance from the nominal roof; DIAGNOSTIC only, NOT a correctness or\n"
      "achievability claim). Each measured value is ONE operating point (M=32) at the stated node/rev; it\n"
      "is not a batch sweep. Small ops are overhead-bound (far from roof); sparse-attend's gap confirms\n"
      "routing to the donor MLA flash. Track latency/useful-throughput as primary; off-roof explains plateaus.")
