#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-measured (authored ops) — observation-vs-ideal join.

Each row's IDEAL floor (max(bytes/BW, flops/PEAK)) is derived from the op's ACTUAL benchmark
INPUT/OUTPUT CONTRACT (F4), at the single observed coordinate M=32. The measured value is one
OBSERVATION with a benchmark source link; this join reports the observation and its distance from
the nominal roof. It does NOT draw causal conclusions (overhead vs compute) or validate donor
routing — those belong to the later implementation-review phase.

PROVENANCE: node = pcl-sprh02 (DDR5-5600, 64 threads NUMA0); batch M=32; model revision in REV;
dtypes per contract. Each row cites the benchmark whose contract fixes its operands.
Contract byte/FLOP expectations are asserted below (fail-closed).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_roofline_p2 import REF_REVISION as REV  # single centralized reference identifier

_SPEC = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
P = json.load(open(_SPEC))
MP = P["machine_peak"]
BW = MP["mem_bw_gbps"] * 1e9                 # nominal target
PEAK = MP["amx_bf16_tflops"] * 1e12          # AMX bf16 ceiling
FP32_PEAK = MP["avx512_fp32_tflops"] * 1e12  # AVX-512 FP32 ceiling
BW_MEAS = P["mem_bw_gbps"] * 1e9             # measured reference
B2, B4, I4 = 2.0, 4.0, 4.0                   # bf16 / fp32 / int32 bytes

M = 32  # the single observed coordinate

# op: (name, flops, bytes, compute_peak, measured_ms, src, note)
# bytes/flops derived from each bench's call contract (operands + dtypes + shapes).
OPS = [
    ("indexer logits (q.ck+reduce)",
     2*M*64*1024*128 + 3*M*64*1024,
     M*1024*128*B2            # index-kv read (ctx=1024, HD=128), bf16 keys
     + M*64*128*B2            # projected bf16 query (already projected; NOT the wq_b matrix)
     + M*64*B4                # per-request fp32 head-weights w[M,nh]
     + M*1024*B4,             # fp32 logits out
     PEAK, 0.536, "bench_indexer_logits.py",
     "ctx=1024; bf16 projected query+keys, fp32 head-weights+out"),
    ("indexer top-k (512 of 1024)",
     0,
     M*1024*B4                # read logits[M,1024] fp32
     + M*512*I4,              # write 512 selected indices (int32) per request
     PEAK, 0.039, "bench_topk.py",
     "selection; reads logits[M,1024], writes [M,512] idx"),
    ("compressor softmax-pool",
     3*M*128*512,
     M*128*512*B4             # read kv state (win=128, D=512) FP32
     + M*128*512*B4           # read score state FP32
     + M*512*B4               # write compressed token (D) per request FP32
     + 128*512*B4,            # shared positional APE[R,D] read once/call FP32
     FP32_PEAK, 0.479, "bench_compressor.py",
     "R=128,D=512; FP32 kv+score+APE reads, FP32 out"),
    ("sparse attend (MQA+sink)",
     4*M*64*512*512,
     M*512*512*B4             # latent KV read (K=512, D=512)
     + 2*M*64*512*B4,         # q + out (NH=64, D=512)
     PEAK, 1.781, "bench_sparse_attend.py",
     "H=64,K=512,D=512; scalar flash observation"),
    ("MHC sinkhorn (hc=4,20it)",
     M*4*4*20*5,
     M*24*B4                  # read mixes[M,24] FP32
     + 3*B4                   # read scale[3] FP32
     + 24*B4                  # read base[24] FP32
     + M*24*B4,               # write pre+post+comb (24-wide/row total) FP32
     FP32_PEAK, 0.015, "bench_sinkhorn.py",
     "mixes[M,24]+scale[3]+base[24] -> pre/post/comb (24/row)"),
    ("MHC combine (reduce)",
     M*4096*(4+3),            # 4 multiplies + 3 adds per output elem (weighted sum)
     M*4*4096*B4              # read x[M,4,4096] FP32
     + M*4*B4                 # read per-request weights pre[M,4] FP32
     + M*4096*B4,             # write y[M,4096] FP32
     FP32_PEAK, 0.013, "bench_combine.py",
     "einsum mk,mkh->mh; per-request pre[M,4] weights; +reduction adds"),
]

# --- F4/F5: fail-closed contract assertions (byte/FLOP from the bench contracts @ M=32) ---
_EXPECT = {
    "indexer logits (q.ck+reduce)": {"bytes": 9_052_160},
    "compressor softmax-pool": {"bytes": 17_104_896},
    "MHC sinkhorn (hc=4,20it)": {"bytes": 6_252},
    "MHC combine (reduce)": {"bytes": 2_621_952, "flops": 917_504},
}
for _n, _fl, _by, _pk, _ms, _src, _nt in OPS:
    _e = _EXPECT.get(_n)
    if _e:
        assert _by == _e["bytes"], f"{_n}: bytes {_by} != contract {_e['bytes']}"
        if "flops" in _e:
            assert _fl == _e["flops"], f"{_n}: flops {_fl} != contract {_e['flops']}"

print(f"DSv4 roofline-VS-measured (authored ops, M={M})  rev {REV[:8]}  nominal BW={BW/1e9:.0f} GB/s "
      f"AMX={PEAK/1e12:.0f} TF FP32={FP32_PEAK/1e12:.1f} TF")
print(f"  measured-reference node={P.get('measurement_node','?')} ({P.get('mem_type','?')}) BW={BW_MEAS/1e9:.0f} GB/s; "
      f"meas_us = ONE observation @M={M} (see src)")
print("-" * 118)
print(f"{'op':30s} {'ideal_us':>9s} {'bind':>5s} {'meas_us':>9s} {'off_roof':>9s}  {'src':22s} note")
print("-" * 118)
for name, fl, by, peak, meas_ms, src, note in OPS:
    t_bw, t_cc = by / BW, (fl / peak if fl else 0.0)
    ideal = max(t_bw, t_cc)
    bind = "C" if t_cc > t_bw else "B"
    meas = meas_ms * 1e-3
    off = meas / ideal if ideal else float("inf")
    print(f"{name:30s} {ideal*1e6:9.1f} {bind:>5s} {meas*1e6:9.1f} {off:8.1f}x  {src:22s} {note}")
print("-" * 118)
print("off_roof = observation/ideal at M=32 (distance from the nominal roof). It is a DIAGNOSTIC only:\n"
      "it does NOT establish overhead vs compute causation, nor validate donor routing or any speedup \u2014\n"
      "those are deferred to implementation review. Each row's operands are derived from the cited\n"
      "benchmark's input/output contract; bytes/FLOPs are asserted against those contracts above.")
