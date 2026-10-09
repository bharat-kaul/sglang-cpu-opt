#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-observation join (authored ops).

Each row's IDEAL floor (max(bytes/BW, flops/peak)) is derived from the op's ACTUAL benchmark
INPUT/OUTPUT CONTRACT (F4) at M=32, with an EXPLICIT compute dtype that selects the compute
resource (G1): bf16 -> AMX, fp32 -> AVX-512 FP32 ceiling. Each row links an AUDITABLE RESULT
RECORD (results/op_passes/*.json + kernel commit) that certifies correctness + speedup-vs-torch.
An absolute node latency is NOT present in those records, so it is WITHHELD (unverified) rather
than invented (G2). No causal/overhead/donor-routing conclusions are drawn here.

PROVENANCE: node = pcl-sprh02 (DDR5-5600, 64 threads NUMA0); batch M=32; model revision in REV;
dtypes + operands per the cited benchmark contract. Contract byte/FLOP expectations are asserted
below (fail-closed).
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
CPEAK = {"bf16": PEAK, "fp32": FP32_PEAK}    # G1: compute dtype -> compute resource
B2, B4, I4 = 2.0, 4.0, 4.0                   # bf16 / fp32 / int32 bytes

M = 32  # the single observed coordinate
_REC = "plugin/validate/results/op_passes"

# op: (name, flops, bytes, compute_dtype, record, commit, certifies, note)
# bytes/flops derived from each bench's call contract (operands + dtypes + shapes).
OPS = [
    ("indexer logits (q.ck+reduce)",
     2*M*64*1024*128 + 3*M*64*1024,
     M*64*128*B4              # q[M,H,D] FP32 public input (bench_idx_logits.py)
     + M*1024*128*B4          # kv[M,S,D] per-request FP32 keys
     + M*64*B4                # weights w[M,H] FP32
     + M*1024*B4,             # logits[M,S] FP32 out
     "fp32", f"{_REC}/indexer_logits.json", "e3fdcb9",
     "cos 1.0; 1.76x vs torch @M=32 (ctx=1024)",
     "bench_idx_logits.py FP32 public boundary, per-request keys; BF16 projected-query (9,052,160 B) is a PROSPECTIVE target, not this observation"),
    ("indexer top-k (512 of 1024)",
     0,
     M*1024*B4                # read logits[M,1024] fp32
     + M*512*I4,              # write 512 selected indices (int32) per request
     "fp32", f"{_REC}/indexer_topk.json", "06faec0",
     "set-match 1.0; 7.62x vs torch.topk @M=32",
     "selection; reads logits[M,1024], writes [M,512] idx"),
    ("compressor softmax-pool",
     3*M*128*512,
     M*128*512*B4             # read kv state (win=128, D=512) FP32
     + M*128*512*B4           # read score state FP32
     + M*512*B4               # write compressed token (D) per request FP32
     + 128*512*B4,            # shared positional APE[R,D] read once/call FP32
     "fp32", f"{_REC}/compressor.json", "pending",
     "cos 1.0; 2.68x vs torch @M=32",
     "R=128,D=512; FP32 kv+score+APE reads, FP32 out"),
    ("sparse attend (MQA+sink)",
     4*M*64*512*512,
     M*512*512*B4             # latent KV read (K=512, D=512)
     + 2*M*64*512*B4,         # q + out (NH=64, D=512)
     "fp32", f"{_REC}/sparse_attend.json", "pending",
     "surfaced: torch BLAS wins at real MQA dims -> route to donor flash",
     "H=64,K=512,D=512; FP32 scalar-flash boundary (bench_sparse_attend.py)"),
    ("MHC sinkhorn (hc=4,20it)",
     M*4*4*20*5,
     M*24*B4                  # read mixes[M,24] FP32
     + 3*B4                   # read scale[3] FP32
     + 24*B4                  # read base[24] FP32
     + M*24*B4,               # write pre+post+comb (24-wide/row total) FP32
     "fp32", f"{_REC}/mhc_sinkhorn.json", "pending",
     "cos 1.0 (pre/post/comb); 20.84x vs torch @M=32",
     "mixes[M,24]+scale[3]+base[24] -> pre/post/comb (24/row)"),
    ("MHC combine (reduce)",
     M*4096*(4+3),            # 4 multiplies + 3 adds per output elem (weighted sum)
     M*4*4096*B4              # read x[M,4,4096] FP32
     + M*4*B4                 # read per-request weights pre[M,4] FP32
     + M*4096*B4,             # write y[M,4096] FP32
     "fp32", f"{_REC}/mhc_combine.json", "pending",
     "cos 1.0; 2.12x vs torch @M=32",
     "einsum mk,mkh->mh; per-request pre[M,4] weights; +reduction adds"),
]

# --- fail-closed contract assertions (byte/FLOP + compute-dtype from the bench contracts @ M=32) ---
_EXPECT = {
    "indexer logits (q.ck+reduce)": {"bytes": 17_965_056, "cdt": "fp32"},   # G2: FP32 public boundary
    "compressor softmax-pool": {"bytes": 17_104_896, "cdt": "fp32"},
    "sparse attend (MQA+sink)": {"cdt": "fp32"},                            # G1: FP32 ceiling, not AMX
    "MHC sinkhorn (hc=4,20it)": {"bytes": 6_252, "cdt": "fp32"},
    "MHC combine (reduce)": {"bytes": 2_621_952, "flops": 917_504, "cdt": "fp32"},
}
for _n, _fl, _by, _cdt, _rec, _cm, _ct, _nt in OPS:
    _e = _EXPECT.get(_n)
    if _e:
        if "bytes" in _e:
            assert _by == _e["bytes"], f"{_n}: bytes {_by} != contract {_e['bytes']}"
        if "flops" in _e:
            assert _fl == _e["flops"], f"{_n}: flops {_fl} != contract {_e['flops']}"
        assert _cdt == _e["cdt"], f"{_n}: compute dtype {_cdt} != {_e['cdt']}"

print(f"DSv4 roofline-VS-observation (authored ops, M={M})  rev {REV[:8]}  nominal BW={BW/1e9:.0f} GB/s "
      f"AMX={PEAK/1e12:.0f} TF FP32={FP32_PEAK/1e12:.1f} TF")
print(f"  node={P.get('measurement_node','?')} ({P.get('mem_type','?')}); absolute latency WITHHELD "
      f"(not in any raw record) \u2014 records certify correctness + speedup only")
print("-" * 124)
print(f"{'op':30s} {'ideal_us':>9s} {'bind':>5s} {'cdt':>5s}  {'record@commit':34s} certifies")
print("-" * 124)
for name, fl, by, cdt, record, commit, certifies, note in OPS:
    peak = CPEAK[cdt]
    t_bw, t_cc = by / BW, (fl / peak if fl else 0.0)
    ideal = max(t_bw, t_cc)
    bind = "C" if t_cc > t_bw else "B"
    rc = f"{os.path.basename(record)}@{commit}"
    print(f"{name:30s} {ideal*1e6:9.1f} {bind:>5s} {cdt:>5s}  {rc:34s} {certifies}")
print("-" * 124)
print("ideal_us = max(bytes/BW, FLOPs/peak) with the row's EXPLICIT compute dtype (bf16->AMX, fp32->AVX-512).\n"
      "Each row's operands come from the cited benchmark's input/output contract (asserted above). The absolute\n"
      "node latency is WITHHELD: no auditable raw record contains it; the linked op_passes record certifies\n"
      "correctness + speedup-vs-torch only. No overhead/compute causation or donor-routing claim is made here.")
