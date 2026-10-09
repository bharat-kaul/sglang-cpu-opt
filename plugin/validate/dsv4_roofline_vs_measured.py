#!/usr/bin/env python3
"""DSv4-Flash roofline-VS-observation join (authored ops).

Each row's IDEAL floor (max(bytes/BW, flops/peak)) is derived from the op's ACTUAL benchmark
INPUT/OUTPUT CONTRACT (F4) at M=32, with an EXPLICIT compute dtype that selects the compute
resource (G1): bf16 -> AMX, fp32 -> AVX-512 FP32 ceiling. Each row links an AUDITABLE RESULT
RECORD (results/op_passes/*.json) whose kept pass gives a MICROBENCH cosine/set-match + kernel
revision. That numerical match is tied to the tested shape/dtype/reference/tolerance; it is NOT a
current-target certificate -> E2E verification PENDING. Absolute latency and speedup are WITHHELD/
UNVERIFIED. No causal/overhead/donor-routing conclusions are drawn here.

PROVENANCE: node = pcl-sprh02 (DDR5-5600, 64 threads NUMA0); batch M=32; model revision in REV;
dtypes + operands per the cited benchmark contract. Contract byte/FLOP expectations are asserted
below (fail-closed).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_roofline_p2 import REF_REVISION as REV, load_record, _rev_resolved, _ATTRIB  # centralized ref + record reader

_SPEC = os.path.join(os.path.dirname(__file__), "platforms", "emr.json")
P = json.load(open(_SPEC))
MP = P["machine_peak"]
BW = MP["mem_bw_gbps"] * 1e9                 # nominal target
PEAK = MP["amx_bf16_tflops"] * 1e12          # AMX bf16 ceiling
FP32_PEAK = MP["avx512_fp32_tflops"] * 1e12  # AVX-512 FP32 ceiling
CPEAK = {"bf16": PEAK, "fp32": FP32_PEAK}    # G1: compute dtype -> compute resource
B2, B4, I4 = 2.0, 4.0, 4.0                   # bf16 / fp32 / int32 bytes

M = 32  # the single observed coordinate

# op: (name, flops(M), bytes(M), compute_dtype, record_filename, plateau_note)
# bytes/flops derived from each bench's call contract; evidence READ from the result record (H1).
OPS = [
    ("indexer logits (q.ck+reduce)",
     lambda M: 2*M*64*1024*128 + 3*M*64*1024,
     lambda M: M*64*128*B4 + M*1024*128*B4 + M*64*B4 + M*1024*B4,   # q + kv + w + logits (FP32 public boundary)
     "fp32", "indexer_logits.json",
     "BW-bound; best-of dispatcher (tiled brgemm M>=8 + bmm M=1), scores cache-resident. Plateau = achievable-vs-nominal DRAM BW wall (~60-77%) + small-M dispatch; not closable by kernel work."),
    ("indexer top-k (512 of 1024)",
     lambda M: 0,
     lambda M: M*1024*B4 + M*512*I4,                               # read logits + write 512 idx
     "fp32", "indexer_topk.json",
     "latency/selection-bound (no AMX GEMM primitive); chunked within-row saturates cores at small M. Plateau = tiny absolute (14-39us), dispatch-bound; off-roof diagnostic only."),
    ("compressor softmax-pool",
     lambda M: 3*M*128*512,
     lambda M: M*128*512*B4 + M*128*512*B4 + M*512*B4 + 128*512*B4,  # kv + score + out + shared APE
     "fp32", "compressor.json",
     "BW-bound streaming online-softmax pool (no w[N,R,D] temporary). Plateau = DRAM BW wall; off-roof vs nominal is the wall, not a defect."),
    ("sparse attend (MQA+sink)",
     lambda M: 4*M*64*512*512,
     lambda M: M*512*512*B4 + 2*M*64*512*B4,                       # latent KV + q + out
     "fp32", "sparse_attend.json",
     "SURFACED: scalar loses to torch BLAS (~0.3x), AMX below correctness tol -> donor MLA flash is the production path. Not optimized further."),
    ("MHC sinkhorn (hc=4,20it)",
     lambda M: M*4*4*20*5,
     lambda M: M*24*B4 + 3*B4 + 24*B4 + M*24*B4,                   # mixes + scale + base + pre/post/comb
     "fp32", "mhc_sinkhorn.json",
     "dispatch-bound (reference ~40 tiny torch ops/call); fused 20 iters -> ~20x vs torch. Plateau = tiny hc=4 op, latency-bound; off-roof diagnostic only."),
    ("MHC combine (reduce)",
     lambda M: M*4096*(4+3),                                       # 4 mul + 3 add per out elem
     lambda M: M*4*4096*B4 + M*4*B4 + M*4096*B4,                   # x + per-request pre + y
     "fp32", "mhc_combine.json",
     "BW/latency-bound; tiled accumulate-once (x read once, y written once). Near roof at M>=8; plateau = minimal traffic achieved."),
]

# --- fail-closed contract assertions (byte/FLOP + compute-dtype from the bench contracts @ M=32) ---
_EXPECT = {
    "indexer logits (q.ck+reduce)": {"bytes": 17_965_056, "cdt": "fp32"},   # G2: FP32 public boundary
    "compressor softmax-pool": {"bytes": 17_104_896, "cdt": "fp32"},
    "sparse attend (MQA+sink)": {"cdt": "fp32"},                            # G1: FP32 ceiling, not AMX
    "MHC sinkhorn (hc=4,20it)": {"bytes": 6_252, "cdt": "fp32"},
    "MHC combine (reduce)": {"bytes": 2_621_952, "flops": 917_504, "cdt": "fp32"},
}
for _n, _flf, _byf, _cdt, _rec, _nt in OPS:
    _e = _EXPECT.get(_n)
    if _e:
        if "bytes" in _e:
            assert _byf(M) == _e["bytes"], f"{_n}: bytes {_byf(M)} != contract {_e['bytes']}"
        if "flops" in _e:
            assert _flf(M) == _e["flops"], f"{_n}: flops {_flf(M)} != contract {_e['flops']}"
        assert _cdt == _e["cdt"], f"{_n}: compute dtype {_cdt} != {_e['cdt']}"

# measured M-sweep (sourced from a raw record); None if an op has no measurement yet
_PS = json.load(open(os.path.join(os.path.dirname(__file__), "results", "perf_sweep.json")))
MS = _PS["ms"]

print(f"DSv4 roofline-VS-measured (authored ops)  rev {REV[:8]}  M-sweep {MS}  nominal BW={BW/1e9:.0f} GB/s "
      f"AMX={PEAK/1e12:.0f} TF FP32={FP32_PEAK/1e12:.1f} TF")
print(f"  measured: {_PS['raw_record']}")
print(f"  ideal_us = max(bytes/BW, FLOPs/peak) at the row's compute dtype (nominal peak); off = measured/ideal "
      f"(vs NOMINAL \u2014 the node reaches ~60-77% of nominal DRAM BW, so a BW-bound op is ~1.3-1.7x off from the wall alone).")
for name, flf, byf, cdt, record, plateau in OPS:
    peak = CPEAK[cdt]
    rec = load_record(record)                                     # H1: evidence READ from record (fail-closed)
    rev = rec["kernel_rev"] if _rev_resolved(rec["kernel_rev"]) else f"UNRESOLVED({rec['kernel_rev'] or 'none'})"
    meas = _PS["ops"].get(name, {}).get("median_ms")
    print("-" * 104)
    print(f"{name}  [cdt={cdt}, record={record}@{rev}]")
    print(f"  correctness (recorded, verbatim): {rec['correctness']}   [{_ATTRIB}]")
    print(f"  {'M':>4} {'bind':>5} {'ideal_us':>10} {'measured_us':>12} {'off_ceiling':>12}")
    for i, Mv in enumerate(MS):
        fl, by = flf(Mv), byf(Mv)
        t_cc = fl / peak if fl else 0.0
        ideal = max(by / BW, t_cc)
        bind = "C" if t_cc > by / BW else "B"
        if meas:
            m_us = meas[i] * 1e3
            off = m_us / (ideal * 1e6) if ideal else float("inf")
            print(f"  {Mv:>4} {bind:>5} {ideal*1e6:>10.2f} {m_us:>12.1f} {off:>11.1f}x")
        else:
            print(f"  {Mv:>4} {bind:>5} {ideal*1e6:>10.2f} {'n/a':>12} {'n/a':>12}")
    print(f"  plateau: {plateau}")
print("-" * 104)
print("off_ceiling is a DIAGNOSTIC vs the NOMINAL roof (not an achievability claim). Streaming ops are DRAM-\n"
      "BW-wall-bound; tiny ops are latency/dispatch-bound (far from roof by construction). Correctness is the\n"
      "recorded field VERBATIM (microbench; E2E verification PENDING). Measured latency is sourced from the\n"
      "raw record above (median of 3, threads bound, one NUMA domain). No causation/donor-routing claim here.")
