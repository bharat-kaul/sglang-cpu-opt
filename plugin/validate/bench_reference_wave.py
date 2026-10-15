#!/usr/bin/env python3
"""REFERENCE-ALGORITHM WAVE: measure the DSA kernels at the dtype/algorithm DeepSeek's reference actually uses
(inference/kernel.py) vs the CORRECTED roofline, isolating the fp32->bf16 CAST confound that sank the earlier
bf16 attempts. The reference runs sparse attention bf16 flash-fused (KV read once) and the indexer in fp8;
our prior bf16 losses fed FP32 inputs, so the kernels paid a per-call cast the reference never pays. Here we
feed BF16-NATIVE inputs (the internal .to(bf16) becomes a no-op; the indexer tiled path takes the kv_bf16
branch and drops the per-tile convert) so the measurement reflects the real serving dtype.

One kernel per node (parallel wave): --kernel {sparse|indexer}. Reports, per (variant, M): streaming latency
(LLC-flushed, 3-replicate median), speedup vs the fp32 baseline, %roof vs the corrected roofline, and cos vs
the fp32 oracle (conformance). Disposition per variant: WIN+conformant / WIN+tolerance-gated / LOSS.

Usage: python bench_reference_wave.py --kernel sparse   (EMR, OMP64 pinned, numactl N0)
"""
import argparse
import json
import os
import statistics as st
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "intel_cpu_models"))
from torch.utils.cpp_extension import load
from bench_replicated import _idx_ref, _sparse_ref

CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
BW, AMX, FP32 = 358.4e9, 124.5184e12, 7.7824e12
MS = [1, 2, 4, 8, 16, 32, 64]

_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0)


def _call(fn, a):
    t0 = time.perf_counter(); fn(*a); return time.perf_counter() - t0


def tstream(fn, a, it=25):
    fn(*a)                                        # warm (compile/pack), not timed
    s = []
    for _ in range(it):
        _flush(); s.append(_call(fn, a))
    return st.median(s) * 1e6


def _cos(x, y):
    x = x.to(torch.float32).flatten(); y = y.to(torch.float32).flatten()
    return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-20))


def _mod(cpp):
    return load(name="wave_" + cpp.split(".")[0], sources=[os.path.join(KD, cpp)], extra_cflags=CF, verbose=False)


def bench_sparse():
    m = _mod("sparse_attend.cpp")
    # corrected roofline: bf16 flash-fused (KV read once, nothing materialized), AMX peak
    def floor(M):
        return max((M*512*512*2 + 2*M*64*512*2) / BW, (4*M*64*512*512) / AMX) * 1e6
    # (label, fn, prep(args)->args, conformant?)  prep casts the operands to the variant's native dtype
    def f32(a): return a
    def bf16(a): return (a[0].to(torch.bfloat16), a[1].to(torch.bfloat16), a[2], a[3])
    variants = [
        ("fp32_bestof_hx (baseline, bit-exact)", lambda a: m.sparse_attend_bestof_hx(*a), f32, True),
        ("fp32_scalar_fused_own (no donor)", lambda a: m.sparse_attend(*a), f32, True),
        ("bf16_dense_native (no cast)", lambda a: m.sparse_attend_bf16bmm_hx(*a), bf16, False),
        ("bf16_flash_native (no cast)", lambda a: m.sparse_attend_blockbf16(*a), bf16, False),
    ]
    def build(M): return (torch.randn(M, 64, 512), torch.randn(M, 512, 512), torch.randn(64), 512 ** -0.5)
    return "sparse", variants, build, lambda a: _sparse_ref(*a), floor, {"bf16_flash_us": floor}


def bench_indexer():
    m = _mod("indexer_logits.cpp")
    # ROOFLINE = native bf16 compute peak + BW (the fixed HW ceiling; %roof basis). fp8 is NOT a compute dtype
    # on AMX -> it is a DATA-MOVEMENT optimization only: storing KV fp8 cuts the length-S read traffic, reported
    # as a projected BW-floor, NOT a roofline. It helps only when BW-bound (long context), ~0 when dispatch-bound.
    def roof_bf16(M):
        return max((M*64*128*2 + M*1024*128*2 + M*64*4 + M*1024*4) / BW, (2*M*64*1024*128) / AMX) * 1e6
    def fp8_traffic_proj(M):
        return max((M*64*128*2 + M*1024*128*1 + M*64*4 + M*1024*4) / BW, (2*M*64*1024*128) / AMX) * 1e6
    def f32(a): return a
    def bf16kv(a): return (a[0], a[1].to(torch.bfloat16), a[2])        # feed bf16 KV -> kv_bf16 branch (no convert)
    variants = [
        ("fp32KV_tiled (baseline)", lambda a: m.indexer_logits(*a), f32, True),
        ("bf16KV_tiled (no convert)", lambda a: m.indexer_logits(*a), bf16kv, True),
        ("fp32avx (AVX-512 fp32 score)", lambda a: m.indexer_logits_fp32avx(*a), f32, False),
        ("bf16_amxbmm (single bf16 bmm)", lambda a: m.indexer_logits_amxbmm(*a), bf16kv, False),
    ]
    def build(M): return (torch.randn(M, 64, 128), torch.randn(M, 1024, 128), torch.randn(M, 64))
    # %roof vs the bf16+BW hardware roofline; fp8 reported separately as a BW-optimization traffic projection
    return "indexer", variants, build, lambda a: _idx_ref(*a), roof_bf16, {"bf16_bw_roofline_us": roof_bf16, "fp8_store_traffic_proj_us": fp8_traffic_proj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=["sparse", "indexer"])
    args = ap.parse_args()
    name, variants, build, oracle, floor, floors = (bench_sparse() if args.kernel == "sparse" else bench_indexer())

    out = {"_schema": "reference-algorithm wave: variants at the reference dtype (bf16-native, no cast) vs the "
                      "corrected roofline; cos vs the fp32 oracle. streaming LLC-flushed 3-replicate median (EMR). "
                      "ONEDNN_MAX_CPU_ISA forces the oneDNN ISA (AVX512_CORE_BF16 = dpbf16, AMX disabled) for the "
                      "donor-first conformance-safe low-M ISA test.",
           "kernel": name, "flags": CF, "node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
           "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA", os.environ.get("DNNL_MAX_CPU_ISA", "default(AMX)")),
           "threads": torch.get_num_threads(), "M": {}}
    hdr = f"{'M':>4} {'variant':<38} {'us':>9} {'spd':>6} {'%roof':>6} {'cos':>10}"
    print(hdr); print("-" * len(hdr))
    for M in MS:
        a = build(M)
        ref = oracle(a)
        base = None
        rowM = {"floor_us": {k: round(fn(M), 2) for k, fn in floors.items()}, "variants": {}}
        for label, fn, prep, conf in variants:
            aa = prep(build(M))
            us = tstream(lambda *x: fn(x), aa)
            cos = _cos(fn(prep(a)), ref)
            if base is None:
                base = us
            fl = floor(M)
            rowM["variants"][label] = {"us": round(us, 1), "speedup_vs_base": round(base / us, 2),
                                       "pct_roof": round(100 * fl / us, 1), "cos": round(cos, 6), "conformant": conf}
            print(f"{M:>4} {label:<38} {us:9.1f} {base/us:5.2f}x {100*fl/us:5.1f}% {cos:10.6f}")
        out["M"][str(M)] = rowM
    outp = os.environ.get("WAVE_OUT", os.path.join(HERE, "results", f"reference_wave_{name}.json"))
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", outp)


if __name__ == "__main__":
    main()
