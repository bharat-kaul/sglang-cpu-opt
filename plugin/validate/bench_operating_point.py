#!/usr/bin/env python3
"""Regenerate the kernel-pilot operating-point data CONSISTENTLY after the re-optimization wave: the PROMOTED
C/C++ variant per kernel (compressor fp32intrin, sparse bestof_hx), the STANDARD 512-bit flags, and BOTH warm
and STREAMING (LLC-flushed, reliable) timing vs the torch baseline each kernel replaces, across the M-sweep.
Writes results/operating_point.json consumed by make_kernel_pilot_plots.py. EMR, OMP64 pinned, numactl N0.
"""
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
from bench_replicated import _idx_ref, _sparse_ref, _compressor_ref, _combine_ref   # the torch baselines

CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]  # STANDARD
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
BW, AMX, FP32 = 358.4e9, 124.5184e12, 7.7824e12
MS = [1, 2, 4, 8, 16, 32, 64]


def _mod(n, cpp):
    return load(name=n, sources=[os.path.join(KD, cpp)], extra_cflags=CF, verbose=False)


# byte/FLOP contracts + nominal peak (identical to make_kernel_pilot_plots KC) for the roofline floor.
# sparse_bestof roofline is the REFERENCE ALGORITHM ceiling (DeepSeek inference/kernel.py sparse_attn_kernel):
# bf16 flash-fused, KV read ONCE, nothing materialized -> AMX peak + bf16 bytes (KV once + q + out), NOT the
# old fp32-dense ceiling (276us@M32, the wrong algorithm). New floor ~58.5us@M32; the shipped fp32 kernel is
# ~12.6% of it (the gap the flash-bf16 rewrite would close).
KC = {
    "indexer_logits": (lambda M: 2*M*64*1024*128 + 3*M*64*1024, lambda M: M*64*128*4 + M*1024*128*4 + M*64*4 + M*1024*4, AMX, False),
    "indexer_topk": (lambda M: 0.0, lambda M: M*1024*4 + M*512*8, FP32, True),
    "compressor_R128": (lambda M: 3.0*M*128*512, lambda M: M*128*512*4*2 + M*512*4 + 128*512*4, FP32, True),
    "sparse_bestof": (lambda M: 4.0*M*64*512*512, lambda M: M*512*512*2 + 2*M*64*512*2, AMX, False),
    "sinkhorn": (lambda M: float(M*4*4*20*5), lambda M: M*24*4 + 3*4 + 24*4 + M*24*4, FP32, True),
    "combine": (lambda M: float(M*4096*7), lambda M: M*4*4096*4 + M*4*4 + M*4096*4, FP32, True),
}

_FLUSH = None
_SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0)
    _SINK += float(_FLUSH[::4096].sum())


def _call(fn, a):
    t0 = time.perf_counter(); fn(*a); return time.perf_counter() - t0


def twarm(fn, a, it=40):
    fn(*a)
    return st.median([_call(fn, a) for _ in range(it)])


def tstream(fn, a, it=25):
    s = []
    for _ in range(it):
        _flush(); s.append(_call(fn, a))
    return st.median(s)


def main():
    il = _mod("op_il", "indexer_logits.cpp"); tkm = _mod("op_tk", "indexer_topk.cpp")
    cp = _mod("op_cp", "compressor.cpp"); spm = _mod("op_sp", "sparse_attend.cpp")
    skm = _mod("op_sk", "sinkhorn.cpp"); cbm = _mod("op_cb", "combine.cpp")
    from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch

    def _idx(M): return (torch.randn(M, 64, 128), torch.randn(M, 1024, 128), torch.randn(M, 64))
    def _sp(M): return (torch.randn(M, 64, 512), torch.randn(M, 512, 512), torch.randn(64), 512 ** -0.5)
    def _cpr(M): return (torch.randn(M, 128, 512), torch.randn(M, 128, 512), torch.randn(128, 512))
    def _tk(M): return (torch.randn(M, 1024), 512)
    def _cb(M): return (torch.randn(M, 4 * 4096), torch.rand(M, 4), 4)
    def _sk(M): return (torch.randn(1, M, 24), torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6)

    # (name, builder, PROMOTED cpp fn, torch ref, pretty, donor, authored)
    specs = [
        ("indexer_logits", _idx, lambda a: il.indexer_logits(*a), lambda a: _idx_ref(*a), "indexer logits", True, "oneDNN BRGEMM + authored epilogue (512-bit)"),
        ("indexer_topk", _tk, lambda a: tkm.indexer_topk(*a), lambda a: torch.topk(a[0], a[1], dim=1, sorted=False).indices, "indexer top-k", False, "pure-authored nth_element selection"),
        ("compressor_R128", _cpr, lambda a: cp.compressor_softmax_pool_fp32intrin(*a), lambda a: _compressor_ref(*a), "compressor", False, "pure-authored, hand-inlined Cephes exp (D1)"),
        ("sparse_bestof", _sp, lambda a: spm.sparse_attend_bestof_hx(*a), lambda a: _sparse_ref(*a), "sparse attend", True, "oneDNN at::bmm + hand-exp softmax (bestof_hx)"),
        ("sinkhorn", _sk, lambda a: skm.mhc_sinkhorn(*a), lambda a: _hc_split_sinkhorn_torch(*a), "MHC sinkhorn", False, "pure-authored fused recurrence"),
        ("combine", _cb, lambda a: cbm.mhc_combine(*a), lambda a: _combine_ref(*a), "MHC combine", False, "pure-authored accumulate-once"),
    ]
    out = {"_schema": "Operating-point regen after the reopt wave: PROMOTED variant per kernel, 512-bit flags, "
                      "WARM + STREAMING (LLC-flushed, reliable) us vs the torch baseline. speedup uses STREAMING. "
                      "floor = max(bytes/BW, FLOPs/peak) vs NOMINAL peak; %roof = floor/stream (streaming).",
           "flags": CF, "node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "kernels": {}}
    print(f"{'kernel':>16} {'M':>4} {'cppW':>7} {'cppS':>7} {'torW':>7} {'torS':>7} {'spdS':>6} {'floor':>8} {'%roof':>6}")
    for name, build, cppf, reff, pretty, donor, authored in specs:
        flf, byf, pk, pure = KC[name]
        out["kernels"][name] = {"pretty": pretty, "donor": donor, "authored": authored, "M": {}}
        for M in MS:
            a = build(M)
            cw = twarm(lambda *x: cppf(x), a) * 1e6
            cs = tstream(lambda *x: cppf(x), a) * 1e6
            tw = twarm(lambda *x: reff(x), a) * 1e6
            ts = tstream(lambda *x: reff(x), a) * 1e6
            floor = max(byf(M) / BW, flf(M) / pk) * 1e6
            out["kernels"][name]["M"][str(M)] = {
                "cpp_warm_us": round(cw, 1), "cpp_stream_us": round(cs, 1),
                "torch_warm_us": round(tw, 1), "torch_stream_us": round(ts, 1),
                "speedup_stream": round(ts / cs, 2) if cs else 0, "floor_us": round(floor, 3),
                "pct_roof_stream": round(100 * floor / cs, 1) if cs else 0}
            print(f"{name:>16} {M:>4} {cw:7.1f} {cs:7.1f} {tw:7.1f} {ts:7.1f} {ts/cs:5.2f}x {floor:8.3f} {100*floor/cs:5.1f}%")
    out_path = os.environ.get("OP_OUT", os.path.join(HERE, "results", "operating_point.json"))
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
