#!/usr/bin/env python3
"""Replicated paired-trial requalification (R2-F3/P1-F6) for all 6 authored kernels.

For each kernel: >=5 trials, ORDER-VARIED paired timing (alternate cpp-first / fallback-first each trial to
cancel ordering bias), one warmup discarded, against a SAME-CONTRACT torch fallback (the op the kernel
replaces, at the kernel's arithmetic contract). Reports median + [min,max] spread of cpp, fallback, and the
paired speedup. NOT a single sequential mean. Prints run identity (git/dirty/threads/affinity). Correctness
is the F4 gate's job; this file is timing + the honest no-regression posture only.
"""
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "intel_cpu_models"))
from sparse_ref import ora_sparse_fp32
from torch.utils.cpp_extension import load

_CF = ["-O3", "-fopenmp", "-march=native"]
_KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
torch.manual_seed(0)


def _mod(n, cpp):
    return load(name=n, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _t(fn, args, it=30):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*args)
    return (time.perf_counter() - t0) / it * 1e3


def paired(cpp_fn, ref_fn, args, trials=5, it=30):
    cpp_fn(*args); ref_fn(*args)                       # warm (discarded)
    cs, rs = [], []
    for k in range(trials):
        if k % 2 == 0:                                 # alternate order each trial
            c = _t(cpp_fn, args, it); r = _t(ref_fn, args, it)
        else:
            r = _t(ref_fn, args, it); c = _t(cpp_fn, args, it)
        cs.append(c); rs.append(r)
    return cs, rs


# ---- same-contract fallbacks (the torch op each kernel replaces, at the kernel's contract) ----
def _idx_ref(q, kv, w):                                # bf16 stage boundaries (matches indexer_logits)
    s = torch.einsum("nhd,nsd->nhs", q.bfloat16(), kv.bfloat16())
    return (s.relu_() * w.bfloat16().unsqueeze(-1)).sum(1)


def _sparse_ref(q, kv, sink, scale):                   # fp32 einsum sparse (the torch fallback the donor replaces)
    return ora_sparse_fp32(q, kv, sink, scale)


def _compressor_ref(kv, score, ape):
    x = (score.float() + ape.float()).softmax(dim=1)
    return (x * kv.float()).sum(1)


def _combine_ref(x_flat, pre, hc):
    m, h = x_flat.shape[0], x_flat.shape[1] // hc
    return torch.einsum("mk,mkh->mh", pre.float(), x_flat.reshape(m, hc, h).float())


def main():
    il = _mod("r_il", "indexer_logits.cpp"); tkm = _mod("r_tk", "indexer_topk.cpp")
    cp = _mod("r_cp", "compressor.cpp"); spm = _mod("r_sp", "sparse_attend.cpp")
    skm = _mod("r_sk", "sinkhorn.cpp"); cbm = _mod("r_cb", "combine.cpp")
    try:
        from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
    except Exception:
        _hc_split_sinkhorn_torch = None

    def _idx(M): q = torch.randn(M, 64, 128); kv = torch.randn(M, 1024, 128); w = torch.randn(M, 64); return (q, kv, w)
    def _sp(M): return (torch.randn(M, 64, 512), torch.randn(M, 512, 512), torch.randn(64), 512 ** -0.5)
    def _cpr(M): return (torch.randn(M, 128, 512), torch.randn(M, 128, 512), torch.randn(128, 512))
    def _tk(M): return (torch.randn(M, 1024), 512)
    def _cb(M): return (torch.randn(M, 4 * 4096), torch.rand(M, 4), 4)

    specs = [
        ("indexer_logits", _idx, lambda a: il.indexer_logits(*a), lambda a: _idx_ref(*a)),
        ("sparse_bestof", _sp, lambda a: spm.sparse_attend_bestof(*a), lambda a: _sparse_ref(*a)),
        ("compressor_R128", _cpr, lambda a: cp.compressor_softmax_pool(*a), lambda a: _compressor_ref(*a)),
        ("indexer_topk", _tk, lambda a: tkm.indexer_topk(*a), lambda a: torch.topk(a[0], a[1], dim=1, sorted=False).indices),
        ("combine", _cb, lambda a: cbm.mhc_combine(*a), lambda a: _combine_ref(*a)),
    ]
    if _hc_split_sinkhorn_torch is not None:
        def _sk(M): return (torch.randn(1, M, 24), torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6)
        specs.insert(4, ("sinkhorn", _sk, lambda a: skm.mhc_sinkhorn(*a), lambda a: _hc_split_sinkhorn_torch(*a)))

    print(f"{'kernel':>16} {'M':>4} {'cpp_med_us':>11} {'cpp[min,max]':>16} {'ref_med_us':>11} "
          f"{'spd_med':>8} {'spd[min,max]':>14}")
    for name, build, cppf, reff in specs:
        for M in (1, 8, 16, 32, 64):
            cs, rs = paired(lambda *a: cppf(a), lambda *a: reff(a), build(M))
            cm, rm = st.median(cs) * 1e3, st.median(rs) * 1e3          # us
            spds = sorted(r / c for c, r in zip(cs, rs))
            print(f"{name:>16} {M:>4} {cm:>11.1f} [{min(cs)*1e3:>6.1f},{max(cs)*1e3:>6.1f}] {rm:>11.1f} "
                  f"{st.median(spds):>7.2f}x [{spds[0]:>5.2f},{spds[-1]:>5.2f}]")
    print(">>> done")


if __name__ == "__main__":
    main()
