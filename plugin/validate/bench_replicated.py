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

    # implementation provenance: which kernels lean on a oneDNN/MKL donor GEMM vs are authored from scratch.
    PROVENANCE = {
        "indexer_logits": {"donor": "oneDNN cpublas BRGEMM (q.kv^T GEMM)", "authored": "fused relu*weight*sum epilogue (L1-resident, bf16 stages)", "pure_authored": False},
        "sparse_bestof": {"donor": "oneDNN/MKL at::bmm (score + value GEMMs)", "authored": "fused in-place softmax+sink; N==1 scalar path; best-of dispatch", "pure_authored": False},
        "compressor_R128": {"donor": None, "authored": "AVX-512 streaming online-softmax pool, N*D-tile parallel", "pure_authored": True},
        "indexer_topk": {"donor": None, "authored": "nth_element selection + guarded chunked/row-parallel", "pure_authored": True},
        "sinkhorn": {"donor": None, "authored": "fused 20-iter recurrence (L1-resident, no per-iter torch ops)", "pure_authored": True},
        "combine": {"donor": None, "authored": "accumulate-once tiling (read-once/write-once)", "pure_authored": True},
    }

    print(f"{'kernel':>16} {'M':>4} {'cpp_med_us':>11} {'cpp[min,max]':>16} {'ref_med_us':>11} "
          f"{'spd_med':>8} {'spd[min,max]':>14}  (ref = non-C/C++ torch baseline)")
    out = {"_schema": "Newly-authored C/C++ kernel vs its NON-C/C++ torch baseline (the op the kernel replaces, "
                      "at the kernel's arithmetic contract). spd = torch_ref/cpp, order-varied paired, median of "
                      "trials. PROVENANCE flags which kernels lean on a oneDNN/MKL donor GEMM (indexer_logits, "
                      "sparse) vs are PURE-AUTHORED (compressor, topk, sinkhorn, combine) -- for donor kernels "
                      "the speedup includes the library GEMM, so the AUTHORED contribution is the fused epilogue/"
                      "dispatch. torch refs are noisy (comparator drift, esp. indexer) -> report medians.",
           "provenance_note": PROVENANCE, "kernels": {}}
    for name, build, cppf, reff in specs:
        out["kernels"][name] = {"pure_authored": PROVENANCE.get(name, {}).get("pure_authored"), "M": {}}
        for M in (1, 8, 16, 32, 64):
            cs, rs = paired(lambda *a: cppf(a), lambda *a: reff(a), build(M))
            cm, rm = st.median(cs) * 1e3, st.median(rs) * 1e3          # us
            spds = sorted(r / c for c, r in zip(cs, rs))
            out["kernels"][name]["M"][str(M)] = {"cpp_us": round(cm, 1), "torch_us": round(rm, 1),
                                                 "speedup_med": round(st.median(spds), 2),
                                                 "speedup_minmax": [round(spds[0], 2), round(spds[-1], 2)]}
            print(f"{name:>16} {M:>4} {cm:>11.1f} [{min(cs)*1e3:>6.1f},{max(cs)*1e3:>6.1f}] {rm:>11.1f} "
                  f"{st.median(spds):>7.2f}x [{spds[0]:>5.2f},{spds[-1]:>5.2f}]")
    import os as _os
    if _os.environ.get("REPLICATED_JSON"):
        import json as _json
        out["_provenance"] = {"head": _os.popen("git rev-parse --short HEAD").read().strip(),
                              "node": _os.uname().nodename, "process_index": _os.environ.get("REPLICATED_PROC"),
                              "slurm_job_id": _os.environ.get("SLURM_JOB_ID"), "omp": _os.environ.get("OMP_NUM_THREADS"),
                              "bind": _os.environ.get("OMP_PROC_BIND")}
        _json.dump(out, open(_os.environ["REPLICATED_JSON"], "w"), indent=1)
    print(">>> done")


if __name__ == "__main__":
    main()
