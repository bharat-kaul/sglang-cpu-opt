#!/usr/bin/env python3
"""S1 sparse blockwise-BF16 measurement (review b51d1eb). Paired vs the shipped fp32-bmm donor and best-of,
across the full M x K sweep. speedup = donor/S1 (>1 means S1 faster). Correctness is screened by F4 / the
blockwise-ref check; this is timing only. Run as >=3 processes (sbatch). EXPERIMENTAL: a measured loss is a
valid disposition (many small packed GEMMs per request vs 2 large donor bmms)."""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_CF = ["-O3", "-fopenmp", "-march=native"]
_KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
torch.manual_seed(0)
from torch.utils.cpp_extension import load


def _t(fn, args, it=30):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*args)
    return (time.perf_counter() - t0) / it * 1e3


def paired(new_fn, old_fn, args, trials=5, it=30):
    new_fn(*args); old_fn(*args)
    ns, os_ = [], []
    for k in range(trials):
        if k % 2 == 0:
            n = _t(new_fn, args, it); o = _t(old_fn, args, it)
        else:
            o = _t(old_fn, args, it); n = _t(new_fn, args, it)
        ns.append(n); os_.append(o)
    return ns, os_


def _row(label, M, K, ns, os_):
    nm, om = st.median(ns) * 1e3, st.median(os_) * 1e3
    spds = sorted(o / n for n, o in zip(ns, os_))
    print(f"{label:>20} {M:>4} {K:>5} {nm:>11.1f} [{min(ns)*1e3:>6.1f},{max(ns)*1e3:>6.1f}] {om:>11.1f} "
          f"{st.median(spds):>7.3f}x [{spds[0]:>5.3f},{spds[-1]:>5.3f}]")


def main():
    sp = load(name="s1_sp", sources=[os.path.join(_KD, "sparse_attend.cpp")], extra_cflags=_CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# run identity: HEAD={sha} threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"bind={os.environ.get('OMP_PROC_BIND')} torch={torch.__version__}")
    print(f"{'experiment':>20} {'M':>4} {'K':>5} {'s1_med_us':>11} {'s1[min,max]':>16} {'base_med_us':>11} "
          f"{'spd_med':>8} {'spd[min,max]':>14}  (spd = base/S1)")
    for M in (1, 8, 16, 32, 64):
        for K in (128, 160, 512, 640):
            q = torch.randn(M, 64, 512); kv = torch.randn(M, K, 512); sink = torch.randn(64); scale = 512 ** -0.5
            ns, os_ = paired(lambda *a: sp.sparse_attend_blockbf16(q, kv, sink, scale),
                             lambda *a: sp.sparse_attend_fp32bmm(q, kv, sink, scale), (q,))
            _row("S1/fp32bmm", M, K, ns, os_)
            ns, os_ = paired(lambda *a: sp.sparse_attend_blockbf16(q, kv, sink, scale),
                             lambda *a: sp.sparse_attend_bestof(q, kv, sink, scale), (q,))
            _row("S1/bestof", M, K, ns, os_)
    print(">>> done")


if __name__ == "__main__":
    main()
