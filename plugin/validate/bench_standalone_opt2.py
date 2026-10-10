#!/usr/bin/env python3
"""Standalone-optimization step-3 paired measurement (review b51d1eb: I3, C2, C3).

  I3: indexer epilogue weight-hoist (bit-exact) -- working tree vs git HEAD3 ($I3_BASELINE_REF, default
      3ed5869 = post-I1/pre-I3). speedup = old/new.
  C2: compressor vectorized one-exp vs the shipped scalar C1 (both in the working-tree module). NOT bit-exact
      (vector exp) -> F4-screened separately. speedup = C1/C2.
  C3: compressor stable two-pass vs the shipped C1. NOT bit-exact -> F4-screened. speedup = C1/C3.

Correctness is the F4 gate's job (C2/C3 screen within the compressor tolerance; I3 is bit-exact). This file
is timing only: order-varied paired trials, median + spread, run identity. Run as >=3 PROCESSES (sbatch).
"""
import os
import statistics as st
import subprocess
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_CF = ["-O3", "-fopenmp", "-march=native"]
_KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
I3_BASELINE_REF = os.environ.get("I3_BASELINE_REF", "3ed5869")
torch.manual_seed(0)
from torch.utils.cpp_extension import load


def _cand(name, cpp):
    return load(name=name, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _baseline(name, cpp, ref):
    src = subprocess.check_output(["git", "-C", ROOT, "show", f"{ref}:plugin/kernels/dsa_pilot/{cpp}"])
    f = os.path.join(tempfile.gettempdir(), f"base_{ref}_{cpp}")
    open(f, "wb").write(src)
    return load(name=name, sources=[f], extra_cflags=_CF, verbose=False)


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


def _row(label, M, ns, os_):
    nm, om = st.median(ns) * 1e3, st.median(os_) * 1e3
    spds = sorted(o / n for n, o in zip(ns, os_))
    print(f"{label:>24} {M:>4} {nm:>11.1f} [{min(ns)*1e3:>6.1f},{max(ns)*1e3:>6.1f}] {om:>11.1f} "
          f"{st.median(spds):>7.3f}x [{spds[0]:>5.3f},{spds[-1]:>5.3f}]")


def main():
    il_new = _cand("s3_il_new", "indexer_logits.cpp")
    il_old = _baseline("s3_il_old", "indexer_logits.cpp", I3_BASELINE_REF)
    cp = _cand("s3_cp", "compressor.cpp")

    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    dirty = bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"]).strip())
    print(f"# run identity: HEAD={sha} dirty={dirty} i3_baseline={I3_BASELINE_REF} "
          f"threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"bind={os.environ.get('OMP_PROC_BIND')} torch={torch.__version__}")
    print(f"{'experiment':>24} {'M':>4} {'new_med_us':>11} {'new[min,max]':>16} {'old_med_us':>11} "
          f"{'spd_med':>8} {'spd[min,max]':>14}  (spd = old/new = candidate speedup)")

    # I3: indexer weight-hoist, bf16-KV and fp32-KV paths
    for tag, bf in (("i3 idx bf16kv", True), ("i3 idx fp32kv", False)):
        for M in (1, 8, 16, 32, 64):
            q = torch.randn(M, 64, 128); kv = torch.randn(M, 1024, 128); kv = kv.bfloat16() if bf else kv
            w = torch.randn(M, 64).bfloat16().float()
            ns, os_ = paired(lambda *a: il_new.indexer_logits(q, kv, w), lambda *a: il_old.indexer_logits(q, kv, w), (q,))
            _row(tag, M, ns, os_)

    # C2 / C3: variants vs the shipped C1 (speedup = C1/variant)
    for (R, D) in ((128, 512), (8, 512), (8, 128)):
        for M in (1, 8, 16, 32, 64):
            kv = torch.randn(M, R, D); sc = torch.randn(M, R, D); ape = torch.randn(R, D)
            ns, os_ = paired(lambda *a: cp.compressor_softmax_pool_vexp(*a), lambda *a: cp.compressor_softmax_pool(*a), (kv, sc, ape))
            _row(f"C2 vexp r{R}d{D}", M, ns, os_)
            ns, os_ = paired(lambda *a: cp.compressor_softmax_pool_multipass(*a), lambda *a: cp.compressor_softmax_pool(*a), (kv, sc, ape))
            _row(f"C3 mpass r{R}d{D}", M, ns, os_)
    print(">>> done")


if __name__ == "__main__":
    main()
