#!/usr/bin/env python3
"""Standalone-optimization paired measurement (review b51d1eb, experiments I1/I2/C1).

Compares the CANDIDATE kernels (working tree) against the PRE-OPTIMIZATION BASELINE kernels (sources read
from git ref $BASELINE_REF, default 020f429) with ORDER-VARIED paired trials so comparator drift cancels.
This measures the OPTIMIZATION DELTA (new native vs old native) -- NOT the no-regression-vs-torch posture
(that stays bench_replicated.py's job). Correctness (bit-exactness vs baseline) is proven separately in F4 /
the old-vs-new equality check; this file is timing only. Run it as >=3 separate PROCESSES (the sbatch does)
and aggregate the per-process medians; a single process is one sample, not a result.

  I1: indexer bf16-KV path, baseline vs candidate (candidate drops the unused FP32-staging allocation).
  I2: candidate indexer, SAME bf16-rounded KV supplied as FP32 vs as BF16 (qualifies the storage-contract
      read reduction as a speed number; both store FP32 output). This is a conditional, caller-dependent win.
  C1: compressor (all 3 shapes), baseline vs candidate (candidate does one std::exp per update).
"""
import os
import statistics as st
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_CF = ["-O3", "-fopenmp", "-march=native"]
_KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
BASELINE_REF = os.environ.get("BASELINE_REF", "020f429")
torch.manual_seed(0)
from torch.utils.cpp_extension import load


def _cand(name, cpp):
    return load(name=name, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _baseline(name, cpp):
    src = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE_REF}:plugin/kernels/dsa_pilot/{cpp}"])
    f = os.path.join(tempfile.gettempdir(), f"baseline_{BASELINE_REF}_{cpp}")
    open(f, "wb").write(src)
    return load(name=name, sources=[f], extra_cflags=_CF, verbose=False)


def _t(fn, args, it=30):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*args)
    return (time.perf_counter() - t0) / it * 1e3


def paired(new_fn, old_fn, args, trials=5, it=30):
    new_fn(*args); old_fn(*args)                       # warm (discarded)
    ns, os_ = [], []
    for k in range(trials):
        if k % 2 == 0:
            n = _t(new_fn, args, it); o = _t(old_fn, args, it)
        else:
            o = _t(old_fn, args, it); n = _t(new_fn, args, it)
        ns.append(n); os_.append(o)
    return ns, os_


def _row(label, M, ns, os_):
    nm, om = st.median(ns) * 1e3, st.median(os_) * 1e3                      # us
    spds = sorted(o / n for n, o in zip(ns, os_))                          # old/new = candidate speedup
    print(f"{label:>22} {M:>4} {nm:>11.1f} [{min(ns)*1e3:>6.1f},{max(ns)*1e3:>6.1f}] {om:>11.1f} "
          f"{st.median(spds):>7.3f}x [{spds[0]:>5.3f},{spds[-1]:>5.3f}]")


def main():
    il_new = _cand("so_il_new", "indexer_logits.cpp")
    il_old = _baseline("so_il_old", "indexer_logits.cpp")
    cp_new = _cand("so_cp_new", "compressor.cpp")
    cp_old = _baseline("so_cp_old", "compressor.cpp")

    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    dirty = bool(subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"]).strip())
    print(f"# run identity: HEAD={sha} dirty={dirty} baseline_ref={BASELINE_REF} "
          f"threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"bind={os.environ.get('OMP_PROC_BIND')} torch={torch.__version__}")
    print(f"{'experiment':>22} {'M':>4} {'new_med_us':>11} {'new[min,max]':>16} {'old_med_us':>11} "
          f"{'spd_med':>8} {'spd[min,max]':>14}  (spd = old/new = candidate speedup)")

    # I1: indexer bf16-KV path, baseline vs candidate (both consume bf16 KV directly)
    for M in (1, 8, 16, 32, 64):
        q = torch.randn(M, 64, 128); kv = torch.randn(M, 1024, 128).bfloat16(); w = torch.randn(M, 64).bfloat16().float()
        ns, os_ = paired(lambda *a: il_new.indexer_logits(*a), lambda *a: il_old.indexer_logits(*a), (q, kv, w))
        _row("I1 indexer bf16kv", M, ns, os_)

    # I2: candidate indexer, same bf16-rounded KV as FP32 vs BF16 (storage-contract read reduction)
    for M in (1, 8, 16, 32, 64):
        q = torch.randn(M, 64, 128); kvr = torch.randn(M, 1024, 128).bfloat16(); w = torch.randn(M, 64).bfloat16().float()
        kv_fp32 = kvr.float(); kv_bf16 = kvr
        ns, os_ = paired(lambda *a: il_new.indexer_logits(q, kv_bf16, w),
                         lambda *a: il_new.indexer_logits(q, kv_fp32, w), (q,))
        _row("I2 bf16kv/fp32kv", M, ns, os_)

    # C1: compressor, all 3 shapes, baseline vs candidate
    for (R, D) in ((128, 512), (8, 512), (8, 128)):
        for M in (1, 8, 16, 32, 64):
            kv = torch.randn(M, R, D); sc = torch.randn(M, R, D); ape = torch.randn(R, D)
            ns, os_ = paired(lambda *a: cp_new.compressor_softmax_pool(*a),
                             lambda *a: cp_old.compressor_softmax_pool(*a), (kv, sc, ape))
            _row(f"C1 compressor r{R}d{D}", M, ns, os_)
    print(">>> done")


if __name__ == "__main__":
    main()
