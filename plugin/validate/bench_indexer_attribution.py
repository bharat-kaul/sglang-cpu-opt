#!/usr/bin/env python3
"""Indexer-logits ATTRIBUTION (Stage 1.5: attribute BEFORE authoring a KV-read optimization).

Splits the measured streaming latency into its components so we know whether the dominant, RECOVERABLE cost is
actually the KV DRAM read (-> optimize the read) or the scalar bf16-staged epilogue / brgemm / convert/pack
(-> a different lever). Measures, streaming, at the M-sweep:
  T_full        full kernel, fp32 KV (shipped path)
  T_noepi       full kernel, fp32 KV, INDEXER_SKIP_EPILOGUE=1  -> epilogue = T_full - T_noepi
  T_bf16kv      full kernel, bf16 KV (half read, no per-tile convert) -> read+convert share = T_full - T_bf16kv
  T_rawread     reduce the fp32 KV tensor (achievable DRAM read time for the KV bytes)
  T_gemmproxy   bf16 at::bmm at the score-GEMM shape (brgemm cost proxy)
  breakdown     INDEXER_BREAKDOWN convert/pack/compute (fp32 KV), from the kernel's own timers
Run >=3 processes (sbatch); take cross-proc medians.
"""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
BW = 358.4e9
MS = [1, 8, 16, 32, 64]
H, D, S = 64, 128, 1024
torch.manual_seed(0)
from torch.utils.cpp_extension import load

_FLUSH = None; _SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0); _SINK += float(_FLUSH[::4096].sum())


def _call(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def tstream(fn, it=25):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6   # us


def main():
    il = load(name="idx_attr", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# identity HEAD={sha} proc={os.environ.get('PROC','0')} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")
    print(f"{'M':>4} {'T_full':>8} {'T_noepi':>8} {'epi':>7} {'epi%':>5} {'T_bf16kv':>9} {'rd+cvt':>7} "
          f"{'T_raw':>7} {'T_gemm':>7} {'kvMB':>6} {'rdGB/s':>7}")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        kvb = kv.to(torch.bfloat16)
        qb = q.to(torch.bfloat16)
        os.environ.pop("INDEXER_SKIP_EPILOGUE", None)
        t_full = tstream(lambda: il.indexer_logits(q, kv, w))
        t_bf16 = tstream(lambda: il.indexer_logits(q, kvb, w))
        os.environ["INDEXER_SKIP_EPILOGUE"] = "1"
        t_noepi = tstream(lambda: il.indexer_logits(q, kv, w))
        os.environ.pop("INDEXER_SKIP_EPILOGUE", None)
        t_raw = tstream(lambda: kv.sum())                                   # fp32 KV read
        t_gemm = tstream(lambda: torch.bmm(kvb, qb.transpose(1, 2)))        # bf16 score GEMM proxy
        epi = t_full - t_noepi
        rdcvt = t_full - t_bf16
        kv_mb = M * S * D * 4 / 1e6
        rd_gbs = (M * S * D * 4) / (t_raw * 1e-6) / 1e9
        print(f"{M:>4} {t_full:>8.1f} {t_noepi:>8.1f} {epi:>7.1f} {100*epi/t_full:>4.0f}% {t_bf16:>9.1f} "
              f"{rdcvt:>7.1f} {t_raw:>7.1f} {t_gemm:>7.1f} {kv_mb:>6.1f} {rd_gbs:>7.1f}")
    # breakdown (convert/pack/compute) for M=32 via the kernel's own stderr timers
    print("\n# INDEXER_BREAKDOWN (fp32 KV), one call each M:")
    os.environ["INDEXER_BREAKDOWN"] = "1"
    for M in (1, 32, 64):
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        il.indexer_logits(q, kv, w)
    os.environ.pop("INDEXER_BREAKDOWN", None)


if __name__ == "__main__":
    main()
