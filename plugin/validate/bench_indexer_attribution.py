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


def twarm(fn, it=40):
    fn()
    return st.median([_call(fn) for _ in range(it)]) * 1e6   # us (warm; buffers hot, no flush)


def main():
    il = load(name="idx_attr", sources=[os.path.join(KD, "indexer_logits.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# identity HEAD={sha} proc={os.environ.get('PROC','0')} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")
    print("# FUSED kernel component decomposition (streaming us). Derived: epilogue=T_full-T_noepi, "
          "brgemm=T_noepi-T_nogemm, convert+pack+KVread+dispatch=T_nogemm.")
    print(f"{'M':>4} {'T_full':>8} {'epi':>6} {'brgemm':>7} {'cvt+pk+rd':>9} {'rawKVrd':>8} {'rdGB/s':>7} "
          f"{'bf16kvΔ':>8} {'T_nopack':>8} {'packcall':>8}")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        kvb = kv.to(torch.bfloat16)
        for v in ("INDEXER_SKIP_EPILOGUE", "INDEXER_SKIP_GEMM", "INDEXER_NO_PACK"):
            os.environ.pop(v, None)
        t_full = tstream(lambda: il.indexer_logits(q, kv, w))
        t_bf16 = tstream(lambda: il.indexer_logits(q, kvb, w))
        os.environ["INDEXER_SKIP_EPILOGUE"] = "1"
        t_noepi = tstream(lambda: il.indexer_logits(q, kv, w))
        os.environ.pop("INDEXER_SKIP_EPILOGUE", None)
        os.environ["INDEXER_SKIP_GEMM"] = "1"
        t_nogemm = tstream(lambda: il.indexer_logits(q, kv, w))
        os.environ.pop("INDEXER_SKIP_GEMM", None)
        os.environ["INDEXER_NO_PACK"] = "1"
        t_nopack = tstream(lambda: il.indexer_logits(q, kv, w))             # force unpacked brgemm (no cpublas::pack)
        os.environ.pop("INDEXER_NO_PACK", None)
        os.environ["INDEXER_NO_PACK"] = "1"; os.environ["INDEXER_SKIP_GEMM"] = "1"
        t_ng_np = tstream(lambda: il.indexer_logits(q, kv, w))              # no cpublas::pack AND no brgemm -> isolates pack-call cost
        os.environ.pop("INDEXER_NO_PACK", None); os.environ.pop("INDEXER_SKIP_GEMM", None)
        t_raw = tstream(lambda: kv.sum())                                  # fp32 KV read alone
        epi = t_full - t_noepi
        gemm = t_noepi - t_nogemm
        packcall = t_nogemm - t_ng_np                                      # cpublas::pack cost (both skip-gemm)
        rd_gbs = (M * S * D * 4) / (t_raw * 1e-6) / 1e9
        print(f"{M:>4} {t_full:>8.1f} {epi:>6.1f} {gemm:>7.1f} {t_nogemm:>9.1f} {t_raw:>8.1f} {rd_gbs:>7.0f} "
              f"{t_full - t_bf16:>8.1f} {t_nopack:>8.1f} {packcall:>8.1f}")
    print("\n# WARM vs STREAMING (warm = buffers hot, no flush -> excludes cold alloc/first-touch):")
    print(f"{'M':>4} {'warm_us':>8} {'stream_us':>10} {'stream/warm':>11}")
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        warm = twarm(lambda: il.indexer_logits(q, kv, w))
        strm = tstream(lambda: il.indexer_logits(q, kv, w))
        print(f"{M:>4} {warm:>8.1f} {strm:>10.1f} {strm/warm:>10.2f}x")
    # breakdown (convert/pack/compute) for M=32 via the kernel's own stderr timers
    print("\n# INDEXER_BREAKDOWN (fp32 KV), one call each M:")
    os.environ["INDEXER_BREAKDOWN"] = "1"
    for M in (1, 32, 64):
        q = torch.randn(M, H, D); kv = torch.randn(M, S, D); w = torch.randn(M, H)
        il.indexer_logits(q, kv, w)
    os.environ.pop("INDEXER_BREAKDOWN", None)


if __name__ == "__main__":
    main()
