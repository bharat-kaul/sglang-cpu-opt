#!/usr/bin/env python3
"""BF16-AMX sparse-attend standalone optimization (user-approved; FP8 = later stretch).

Answers three questions in ONE run, so a single EMR job dispositions the whole BF16 lever:
  (A) INSTRUMENT CHECK  — does bf16 `at::bmm` dispatch to an AMX kernel and actually beat fp32 `at::bmm`
      at the REAL sparse-attend GEMM shapes? If not, the premise is dead (compressor 256-bit lesson:
      validate the instrument before optimizing). Measured streaming, both the score and value GEMMs.
  (B) VARIANT PERF      — torch ref vs the shipped fp32 donor (sparse_attend_fp32bmm_hx) vs the NEW
      sparse_attend_bf16bmm_hx (big bf16 bmms + FUSED hand-exp fp32 softmax), streaming, M-sweep.
      This is the config the prior bf16 losers missed (amx = materialized softmax tax; blockbf16 =
      N<=64 micro-GEMM/pack tax).
  (C) TOLERANCE         — cos / max-rel / mae of fp32bmm_hx (should be ~bit-exact) and bf16bmm_hx
      (bf16 operands) vs the FP32 sparse oracle, on a NORMAL and a SHARPER (randn*4) distribution
      (peaked softmax stresses bf16 differently). Standalone tolerance is a PROXY; the authoritative
      gate is full-model accuracy. Run >=3 processes (sbatch); take cross-proc medians.
"""
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
import sys
sys.path.insert(0, HERE)
from bench_replicated import _sparse_ref                       # fp32 torch oracle
from torch.utils.cpp_extension import load

CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]  # STANDARD (512-bit)
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
BW, AMX, FP32 = 358.4e9, 124.5184e12, 7.7824e12
MS = [1, 8, 16, 32, 64]
torch.manual_seed(0)

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


def tstream(fn, a, it=25):
    fn(*a)
    return st.median([(_flush(), _call(fn, a))[1] for _ in range(it)])


def _tol(cand, ref):
    c = cand.flatten().double(); r = ref.flatten().double()
    cos = float(torch.dot(c, r) / (c.norm() * r.norm() + 1e-30))
    mae = float((c - r).abs().mean())
    rel = float((c - r).abs().max() / (r.abs().max() + 1e-30))
    return cos, mae, rel


def main():
    sp = load(name="spbf16", sources=[os.path.join(KD, "sparse_attend.cpp")], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    proc = os.environ.get("PROC", "0")
    print(f"# identity: HEAD={sha} proc={proc} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} bind={os.environ.get('OMP_PROC_BIND')} "
          f"node={os.uname().nodename} torch={torch.__version__}")

    # ---- (A) INSTRUMENT CHECK: fp32 vs bf16 at::bmm at the real shapes (score + value GEMMs) ----
    print("\n[A] at::bmm fp32-vs-bf16 at real sparse shapes (streaming us; spd = fp32/bf16, >1 => bf16 wins)")
    print(f"{'gemm':>8} {'M':>4} {'fp32_us':>9} {'bf16_us':>9} {'spd':>6}")
    for M in MS:
        q = torch.randn(M, 64, 512); kvt = torch.randn(M, 512, 512)           # score: [H,D]x[D,K]
        qb, kbt = q.to(torch.bfloat16), kvt.to(torch.bfloat16)
        f = tstream(lambda a, b: at_bmm(a, b), (q, kvt.transpose(1, 2))) * 1e6
        g = tstream(lambda a, b: at_bmm(a, b), (qb, kbt.transpose(1, 2))) * 1e6
        print(f"{'score':>8} {M:>4} {f:>9.1f} {g:>9.1f} {f/g:>6.2f}")
        w = torch.randn(M, 64, 512); kv = torch.randn(M, 512, 512)            # value: [H,K]x[K,D]
        wb, kvb = w.to(torch.bfloat16), kv.to(torch.bfloat16)
        f = tstream(lambda a, b: at_bmm(a, b), (w, kv)) * 1e6
        g = tstream(lambda a, b: at_bmm(a, b), (wb, kvb)) * 1e6
        print(f"{'value':>8} {M:>4} {f:>9.1f} {g:>9.1f} {f/g:>6.2f}")

    # ---- (B) VARIANT PERF: torch vs fp32bmm_hx (shipped) vs bf16bmm_hx (new), streaming ----
    print("\n[B] sparse-attend variant perf (streaming us; spd vs torch; spd_vs_fp32 = fp32bmm_hx/bf16bmm_hx)")
    print(f"{'M':>4} {'torch':>8} {'fp32hx':>8} {'bf16hx':>8} {'spd_tor':>8} {'spd_fp32':>9} {'floor_fp32':>10} {'floor_bf16':>10}")
    for M in MS:
        q = torch.randn(M, 64, 512); kv = torch.randn(M, 512, 512); sink = torch.randn(64); sc = 512 ** -0.5
        a = (q, kv, sink, sc)
        t_tor = tstream(lambda *_: _sparse_ref(*a), ()) * 1e6
        t_f32 = tstream(lambda *_: sp.sparse_attend_fp32bmm_hx(*a), ()) * 1e6
        t_bf16 = tstream(lambda *_: sp.sparse_attend_bf16bmm_hx(*a), ()) * 1e6
        flop = 4.0 * M * 64 * 512 * 512
        fl_f32 = max(flop / FP32, (M * 512 * 512 * 4 + 2 * M * 64 * 512 * 4) / BW) * 1e6
        fl_bf16 = max(flop / AMX, (M * 512 * 512 * 2 + 2 * M * 64 * 512 * 2) / BW) * 1e6
        print(f"{M:>4} {t_tor:>8.1f} {t_f32:>8.1f} {t_bf16:>8.1f} {t_tor/t_bf16:>8.2f} {t_f32/t_bf16:>9.2f} "
              f"{fl_f32:>10.1f} {fl_bf16:>10.1f}")

    # ---- (C) TOLERANCE vs fp32 oracle on NORMAL and SHARPER distributions ----
    print("\n[C] tolerance vs fp32 sparse oracle (cos / mae / max-rel); dist=normal(randn) and sharp(randn*4)")
    print(f"{'dist':>7} {'M':>4} {'variant':>8} {'cos':>12} {'mae':>10} {'maxrel':>10}")
    for dname, mul in (("normal", 1.0), ("sharp", 4.0)):
        for M in (1, 32, 64):
            q = torch.randn(M, 64, 512) * mul; kv = torch.randn(M, 512, 512) * mul
            sink = torch.randn(64); sc = 512 ** -0.5
            ref = _sparse_ref(q, kv, sink, sc).to(torch.float32)
            for vn, fn in (("fp32hx", sp.sparse_attend_fp32bmm_hx), ("bf16hx", sp.sparse_attend_bf16bmm_hx)):
                cos, mae, rel = _tol(fn(q, kv, sink, sc), ref)
                print(f"{dname:>7} {M:>4} {vn:>8} {cos:>12.8f} {mae:>10.3e} {rel:>10.3e}")


def at_bmm(a, b):
    return torch.bmm(a, b)


if __name__ == "__main__":
    main()
