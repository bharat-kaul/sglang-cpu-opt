#!/usr/bin/env python3
"""Performance-vs-roofline M-sweep for ALL shipped authored kernels (current build, incl. the I1/C1 wins).

Measures median latency per (kernel, M) and joins it to the per-M roofline FLOOR derived from each kernel's
actual byte/FLOP contract and the EMR machine peak (BW 358.4 GB/s, AMX bf16 124.5184 TF, AVX-512 FP32 7.7824
TF). achieved% = floor_us / measured_us is a fraction of the NOMINAL REFERENCE, not of an absolute hardware
ceiling. Byte/FLOP contracts are the same ones asserted in dsv4_roofline_vs_measured.py, evaluated across
the whole M-sweep (simplified: the compressor FLOP count omits its exp/recurrence cost). The machine peak is
the NOMINAL base-clock (1.9 GHz) reference (NOT a measured achievable BW/clock); the regime tag is the larger
theoretical term (a diagnostic, NOT a measured bottleneck; cache-resident rows are not DRAM-saturated). The
residual gap is surfaced, not assumed recoverable. Timing only; correctness is the F4 gate's job. Run as >=3
processes (sbatch) and aggregate with aggregate_roofline.py."""
import json
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_CF = ["-O3", "-fopenmp", "-march=native"]
_KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
torch.manual_seed(0)
from torch.utils.cpp_extension import load

BW = 358.4e9
AMX = 124.5184e12
FP32 = 7.7824e12
B4 = 4.0; B2 = 2.0; I8 = 8.0


def _mod(n, cpp):
    return load(name=n, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _med(fn, args, trials=5, it=30):
    fn(args)
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(it):
            fn(args)
        ts.append((time.perf_counter() - t0) / it)
    return st.median(ts)


def _floor_us(flops, nbytes, peak):
    return max(nbytes / BW, flops / peak) * 1e6


def main():
    il = _mod("rl_il", "indexer_logits.cpp"); tk = _mod("rl_tk", "indexer_topk.cpp")
    cp = _mod("rl_cp", "compressor.cpp"); spm = _mod("rl_sp", "sparse_attend.cpp")
    sk = _mod("rl_sk", "sinkhorn.cpp"); cb = _mod("rl_cb", "combine.cpp")

    # (label, build(M), fn, flops(M), bytes(M), compute_peak)
    H, S, D = 64, 1024, 128
    specs = [
        ("indexer_logits/tiled", lambda M: (torch.randn(M, H, D), torch.randn(M, S, D), torch.randn(M, H)),
         lambda a: il.indexer_logits(*a), lambda M: 2 * M * H * S * D + 3 * M * H * S,
         lambda M: M * H * D * B4 + M * S * D * B4 + M * H * B4 + M * S * B4, AMX),
        ("indexer_topk", lambda M: (torch.randn(M, S), 512),
         lambda a: tk.indexer_topk(*a), lambda M: 0.0,
         lambda M: M * S * B4 + M * 512 * I8, FP32),
        ("compressor/r128d512", lambda M: (torch.randn(M, 128, 512), torch.randn(M, 128, 512), torch.randn(128, 512)),
         lambda a: cp.compressor_softmax_pool(*a), lambda M: 3.0 * M * 128 * 512,
         lambda M: M * 128 * 512 * B4 * 2 + M * 512 * B4 + 128 * 512 * B4, FP32),
        ("compressor/r8d512", lambda M: (torch.randn(M, 8, 512), torch.randn(M, 8, 512), torch.randn(8, 512)),
         lambda a: cp.compressor_softmax_pool(*a), lambda M: 3.0 * M * 8 * 512,
         lambda M: M * 8 * 512 * B4 * 2 + M * 512 * B4 + 8 * 512 * B4, FP32),
        ("compressor/r8d128", lambda M: (torch.randn(M, 8, 128), torch.randn(M, 8, 128), torch.randn(8, 128)),
         lambda a: cp.compressor_softmax_pool(*a), lambda M: 3.0 * M * 8 * 128,
         lambda M: M * 8 * 128 * B4 * 2 + M * 128 * B4 + 8 * 128 * B4, FP32),
        ("sparse/bestof", lambda M: (torch.randn(M, 64, 512), torch.randn(M, 512, 512), torch.randn(64), 512 ** -0.5),
         lambda a: spm.sparse_attend_bestof(*a), lambda M: 4.0 * M * 64 * 512 * 512,
         lambda M: M * 512 * 512 * B4 + 2 * M * 64 * 512 * B4, FP32),
        ("sinkhorn", lambda M: (torch.randn(1, M, 24), torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6),
         lambda a: sk.mhc_sinkhorn(*a), lambda M: float(M * 4 * 4 * 20 * 5),
         lambda M: M * 24 * B4 + 3 * B4 + 24 * B4 + M * 24 * B4, FP32),
        ("combine", lambda M: (torch.randn(M, 4 * 4096), torch.rand(M, 4), 4),
         lambda a: cb.mhc_combine(*a), lambda M: float(M * 4096 * 7),
         lambda M: M * 4 * 4096 * B4 + M * 4 * B4 + M * 4096 * B4, FP32),
    ]
    sha = os.popen("git rev-parse --short HEAD").read().strip()
    print(f"# roofline-vs-measured | HEAD={sha} threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"bind={os.environ.get('OMP_PROC_BIND')} | BW=358.4GB/s AMXbf16=124.52TF FP32=7.78TF (NOMINAL reference @1.9GHz base, NOT a measured ceiling)")
    print(f"{'kernel':>22} {'M':>4} {'meas_us':>9} {'floor_us':>9} {'%ofNomRef':>9} {'regime':>7}")
    out = {}
    for label, build, fn, flf, byf, peak in specs:
        out[label] = {}
        for M in (1, 8, 16, 32, 64):
            meas = _med(fn, build(M)) * 1e6
            fl = _floor_us(flf(M), byf(M), peak)
            regime = "BW" if byf(M) / BW >= flf(M) / peak else "compute"   # theoretical-larger-term diagnostic, NOT a measured bottleneck
            ach = 100.0 * fl / meas                                        # fraction of the NOMINAL reference, not an absolute ceiling
            out[label][M] = {"meas_us": round(meas, 2), "floor_us": round(fl, 4), "achieved_pct": round(ach, 2), "regime": regime}
            print(f"{label:>22} {M:>4} {meas:>9.1f} {fl:>9.4f} {ach:>8.2f}% {regime:>7}")
    if os.environ.get("ROOFLINE_JSON"):
        prov = {"head": sha, "node": os.uname().nodename, "threads": torch.get_num_threads(),
                "omp": os.environ.get("OMP_NUM_THREADS"), "bind": os.environ.get("OMP_PROC_BIND"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
                "peak_nominal": {"mem_bw_gbps": 358.4, "amx_bf16_tflops": 124.5184, "avx512_fp32_tflops": 7.7824,
                                 "note": "NOMINAL @1.9GHz base, NOT measured achievable BW/clock"}}
        json.dump({"_provenance": prov, "kernels": out}, open(os.environ["ROOFLINE_JSON"], "w"), indent=1)
    print(">>> done")


if __name__ == "__main__":
    main()
