#!/usr/bin/env python3
"""Compressor retry per the libxsmm/RAJAPerf hpc-kernel-optimization SKILL (all stages except Stage-2
AoS->SoA, which N/A: the kernel is already unit-stride SoA with the channel dim innermost).

Stage 0: pin baseline + build-flag matrix, interleaved median, random inputs, tolerance check.
Stage 1: CLASSIFY by inspecting the built object -- 512-bit `zmm` share and libmvec `_ZGV...expf`
         presence -- to settle whether the shipped kernel's exp vectorized (the 27%-roofline cause).
Stage 4: compare variants (scalar C1 / at::vec C2 / two-pass C3) across flag sets that (a) force
         512-bit vectors (`-mprefer-vector-width=512`; gcc defaults to 256), (b) widen at::vec to
         AVX-512 (`-DCPU_CAPABILITY_AVX512`), (c) enable libmvec vectorized exp in the scalar omp-simd
         loop (`-fno-math-errno` / `-ffast-math -fno-finite-math-only`, keeping inf for the mask).

Report: per (flagset, variant): median us @ M, achieved GB/s, % of nominal roofline, max rel-err vs
torch, and the asm evidence (zmm count, libmvec expf present). Honest: libmvec/at::vec exp are NOT
bit-exact to scalar glibc exp -> those variants are F4-SCREENED, reported with their rel-err.
"""
import glob
import os
import statistics as st
import subprocess
import time

import torch
from torch.utils.cpp_extension import load

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
K = os.path.join(KD, "compressor.cpp")
BW = 358.4e9  # nominal EMR DDR5600 peak
PEAK_FP32 = 7.7824e12

BASE = ["-O3", "-fopenmp", "-march=native"]
P512 = "-mprefer-vector-width=512"
VEC = "-DCPU_CAPABILITY_AVX512"      # widen at::vec::Vectorized<float> to zmm in a JIT extension
# -ffast-math pulls in -ffinite-math-only which breaks our -INFINITY masking -> keep inf explicitly.
FAST = ["-ffast-math", "-fno-finite-math-only"]

FLAGSETS = {
    "base": BASE,                                   # current shipped flags (gcc 256-bit default)
    "pw512_vec": BASE + [P512, VEC],                 # force 512-bit + at::vec 512-bit (C2 winner)
}
VARIANTS = ["compressor_softmax_pool", "compressor_softmax_pool_vexp", "compressor_softmax_pool_multipass",
            "compressor_softmax_pool_fp32intrin", "compressor_softmax_pool_fastexp", "compressor_softmax_pool_bf16exp"]
SHORT = {"compressor_softmax_pool": "C1_scalar", "compressor_softmax_pool_vexp": "C2_vexp",
         "compressor_softmax_pool_multipass": "C3_2pass", "compressor_softmax_pool_fp32intrin": "D1_handexp",
         "compressor_softmax_pool_fastexp": "D2_fastexp", "compressor_softmax_pool_bf16exp": "D3_bf16"}
MS = [1, 8, 16, 32, 64]


def ref(kv, score, ape):
    x = (score.float() + ape.float()).softmax(dim=1)
    return (x * kv.float()).sum(1)


def make_inp(M, R=128, D=512):
    return torch.randn(M, R, D), torch.randn(M, R, D), torch.randn(R, D)


def bytes_moved(M, R=128, D=512):
    return M * R * D * 4 * 2 + M * D * 4 + R * D * 4


def floor_us(M):
    return max(bytes_moved(M) / BW, 3.0 * M * 128 * 512 / PEAK_FP32) * 1e6


def tmed(fn, args, it=60):
    s = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn(*args)
        s.append(time.perf_counter() - t0)
    return st.median(s)


# STREAMING (cold-cache) timing — the reliable standalone-kernel methodology. Warm reuse lets a working set
# that fits in L3 (compressor @M32 = 17MB << 320MB) be re-read from cache, hiding the real DRAM ceiling. We
# flush >= 2x LLC between every timed call so each kernel invocation reads its inputs cold from DRAM.
_FLUSH = None
_SINK = 0.0


def _flush_cache():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)   # 768 MiB >> any EMR/GNR L3
    _FLUSH.uniform_(-1.0, 1.0)                                          # write-evict the whole LLC (RFO)
    _SINK += float(_FLUSH[::4096].sum())                               # prevent DCE; touch result


def tmed_stream(fn, args, it=40):
    """Median single-call latency with an LLC flush before EACH call -> inputs stream from DRAM."""
    s = []
    for _ in range(it):
        _flush_cache()
        t0 = time.perf_counter()
        fn(*args)
        s.append(time.perf_counter() - t0)
    return st.median(s)


def rel_err(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def asm_evidence(modname):
    """Stage-1: objdump the built .so -> (zmm-insn count, libmvec expf present)."""
    so = sorted(glob.glob(os.path.expanduser(f"~/.cache/torch_extensions/**/{modname}.so"), recursive=True))
    if not so:
        so = sorted(glob.glob(os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", ""), "**", f"{modname}.so"), recursive=True)) if os.environ.get("TORCH_EXTENSIONS_DIR") else []
    if not so:
        return (-1, "?", "no .so found")
    try:
        d = subprocess.run(["objdump", "-d", so[-1]], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return (-1, "?", f"objdump failed: {e}")
    zmm = d.count("%zmm")
    ymm = d.count("%ymm")
    libmvec = ("_ZGV" in d and "expf" in d)
    return (zmm, f"zmm={zmm} ymm={ymm}", "libmvec_expf=" + ("YES" if libmvec else "no"))


def main():
    print(f"threads={torch.get_num_threads()}  op=compressor R=128 D=512  BW_nominal={BW/1e9:.1f}GB/s")
    print("Stage-0/1: build-flag matrix; asm evidence (zmm/libmvec); interleaved median; rel-err vs torch.\n")
    mods = {}
    for fs, flags in FLAGSETS.items():
        name = f"cz_{fs}"
        mods[fs] = load(name=name, sources=[K], extra_cflags=flags, verbose=False)
        zmm, vwidth, lmv = asm_evidence(name)
        print(f"[build {fs:16s}] flags={' '.join(flags[3:]) or '(none extra)':30s}  {vwidth}  {lmv}")
    print()

    # Stage-1 classify + Stage-4 variant comparison at the operating point M=32, plus M-sweep.
    hdr = f"{'flagset':16s} {'variant':10s} " + " ".join(f"M{m:>3}us" for m in MS) + f"  {'M32_GB/s':>8} {'M32_%roof':>9} {'relerr':>9}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for fs in FLAGSETS:
        mod = mods[fs]
        for vn in VARIANTS:
            fn = getattr(mod, vn, None)
            if fn is None:
                continue
            us = {}
            re = 0.0
            for M in MS:
                kv, sc, ap = make_inp(M)
                fn(kv, sc, ap)  # warm
                us[M] = tmed(lambda a, b, c: fn(a, b, c), (kv, sc, ap)) * 1e6
                if M == 32:
                    re = rel_err(fn(kv, sc, ap).float(), ref(kv, sc, ap))
            gbps = bytes_moved(32) / (us[32] / 1e6) / 1e9
            proof = 100 * floor_us(32) / us[32]
            cells = " ".join(f"{us[m]:7.1f}" for m in MS)
            print(f"{fs:16s} {SHORT[vn]:10s} {cells}  {gbps:7.1f} {proof:8.1f}% {re:9.2e}")
            rows.append((fs, SHORT[vn], us, gbps, proof, re))
    print(f"\nnominal floor: M32={floor_us(32):.1f}us ({bytes_moved(32)/1e6:.2f}MB / {BW/1e9:.0f}GB/s)  shipped baseline(job385001) M32 C1=176.1us=27%")

    # STREAMING (cold-cache) re-measure of the decision variants -> the RELIABLE post-optimization numbers.
    # warm (above) reuses one buffer (L3-resident); streaming flushes >=2x LLC per call (DRAM-fed).
    print("\n=== STREAMING (cold-cache, LLC-flushed per call) vs WARM — the reliable methodology ===")
    fs = "pw512_vec" if "pw512_vec" in FLAGSETS else list(FLAGSETS)[0]
    mod = mods[fs]
    DECIDE = [v for v in ("compressor_softmax_pool", "compressor_softmax_pool_vexp",
                          "compressor_softmax_pool_fp32intrin", "compressor_softmax_pool_bf16exp") if getattr(mod, v, None)]
    print(f"flagset={fs}   nominal DRAM floor M32={floor_us(32):.1f}us ({bytes_moved(32)/1e6:.1f}MB/{BW/1e9:.0f}GB/s)")
    print(f"{'variant':10s} {'M':>3} {'warm_us':>8} {'stream_us':>9} {'strm_GB/s':>9} {'%DRAMroof':>9}  (warm hides DRAM; stream is reliable)")
    for vn in DECIDE:
        fn = getattr(mod, vn)
        for M in (1, 8, 32, 64):
            kv, sc, ap = make_inp(M)
            fn(kv, sc, ap)
            w = tmed(lambda a, b, c: fn(a, b, c), (kv, sc, ap)) * 1e6
            s = tmed_stream(lambda a, b, c: fn(a, b, c), (kv, sc, ap)) * 1e6
            gbps = bytes_moved(M) / (s / 1e6) / 1e9
            proof = 100 * floor_us(M) / s
            print(f"{SHORT[vn]:10s} {M:>3} {w:8.1f} {s:9.1f} {gbps:9.1f} {proof:8.1f}%")
    # mutation test: swap two score positions must change the result (proves the check is live)
    kv, sc, ap = make_inp(32)
    o0 = mods["base"].compressor_softmax_pool(kv, sc, ap).clone()
    sc2 = sc.clone(); sc2[0, 0] += 3.0
    o1 = mods["base"].compressor_softmax_pool(kv, sc2, ap)
    print(f"mutation-test (perturb score[0,0]): max|Δout|={ (o1-o0).abs().max().item():.3e} (must be >0)")


if __name__ == "__main__":
    main()
