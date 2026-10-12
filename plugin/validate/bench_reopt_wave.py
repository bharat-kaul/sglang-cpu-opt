#!/usr/bin/env python3
"""Parallel re-optimization WAVE harness (one kernel per invocation via KERNEL env), applying the compressor
learnings + the libxsmm/RAJAPerf hpc-kernel-optimization SKILL (Stage 2 = AoS->SoA N/A; our kernels are
already unit-stride SoA):
  Stage-0/1: build-flag matrix {base, pw512_vec}; objdump the built .so for %zmm + libmvec (classify);
  RELIABLE timing = STREAMING (LLC-flushed per call, not warm reuse) for torch baseline AND each C/C++ variant
  across the M-sweep; accuracy vs the torch oracle where continuous.
Emits the per-kernel JOURNEY: torch baseline -> shipped C/C++ -> re-optimized (flags / hand-exp), warm+stream.
KERNEL in {indexer_logits, indexer_topk, sparse_bestof, sinkhorn, combine}.
"""
import os
import statistics as st
import subprocess
import time
import glob

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "intel_cpu_models"))
from torch.utils.cpp_extension import load

BW = 358.4e9
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
BASE = ["-O3", "-fopenmp", "-march=native"]
FLAGSETS = {"base": BASE, "pw512_vec": BASE + ["-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]}
MS = [1, 8, 16, 32, 64]
KERNEL = os.environ.get("KERNEL", "combine")


# ---- torch refs (the non-C/C++ baseline each kernel replaces) ----
def _idx_ref(q, kv, w):
    s = torch.einsum("nhd,nsd->nhs", q.bfloat16(), kv.bfloat16())
    return (s.relu_() * w.bfloat16().unsqueeze(-1)).sum(1)


def _combine_ref(x_flat, pre, hc):
    m, h = x_flat.shape[0], x_flat.shape[1] // hc
    return torch.einsum("mk,mkh->mh", pre.float(), x_flat.reshape(m, hc, h).float())


def _sparse_ref(q, kv, sink, scale):
    from sparse_ref import ora_sparse_fp32
    return ora_sparse_fp32(q, kv, sink, scale)


# ---- per-kernel config: (source, builder(M), torch_ref, {variant_label: cpp_fn_name}, bytes(M), kind) ----
def _cfg(mod):
    H, S, D, Dl, K, R = 64, 1024, 128, 512, 512, 128
    if KERNEL == "indexer_logits":
        return (lambda M: (torch.randn(M, H, D), torch.randn(M, S, D), torch.randn(M, H)),
                lambda a: _idx_ref(*a), {"indexer_logits": "indexer_logits"},
                lambda M: M*H*D*4 + M*S*D*4 + M*H*4 + M*S*4, "cont")
    if KERNEL == "indexer_topk":
        return (lambda M: (torch.randn(M, S), 512),
                lambda a: torch.topk(a[0], a[1], dim=1, sorted=False).indices, {"indexer_topk": "indexer_topk"},
                lambda M: M*S*4 + M*512*8, "sel")
    if KERNEL == "sparse_bestof":
        return (lambda M: (torch.randn(M, H, Dl), torch.randn(M, Dl, Dl), torch.randn(H), Dl ** -0.5),
                lambda a: _sparse_ref(*a),
                {"bestof": "sparse_attend_bestof", "bestof_hx": "sparse_attend_bestof_hx"},
                lambda M: M*Dl*Dl*4 + 2*M*H*Dl*4, "cont")
    if KERNEL == "sinkhorn":
        from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
        return (lambda M: (torch.randn(1, M, 24), torch.rand(3) + 0.5, torch.randn(24), 4, 20, 1e-6),
                lambda a: _hc_split_sinkhorn_torch(*a), {"sinkhorn": "mhc_sinkhorn"},
                lambda M: M*24*4*2, "tuple3")
    if KERNEL == "combine":
        return (lambda M: (torch.randn(M, 4 * 4096), torch.rand(M, 4), 4),
                lambda a: _combine_ref(*a), {"combine": "mhc_combine"},
                lambda M: M*4*4096*4 + M*4*4 + M*4096*4, "cont")
    raise SystemExit(f"unknown KERNEL={KERNEL}")


_SRC = {"indexer_logits": "indexer_logits.cpp", "indexer_topk": "indexer_topk.cpp",
        "sparse_bestof": "sparse_attend.cpp", "sinkhorn": "sinkhorn.cpp", "combine": "combine.cpp"}

_FLUSH = None
_SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)   # 768 MiB >> any EMR/GNR L3
    _FLUSH.uniform_(-1.0, 1.0)
    _SINK += float(_FLUSH[::4096].sum())


def twarm(fn, args, it=40):
    fn(*args)
    s = [(_t(fn, args)) for _ in range(it)]
    return st.median(s)


def tstream(fn, args, it=25):
    s = []
    for _ in range(it):
        _flush()
        s.append(_t(fn, args))
    return st.median(s)


def _t(fn, args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def objdump_zmm(modname):
    so = sorted(glob.glob(os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", ""), "**", f"{modname}.so"), recursive=True))
    if not so:
        return "no .so"
    d = subprocess.run(["objdump", "-d", so[-1]], capture_output=True, text=True, timeout=180).stdout
    return f"zmm={d.count('%zmm')} ymm={d.count('%ymm')} libmvec_expf={'YES' if ('_ZGV' in d and 'expf' in d) else 'no'}"


def relerr(a, b):
    if isinstance(a, (tuple, list)):
        return max(relerr(x, y) for x, y in zip(a, b))
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def main():
    print(f"=== REOPT WAVE: {KERNEL}  threads={torch.get_num_threads()}  (streaming=reliable) ===")
    build, ref_fn, variants, bytes_fn, kind = None, None, None, None, None
    mods = {}
    for fs, flags in FLAGSETS.items():
        name = f"rw_{KERNEL}_{fs}"
        mods[fs] = load(name=name, sources=[os.path.join(KD, _SRC[KERNEL])], extra_cflags=flags, verbose=False)
        print(f"[build {fs:10s}] {' '.join(flags[3:]) or '(none)':44s} {objdump_zmm(name)}")
    build, ref_fn, variants, bytes_fn, kind = _cfg(mods["base"])

    # torch baseline journey (warm + streaming) + each cpp variant under each flagset
    print(f"\n{'impl':28s} {'flags':10s} " + " ".join(f"wM{m}" for m in MS) + "   " + " ".join(f"sM{m}" for m in MS) + "   relerr")
    # torch
    tw, ts = {}, {}
    for M in MS:
        a = build(M)
        tw[M] = twarm(lambda *x: ref_fn(x), a) * 1e6
        ts[M] = tstream(lambda *x: ref_fn(x), a) * 1e6
    print(f"{'torch_baseline':28s} {'-':10s} " + " ".join(f"{tw[m]:5.0f}" for m in MS) + "   " + " ".join(f"{ts[m]:5.0f}" for m in MS))
    for fs in FLAGSETS:
        mod = mods[fs]
        for lbl, fname in variants.items():
            fn = getattr(mod, fname, None)
            if fn is None:
                continue
            cw, cs, re = {}, {}, 0.0
            for M in MS:
                a = build(M)
                cw[M] = twarm(lambda *x: fn(*x), a) * 1e6
                cs[M] = tstream(lambda *x: fn(*x), a) * 1e6
                if M == 32 and kind in ("cont", "tuple3"):
                    try:
                        re = relerr(fn(*a), ref_fn(a))
                    except Exception as e:
                        re = -1.0
            sM32 = cs[32]
            gbps = bytes_fn(32) / (sM32 / 1e6) / 1e9
            print(f"{lbl:28s} {fs:10s} " + " ".join(f"{cw[m]:5.0f}" for m in MS) + "   " + " ".join(f"{cs[m]:5.0f}" for m in MS) + f"   {re:.2e}")
            # journey line at M=32 (streaming, the reliable number)
            spd = ts[32] / sM32 if sM32 else 0
            print(f"    -> M32 STREAM: torch {ts[32]:.0f}us -> {lbl} {sM32:.0f}us = {spd:.2f}x   ({gbps:.0f} GB/s)")
    print(f"[done wave {KERNEL}]")


if __name__ == "__main__":
    main()
