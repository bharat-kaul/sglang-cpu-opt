#!/usr/bin/env python3
"""Level-2 kernel ATTRIBUTION (convert ASSERTED ceilings -> MEASURED, per the gap-ledger rules).

For each of the 4 remaining kernels, decompose the streaming latency into its components so the regime
claim ("compute / selection / recurrence / cache-resident") is MEASURED, not asserted, and the absolute
OPERATIVE FLOOR is pinned. Mirrors the indexer attribution method (job 385397).

  KERNEL=sparse    : score-GEMM / softmax / value-GEMM / dispatch   (env SPARSE_SKIP_SOFTMAX / SKIP_VALUE_GEMM)
  KERNEL=topk      : read / nth_element selection / parallel-dispatch (env TOPK_SKIP_SELECT / TOPK_SERIAL)
  KERNEL=sinkhorn  : recurrence-per-iter / fixed (sigmoid+dispatch)  (iters sweep; iters is a kernel arg)
  KERNEL=combine   : read-BW / residency (warm vs cold) / dispatch   (warm-vs-stream + raw-read proxy)

Run on EMR (portable). Single process; tstream = median of 25 LLC-flushed calls. Writes results/<kernel>_attribution.json.
"""
import json
import os
import statistics as st
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
BW = 358.4e9           # nominal DRAM BW (EMR), for the read-BW reference
FP32_TF = 7.78e12      # nominal fp32 AVX-512 peak
MS = [1, 2, 4, 8, 16, 32, 64]
torch.manual_seed(0)
from torch.utils.cpp_extension import load

_FLUSH = None
_SINK = 0.0


def _flush():
    global _FLUSH, _SINK
    if _FLUSH is None:
        _FLUSH = torch.empty(192 * 1024 * 1024, dtype=torch.float32)
    _FLUSH.uniform_(-1.0, 1.0)
    _SINK += float(_FLUSH[::4096].sum())


def _call(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


def tstream(fn, it=25):
    fn()
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6   # us


def twarm(fn, it=40):
    fn()
    return st.median([_call(fn) for _ in range(it)]) * 1e6                   # us (buffers hot, no flush)


def _setenv(**kv):
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def attr_sparse(mod):
    H, D, K = 64, 512, 512
    rows = []
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, K, D); sink = torch.randn(H); sc = D ** -0.5
        qc, kc = q.contiguous(), kv.contiguous()
        for v in ("SPARSE_SKIP_SOFTMAX", "SPARSE_SKIP_VALUE_GEMM"):
            os.environ.pop(v, None)
        t_full = tstream(lambda: mod.sparse_attend_bestof_hx(q, kv, sink, sc))
        t_score = tstream(lambda: torch.bmm(qc, kc.transpose(1, 2)))          # donor score GEMM proxy
        scores = torch.bmm(qc, kc.transpose(1, 2))
        t_value = tstream(lambda: torch.bmm(scores, kc))                      # donor value GEMM proxy
        t_rawrd = tstream(lambda: kv.sum())
        row = {"M": M, "t_full": t_full, "t_score_proxy": t_score, "t_value_proxy": t_value, "t_rawread": t_rawrd}
        if M >= 2:                                                            # switches bite only on the fp32bmm_hx path
            _setenv(SPARSE_SKIP_SOFTMAX=1)
            t_nosm = tstream(lambda: mod.sparse_attend_bestof_hx(q, kv, sink, sc))
            _setenv(SPARSE_SKIP_SOFTMAX=None, SPARSE_SKIP_VALUE_GEMM=1)
            t_novg = tstream(lambda: mod.sparse_attend_bestof_hx(q, kv, sink, sc))
            _setenv(SPARSE_SKIP_VALUE_GEMM=None)
            softmax = t_full - t_nosm
            value_gemm = t_full - t_novg
            score_plus_disp = t_novg - softmax                               # score-GEMM + dispatch (value removed, softmax removed)
            row.update({"softmax": softmax, "value_gemm": value_gemm, "score_gemm+disp": score_plus_disp,
                        "flops_floor_us": (4.0 * M * H * K * D) / FP32_TF * 1e6})
        rows.append(row)
    return rows


def attr_topk(mod):
    S, k = 1024, 512
    rows = []
    for M in MS:
        logits = torch.randn(M, S)
        for v in ("TOPK_SERIAL", "TOPK_SKIP_SELECT"):
            os.environ.pop(v, None)
        t_full = tstream(lambda: mod.indexer_topk(logits, k))
        _setenv(TOPK_SKIP_SELECT=1)
        t_nosel = tstream(lambda: mod.indexer_topk(logits, k))               # read + dispatch, no nth_element
        _setenv(TOPK_SKIP_SELECT=None, TOPK_SERIAL=1)
        t_serial = tstream(lambda: mod.indexer_topk(logits, k))              # single-thread selection (no team)
        _setenv(TOPK_SERIAL=None)
        t_rawrd = tstream(lambda: logits.sum())
        rows.append({"M": M, "t_full": t_full, "t_noselect": t_nosel, "selection": t_full - t_nosel,
                     "t_serial": t_serial, "dispatch_gain": t_serial - t_full, "t_rawread": t_rawrd,
                     "read_GBs": (M * S * 4) / (t_rawrd * 1e-6) / 1e9})
    return rows


def attr_sinkhorn(mod):
    hc, eps = 4, 1e-6
    base = torch.randn((2 + hc) * hc); scale = torch.rand(3) + 0.5
    rows = []
    for M in MS:
        mixes = torch.randn(1, M, (2 + hc) * hc)
        iters_sweep = {}
        for it in (1, 2, 5, 10, 20):
            iters_sweep[it] = tstream(lambda: mod.mhc_sinkhorn(mixes, scale, base, hc, it, eps))
        per_iter = (iters_sweep[20] - iters_sweep[1]) / 19.0                 # recurrence cost per iter
        fixed = iters_sweep[1] - per_iter                                    # sigmoid pre/post + dispatch + alloc
        rows.append({"M": M, "iters_sweep_us": iters_sweep, "t_full": iters_sweep[20],
                     "per_iter_us": per_iter, "recurrence_20_us": per_iter * 20, "fixed_us": fixed,
                     "recurrence_frac": (per_iter * 20) / iters_sweep[20] if iters_sweep[20] else 0})
    return rows


def attr_combine(mod):
    hc, H = 4, 4096
    rows = []
    for M in MS:
        x = torch.randn(M, hc * H); pre = torch.rand(M, hc)
        t_stream = tstream(lambda: mod.mhc_combine(x, pre, hc))
        t_warm = twarm(lambda: mod.mhc_combine(x, pre, hc))
        t_rawrd = tstream(lambda: x.sum())
        bytes_rd = M * hc * H * 4
        rows.append({"M": M, "t_stream": t_stream, "t_warm": t_warm, "residency_ratio": t_stream / t_warm,
                     "t_rawread": t_rawrd, "read_GBs_stream": bytes_rd / (t_rawrd * 1e-6) / 1e9,
                     "op_read_GBs": bytes_rd / (t_stream * 1e-6) / 1e9,
                     "flops_floor_us": (2.0 * M * hc * H) / FP32_TF * 1e6})
    return rows


KERNELS = {
    "sparse": ("sparse_attend.cpp", "op_sp_attr", attr_sparse),
    "topk": ("indexer_topk.cpp", "op_tk_attr", attr_topk),
    "sinkhorn": ("sinkhorn.cpp", "op_sk_attr", attr_sinkhorn),
    "combine": ("combine.cpp", "op_cb_attr", attr_combine),
}


def main():
    kernel = os.environ.get("KERNEL", "sparse")
    src, name, fn = KERNELS[kernel]
    mod = load(name=name, sources=[os.path.join(KD, src)], extra_cflags=CF, verbose=False)
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# KERNEL={kernel} HEAD={sha} proc={os.environ.get('PROC', '0')} "
          f"threads={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")
    rows = fn(mod)
    for r in rows:
        print("  " + "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    out = {"kernel": kernel, "head": sha, "node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
           "threads": torch.get_num_threads(), "omp": os.environ.get("OMP_NUM_THREADS"),
           "nominal_BW_GBs": BW / 1e9, "fp32_TF": FP32_TF / 1e12, "MS": MS, "rows": rows}
    out_path = os.environ.get("ATTR_OUT", os.path.join(HERE, "results", f"{kernel}_attribution.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
