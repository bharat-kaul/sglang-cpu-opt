#!/usr/bin/env python3
"""Level-2 RECOVERY wave (strict rule application): test the attribution-motivated knob for each kernel with a
recoverable gap, BEFORE any at-ceiling close. Measured win -> ship; measured loss -> published disposition.

  sparse   : score-only bf16 hybrid vs fp32bmm_hx (standalone + bf16-KV-provided sim) + tolerance vs fp32 oracle  [R9]
  topk     : TOPK_CHUNKS forced within-row chunking vs shipped dispatch (core-fill at M<nthr); set-match gate    [R6/R7]
  sinkhorn : SINK_SERIAL (one task) vs parallel across M (is the fixed cost dispatch we can cut at low M?)        [R6]

combine is NOT here: attribution already showed op_read ~= raw-read BW (at-ceiling confirmed, no knob).
EMR, single process, tstream = median of 25 LLC-flushed calls. Writes results/recovery_<kernel>.json.
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
    return st.median([(_flush(), _call(fn))[1] for _ in range(it)]) * 1e6


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rec_sparse():
    sp = load(name="rec_sp", sources=[os.path.join(KD, "sparse_attend.cpp")], extra_cflags=CF, verbose=False)
    H, D, K = 64, 512, 512
    rows = []
    for M in MS:
        q = torch.randn(M, H, D); kv = torch.randn(M, K, D); sink = torch.randn(H); sc = D ** -0.5
        kvb = kv.to(torch.bfloat16)
        base = sp.sparse_attend_bestof_hx(q, kv, sink, sc)                    # fp32 donor (shipped)
        hyb = sp.sparse_attend_scorehybrid(q, kv, sink, sc)                   # score-bf16 hybrid (fp32 input)
        t_base = tstream(lambda: sp.sparse_attend_bestof_hx(q, kv, sink, sc))
        t_hyb = tstream(lambda: sp.sparse_attend_scorehybrid(q, kv, sink, sc))
        t_hyb_bf16kv = tstream(lambda: sp.sparse_attend_scorehybrid(q, kvb, sink, sc))  # bf16-KV provided (integration sim)
        rows.append({"M": M, "t_fp32_donor": t_base, "t_scorehybrid": t_hyb, "t_scorehybrid_bf16kv": t_hyb_bf16kv,
                     "speedup_vs_donor": t_base / t_hyb, "speedup_bf16kv": t_base / t_hyb_bf16kv,
                     "cos_vs_fp32": _cos(hyb, base)})
    return "sparse", rows


def rec_topk():
    tk = load(name="rec_tk", sources=[os.path.join(KD, "indexer_topk.cpp")], extra_cflags=CF, verbose=False)
    S, k = 1024, 512
    rows = []
    for M in MS:
        logits = torch.randn(M, S)
        for v in ("TOPK_CHUNKS", "TOPK_SERIAL", "TOPK_SKIP_SELECT"):
            os.environ.pop(v, None)
        o_base = tk.indexer_topk(logits, k)
        t_base = tstream(lambda: tk.indexer_topk(logits, k))
        os.environ["TOPK_CHUNKS"] = "1"
        o_chunk = tk.indexer_topk(logits, k)
        t_chunk = tstream(lambda: tk.indexer_topk(logits, k))
        os.environ.pop("TOPK_CHUNKS", None)
        setmatch = set(o_base[0].tolist()) == set(o_chunk[0].tolist())
        rows.append({"M": M, "t_shipped": t_base, "t_chunked": t_chunk, "speedup": t_base / t_chunk,
                     "setmatch": setmatch})
    return "topk", rows


def rec_sinkhorn():
    sk = load(name="rec_sk", sources=[os.path.join(KD, "sinkhorn.cpp")], extra_cflags=CF, verbose=False)
    hc, iters, eps = 4, 20, 1e-6
    base = torch.randn((2 + hc) * hc); scale = torch.rand(3) + 0.5
    rows = []
    for M in MS:
        mixes = torch.randn(1, M, (2 + hc) * hc)
        os.environ.pop("SINK_SERIAL", None)
        t_par = tstream(lambda: sk.mhc_sinkhorn(mixes, scale, base, hc, iters, eps))
        os.environ["SINK_SERIAL"] = "1"
        t_ser = tstream(lambda: sk.mhc_sinkhorn(mixes, scale, base, hc, iters, eps))
        os.environ.pop("SINK_SERIAL", None)
        rows.append({"M": M, "t_parallel": t_par, "t_serial": t_ser, "serial_speedup": t_par / t_ser})
    return "sinkhorn", rows


KERNELS = {"sparse": rec_sparse, "topk": rec_topk, "sinkhorn": rec_sinkhorn}


def main():
    kernel = os.environ.get("KERNEL", "sparse")
    name, rows = KERNELS[kernel]()
    sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()[:10]
    print(f"# RECOVERY KERNEL={name} HEAD={sha} threads={torch.get_num_threads()} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} node={os.uname().nodename}")
    for r in rows:
        print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    out = {"kernel": name, "head": sha, "node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
           "threads": torch.get_num_threads(), "MS": MS, "rows": rows}
    out_path = os.environ.get("REC_OUT", os.path.join(HERE, "results", f"recovery_{name}.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
