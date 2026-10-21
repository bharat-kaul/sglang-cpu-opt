"""Phase-2 integration: optional dispatch of the DSA Python ops to the authored C/C++ kernels.

The DSA ops default to their vectorized torch ports (bit-stable, the shipped Phase-1 path). Setting
`INTEL_CPU_DSV4_DSA_CKERNEL=1` routes the clean-signature ops (indexer logits/top-k, compressor softmax-pool)
to the microbench-validated C/C++ kernels in `plugin/kernels/dsa_pilot/` instead — the standalone 4–24× wins,
now wired into the forward. Default-OFF so gate-off reproduces the Phase-1 path byte-for-byte; the C kernels
carry the model.py-faithful stage rounding (bf16 boundaries / ~1-ULP Cephes exp), so the swap is a
tolerance-gated numeric move toward the reference (Phase-3 accuracy sign-off), not a pure perf transform.

`INTEL_CPU_DSV4_DSA_CKERNEL_CHECK=1` additionally asserts the C output matches the torch port per call (cos),
so an integration shape/dtype mismatch fails loud instead of silently diverging.
"""
from __future__ import annotations

import atexit
import collections
import functools
import os
import time

import torch

_KD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "dsa_pilot")
_CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
_COS_MIN = 0.9995   # tolerance for the per-call parity check (bf16 stage boundaries / ~1-ULP exp)

_DBG = collections.defaultdict(lambda: [0.0, 0.0, 0])   # op -> [c_time, torch_time, calls] (INTEL_CPU_DSV4_CK_DBG=1)
_DBG_SHAPE = {}


def _dbg() -> bool:
    return os.environ.get("INTEL_CPU_DSV4_CK_DBG", "0") == "1"


@atexit.register
def _dbg_dump():
    if not _DBG:
        return
    lines = [f"[CK_DBG] {op}: calls={n} C={ct*1e3:.1f}ms ({ct/n*1e6:.1f}us/call) "
             f"torch={tt*1e3:.1f}ms ({tt/n*1e6:.1f}us/call) C/torch={ct/max(tt,1e-9):.2f}x shape={_DBG_SHAPE.get(op)}"
             for op, (ct, tt, n) in sorted(_DBG.items()) if n]
    out = os.environ.get("INTEL_CPU_DSV4_CK_DBG_OUT")   # worker subprocess: write to a file, not stdout
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln, flush=True)


def enabled(op: str | None = None) -> bool:
    """Master toggle INTEL_CPU_DSV4_DSA_CKERNEL=1; per-op override INTEL_CPU_DSV4_CK_<OP>=0/1 (for one-at-a-time
    bisection, Phase-2 rule). op in {indexer, topk, compressor, sinkhorn, combine, sparse_attend}.

    A control FILE (INTEL_CPU_DSV4_CK_CTL_FILE, one int re-read each call) overrides the master so a
    PERSISTENT process can do a HELD-FIXED eager-vs-C A/B on ONE resident load (write 0, measure; write
    1, measure) — the measurement-integrity rule A1."""
    ctl = os.environ.get("INTEL_CPU_DSV4_CK_CTL_FILE")
    if ctl:
        try:
            with open(ctl) as _f:
                master = _f.read().strip() == "1"
        except Exception:
            master = os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL", "0") == "1"
    else:
        master = os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL", "0") == "1"
    if op is None:
        return master
    v = os.environ.get("INTEL_CPU_DSV4_CK_" + op.upper())
    return (v == "1") if v is not None else master


def _check() -> bool:
    return os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL_CHECK", "0") == "1"


# --- Dispatch-provenance registry (playbook gate H) -------------------------------------------
# Records, per hot op, which implementation ACTUALLY RAN: 'c_kernel' (authored C/C++ via this
# module) vs 'eager' (a Python/eager-torch reimplementation in the serving forward). The gate
# (prov_gate) asserts the serving path dispatches to the C kernel and NO eager op remains in the
# hot chain — the check that catches an authored-but-bypassed kernel (the DSV4 D2 miss).
_PROV: "dict[str, dict[str, int]]" = collections.defaultdict(lambda: {"c_kernel": 0, "eager": 0})


def mark(op: str, impl: str) -> None:
    """impl in {'c_kernel','eager'}. Call at the ACTUAL dispatch site of a hot op. Write-through to
    INTEL_CPU_DSV4_PROV_OUT each call (the chain runs in a WORKER subprocess whose atexit may not
    fire on kill — lesson D3; the dict is tiny and stabilizes after layer 0, so writes are idempotent)."""
    _PROV[op][impl] = _PROV[op].get(impl, 0) + 1
    out = os.environ.get("INTEL_CPU_DSV4_PROV_OUT")
    if out:
        try:
            prov_dump(out)
        except Exception:
            pass


def prov_snapshot() -> dict:
    return {op: dict(d) for op, d in _PROV.items()}


def prov_dump(path: str) -> None:
    import json
    with open(path, "w") as f:
        json.dump(prov_snapshot(), f, indent=2)


def prov_gate(hot_ops) -> "list[str]":
    """Return the list of GAP messages: a hot op that ran eager (>0) or never dispatched to its
    C kernel (c_kernel==0). Empty list = gate PASS."""
    gaps = []
    for op in hot_ops:
        d = _PROV.get(op, {"c_kernel": 0, "eager": 0})
        if d.get("eager", 0) > 0:
            gaps.append(f"{op}: ran EAGER {d['eager']}x in the hot chain (wire it to the C kernel)")
        if d.get("c_kernel", 0) == 0:
            gaps.append(f"{op}: C kernel NEVER dispatched (authored-but-bypassed WIRING GAP)")
    return gaps


@atexit.register
def _prov_dump_atexit():
    out = os.environ.get("INTEL_CPU_DSV4_PROV_OUT")
    if out and _PROV:
        try:
            prov_dump(out)
        except Exception:
            pass



@functools.lru_cache(maxsize=None)
def _mod(name: str, cpp: str):
    from torch.utils.cpp_extension import load
    return load(name="dsv4ck_" + name, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    na, nb = a.norm(), b.norm()
    # Both (near-)zero vectors are a MATCH, not a mismatch — cos is undefined (0/0) there. The DSA
    # indexer's relu can legitimately zero all logits on dummy-weight / all-negative-score inputs;
    # without this guard the parity check false-fails on an otherwise-correct kernel.
    if float(na) < 1e-12 and float(nb) < 1e-12:
        return 1.0
    return float(torch.dot(a, b) / (na * nb + 1e-20))


def _parity(name: str, c_out: torch.Tensor, torch_fn) -> torch.Tensor:
    if _check():
        ref = torch_fn()
        c = _cos(c_out, ref)
        assert c >= _COS_MIN, f"DSA C-kernel {name} parity cos={c:.6f} < {_COS_MIN} (shape {tuple(c_out.shape)})"
    return c_out


def indexer_logits(q, kv, weight, torch_fn):
    out = _mod("il", "indexer_logits.cpp").indexer_logits(q.contiguous(), kv.contiguous(), weight.contiguous())
    mark("indexer", "c_kernel")  # after success — a failed brgemm raises before this
    return _parity("indexer_logits", out, torch_fn)


def indexer_topk(logits, k, torch_fn):
    # top-k returns INDICES (int64) — parity is a per-row SET match (order-free), not cos
    out = _mod("tk", "indexer_topk.cpp").indexer_topk(logits.contiguous(), int(k))
    mark("topk", "c_kernel")
    out = _mod("tk", "indexer_topk.cpp").indexer_topk(logits.contiguous(), int(k))
    if _check():
        ref = torch_fn()
        inter = sum(len(set(out[n].tolist()) & set(ref[n].tolist())) for n in range(out.shape[0]))
        frac = inter / max(1, out.shape[0] * out.shape[1])
        assert frac >= 0.999, f"DSA C-kernel indexer_topk set-match {frac:.4f} < 0.999"
    return out


def compress_softmax_pool(kv, score, ape, torch_fn):
    mark("compressor", "c_kernel")
    if _dbg():
        t0 = time.perf_counter()
        out = _mod("cp", "compressor.cpp").compressor_softmax_pool_fp32intrin(
            kv.contiguous(), score.contiguous(), ape.contiguous())
        t1 = time.perf_counter(); torch_fn(); t2 = time.perf_counter()
        d = _DBG["compressor"]; d[0] += t1 - t0; d[1] += t2 - t1; d[2] += 1
        _DBG_SHAPE["compressor"] = tuple(kv.shape)
        return out
    out = _mod("cp", "compressor.cpp").compressor_softmax_pool_fp32intrin(
        kv.contiguous(), score.contiguous(), ape.contiguous())
    return _parity("compressor", out, torch_fn)


def mhc_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps, torch_fn):
    mark("sinkhorn", "c_kernel")
    if _dbg():
        t0 = time.perf_counter()
        r = _mod("sk", "sinkhorn.cpp").mhc_sinkhorn(
            mixes.contiguous(), hc_scale.contiguous(), hc_base.contiguous(), int(hc_mult), int(iters), float(eps))
        t1 = time.perf_counter(); torch_fn(); t2 = time.perf_counter()
        d = _DBG["sinkhorn"]; d[0] += t1 - t0; d[1] += t2 - t1; d[2] += 1
        _DBG_SHAPE["sinkhorn"] = tuple(mixes.shape)
        return r[0], r[1], r[2]
    pre, post, comb = _mod("sk", "sinkhorn.cpp").mhc_sinkhorn(
        mixes.contiguous(), hc_scale.contiguous(), hc_base.contiguous(), int(hc_mult), int(iters), float(eps))
    if _check():
        rp, ro, rc = torch_fn()
        for nm, c, r in (("pre", pre, rp), ("post", post, ro), ("comb", comb, rc)):
            cc = _cos(c, r)
            assert cc >= _COS_MIN, f"DSA C-kernel mhc_sinkhorn.{nm} parity cos={cc:.6f} < {_COS_MIN}"
    return pre, post, comb


def mhc_combine(x_flat, pre, hc, out_dtype, torch_fn):
    mark("combine", "c_kernel")
    out = _mod("cb", "combine.cpp").mhc_combine(
        x_flat.float().contiguous(), pre.float().contiguous(), int(hc)).to(out_dtype)
    return _parity("mhc_combine", out, torch_fn)


def sparse_attend_masked(q, kv, mask, sink, scale, torch_fn):
    # SERVING sparse attend: shared kv[K,D] + per-query mask[N,K] + per-head sink, fused online-softmax.
    out = _mod("sa", "sparse_attend.cpp").sparse_attend_masked(
        q.contiguous(), kv.contiguous(), mask.contiguous(), sink.contiguous(), float(scale))
    mark("sparse_attend", "c_kernel")  # after success
    return _parity("sparse_attend", out, torch_fn)
