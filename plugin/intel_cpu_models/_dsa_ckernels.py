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

import functools
import os

import torch

_KD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "dsa_pilot")
_CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]
_COS_MIN = 0.9995   # tolerance for the per-call parity check (bf16 stage boundaries / ~1-ULP exp)


def enabled() -> bool:
    return os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL", "0") == "1"


def _check() -> bool:
    return os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL_CHECK", "0") == "1"


@functools.lru_cache(maxsize=None)
def _mod(name: str, cpp: str):
    from torch.utils.cpp_extension import load
    return load(name="dsv4ck_" + name, sources=[os.path.join(_KD, cpp)], extra_cflags=_CF, verbose=False)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-20))


def _parity(name: str, c_out: torch.Tensor, torch_fn) -> torch.Tensor:
    if _check():
        ref = torch_fn()
        c = _cos(c_out, ref)
        assert c >= _COS_MIN, f"DSA C-kernel {name} parity cos={c:.6f} < {_COS_MIN} (shape {tuple(c_out.shape)})"
    return c_out


def indexer_logits(q, kv, weight, torch_fn):
    out = _mod("il", "indexer_logits.cpp").indexer_logits(q.contiguous(), kv.contiguous(), weight.contiguous())
    return _parity("indexer_logits", out, torch_fn)


def indexer_topk(logits, k, torch_fn):
    # top-k returns INDICES (int64) — parity is a per-row SET match (order-free), not cos
    out = _mod("tk", "indexer_topk.cpp").indexer_topk(logits.contiguous(), int(k))
    if _check():
        ref = torch_fn()
        inter = sum(len(set(out[n].tolist()) & set(ref[n].tolist())) for n in range(out.shape[0]))
        frac = inter / max(1, out.shape[0] * out.shape[1])
        assert frac >= 0.999, f"DSA C-kernel indexer_topk set-match {frac:.4f} < 0.999"
    return out


def compress_softmax_pool(kv, score, ape, torch_fn):
    out = _mod("cp", "compressor.cpp").compressor_softmax_pool_fp32intrin(
        kv.contiguous(), score.contiguous(), ape.contiguous())
    return _parity("compressor", out, torch_fn)
