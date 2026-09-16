"""Numeric parity gate for the CPU DSA compressor compute (Thesis-2 new-kernel leg).

Validates the vectorized CPU port against the slow reference oracle across both
compression ratios. This is the correctness gate a new kernel must pass before any
perf work (kernel-authoring). Run:
  PYTHONPATH=plugin python plugin/validate/test_dsa_compressor_cpu.py
"""
import torch

from intel_cpu_models.dsa_compressor_cpu import (
    compress_softmax_pool,
    compress_softmax_pool_reference,
)


def main() -> None:
    torch.manual_seed(0)
    for ratio in (4, 128):
        N, D = 6, 32
        kv = torch.randn(N, ratio, D)
        score = torch.randn(N, ratio, D)
        ape = torch.randn(ratio, D)
        ref = compress_softmax_pool_reference(kv, score, ape)
        got = compress_softmax_pool(kv, score, ape)
        err = (got - ref).abs().max().item()
        assert err < 1e-4, f"ratio={ratio} FAIL max_err={err}"
        print(f"ratio={ratio}: PASS  max_abs_err={err:.2e}")
    print("DSA compressor CPU compute: numeric parity gate PASS")


if __name__ == "__main__":
    main()
