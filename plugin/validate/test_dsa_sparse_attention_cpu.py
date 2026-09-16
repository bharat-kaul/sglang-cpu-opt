"""Numeric parity gate for the CPU DSA sparse-prefill attention (Thesis-2)."""
import torch

from intel_cpu_models.dsa_sparse_attention_cpu import (
    sparse_attention,
    sparse_attention_reference,
)


def main() -> None:
    torch.manual_seed(0)
    N, H, D, K, Dv = 3, 8, 16, 12, 16
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, K, D)
    v = torch.randn(N, H, K, Dv)
    ref = sparse_attention_reference(q, k, v)
    got = sparse_attention(q, k, v)
    err = (got - ref).abs().max().item()
    assert err < 1e-4, f"sparse_attention FAIL max_err={err}"
    print(f"sparse_attention: PASS  max_abs_err={err:.2e}")
    print("DSA sparse-prefill attention CPU compute: numeric parity gate PASS")


if __name__ == "__main__":
    main()
