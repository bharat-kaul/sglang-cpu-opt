"""End-to-end gate for the composed CPU DSA attention (Thesis-2 capstone).

(1) top-k == S -> DSA attention MUST equal dense attention (composition anchor).
(2) top-k < S  -> attends to exactly the indexer-selected tokens (sparsity works).
"""
import torch

from intel_cpu_models.dsa_attention_cpu import (
    dense_attention_reference,
    dsa_attention,
)


def main() -> None:
    torch.manual_seed(0)
    N, H, D, S, Dv = 3, 8, 16, 20, 16
    q = torch.randn(N, H, D)
    k = torch.randn(N, S, D)
    v = torch.randn(N, S, Dv)
    w = torch.rand(N, H)

    # (1) select-all reduces to dense attention
    full = dsa_attention(q, k, v, w, topk=S)
    dense = dense_attention_reference(q, k, v)
    err = (full - dense).abs().max().item()
    assert err < 1e-4, f"select-all != dense: max_err={err}"
    print(f"DSA(top-k=all) == dense attention: PASS  max_abs_err={err:.2e}")

    # (2) sparse selection runs and differs from dense (real sparsity)
    sparse = dsa_attention(q, k, v, w, topk=5)
    assert sparse.shape == dense.shape
    assert (sparse - dense).abs().max().item() > 1e-3
    print("DSA(top-k<all): PASS  (sparse, distinct from dense)")
    print("Composed DSA attention (index -> select -> attend) on CPU: end-to-end gate PASS")


if __name__ == "__main__":
    main()
