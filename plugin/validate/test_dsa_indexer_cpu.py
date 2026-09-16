"""Numeric parity gate for the CPU DSA lightning-indexer compute (Thesis-2)."""
import torch

from intel_cpu_models.dsa_indexer_cpu import (
    indexer_logits,
    indexer_logits_reference,
    indexer_topk,
)


def main() -> None:
    torch.manual_seed(0)
    N, H, D, S = 4, 8, 16, 40
    q = torch.randn(N, H, D)
    kv = torch.randn(N, S, D)
    weight = torch.rand(N, H)
    ref = indexer_logits_reference(q, kv, weight)
    got = indexer_logits(q, kv, weight)
    err = (got - ref).abs().max().item()
    assert err < 1e-3, f"indexer_logits FAIL max_err={err}"
    print(f"indexer_logits: PASS  max_abs_err={err:.2e}")

    idx = indexer_topk(got, k=8)
    # top-k must be the 8 highest-logit positions (order-agnostic)
    exp = set(torch.topk(ref, 8, dim=1).indices[0].tolist())
    assert set(idx[0].tolist()) == exp, "indexer_topk selected wrong positions"
    print("indexer_topk: PASS  (selects the true top-k)")
    print("DSA lightning-indexer CPU compute: numeric parity gate PASS")


if __name__ == "__main__":
    main()
