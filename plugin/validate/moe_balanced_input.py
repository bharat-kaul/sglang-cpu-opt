#!/usr/bin/env python3
"""MoE anti-mode-collapse input for end-to-end wall-time timing (dummy weights).

Why this exists: the DSv4 wall-time ledger times the model end-to-end with
DETERMINISTIC_DUMMY weights (tiny uniform [-1e-3,1e-3] gate rows). If the input
tokens are not diverse, activations collapse to a common direction and the MoE
router sends every token to the SAME few experts ("mode collapse"). Then only a
handful of the E=256 experts are exercised and the measured wall time is NOT
representative of the real routed-expert traffic the roofline assumes
(distinct-experts = E*(1-(1-1/E)^(topk*M))).

This module (a) builds an input that disperses routing across experts and
(b) VERIFIES the realized dispersion so a collapsed run is caught, not trusted.
Used for TIMING only (no accuracy claim); routing semantics mirror the gate
selection (top-k over sigmoid/softmax scores) so the realized load matches what
the real router would see for decorrelated tokens.
"""
from __future__ import annotations

import numpy as np
import torch

E_DEFAULT = 256      # DSv4-Flash n_routed_experts
TOPK_DEFAULT = 6     # num_experts_per_tok
H_DEFAULT = 4096     # hidden size


def expected_distinct(n_experts: int, topk: int, m: int) -> float:
    """Expected #distinct experts hit by m tokens x topk picks under uniform routing."""
    return n_experts * (1.0 - (1.0 - 1.0 / n_experts) ** (topk * m))


def gate_select(hidden: torch.Tensor, gate_weight: torch.Tensor, topk: int,
                scoring: str = "sigmoid") -> torch.Tensor:
    """Return [M, topk] selected expert ids. gate_weight: [E, H]; hidden: [M, H]."""
    logits = hidden.to(torch.float32) @ gate_weight.to(torch.float32).t()  # [M, E]
    scores = torch.sigmoid(logits) if scoring == "sigmoid" else torch.softmax(logits, dim=-1)
    return torch.topk(scores, topk, dim=-1).indices


def expert_load(sel: torch.Tensor, n_experts: int) -> dict:
    """Routing dispersion stats from a [M, topk] selection tensor."""
    flat = sel.reshape(-1)
    counts = torch.bincount(flat, minlength=n_experts).to(torch.float32)
    picks = int(flat.numel())
    distinct = int((counts > 0).sum())
    max_share = float(counts.max() / picks) if picks else 0.0
    p = counts / picks
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {"picks": picks, "distinct": distinct, "max_share": max_share,
            "entropy": entropy, "entropy_uniform": float(np.log(min(n_experts, picks)))}


def verify_no_collapse(sel: torch.Tensor, n_experts: int, topk: int, m: int,
                       min_distinct_frac: float = 0.8, max_share_abs: float = 0.25) -> tuple[bool, dict]:
    """Collapse gate. The robust signal is realized distinct-experts vs the uniform
    expectation E*(1-(1-1/E)^(topk*m)); collapse = far fewer distinct than expected,
    OR one expert absorbing an outsized absolute share of picks. (A per-pick uniform
    share of 1/picks is NOT collapse when picks < E, so max share is gated absolutely.)
    Returns (ok, stats)."""
    st = expert_load(sel, n_experts)
    exp = expected_distinct(n_experts, topk, m)
    st["expected_distinct"] = exp
    st["distinct_frac"] = st["distinct"] / exp if exp else 1.0
    st["uniform_share"] = 1.0 / min(n_experts, st["picks"]) if st["picks"] else 0.0
    ok = (st["distinct_frac"] >= min_distinct_frac) and (st["max_share"] <= max_share_abs)
    st["ok"] = ok
    return ok, st


def build_balanced_hidden(m: int, gate_weight: torch.Tensor, topk: int = TOPK_DEFAULT,
                          scoring: str = "sigmoid", seed: int = 0, mode: str = "aligned",
                          scale: float = 1.0) -> torch.Tensor:
    """Build a [M, H] hidden-state batch that does NOT collapse MoE routing.

    mode="aligned"  (default, strongest guarantee): token i is aligned to expert
        (i mod E)'s gate row so its top-1 is that expert -> round-robin top-1
        coverage, uniform by construction. Best for small decode M.
    mode="iid": per-token i.i.d. N(0,1) (decorrelated). The key anti-collapse move
        vs the common trap of broadcasting ONE vector across the batch.
    """
    E, H = gate_weight.shape
    g = torch.Generator().manual_seed(seed)
    if mode == "iid":
        return torch.randn(m, H, generator=g) * scale
    if mode == "aligned":
        gw = gate_weight.to(torch.float32)
        rows = gw[torch.arange(m) % E]                       # [M, H] target expert rows
        rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        jitter = torch.randn(m, H, generator=g) * 0.01       # break ties in the other topk-1
        return (rows + jitter) * scale
    raise ValueError(f"unknown mode {mode!r}")


def build_verified_input(m: int, gate_weight: torch.Tensor, topk: int = TOPK_DEFAULT,
                         n_experts: int | None = None, scoring: str = "sigmoid",
                         seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Build the anti-collapse hidden batch and assert it passes the collapse gate.
    Tries iid first (matches the natural E*(1-(1-1/E)^(kM)) routing distribution the
    roofline assumes), falls back to aligned (guaranteed round-robin coverage)."""
    E = n_experts or gate_weight.shape[0]
    for mode in ("iid", "aligned"):
        for s in range(seed, seed + 8):
            h = build_balanced_hidden(m, gate_weight, topk, scoring, s, mode)
            sel = gate_select(h, gate_weight, topk, scoring)
            ok, st = verify_no_collapse(sel, E, topk, m)
            st["mode"], st["seed"] = mode, s
            if ok:
                return h, st
    return h, st  # return last attempt with its (failing) stats for the caller to surface


if __name__ == "__main__":
    torch.manual_seed(0)
    E, H, topk = E_DEFAULT, H_DEFAULT, TOPK_DEFAULT
    gate = torch.randn(E, H) * 1e-3  # mirrors DETERMINISTIC_DUMMY tiny uniform-ish gate
    print(f"MoE anti-collapse check  E={E} H={H} topk={topk}")
    print("-" * 78)
    for m in (1, 8, 16, 32, 64):
        # the TRAP: one vector broadcast across the batch -> collapse
        bad = gate_select(torch.randn(1, H).expand(m, H), gate, topk)
        _, bst = verify_no_collapse(bad, E, topk, m)
        # the FIX: verified anti-collapse input
        _, gst = build_verified_input(m, gate, topk, E)
        print(f"M={m:3d}  expected_distinct={gst['expected_distinct']:6.1f} | "
              f"BROADCAST(trap): distinct={bst['distinct']:3d} max_share={bst['max_share']:.2f} "
              f"collapse_ok={bst['ok']} | "
              f"FIXED[{gst['mode']}]: distinct={gst['distinct']:3d} "
              f"max_share={gst['max_share']:.3f} ok={gst['ok']}")
