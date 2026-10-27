"""Offline bit-exact equivalence test for the DSV4 prefill attention mask construction.

The serving path builds the `mask_main` [T, n] boolean coverage mask with a PER-TOKEN PYTHON LOOP
(_batched_mla_attend, the `else` branch) — measured at ~19% of prefill wall-time (job 393238,
`mla.b.mask` = 8.5s). This test proves a VECTORIZED replacement produces a BIT-IDENTICAL mask on
thousands of fuzzed inputs (covering every branch: idx<0, variable topk_length, pos_kept filtering,
the empty-fallback, duplicate/out-of-range slots), so it can be dropped in with ZERO accuracy change
(same mask -> same attention -> same logits).

Run (in the sglang venv): python plugin/validate/test_vectorized_mask.py
"""
import torch


def mask_loop(T, n, vslots, indices, topk_length, pos_kept):
    """EXACT replica of the serving per-token loop (the reference)."""
    inv = torch.full((int(vslots.max()) + 1,), -1, dtype=torch.long)
    inv[vslots] = torch.arange(n)
    mask_main = torch.zeros(T, n, dtype=torch.bool)
    for t in range(T):
        L = int(topk_length[t])
        idx = indices[t].reshape(-1)[:L].long()
        idx = idx[idx >= 0]
        if pos_kept is not None and t < pos_kept.shape[0] and idx.numel() > 0:
            cov = int(pos_kept.shape[1])
            m = min(cov, idx.shape[0])
            keep = torch.ones(idx.shape[0], dtype=torch.bool)
            if m > 0:
                keep[:m] = pos_kept[t, :m].bool()
            idx = idx[keep]
        if idx.numel() == 0:
            cl = max(0, min(n, n - (T - 1) + t))
            mask_main[t, :cl] = True
        else:
            pos = inv[idx]
            mask_main[t, pos[pos >= 0]] = True
    return mask_main


def mask_vectorized(T, n, vslots, indices, topk_length, pos_kept):
    """Vectorized equivalent — no per-token Python loop."""
    inv = torch.full((int(vslots.max()) + 1,), -1, dtype=torch.long)
    inv[vslots] = torch.arange(n)
    C = indices.reshape(T, -1).shape[1]
    idxC = indices.reshape(T, -1).long()                      # [T, C] raw slot indices
    col = torch.arange(C)
    within_L = col.unsqueeze(0) < topk_length.reshape(T, 1).long()   # col < topk_length[t]
    valid = within_L & (idxC >= 0)                            # [T, C] the loop's `idx[idx>=0]` after [:L]

    if pos_kept is not None:
        cov = int(pos_kept.shape[1])
        Tp = pos_kept.shape[0]
        # rank = position among VALID entries (the index into the compacted `idx` the loop filters)
        rank = torch.cumsum(valid.long(), dim=1) - 1          # [T, C], -1 where not valid
        rank_cl = rank.clamp(min=0, max=cov - 1)
        pk = pos_kept.bool()
        pk_gather = pk[torch.arange(Tp).reshape(Tp, 1), rank_cl[:Tp]]  # [Tp, C] pos_kept[t, rank]
        # keep: for tokens t<Tp with >=1 valid, apply pos_kept to ranks<cov, keep ranks>=cov (True)
        keep = valid.clone()
        has_valid = valid.any(dim=1)                          # loop's `idx.numel() > 0` guard
        apply = torch.zeros(T, dtype=torch.bool)
        apply[:Tp] = has_valid[:Tp]                           # only t<Tp AND has a valid idx
        pk_full = (rank >= cov) | torch.cat(
            [pk_gather, torch.zeros(T - Tp, C, dtype=torch.bool)], dim=0
        )                                                     # [T, C]
        keep = torch.where(apply.reshape(T, 1), valid & pk_full, valid)
    else:
        keep = valid

    # map kept slot indices -> compact positions, scatter True
    mask_main = torch.zeros(T, n, dtype=torch.bool)
    kept_t, kept_c = keep.nonzero(as_tuple=True)
    if kept_t.numel() > 0:
        slots = idxC[kept_t, kept_c]
        pos = inv[slots]
        ok = pos >= 0
        mask_main[kept_t[ok], pos[ok]] = True

    # empty-fallback: tokens with NO kept entry -> causal prefix mask_main[t,:cl], cl=clamp(n-(T-1)+t,0,n)
    num_kept = keep.sum(dim=1)
    empties = (num_kept == 0).nonzero(as_tuple=True)[0]
    if empties.numel() > 0:
        tt = torch.arange(T)
        cl = (n - (T - 1) + tt).clamp(min=0, max=n)           # [T]
        colidx = torch.arange(n).unsqueeze(0)                 # [1, n]
        fill = colidx < cl.reshape(T, 1)                      # [T, n]
        mask_main[empties] = fill[empties]
    return mask_main


def _rand_case(rng):
    T = int(rng.integers(1, 40))
    n = int(rng.integers(1, 60))
    C = int(rng.integers(1, 24))
    # slot universe: valid slots are a random subset of [0, Smax)
    Smax = int(rng.integers(n, n + 40))
    perm = torch.randperm(Smax)[:n]
    vslots = perm.sort().values                               # n distinct slots, sorted
    maxslot = int(vslots.max())
    # indices: per-token candidate slots in [0, maxslot] or -1 (so inv[idx] never OOB), fuzzed
    indices = torch.randint(-2, maxslot + 1, (T, C))
    indices[indices < 0] = -1
    # some tokens: force all -1 (empty), some: all valid
    for t in range(T):
        r = rng.random()
        if r < 0.15:
            indices[t] = -1
        elif r < 0.25:
            indices[t] = vslots[torch.randint(0, n, (C,))]    # all real slots
    topk_length = torch.randint(0, C + 1, (T,))
    # pos_kept: optional, [Tp, cov] bool; Tp may be < T (loop guards t < pos_kept.shape[0])
    if rng.random() < 0.3:
        pos_kept = None
    else:
        Tp = int(rng.integers(1, T + 1))
        cov = int(rng.integers(1, C + 3))
        pos_kept = torch.rand(Tp, cov) < 0.5
    return T, n, vslots, indices, topk_length, pos_kept


def main():
    import numpy as np
    rng = np.random.default_rng(0)
    N = 4000
    fails = 0
    for i in range(N):
        args = _rand_case(rng)
        a = mask_loop(*args)
        b = mask_vectorized(*args)
        if not torch.equal(a, b):
            fails += 1
            if fails <= 3:
                T, n, vslots, indices, topk_length, pos_kept = args
                diff = (a ^ b).nonzero()
                print(f"[FAIL {i}] T={T} n={n} pos_kept={None if pos_kept is None else tuple(pos_kept.shape)} "
                      f"ndiff={int((a ^ b).sum())} first_diff={diff[0].tolist() if diff.numel() else None}")
    if fails == 0:
        print(f"PASS — {N}/{N} fuzzed cases BIT-IDENTICAL (loop == vectorized)")
    else:
        print(f"FAIL — {fails}/{N} cases differ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
