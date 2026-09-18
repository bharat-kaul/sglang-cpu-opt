"""Parity + speed for the MLA KV gather: OLD dict+stack vs the plugin's contiguous
buffer + index_select (_stash_kv_write / _stash_kv_gather).

The decode MLA attention is a softmax-weighted sum over the selected keys, so the key
SET (not order) determines the output. We therefore compare the final attention output of
the two gather paths (bit-comparable up to reduction order), and time the gather itself,
which was ~46s of decode dispatch overhead (per-token .tolist()+list-comp+dict+stack).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Import the real helpers under test (module import does not call install()).
from intel_cpu_models._dsv4_cpu_infra import (  # noqa: E402
    _KV_BUF,
    _KV_VALID,
    _stash_kv_gather,
    _stash_kv_write,
)

torch.manual_seed(0)


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # ms median


# Flash decode sizes: key_dim 576 (512 nope + 64 rope), head_dim_v 512, 64 heads.
H, D, V = 64, 576, 512
scale = 1.0 / (D**0.5)


def old_gather(swa, locs, pos_kept):
    if pos_kept is not None:
        cov = len(pos_kept)
        locs = [l for p, l in enumerate(locs) if p >= cov or bool(pos_kept[p])]
    keys = [swa[l] for l in locs if l in swa]
    if not keys:
        return None
    return torch.stack(keys).float()


def attn(K, q):
    s = (q.float() @ K[:, :D].t()) * scale
    p = s.softmax(dim=-1)
    return p @ K[:, :V]


for STASH, TOPK in ((600, 512), (2048, 512), (8192, 1024)):
    _KV_BUF.clear()
    _KV_VALID.clear()
    dp = 12345  # fake data_ptr key

    all_locs = torch.arange(STASH)
    kv = torch.randn(STASH, D, dtype=torch.bfloat16)
    swa = {int(l): kv[i] for i, l in enumerate(all_locs.tolist())}
    _stash_kv_write(dp, all_locs, kv)

    locs_t = torch.randint(0, STASH, (TOPK,))
    locs = [int(x) for x in locs_t.tolist()]
    q = torch.randn(H, D)

    for label, pos_kept in (("no-mask", None), ("pos_kept", (torch.rand(TOPK) > 0.4))):
        pk_list = pos_kept.tolist() if pos_kept is not None else None
        Kold = old_gather(swa, locs, pk_list)

        idx = locs_t.long()
        if pos_kept is not None:
            n = idx.shape[0]
            keep = torch.ones(n, dtype=torch.bool)
            m = min(pos_kept.shape[0], n)
            keep[:m] = pos_kept[:m]
            idx = idx[keep]
        Knew = _stash_kv_gather(dp, idx)

        oa, na = attn(Kold, q), attn(Knew, q)
        err = (oa - na).abs().max().item()
        set_old = (
            set(locs)
            if pos_kept is None
            else set(l for p, l in enumerate(locs) if p >= len(pk_list) or bool(pk_list[p]))
        )
        set_ok = set_old == set(idx.tolist())

        def _old(_pk=pk_list):
            return old_gather(swa, locs, _pk)

        def _new(_pk=pos_kept):
            ii = locs_t.long()
            if _pk is not None:
                n = ii.shape[0]
                keep = torch.ones(n, dtype=torch.bool)
                m = min(_pk.shape[0], n)
                keep[:m] = _pk[:m]
                ii = ii[keep]
            return _stash_kv_gather(dp, ii)

        t_old, t_new = bench(_old), bench(_new)
        print(
            f"STASH={STASH:5d} TOPK={TOPK:4d} {label:8s}: old={t_old:.3f}ms new={t_new:.3f}ms "
            f"speedup={t_old/t_new:5.2f}x  attn_err={err:.2e}  set_match={set_ok}"
        )
