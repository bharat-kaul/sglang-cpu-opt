"""Micro-attribution of _torch_flash_mla_with_kvcache cost at realistic decode sizes.

Replicates the per-call work (dict gather + stack + .float + QK/softmax/PV) and a fully
vectorized variant (dense loc->row map + index_select), so we optimize the part that is
actually slow and prove the vectorized path is numerically identical.
"""
import time

import torch

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


# Realistic decode-ish sizes (Flash): key_dim 576 (512 nope + 64 rope), head_dim_v 512,
# H=64 heads, topk=512 selected keys, stash ~ a few hundred to a few thousand tokens.
H, D, V, TOPK = 64, 576, 512, 512
for STASH in (600, 2048, 8192):
    swa = {l: torch.randn(D, dtype=torch.bfloat16) for l in range(STASH)}
    locs_all = torch.randint(0, STASH, (TOPK,))
    q = torch.randn(1, 1, H, D)
    scale = 1.0 / (D**0.5)

    # dense loc->row map + contiguous matrix (what the vectorized path would maintain on write)
    row_map = torch.full((STASH,), -1, dtype=torch.long)
    mat = torch.empty(STASH, D, dtype=torch.bfloat16)
    for r, (l, v) in enumerate(swa.items()):
        row_map[l] = r
        mat[r] = v

    def current():
        locs = [int(x) for x in locs_all.reshape(-1).tolist() if x >= 0]
        keys = [swa[l] for l in locs if l in swa]
        K = torch.stack(keys).float()
        s = (q[0, 0].float() @ K[:, :D].t()) * scale
        p = s.softmax(dim=-1)
        return p @ K[:, :V]

    def vectorized():
        rows = row_map.index_select(0, locs_all.reshape(-1).clamp(min=0))
        rows = rows[rows >= 0]
        K = mat.index_select(0, rows).float()
        s = (q[0, 0].float() @ K[:, :D].t()) * scale
        p = s.softmax(dim=-1)
        return p @ K[:, :V]

    # parity (same selected set/order -> bit comparable; here sets match by construction)
    a, b = current(), vectorized()
    err = (a - b).abs().max().item()
    t_gather = bench(lambda: [swa[l] for l in [int(x) for x in locs_all.tolist()] if l in swa])
    t_cur = bench(current)
    t_vec = bench(vectorized)
    print(
        f"STASH={STASH:5d}: dict-gather+stack-only={t_gather:.3f}ms  current={t_cur:.3f}ms  "
        f"vectorized={t_vec:.3f}ms  speedup={t_cur/t_vec:.2f}x  parity_err={err:.2e}"
    )
