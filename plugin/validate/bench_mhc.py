"""Parity + speed for the MHC torch reformulations (materialize-then-sum -> einsum).

Both _mhc_post_torch and _cpu_hc_combine build a large broadcast product and .sum() it;
the einsum/bmm form is the SAME math without materializing the intermediate. Verify
bit-parity (fp32) and the speedup before patching them in the plugin.
"""
import time

import torch

torch.manual_seed(0)


def bench(fn, *a, it=50, wu=5):
    for _ in range(wu):
        fn(*a)
    t0 = time.perf_counter()
    for _ in range(it):
        fn(*a)
    return (time.perf_counter() - t0) / it * 1e3


def post_ref(x, residual, post, comb):
    plm = post.unsqueeze(-1)
    return (plm * x.unsqueeze(1) + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)).type_as(x)


def post_fast(x, residual, post, comb):
    plm = post.unsqueeze(-1)
    # einsum/bmm needs matching operand dtypes; cast to fp32 (matches the broadcast-mul promotion).
    return (plm * x.unsqueeze(1) + torch.einsum("sjk,sjh->skh", comb.float(), residual.float())).type_as(x)


def comb_ref(x_flat, pre, hc):
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    xr = x_flat.reshape(m, hc, h).float()
    return (pre.float().unsqueeze(-1) * xr).sum(dim=1)


def comb_fast(x_flat, pre, hc):
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    xr = x_flat.reshape(m, hc, h).float()
    return torch.einsum("mk,mkh->mh", pre.float(), xr)


print("=== _mhc_post (materialize [s,n,n,h] -> einsum) ===")
for s, n, h in [(1, 16, 4096), (8, 16, 4096), (1, 32, 4096)]:
    x = torch.randn(s, h)
    residual = torch.randn(s, n, h)
    post = torch.randn(s, n)
    comb = torch.randn(s, n, n)
    a, b = post_ref(x, residual, post, comb), post_fast(x, residual, post, comb)
    err = (a - b).abs().max().item()
    tr, tf = bench(post_ref, x, residual, post, comb), bench(post_fast, x, residual, post, comb)
    print(f"s={s} n={n} h={h}: err={err:.2e}  ref={tr:.3f}ms fast={tf:.3f}ms  speedup={tr/tf:.1f}x")

# MIXED dtypes (the real call: fp32 comb, bf16 residual/x) — einsum needs .float() operands.
print("=== _mhc_post MIXED dtype (bf16 x/residual, fp32 comb/post) ===")
for s, n, h in [(1, 16, 4096), (8, 32, 4096)]:
    x = torch.randn(s, h, dtype=torch.bfloat16)
    residual = torch.randn(s, n, h, dtype=torch.bfloat16)
    post = torch.randn(s, n)
    comb = torch.randn(s, n, n)
    a, b = post_ref(x, residual, post, comb), post_fast(x, residual, post, comb)
    err = (a.float() - b.float()).abs().max().item()
    print(f"s={s} n={n} h={h}: dtype_out={b.dtype} err={err:.2e}")

print("=== hc_combine (materialize [m,hc,h] -> einsum) ===")
for m, hc, h in [(1, 16, 4096), (8, 16, 4096), (1, 64, 4096)]:
    x_flat = torch.randn(m, hc * h)
    pre = torch.randn(m, hc)
    a, b = comb_ref(x_flat, pre, hc), comb_fast(x_flat, pre, hc)
    err = (a - b).abs().max().item()
    tr, tf = bench(comb_ref, x_flat, pre, hc), bench(comb_fast, x_flat, pre, hc)
    print(f"m={m} hc={hc} h={h}: err={err:.2e}  ref={tr:.3f}ms fast={tf:.3f}ms  speedup={tr/tf:.1f}x")
