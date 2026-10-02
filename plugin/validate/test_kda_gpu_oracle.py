"""KDA GPU-oracle parity: the CPU reference recurrence (kda_recurrent) vs the REAL
Triton kernel (fused_recurrent_kda) on identical inputs. This is the INDEPENDENT
spec (accuracy-oracle: a self-check twin shares blind spots; the kernel is the truth).

Runs on an H200 node in the same venv (torch+triton+sglang). Compares the output
o[B,T,H,V] with cosine + max-abs + magnitude ratio, in fp32 (tight) and bf16 (realistic),
at both a small shape and the real GLM-5.3 KDA dims (H=64, head_dim=128).
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/nfs_home/bkaul/sglang-cpu-opt/plugin")
from intel_cpu_models.kda_linear_attention_cpu import kda_recurrent  # noqa: E402

from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda  # noqa: E402

LB = -5.0


def run_case(T, H, K, V, dtype, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, T, H, K, device=dev, dtype=dtype)
    k = torch.randn(1, T, H, K, device=dev, dtype=dtype)
    v = torch.randn(1, T, H, V, device=dev, dtype=dtype)
    a = torch.randn(1, T, H, K, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(H, K, device=dev) * 0.1
    A_log = torch.randn(H, 1, device=dev) * 0.1
    b = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g = LB * torch.sigmoid(torch.exp(A_log) * (a + dt_bias))  # [1,T,H,K]
    beta = torch.sigmoid(b)  # [1,T,H]
    scale = K ** -0.5
    init = torch.zeros(1, H, V, K, device=dev, dtype=torch.float32)

    o_gpu, _ = fused_recurrent_kda(
        q.contiguous(), k.contiguous(), v.contiguous(),
        g.to(dtype).contiguous(), beta.to(dtype).contiguous(),
        scale=scale, initial_state=init, use_qk_l2norm_in_kernel=True,
    )
    o_gpu = o_gpu.reshape(1, T, H, V)[0].float().cpu()

    o_cpu, _ = kda_recurrent(
        q[0].float().cpu(), k[0].float().cpu(), v[0].float().cpu(),
        g[0].float().cpu(), beta[0].float().cpu(), scale=scale,
    )
    cos = F.cosine_similarity(o_gpu.flatten(), o_cpu.flatten(), dim=0).item()
    maxabs = (o_gpu - o_cpu).abs().max().item()
    denom = o_gpu.abs().max().item() + 1e-9
    rel = maxabs / denom
    magratio = (o_cpu.norm() / (o_gpu.norm() + 1e-9)).item()
    tag = f"T={T} H={H} K={K} V={V} {str(dtype).split('.')[-1]}"
    print(f"[KDA ORACLE] {tag:34s} cos={cos:.6f} relmax={rel:.3e} magratio={magratio:.5f}")
    return cos, rel


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        results.append(run_case(24, 4, 64, 64, dtype))
        results.append(run_case(31, 8, 128, 128, dtype))          # GLM head_dim=128
        results.append(run_case(64, 64, 128, 128, dtype, seed=1))  # GLM real H=64
    # fp32 must be tight; bf16 looser (kernel accumulation). Gate on fp32 cases.
    fp32 = results[:3]
    ok = all(c > 0.999 and r < 1e-2 for c, r in fp32)
    print(f"[KDA ORACLE] fp32 gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
