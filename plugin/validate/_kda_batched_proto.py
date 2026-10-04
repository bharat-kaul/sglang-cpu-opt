"""Batch-across-chunks KDA (WY form) vs the sequential oracle + the current per-chunk
loop. Validates faithfulness + measures the torch-dispatch-overhead reduction.

WY split per chunk: v'_i = a_i - b_i@S  (a,b state-INDEPENDENT -> batch over all chunks);
only the inter-chunk state scan S_{c+1}=diag(Plast)((I-W)S_c+U) stays sequential (NC cheap
[H,K,K]@[H,K,V] steps). Mirrors the GPU chunk kernel (fwd_intra batched, fwd_h scan).
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intel_cpu_models"))
from kda_linear_attention_cpu import kda_chunked, kda_recurrent, l2norm  # noqa: E402

CH = 16


def kda_chunked_batched(q, k, v, g, beta, scale=None, initial_state=None,
                        use_qk_l2norm=True, chunk=CH):
    T, H, Kd = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = Kd ** -0.5
    qf, kf, vf, gf, bf = q.float(), k.float(), v.float(), g.float(), beta.float()
    if use_qk_l2norm:
        qf = l2norm(qf)
        kf = l2norm(kf)
    qf = qf * scale
    S0 = (torch.zeros(H, Kd, V) if initial_state is None else initial_state.float().clone())
    # pad T up to a multiple of chunk (zeros -> decay 1, k=v=q=0 contribute nothing)
    NC = (T + chunk - 1) // chunk
    pad = NC * chunk - T
    if pad:
        zp = lambda x, d: torch.cat([x, x.new_zeros(pad, *x.shape[1:])], 0)
        qf, kf, vf, gf = zp(qf, 0), zp(kf, 0), zp(vf, 0), zp(gf, 0)
        bf = torch.cat([bf, bf.new_zeros(pad, H)], 0)
    # reshape to [NC, C, H, *]
    qc = qf.view(NC, chunk, H, Kd); kc = kf.view(NC, chunk, H, Kd)
    vc = vf.view(NC, chunk, H, V); gc = gf.view(NC, chunk, H, Kd)
    bc = bf.view(NC, chunk, H)
    G = torch.cumsum(gc, dim=1)
    expG = torch.exp(G)
    KP = kc * expG; QP = qc * expG; KD = kc * torch.exp(-G)
    # intra matrices [NC,H,C,C]
    A_ki = torch.einsum("nihk,njhk->nhij", KP, KD)
    A_qi = torch.einsum("nihk,njhk->nhij", QP, KD)
    tri_s = torch.tril(torch.ones(chunk, chunk), -1)
    tri_i = torch.tril(torch.ones(chunk, chunk), 0)
    A_ki = A_ki * tri_s
    A_qi = A_qi * tri_i
    beta_h = bc.permute(0, 2, 1).unsqueeze(-1)  # [NC,H,C,1]
    eye = torch.eye(chunk).view(1, 1, chunk, chunk)
    M = eye + beta_h * A_ki  # [NC,H,C,C]
    betaV = (beta_h * vc.permute(0, 2, 1, 3))        # [NC,H,C,V]
    betaKP = (beta_h * KP.permute(0, 2, 1, 3))       # [NC,H,C,K]
    X = torch.linalg.solve_triangular(M, torch.cat([betaV, betaKP], -1), upper=False)
    Au = X[..., :V]   # "u"  [NC,H,C,V]  (state-independent part of v')
    Bw = X[..., V:]   # "w"  [NC,H,C,K]
    U = torch.einsum("njhk,nhjv->nhkv", KD, Au)   # [NC,H,K,V]
    W = torch.einsum("njhk,nhjl->nhkl", KD, Bw)   # [NC,H,K,K]
    tA_qi = A_qi
    QPe = QP.permute(0, 2, 1, 3) - torch.einsum("nhij,nhjk->nhik", tA_qi, Bw)  # [NC,H,C,K]
    Ointra = torch.einsum("nhij,nhjv->nhiv", tA_qi, Au)  # [NC,H,C,V]
    P_last = expG[:, -1]  # [NC,H,K]
    # sequential inter-chunk scan (NC cheap steps)
    S = S0
    S_list = []
    for c in range(NC):
        S_list.append(S)
        S = P_last[c].unsqueeze(-1) * (U[c] + S - torch.bmm(W[c], S))
    S_stack = torch.stack(S_list, 0)  # [NC,H,K,V]
    o = torch.einsum("nhck,nhkv->nhcv", QPe, S_stack) + Ointra  # [NC,H,C,V]
    o = o.permute(0, 2, 1, 3).reshape(NC * chunk, H, V)[:T]
    return o, S


def _mk(T, H, Kd, V, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, H, Kd); k = torch.randn(T, H, Kd); v = torch.randn(T, H, V)
    a = torch.randn(T, H, Kd); dt = torch.randn(H, Kd) * 0.1; Al = torch.randn(H, 1) * 0.1
    g = -5.0 * torch.sigmoid(torch.exp(Al.unsqueeze(0)) * (a + dt.unsqueeze(0)))
    beta = torch.sigmoid(torch.randn(T, H))
    return q, k, v, g, beta


def _rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


if __name__ == "__main__":
    for (T, H, Kd, V) in [(31, 3, 16, 16), (256, 64, 128, 128), (255, 64, 128, 128)]:
        q, k, v, g, beta = _mk(T, H, Kd, V)
        o_ref, s_ref = kda_recurrent(q, k, v, g, beta)
        o_b, s_b = kda_chunked_batched(q, k, v, g, beta)
        # seeded-state check
        S0 = torch.randn(H, Kd, V)
        o_r2, s_r2 = kda_recurrent(q, k, v, g, beta, initial_state=S0)
        o_b2, s_b2 = kda_chunked_batched(q, k, v, g, beta, initial_state=S0)
        print(f"T={T} H={H} K={Kd}: batched vs scan  o_rel={_rel(o_b,o_ref):.2e} "
              f"s_rel={_rel(s_b,s_ref):.2e} | seeded o_rel={_rel(o_b2,o_r2):.2e} "
              f"finite={bool(torch.isfinite(o_b).all())}")
        if T == 256:
            N = 20
            for name, fn in [("loop", kda_chunked), ("batched", kda_chunked_batched)]:
                for _ in range(3):
                    fn(q, k, v, g, beta)
                t = time.perf_counter()
                for _ in range(N):
                    fn(q, k, v, g, beta)
                print(f"   {name:8s} {(time.perf_counter()-t)/N*1e3:.2f}ms")
