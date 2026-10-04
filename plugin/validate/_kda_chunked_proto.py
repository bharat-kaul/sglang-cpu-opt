"""Prototype + numerical validation of a CHUNKED (parallel, high-AI) gated delta-rule
scan against the sequential kda_recurrent oracle. Login-node only, no weights.

Goal: turn the per-token sequential scan (low arithmetic intensity, ~90% of GLM KDA
prefill) into blocked matmuls (AMX-friendly, high AI) while matching kda_recurrent to
fp32 tolerance. Hazard: GLM gate_lower_bound=-5 with PER-KEY gates -> cumulative
log-decay G=cumsum(g) over a 64-chunk reaches -320, so a separable exp(-G) overflows
fp32. This script tests which formulation stays stable AND exact.

Run: python plugin/validate/_kda_chunked_proto.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intel_cpu_models"))
from kda_linear_attention_cpu import kda_recurrent, kda_sigmoid_gate, l2norm  # noqa: E402


def kda_chunked(
    q, k, v, g, beta, scale=None, initial_state=None, use_qk_l2norm=True, chunk=64,
):
    """Chunked gated delta-rule. Matches kda_recurrent.

    q,k,g [T,H,K]; v [T,H,V]; beta [T,H]. Returns (o [T,H,V], state [H,K,V]).

    Derivation (per head, state S[K,V], decay a_i=exp(g_i) applied to the state
    carried INTO step i, i.e. P_i=cumprod_{<=i} a):
      v'_i = beta_i (v_i - (k_i*P_i)@S - sum_{j<i} <k_i*P_i, k_j/P_j> v'_j)
      o_i  = (q_i*P_i)@S + sum_{j<=i} <q_i*P_i, k_j/P_j> v'_j
      S_next = diag(P_last) (S + sum_j (k_j/P_j) x v'_j)
    All intra-chunk decay factors <k_i*P_i, k_j/P_j> = sum_K kk*exp(G_i-G_j) with
    i>=j are <=1 (bounded); the k/P (=exp(-G)) intermediate is the overflow risk.
    """
    T, H, Kd = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = Kd ** -0.5
    qf, kf, vf, gf, bf = q.float(), k.float(), v.float(), g.float(), beta.float()
    if use_qk_l2norm:
        qf = l2norm(qf)
        kf = l2norm(kf)
    qf = qf * scale
    S = (
        torch.zeros(H, Kd, V, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(T, H, V, dtype=torch.float32)
    eye = torch.eye(chunk, dtype=torch.float32)
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        C = c1 - c0
        qc = qf[c0:c1]           # [C,H,K]
        kc = kf[c0:c1]
        vc = vf[c0:c1]           # [C,H,V]
        gc = gf[c0:c1]           # [C,H,K]
        bc = bf[c0:c1]           # [C,H]
        G = torch.cumsum(gc, dim=0)              # [C,H,K] log-decay, <=0, decreasing
        expG = torch.exp(G)                      # P_i, in (0,1]
        expNG = torch.exp(-G)                    # 1/P_j, OVERFLOW risk
        KP = kc * expG                           # k_i*P_i
        QP = qc * expG                           # q_i*P_i
        KD = kc * expNG                          # k_j/P_j
        # intra decay matrices [H,C,C]: A[h,i,j] = sum_k (r_i P_i)(k_j/P_j)
        A_kk = torch.einsum("ihk,jhk->hij", KP, KD)   # for the solve (strict lower)
        A_qk = torch.einsum("ihk,jhk->hij", QP, KD)   # for output (lower incl diag)
        tri_strict = torch.tril(torch.ones(C, C), -1)
        tri_incl = torch.tril(torch.ones(C, C), 0)
        A_kk = A_kk * tri_strict
        A_qk = A_qk * tri_incl
        # state-read terms [C,H,V] -> [H,C,V]
        Su = torch.einsum("ihk,hkv->hiv", KP, S)      # (k_i P_i)@S
        So = torch.einsum("ihk,hkv->hiv", QP, S)      # (q_i P_i)@S
        beta_h = bc.transpose(0, 1).unsqueeze(-1)     # [H,C,1]
        rhs = beta_h * (vc.transpose(0, 1) - Su)      # [H,C,V]
        M = eye[:C, :C].unsqueeze(0) + beta_h * A_kk  # (I + tril(beta*A_kk,-1))
        Vp = torch.linalg.solve_triangular(M, rhs, upper=False)   # [H,C,V] = v'
        O = So + torch.bmm(A_qk, Vp)                  # [H,C,V]
        out[c0:c1] = O.transpose(0, 1)
        P_last = expG[-1]                             # [H,K]
        S = P_last.unsqueeze(-1) * (S + torch.einsum("jhk,hjv->hkv", KD, Vp))
    return out, S


def _mk_inputs(T, H, Kd, V, lb=-5.0, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, H, Kd)
    k = torch.randn(T, H, Kd)
    v = torch.randn(T, H, V)
    a = torch.randn(T, H, Kd)
    dt_bias = torch.randn(H, Kd) * 0.1
    A_log = torch.randn(H, 1) * 0.1
    b = torch.randn(T, H)
    g = kda_sigmoid_gate(a, dt_bias.unsqueeze(0), A_log.unsqueeze(0), lb)
    beta = torch.sigmoid(b)
    return q, k, v, g, beta


def _cmp(name, o_ref, o_got):
    cos = torch.nn.functional.cosine_similarity(
        o_ref.flatten(), o_got.flatten(), dim=0
    ).item()
    maxerr = (o_ref - o_got).abs().max().item()
    denom = o_ref.abs().max().item() + 1e-30
    rel = maxerr / denom
    finite = bool(torch.isfinite(o_got).all())
    print(f"{name}: cos={cos:.6f} max_abs={maxerr:.3e} rel={rel:.3e} finite={finite}")
    return cos, rel, finite


if __name__ == "__main__":
    # GLM KDA head dims: head_k_dim/head_v_dim ~ 128; local heads vary. Use realistic.
    for (T, H, Kd, V) in [(31, 2, 16, 16), (256, 4, 128, 128)]:
        print(f"\n=== T={T} H={H} K={Kd} V={V} (lb=-5, per-key gate) ===")
        q, k, v, g, beta = _mk_inputs(T, H, Kd, V)
        G = torch.cumsum(g, dim=0)
        print(f"  gate cumsum min over seq = {G.min().item():.1f} "
              f"(exp(-min)={torch.exp(-G.min()).item():.2e} overflow if >3.4e38)")
        t0 = time.perf_counter()
        o_ref, s_ref = kda_recurrent(q, k, v, g, beta)
        t_ref = time.perf_counter() - t0
        for chunk in (16, 32, 64):
            t0 = time.perf_counter()
            o_c, s_c = kda_chunked(q, k, v, g, beta, chunk=chunk)
            t_c = time.perf_counter() - t0
            cos, rel, finite = _cmp(f"  chunk={chunk:>2} out", o_ref, o_c)
            _cmp(f"  chunk={chunk:>2} state", s_ref, s_c)
            print(f"            t_ref={t_ref*1e3:.1f}ms t_chunk={t_c*1e3:.1f}ms")
