"""Reference-first CPU implementation of KDA (Kimi Delta / gated-delta-rule) linear
attention — the sole authoring GAP for GLM-5.3 Flash (34/45 layers).

Authored against the SGLang GPU kernels as the numeric oracle (reference-first, per
the kernel-authoring + accuracy-oracle skills). Every formula here mirrors a specific
GPU kernel line so a CPU-vs-GPU full-tensor diff should match to bf16 tolerance:

  * gated delta-rule recurrence + L2-norm(q,k) + sigmoid beta + safe gate:
      sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py
        g    = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))     # USE_LOWER_BOUND (KDA)
        beta = sigmoid(b)
        q,k  = l2norm(.)            ; q *= scale (= head_k_dim**-0.5)
        h   *= exp(g)[:, None]      ; v -= (h * k[:, None]).sum(0)
        v   *= beta                 ; h += k[:, None] * v[None, :]
        o    = (h * q[:, None]).sum(0)
  * scale default = k.shape[-1] ** -0.5   (fla/kda.py:fused_recurrent_kda)
  * gated RMSNorm output (activation="sigmoid"):
      fla/fused_norm_gate.py:  y = (x * rstd) * w ; y = y * sigmoid(gate)
  * short causal depthwise conv1d (kernel=4) + silu on the packed qkv, per the
    RadixLinearAttention(activation="silu") + qkv_conv1d wiring in models/glm5_next.py.

The sequential scan is EXACT for both prefill and decode (the GPU chunk kernel is a
perf reformulation of the same recurrence), so this doubles as the correctness oracle
for a later AMX fast path. NOT yet AMX-accelerated — correctness first.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "kda_sigmoid_gate",
    "l2norm",
    "rms_norm_gated",
    "causal_conv1d",
    "causal_conv1d_update",
    "kda_recurrent",
]


def kda_sigmoid_gate(
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """KDA safe per-key log-decay gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias)).

    Mirrors fused_sigmoid_gating_recurrent.py (USE_LOWER_BOUND branch). All in fp32.
    Shapes: a,dt_bias [..., K]; A_log broadcast per head. Returns g [..., K] (<= 0).
    """
    x = a.float() + dt_bias.float()
    return lower_bound * torch.sigmoid(torch.exp(A_log.float()) * x)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row L2 normalization over the last dim, matching the in-kernel form
    x / sqrt(sum(x*x) + 1e-6) (note: eps INSIDE the sqrt, not added to the norm)."""
    xf = x.float()
    return xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + eps)


def rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """Gated RMSNorm output: y = (x * rstd) * weight ; y *= act(gate).

    Mirrors fla/fused_norm_gate.py:layer_norm_gated_fwd_kernel (IS_RMS_NORM, no bias).
    activation 'sigmoid' -> y*sigmoid(g); 'swish'/'silu' -> y*g*sigmoid(g).
    """
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rstd * weight.float()
    g = gate.float()
    if activation in ("swish", "silu"):
        y = y * g * torch.sigmoid(g)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(g)
    else:
        raise ValueError(f"unsupported gated-norm activation: {activation}")
    return y


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Depthwise causal conv1d over a sequence.

    x       [T, C]            (tokens x channels)
    weight  [C, K]            (per-channel kernel; K = conv window, e.g. 4)
    returns [T, C]            causal (left-padded K-1), optional silu.
    """
    T, C = x.shape
    K = weight.shape[-1]
    xf = x.float().transpose(0, 1)  # [C, T]
    xpad = torch.nn.functional.pad(xf, (K - 1, 0))  # left causal pad
    # depthwise conv: out[c,t] = sum_j w[c,j] * xpad[c, t+j]
    out = torch.zeros_like(xf)
    for j in range(K):
        out += weight[:, j].float().unsqueeze(1) * xpad[:, j : j + T]
    if bias is not None:
        out += bias.float().unsqueeze(1)
    out = out.transpose(0, 1)  # [T, C]
    if activation == "silu":
        out = out * torch.sigmoid(out)
    elif activation not in (None, "identity"):
        raise ValueError(f"unsupported conv activation: {activation}")
    return out


def causal_conv1d_update(
    x_t: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step causal conv for decode, using a rolling conv_state ring.

    x_t         [C]              current-token channels
    conv_state  [C, K-1]         previous K-1 inputs (oldest..newest)
    weight      [C, K]
    returns (out_t [C], new_conv_state [C, K-1]).
    """
    C, Km1 = conv_state.shape
    K = Km1 + 1
    window = torch.cat([conv_state.float(), x_t.float().unsqueeze(1)], dim=1)  # [C, K]
    out = (window * weight.float()).sum(1)  # [C]
    if bias is not None:
        out = out + bias.float()
    if activation == "silu":
        out = out * torch.sigmoid(out)
    elif activation not in (None, "identity"):
        raise ValueError(f"unsupported conv activation: {activation}")
    new_state = window[:, 1:].to(conv_state.dtype)  # drop oldest, keep newest K-1
    return out.to(x_t.dtype), new_state


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential gated-delta-rule scan — the EXACT recurrence of the GPU KDA kernel.

    Inputs (per request, single head group), fp-anything (computed in fp32):
      q,k      [T, H, K]        query/key per head
      v        [T, H, V]
      g        [T, H, K]        per-key log-decay gate (<= 0), from kda_sigmoid_gate
      beta     [T, H]           sigmoid(b)
      scale    float            default K**-0.5, applied to q after l2norm
      initial_state [H, K, V]   carried recurrent matrix (None -> zeros)
    Returns:
      o        [T, H, V]
      state    [H, K, V]        final recurrent state (for the decode cache)
    """
    T, H, Kd = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = Kd ** -0.5
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g.float()
    bf = beta.float()
    h = (
        torch.zeros(H, Kd, V, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty(T, H, V, dtype=torch.float32)
    for t in range(T):
        qt, kt, vt = qf[t], kf[t], vf[t]           # [H,K],[H,K],[H,V]
        if use_qk_l2norm:
            qt = l2norm(qt)
            kt = l2norm(kt)
        qt = qt * scale
        decay = torch.exp(gf[t])                    # [H,K]
        h = h * decay.unsqueeze(-1)                 # h[H,K,V] *= exp(g)[:,:,None]
        # v -= sum_K h * k
        vt = vt - (h * kt.unsqueeze(-1)).sum(1)     # [H,V]
        vt = vt * bf[t].unsqueeze(-1)               # *= beta
        h = h + kt.unsqueeze(-1) * vt.unsqueeze(1)  # += k[:,:,None]*v[:,None,:]
        out[t] = (h * qt.unsqueeze(-1)).sum(1)      # o = sum_K h*q
    return out, h


# --------------------------------------------------------------------------------
# Weight-free, ISA-portable self-consistency parity gates (accuracy-oracle Layer 0.5).
# These prove the recurrence + decode-cache carry WITHOUT the checkpoint or a GPU.
# The GPU full-tensor diff (H200 Triton oracle) is the independent spec, run later.
# --------------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    T, H, Kd, V = 17, 3, 16, 16
    lb = -5.0
    q = torch.randn(T, H, Kd)
    k = torch.randn(T, H, Kd)
    v = torch.randn(T, H, V)
    a = torch.randn(T, H, Kd)
    dt_bias = torch.randn(H, Kd) * 0.1
    A_log = torch.randn(H, 1) * 0.1
    b = torch.randn(T, H)
    g = kda_sigmoid_gate(a, dt_bias.unsqueeze(0), A_log.unsqueeze(0), lb)
    beta = torch.sigmoid(b)

    # (1) Prefill (full scan) vs decode (step-by-step with carried state) must match.
    o_full, s_full = kda_recurrent(q, k, v, g, beta)
    o_dec = torch.empty_like(o_full)
    state = None
    for t in range(T):
        o_t, state = kda_recurrent(
            q[t : t + 1], k[t : t + 1], v[t : t + 1], g[t : t + 1], beta[t : t + 1],
            initial_state=state,
        )
        o_dec[t] = o_t[0]
    cos_pd = torch.nn.functional.cosine_similarity(
        o_full.flatten(), o_dec.flatten(), dim=0
    ).item()
    max_pd = (o_full - o_dec).abs().max().item()
    assert cos_pd > 1 - 1e-6 and max_pd < 1e-4, f"prefill!=decode cos={cos_pd} max={max_pd}"

    # (2) Split-sequence state carry: scan [0:n] then [n:T] with carried state ==
    #     one-shot scan over [0:T] (proves initial_state threading).
    n = 7
    o1, s1 = kda_recurrent(q[:n], k[:n], v[:n], g[:n], beta[:n])
    o2, s2 = kda_recurrent(q[n:], k[n:], v[n:], g[n:], beta[n:], initial_state=s1)
    o_split = torch.cat([o1, o2], 0)
    cos_sp = torch.nn.functional.cosine_similarity(
        o_full.flatten(), o_split.flatten(), dim=0
    ).item()
    assert cos_sp > 1 - 1e-6, f"split-carry mismatch cos={cos_sp}"
    assert (s_full - s2).abs().max().item() < 1e-4, "final-state carry mismatch"

    # (3) causal_conv1d sequence form vs the decode-update ring must match.
    C, Kc = 8, 4
    x = torch.randn(T, C)
    w = torch.randn(C, Kc)
    bconv = torch.randn(C)
    seq = causal_conv1d(x, w, bconv, activation="silu")
    cs = torch.zeros(C, Kc - 1)
    upd = torch.empty_like(seq)
    for t in range(T):
        upd[t], cs = causal_conv1d_update(x[t], cs, w, bconv, activation="silu")
    assert (seq - upd).abs().max().item() < 1e-4, "conv seq!=update"

    # (4) Gated RMSNorm sanity: sigmoid gate in (0,1), shape preserved.
    xo = torch.randn(T, H, V)
    gate = torch.randn(T, H, V)
    wn = torch.randn(V)
    y = rms_norm_gated(xo, gate, wn, eps=1e-5, activation="sigmoid")
    assert y.shape == xo.shape

    print(
        f"KDA CPU reference self-consistency PASS | prefill==decode cos={cos_pd:.7f} "
        f"max={max_pd:.2e} | split-carry cos={cos_sp:.7f} | conv seq==update OK | gated-norm OK"
    )


if __name__ == "__main__":
    _selftest()
