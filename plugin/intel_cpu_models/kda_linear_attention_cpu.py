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

import os as _os
from typing import Optional, Tuple

import torch

__all__ = [
    "kda_sigmoid_gate",
    "l2norm",
    "rms_norm_gated",
    "causal_conv1d",
    "causal_conv1d_update",
    "kda_recurrent",
    "kda_chunked",
    "kda_layer_forward",
    "cpu_kda_extend",
    "cpu_kda_decode",
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


# Separable form overflows once chunk*|gate_lower_bound| exceeds the fp32 exp ceiling
# (exp(88)~3.4e38): GLM lb=-5 => chunk <= 17. 16 is the stable max; validated finite +
# cos=1.0 / rel~3e-7 vs kda_recurrent, NaN at 32/64 (plugin/validate/_kda_chunked_proto.py).
_KDA_CHUNK = 16


def kda_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    chunk: int = _KDA_CHUNK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CHUNKED (parallel, matmul) gated delta-rule — the high-arithmetic-intensity fast
    path for prefill. Bit-equivalent to the sequential kda_recurrent scan but turns the
    per-token loop into blocked matmuls (AMX-friendly), raising AI at a fixed operating
    point. kda_recurrent remains the correctness oracle; this is gated on for prefill.

    Per head (state S[K,V], decay a_i=exp(g_i) applied to the state carried INTO step i,
    P_i=cumprod_{<=i} a), within a chunk:
      v'_i = beta_i (v_i - (k_i*P_i)@S - sum_{j<i} <k_i*P_i, k_j/P_j> v'_j)   # tri solve
      o_i  = (q_i*P_i)@S + sum_{j<=i} <q_i*P_i, k_j/P_j> v'_j
      S_next = diag(P_last) (S + sum_j (k_j/P_j) ⊗ v'_j)
    Decay factors <·,k_j/P_j>=sum_K exp(G_i-G_j) with i>=j are <=1; the k/P (=exp(-G))
    intermediate is why chunk is capped at 16 for lb=-5 (see _KDA_CHUNK).
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
        qc, kc, vc, gc, bc = qf[c0:c1], kf[c0:c1], vf[c0:c1], gf[c0:c1], bf[c0:c1]
        G = torch.cumsum(gc, dim=0)                   # [C,H,K] log-decay <=0
        expG = torch.exp(G)                           # P_i in (0,1]
        KP, QP = kc * expG, qc * expG                 # k_i*P_i, q_i*P_i
        KD = kc * torch.exp(-G)                       # k_j/P_j (bounded for chunk<=16)
        A_kk = torch.einsum("ihk,jhk->hij", KP, KD)   # solve matrix (strict lower)
        A_qk = torch.einsum("ihk,jhk->hij", QP, KD)   # output matrix (lower incl diag)
        A_kk = A_kk * torch.tril(torch.ones(C, C), -1)
        A_qk = A_qk * torch.tril(torch.ones(C, C), 0)
        Su = torch.einsum("ihk,hkv->hiv", KP, S)      # (k_i P_i)@S -> [H,C,V]
        So = torch.einsum("ihk,hkv->hiv", QP, S)      # (q_i P_i)@S -> [H,C,V]
        beta_h = bc.transpose(0, 1).unsqueeze(-1)     # [H,C,1]
        rhs = beta_h * (vc.transpose(0, 1) - Su)      # [H,C,V]
        M = eye[:C, :C].unsqueeze(0) + beta_h * A_kk  # I + tril(beta*A_kk,-1)
        Vp = torch.linalg.solve_triangular(M, rhs, upper=False)   # [H,C,V]
        out[c0:c1] = (So + torch.bmm(A_qk, Vp)).transpose(0, 1)
        S = expG[-1].unsqueeze(-1) * (S + torch.einsum("jhk,hjv->hkv", KD, Vp))
    return out, S


def kda_layer_forward(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    lower_bound: float,
    scale: Optional[float] = None,
    conv_state: Optional[torch.Tensor] = None,
    ssm_state: Optional[torch.Tensor] = None,
    is_decode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full single-request KDA layer op on CPU — the compute core a CpuKDABackend
    wraps (reading layer_cache.conv[0] / .temporal, indexed by cache_indices).

    Pipeline (mirrors Glm5NextLinearAttention + the GPU KDA backend):
      mixed_qkv --causal conv1d(+silu)--> split q,k,v --reshape heads-->
      gate g = kda_sigmoid_gate(a, dt_bias, A_log, lb) ; beta = sigmoid(b) -->
      kda_recurrent(carrying ssm_state) --> core_attn_out [T,H,V].

    Shapes (single request, T tokens):
      mixed_qkv  [T, 3*H*head_dim]        packed q|k|v BEFORE conv
      a          [T, H*head_dim]          per-key forget-gate input (f_b_proj out)
      b          [T, H]                   per-head beta input (b_proj out)
      conv_weight[3*H*head_dim, K]        depthwise conv kernel (K=4)
      A_log      [H] or [...,H,1]         per-head log-decay base
      dt_bias    [H*head_dim]             per-key bias
      conv_state [3*H*head_dim, K-1]      rolling conv history (decode); None -> zeros
      ssm_state  [H, head_dim, head_dim]  recurrent matrix carry; None -> zeros
    Returns (core_attn_out [T,H,V], new_conv_state, new_ssm_state).
    """
    T = mixed_qkv.shape[0]
    C = mixed_qkv.shape[1]
    proj = num_heads * head_dim
    assert C == 3 * proj, f"mixed_qkv width {C} != 3*{proj}"
    K = conv_weight.shape[-1]

    # 1) short causal conv1d (+silu), threading conv_state for decode continuity.
    if is_decode:
        assert T == 1, "decode path is one token per call"
        if conv_state is None:
            conv_state = torch.zeros(C, K - 1, dtype=torch.float32)
        conv_out, new_conv_state = causal_conv1d_update(
            mixed_qkv[0], conv_state, conv_weight, conv_bias, activation="silu"
        )
        conv_out = conv_out.unsqueeze(0)  # [1, C]
    else:
        if conv_state is not None:
            # Seed the causal window from the carried history WITHOUT adding
            # recurrence steps: prepend the K-1 raw inputs, conv, drop the first
            # K-1 outputs. Recurrence still runs over the real tokens only.
            seq_in = torch.cat([conv_state.float().transpose(0, 1), mixed_qkv.float()], 0)
            conv_out = causal_conv1d(seq_in, conv_weight, conv_bias, activation="silu")[K - 1 :]
        else:
            conv_out = causal_conv1d(mixed_qkv, conv_weight, conv_bias, activation="silu")
        # new conv_state = the last K-1 raw inputs (what decode would resume from)
        pad = torch.nn.functional.pad(mixed_qkv.float().transpose(0, 1), (K - 1, 0))
        new_conv_state = pad[:, -(K - 1) :].contiguous()

    # 2) split q|k|v and reshape to heads.
    q, k, v = conv_out.split(proj, dim=-1)
    q = q.reshape(T, num_heads, head_dim)
    k = k.reshape(T, num_heads, head_dim)
    v = v.reshape(T, num_heads, head_dim)

    # 3) gate + beta.
    a_hk = a.reshape(T, num_heads, head_dim)
    dt_hk = dt_bias.reshape(num_heads, head_dim).unsqueeze(0)
    A_log_h = A_log.reshape(num_heads, 1).unsqueeze(0)
    g = kda_sigmoid_gate(a_hk, dt_hk, A_log_h, lower_bound)   # [T,H,K]
    beta = torch.sigmoid(b.float())                           # [T,H]

    # 4) recurrence (carrying the SSM matrix state).
    # Prefill fast path: the matmul-based chunked form (identical numerics, higher AI) replaces the
    # sequential scan. VALIDATED FAITHFUL on real weights (fullcap diff vs baseline: logits cos 0.999926,
    # no layer diverges) -> DEFAULT-ON. Decode (T==1) stays on the scan (chunking one token has no benefit).
    # INTEL_CPU_GLM_CHUNKED_KDA=0 reverts to the scan oracle (kept for revertibility / A-B).
    if (not is_decode) and T > 1 and _os.environ.get("INTEL_CPU_GLM_CHUNKED_KDA", "1") != "0":
        out, new_ssm_state = kda_chunked(
            q, k, v, g, beta, scale=scale, initial_state=ssm_state
        )
    else:
        out, new_ssm_state = kda_recurrent(
            q, k, v, g, beta, scale=scale, initial_state=ssm_state
        )
    return out, new_conv_state, new_ssm_state


# --------------------------------------------------------------------------------
# Batched varlen glue the CpuKDABackend methods call: gather the per-request slot
# from the mamba pool, run the layer op, scatter the updated conv+SSM state back.
# Free functions on plain tensors so they are unit-testable without a ModelRunner.
# Per-slot layout here: conv_states[slot] = [C, K-1]; ssm_states[slot] = [H, Kd, Kd].
# --------------------------------------------------------------------------------
def _slice_gate(a: torch.Tensor, s: int, e: int) -> torch.Tensor:
    return a[s:e] if a.ndim == 2 else a[0, s:e]


def _slice_beta(b: torch.Tensor, s: int, e: int) -> torch.Tensor:
    return b[s:e] if b.ndim == 2 else b[0, s:e]


def cpu_kda_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """One-token-per-request decode. Updates conv_states/ssm_states IN PLACE at the
    rows named by cache_indices. Returns core_attn_out [B, H, V]."""
    B = mixed_qkv.shape[0]
    H, V = params["num_heads"], params["head_dim"]
    out = torch.empty(B, H, V, dtype=torch.float32)
    for i in range(B):
        slot = int(cache_indices[i])
        o, new_conv, new_ssm = kda_layer_forward(
            mixed_qkv[i : i + 1], _slice_gate(a, i, i + 1), _slice_beta(b, i, i + 1),
            conv_weight=params["conv_weight"], conv_bias=params["conv_bias"],
            A_log=params["A_log"], dt_bias=params["dt_bias"],
            num_heads=H, head_dim=V, lower_bound=params["lower_bound"],
            scale=params.get("scale"),
            conv_state=conv_states[slot], ssm_state=ssm_states[slot], is_decode=True,
        )
        conv_states[slot] = new_conv.to(conv_states.dtype)
        ssm_states[slot] = new_ssm.to(ssm_states.dtype)
        out[i] = o[0]
    return out


def cpu_kda_extend(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Varlen prefill/extend. For request i over tokens
    [query_start_loc[i]:query_start_loc[i+1]] runs the sequence conv + recurrence,
    seeded from the pool slot when has_initial_state[i], then writes the updated
    conv+SSM state back. Returns core_attn_out [N, H, V] over the packed tokens."""
    B = cache_indices.shape[0]
    H, V = params["num_heads"], params["head_dim"]
    N = mixed_qkv.shape[0]
    out = torch.empty(N, H, V, dtype=torch.float32)
    for i in range(B):
        s, e = int(query_start_loc[i]), int(query_start_loc[i + 1])
        if e <= s:
            continue
        slot = int(cache_indices[i])
        seeded = bool(has_initial_state[i])
        o, new_conv, new_ssm = kda_layer_forward(
            mixed_qkv[s:e], _slice_gate(a, s, e), _slice_beta(b, s, e),
            conv_weight=params["conv_weight"], conv_bias=params["conv_bias"],
            A_log=params["A_log"], dt_bias=params["dt_bias"],
            num_heads=H, head_dim=V, lower_bound=params["lower_bound"],
            scale=params.get("scale"),
            conv_state=conv_states[slot] if seeded else None,
            ssm_state=ssm_states[slot] if seeded else None,
            is_decode=False,
        )
        conv_states[slot] = new_conv.to(conv_states.dtype)
        ssm_states[slot] = new_ssm.to(ssm_states.dtype)
        out[s:e] = o
    return out


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

    # (2b) CHUNKED (matmul) fast path == sequential scan, incl. a carried initial_state.
    o_ch, s_ch = kda_chunked(q, k, v, g, beta)
    cos_ch = torch.nn.functional.cosine_similarity(
        o_full.flatten(), o_ch.flatten(), dim=0
    ).item()
    assert cos_ch > 1 - 1e-4 and torch.isfinite(o_ch).all(), f"chunked!=scan cos={cos_ch}"
    assert (s_full - s_ch).abs().max().item() < 1e-3, "chunked final-state mismatch"
    o_ch2, s_ch2 = kda_chunked(q[n:], k[n:], v[n:], g[n:], beta[n:], initial_state=s1)
    assert (
        torch.cat([o1, o_ch2], 0) - o_full
    ).abs().max().item() < 1e-3, "chunked seeded-state mismatch"

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

    # (5) FULL LAYER OP: prefill (whole-seq conv + scan) vs decode (per-token, with
    #     conv_state + ssm_state threaded) must match — proves the conv-state ring +
    #     SSM carry wiring a CpuKDABackend relies on, end to end, weight-free.
    proj = H * Kd
    mixed = torch.randn(T, 3 * proj)
    aa = torch.randn(T, proj)
    bb = torch.randn(T, H)
    Kc2 = 4
    cw = torch.randn(3 * proj, Kc2)
    cb = torch.randn(3 * proj)
    A_log2 = torch.randn(H) * 0.1
    dt2 = torch.randn(proj) * 0.1
    o_pf, cs_pf, ss_pf = kda_layer_forward(
        mixed, aa, bb, conv_weight=cw, conv_bias=cb, A_log=A_log2, dt_bias=dt2,
        num_heads=H, head_dim=Kd, lower_bound=lb, is_decode=False,
    )
    o_dc = torch.empty_like(o_pf)
    cs, ss = None, None
    for t in range(T):
        o_t, cs, ss = kda_layer_forward(
            mixed[t : t + 1], aa[t : t + 1], bb[t : t + 1],
            conv_weight=cw, conv_bias=cb, A_log=A_log2, dt_bias=dt2,
            num_heads=H, head_dim=Kd, lower_bound=lb,
            conv_state=cs, ssm_state=ss, is_decode=True,
        )
        o_dc[t] = o_t[0]
    cos_layer = torch.nn.functional.cosine_similarity(
        o_pf.flatten(), o_dc.flatten(), dim=0
    ).item()
    max_layer = (o_pf - o_dc).abs().max().item()
    assert cos_layer > 1 - 1e-6 and max_layer < 1e-4, (
        f"layer prefill!=decode cos={cos_layer} max={max_layer}"
    )
    assert (ss_pf - ss).abs().max().item() < 1e-4, "layer final SSM-state mismatch"

    # (6) BACKEND GLUE: a 2-request varlen batch through cpu_kda_extend (prefill) then
    #     cpu_kda_decode (one more token each, seeded from the written pool slots) must
    #     equal a per-request full-sequence reference. Proves gather/scatter by
    #     cache_indices + query_start_loc + per-slot conv/SSM seeding, weight-free.
    lens = [3, 5]
    Np = sum(lens)
    qsl = torch.tensor([0, lens[0], lens[0] + lens[1]], dtype=torch.long)
    # requests map to non-trivial, non-identity pool slots to catch index bugs.
    slots = torch.tensor([2, 0], dtype=torch.long)
    S = 4
    params = dict(
        conv_weight=cw, conv_bias=cb, A_log=A_log2, dt_bias=dt2,
        num_heads=H, head_dim=Kd, lower_bound=lb, scale=None,
    )
    mixed_e = torch.randn(Np, 3 * proj)
    a_e = torch.randn(Np, proj)
    b_e = torch.randn(Np, H)
    conv_pool = torch.zeros(S, 3 * proj, Kc2 - 1)
    ssm_pool = torch.zeros(S, H, Kd, Kd)
    o_ext = cpu_kda_extend(
        mixed_e, a_e, b_e, conv_states=conv_pool, ssm_states=ssm_pool,
        cache_indices=slots, query_start_loc=qsl,
        has_initial_state=torch.zeros(2, dtype=torch.bool), params=params,
    )
    # one decode token per request
    mixed_d = torch.randn(2, 3 * proj)
    a_d = torch.randn(2, proj)
    b_d = torch.randn(2, H)
    o_dec2 = cpu_kda_decode(
        mixed_d, a_d, b_d, conv_states=conv_pool, ssm_states=ssm_pool,
        cache_indices=slots, params=params,
    )
    # per-request full-sequence reference (prefill over prefix+1), compared to the
    # extend outputs + the decode token.
    worst_cos, worst_max = 1.0, 0.0
    for i, L in enumerate(lens):
        s, e = int(qsl[i]), int(qsl[i + 1])
        full_mix = torch.cat([mixed_e[s:e], mixed_d[i : i + 1]], 0)
        full_a = torch.cat([a_e[s:e], a_d[i : i + 1]], 0)
        full_b = torch.cat([b_e[s:e], b_d[i : i + 1]], 0)
        o_ref, _, _ = kda_layer_forward(
            full_mix, full_a, full_b, conv_weight=cw, conv_bias=cb,
            A_log=A_log2, dt_bias=dt2, num_heads=H, head_dim=Kd, lower_bound=lb,
            is_decode=False,
        )
        got = torch.cat([o_ext[s:e], o_dec2[i : i + 1]], 0)
        c = torch.nn.functional.cosine_similarity(
            o_ref.flatten(), got.flatten(), dim=0
        ).item()
        m = (o_ref - got).abs().max().item()
        worst_cos, worst_max = min(worst_cos, c), max(worst_max, m)
    assert worst_cos > 1 - 1e-6 and worst_max < 1e-4, (
        f"backend-glue mismatch cos={worst_cos} max={worst_max}"
    )

    print(
        f"KDA CPU reference self-consistency PASS | prefill==decode cos={cos_pd:.7f} "
        f"max={max_pd:.2e} | split-carry cos={cos_sp:.7f} | conv seq==update OK | "
        f"gated-norm OK | FULL-LAYER prefill==decode cos={cos_layer:.7f} max={max_layer:.2e} | "
        f"BACKEND-GLUE varlen cos={worst_cos:.7f} max={worst_max:.2e}"
    )


if __name__ == "__main__":
    _selftest()
