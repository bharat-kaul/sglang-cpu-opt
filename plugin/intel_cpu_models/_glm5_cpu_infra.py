"""Plugin-side CPU enablement for GLM-5.3 Flash (Glm5Next) — the KDA linear-attention
backend + its mamba-style dual cache, the sole authoring GAP (34/45 layers).

GLM-5.3 Flash selects the KDA linear-attention backend via
``configs/linear_attn_model_registry.py`` -> ``KDAAttnBackend``, whose extend/decode
kernels (``fused_recurrent_kda`` / ``chunk_kda``) are Triton/CUDA-only. On CPU we keep
ALL of ``KDAAttnBackend``'s pool + metadata machinery and swap only the two compute
methods for the parity-proven reference in ``kda_linear_attention_cpu`` (self-consistency
+ GPU-oracle gated). No ``sglang/`` edit — installed as a monkeypatch from the plugin,
mirroring ``_dsv4_cpu_infra``.

State access mirrors the GPU backend exactly:
  layer_cache = req_to_token_pool.mamba2_layer_cache(layer.layer_id)
  conv_states = layer_cache.conv[0].transpose(-1, -2)   # [slots, C, K-1] (KDA stores [K-1, C])
  ssm_states  = layer_cache.temporal                    # [slots, H, Dk, Dv]
  cache_indices = forward_metadata.mamba_cache_indices
  query_start_loc = forward_metadata.query_start_loc    # extend varlen
"""

from __future__ import annotations

import logging

import torch

from intel_cpu_models.kda_linear_attention_cpu import cpu_kda_decode, cpu_kda_extend

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_cpu_resolution_guards() -> None:
    """Make CPU-unsafe device probes in sglang's arg-resolution survive on CPU-only torch.

    GLM-5.3 Flash is in sglang's DSA-family list (arg_groups/model_hook.py), so
    ``is_deepseek_dsa(cfg)`` routes it into a CUDA/ROCm branch that calls
    ``torch.cuda.get_device_capability()`` during ``resolve_once()`` — which raises on a
    CPU-only torch build (DeepSeek-V4 never hit this: it is NOT in that arch list). This
    runs BEFORE the external model package's model modules import, so the guard must install
    at plugin-package import. Inert on any CUDA build (guarded by is_available)."""
    import torch

    if torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability
    if getattr(cap, "_glm5_cpu_shim", False):
        return

    def _cpu_safe_capability(device=None):
        # Report a Hopper-class capability so DSA resolution picks sane defaults; the
        # actual CPU attention/KV wiring overrides execution downstream.
        return (9, 0)

    _cpu_safe_capability._glm5_cpu_shim = True
    torch.cuda.get_device_capability = _cpu_safe_capability
    logger.info("GLM5 CPU resolution guard: torch.cuda.get_device_capability shimmed (CPU).")

    # Hybrid-cache resolution clash (GLM is mamba/KDA + MLA/DSA): the mamba path forces
    # no_buffer (requires page_size=1) while the MLA backends force page_size=64. On CPU
    # the GPU-FLA extra_buffer strategy is unavailable, so no_buffer is correct; relax its
    # page_size==1 resolution assert on CPU (the page geometry is handled by the CPU KV/
    # mamba pools, not this FLA guard). Keeps the non-CPU assert intact.
    try:
        from sglang.srt.arg_groups import mamba_hook as _mh

        _orig_no_buffer = _mh.validate_mamba_no_buffer

        def _cpu_validate_mamba_no_buffer(view, model_arch, *a, **k):
            from sglang.srt.utils import is_cpu as _is_cpu

            if _is_cpu():
                return  # CPU mamba/KV pools own the page geometry; skip FLA-oriented asserts
            return _orig_no_buffer(view, model_arch, *a, **k)

        _mh.validate_mamba_no_buffer = _cpu_validate_mamba_no_buffer
        logger.info("GLM5 CPU resolution guard: mamba no_buffer page_size assert relaxed (CPU).")
    except Exception as _e:
        logger.warning("GLM5 CPU mamba no_buffer guard not installed: %s", _e)

    # KDA conv1d weight is fp32 and consumed LOGICALLY by our torch causal_conv1d
    # (cpu_kda_*), NOT sglang's AMX causal_conv1d kernel. The stock loader routes conv
    # weights to causal_conv1d_weight_pack, which (a) is unimplemented for Float and
    # (b) would scramble the weight our backend reads row-major (same bug class as the
    # DSV4 wo_a VNNI-pack-vs-einsum mismatch). Skip the pack for fp32 conv weights on CPU
    # -> keep them logical. bf16 conv weights (other mamba models) still pack normally.
    try:
        from sglang.srt.layers import amx_utils as _au

        _orig_pack = _au.amx_process_weight_after_loading

        def _cpu_skip_fp32_conv_pack(weight, is_conv=False):
            if is_conv and weight.dtype == torch.float32:
                # Return the LOGICAL conv weight as 2D [C, K] (what both the loader's
                # in-place copy and our torch cpu_kda conv expect), skipping the pack.
                return weight.view(-1, weight.size(-1))
            if is_conv and weight.dim() == 5:
                # 5D conv3d = the vision patch-embed. Vision is OUT OF SCOPE for the
                # text "Flash" path (tower built but never run); its AMX pack also
                # rejects IC%32!=0. Keep it logical so text bring-up proceeds.
                return weight
            return _orig_pack(weight, is_conv)

        _au.amx_process_weight_after_loading = _cpu_skip_fp32_conv_pack
        logger.info("GLM5 CPU guard: fp32 KDA conv1d weight kept logical (skip AMX conv pack).")
    except Exception as _e:
        logger.warning("GLM5 CPU conv-pack guard not installed: %s", _e)


def _layer_params(layer) -> dict:
    """Build the kda_layer_forward param bag from a RadixLinearAttention layer."""
    return dict(
        conv_weight=layer.conv_weights,
        conv_bias=layer.bias,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_heads=layer.num_v_heads,
        head_dim=layer.head_v_dim,
        lower_bound=layer.lower_bound,
        scale=layer.head_k_dim ** -0.5,
    )


def _cpu_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0].transpose(-1, -2)   # view -> writes propagate
    ssm_states = layer_cache.temporal
    cache_indices = self.forward_metadata.mamba_cache_indices
    out = cpu_kda_decode(
        mixed_qkv, a, b,
        conv_states=conv_states, ssm_states=ssm_states,
        cache_indices=cache_indices, params=_layer_params(layer),
    )
    # recurrence runs in fp32; cast back to the model dtype so the downstream
    # o_proj AMX weight_packed_linear gets bf16 activations (not Float).
    return out.to(mixed_qkv.dtype).unsqueeze(0)  # [1, B, H, V]


def _cpu_forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    query_start_loc = self.forward_metadata.query_start_loc
    cache_indices = self.forward_metadata.mamba_cache_indices
    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0].transpose(-1, -2)
    ssm_states = layer_cache.temporal

    if forward_batch.extend_prefix_lens is None:
        has_initial_state = torch.zeros(
            cache_indices.shape[0], dtype=torch.bool
        )
    else:
        has_initial_state = forward_batch.extend_prefix_lens > 0

    # Trim DP/padded physical rows beyond the logical varlen layout, as the GPU path does.
    physical = mixed_qkv.shape[0]
    logical = int(query_start_loc[-1])
    if logical < physical:
        mixed_qkv = mixed_qkv[:logical]

    out = cpu_kda_extend(
        mixed_qkv, a, b,
        conv_states=conv_states, ssm_states=ssm_states,
        cache_indices=cache_indices, query_start_loc=query_start_loc,
        has_initial_state=has_initial_state, params=_layer_params(layer),
    )
    N, H, V = out.shape
    if logical < physical:
        pad = out.new_zeros(physical - logical, H, V)
        out = torch.cat([out, pad], 0)
    # recurrence runs in fp32; cast to model dtype for the downstream o_proj AMX GEMM.
    return out.to(mixed_qkv.dtype).unsqueeze(0)  # [1, N, H, V]


def _cpu_kda_init(self, model_runner):
    """CPU KDAAttnBackend init: reuse the Mamba base (pool + metadata) and the conv
    shape probe, but SKIP the Triton KDAKernelDispatcher (CUDA-only) — the CPU
    forward_{extend,decode} overrides never call it."""
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )

    MambaAttnBackendBase.__init__(self, model_runner)
    self.conv_states_shape = (
        model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
        .transpose(-1, -2)
        .shape
    )


def install() -> None:
    """Monkeypatch KDAAttnBackend for CPU. Idempotent; no-op off CPU."""
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.srt.utils import is_cpu

    if not is_cpu():
        _INSTALLED = True
        return

    from sglang.srt.layers.attention.linear import kda_backend as _kb

    _kb.KDAAttnBackend.__init__ = _cpu_kda_init
    _kb.KDAAttnBackend.forward_decode = _cpu_forward_decode
    _kb.KDAAttnBackend.forward_extend = _cpu_forward_extend
    logger.info("GLM5 CPU KDA backend installed (CPU recurrence + dual cache).")

    # KDA o_norm is FusedRMSNormGated(activation="sigmoid"), but the CPU sgl_kernel
    # fused_rmsnorm_gated_cpu only implements silu. Route sigmoid through our
    # parity-tested torch rms_norm_gated (the GPU uses the same torch path for sigmoid).
    try:
        from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated

        _orig_norm_fwd = FusedRMSNormGated.forward

        def _cpu_gated_norm_forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
            if is_cpu() and self.activation == "sigmoid":
                from intel_cpu_models.kda_linear_attention_cpu import rms_norm_gated as _rng

                assert residual is None and not prenorm, (
                    "CPU sigmoid gated-norm: residual/prenorm not supported"
                )
                return _rng(x, g, self.weight, eps=self.eps, activation="sigmoid").to(x.dtype)
            return _orig_norm_fwd(self, x, g, residual, prenorm, residual_in_fp32)

        FusedRMSNormGated.forward = _cpu_gated_norm_forward
        logger.info("GLM5 CPU: FusedRMSNormGated sigmoid routed through torch rms_norm_gated.")
    except Exception as _e:
        logger.warning("GLM5 CPU gated-norm patch not installed: %s", _e)

    # MHC hc_post: GLM passes post_layer_mix already as [s, hc_mult, 1] (3D), but the DSV4
    # CPU patch's mhc_post lambda does post.unsqueeze(-1) (it assumed DSV4's 2D [s,n] input)
    # -> 4D -> broadcast error. Re-patch with a SHAPE-ADAPTIVE mhc_post that only unsqueezes a
    # 2D post, so it is correct for BOTH GLM (3D) and DSV4 (2D). Math is unchanged:
    #   term2[s,k,h] = sum_j comb[s,j,k]*residual[s,j,h]; out = post*x.unsqueeze(1) + term2.
    try:
        import sglang.kernels.ops.layernorm.mhc as _mhc

        def _adaptive_mhc_post(x, residual, post, comb):
            if post.dim() == 2:
                post = post.unsqueeze(-1)  # DSV4 convention; GLM already [s,hc,1]
            term2 = torch.einsum("sjk,sjh->skh", comb.float(), residual.float())
            return (post * x.unsqueeze(1) + term2).type_as(x)

        _mhc.mhc_post = _adaptive_mhc_post
        logger.info("GLM5 CPU: MHC mhc_post made shape-adaptive (GLM 3D post_layer_mix).")
    except Exception as _e:
        logger.warning("GLM5 CPU mhc_post patch not installed: %s", _e)

    # SwiGLU clamp (GLM swiglu_limit): silu_and_mul_clamp is CUDA-JIT only (nvcc absent on CPU);
    # DSV4 had no swiglu_limit so never hit it. Route to torch on CPU. Math (per the NPU ref
    # _apply_swiglu_limit_npu): gate=first half clamp(max=L); up=second half clamp(-L,L); out=silu(gate)*up.
    try:
        import torch.nn.functional as _Fn

        def _cpu_silu_and_mul_clamp(inp, output, swiglu_limit):
            gate, up = inp.chunk(2, dim=-1)
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            output.copy_((_Fn.silu(gate.float()) * up.float()).to(output.dtype))

        import sglang.kernels.ops.attention.dsv4.moe as _dmoe

        _dmoe.silu_and_mul_clamp = _cpu_silu_and_mul_clamp
        import sglang.srt.models.deepseek_v2 as _dv2

        if hasattr(_dv2, "silu_and_mul_clamp"):
            _dv2.silu_and_mul_clamp = _cpu_silu_and_mul_clamp
        logger.info("GLM5 CPU: silu_and_mul_clamp routed to torch (swiglu_limit).")
    except Exception as _e:
        logger.warning("GLM5 CPU silu_and_mul_clamp patch not installed: %s", _e)

    # DSA lightning Indexer (GLM MLA layers) has no CPU kernel (forward_native raises). On CPU
    # return None -> the MLA attends DENSE over all valid KV. This is EXACT at short context
    # (seq <= index_topk: top-k selects all = dense == sparse), which covers the make-it-work
    # bring-up + the short-prefill parity fingerprint. Long-context / incremental-decode sparse
    # selection is a documented perf/accuracy TODO (mirrors DSV4's incremental-sparse decode).
    try:
        from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

        def _cpu_indexer_forward_native(
            self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
        ):
            return None  # dense fallback; sparse selection is a perf-phase TODO

        Indexer.forward_native = _cpu_indexer_forward_native
        logger.info("GLM5 CPU: DSA Indexer -> dense fallback (None); sparse selection = perf TODO.")
    except Exception as _e:
        logger.warning("GLM5 CPU DSA Indexer fallback not installed: %s", _e)

    _INSTALLED = True
