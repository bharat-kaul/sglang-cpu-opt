"""Plugin-side CPU enablement for the DeepSeek-V4 (DSA) KV-cache / memory-pool layer.

Surfaced by the make-it-work bring-up (see plugin/coverage/deepseek_v4_flash_coverage.yaml
`cpu_infra_gaps`): ``KVCacheConfigurator.configure()`` has a generic CPU-FP8-KV guard that
assumes fp8 KV => MHA + the ``intel_amx`` attention backend, and therefore REJECTS DSV4's
native packed fp8 MLA pool (584 B/token, ``store_dtype=uint8``) served by the ``dsv4``
backend. The guard is correct for generic MHA but wrong for DSV4, which manages its own
pool + accessors. This module overrides ``configure()`` to KEEP the guard for every
non-DSV4 case and SKIP it only for CPU + DSV4, mirroring the upstream body otherwise.

No ``sglang/`` edit — the override is installed as a monkeypatch from the plugin, exactly
the kind of targeted CPU-infra wiring the ``cpu-model-wiring`` skill prescribes. A human can
lift this into an upstream ``configure()`` fix (exempt the DSV4 packed pool) from the diff.
"""

from __future__ import annotations

import logging

import sglang.srt.mem_cache.kv_cache_configurator as kcc
from sglang.srt.platforms import current_platform

logger = logging.getLogger(__name__)

_orig_configure = kcc.KVCacheConfigurator.configure
_INSTALLED = False


# Env-gated per-function wall-time counters (INTEL_CPU_DSV4_TIMEIT=1). Inert otherwise:
# _timed returns the original function unchanged when off, so the normal path is untouched.
import os as _os

_TIMEIT_ON = _os.environ.get("INTEL_CPU_DSV4_TIMEIT", "0") == "1"
_TIMES: dict = {}
# Overhead category per timed op, so a run separates optimized-KERNEL time (AMX) from
# unoptimized-TORCH compute and FRAMEWORK orchestration. "parent" = contains other timed
# leaves (excluded from the category split to avoid double counting).
_TIMES_KIND: dict = {}
# One-time MoE kernel-isolation diagnostic (tokens/expert-shapes/dtype/threads), env-gated.
_MOE_DIAG: dict = {}


def _timed(name: str, kind: str = ""):
    """Decorator that accumulates wall time per call under `name` (no-op when off).

    `kind` in {kernel, torch, framework, parent} categorizes the op for the
    framework-vs-kernel overhead split printed at exit.
    """
    if not _TIMEIT_ON:
        return lambda fn: fn

    import functools
    import time

    def deco(fn):
        _TIMES_KIND[name] = kind

        @functools.wraps(fn)
        def wrap(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                rec = _TIMES.setdefault(name, [0.0, 0])
                rec[0] += time.perf_counter() - t0
                rec[1] += 1

        return wrap

    return deco


def _dump_times() -> None:
    if not _TIMES:
        return
    rows = sorted(_TIMES.items(), key=lambda kv: kv[1][0], reverse=True)
    total = sum(v[0] for _, v in rows)
    lines = [f"[DSV4 TIMEIT] pid={_os.getpid()} total={total:.3f}s over {len(rows)} ops"]
    for name, (t, n) in rows:
        kind = _TIMES_KIND.get(name, "")
        lines.append(f"  {t:8.3f}s  {100*t/total:5.1f}%  n={n:<6d} {name}" + (f"  [{kind}]" if kind else ""))
    # Framework-vs-kernel split over LEAF ops (exclude parents to avoid double counting).
    cat = {}
    for name, (t, _) in rows:
        k = _TIMES_KIND.get(name, "")
        if k and k != "parent":
            cat[k] = cat.get(k, 0.0) + t
    if cat:
        catsum = sum(cat.values())
        lines.append("  -- overhead split (leaf ops) --")
        for k, t in sorted(cat.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"     {t:8.3f}s  {100*t/catsum:5.1f}%  {k}")
    logger.warning("\n".join(lines))


def _tacc(name: str, t0: float, kind: str = "") -> None:
    """Accumulate an elapsed interval under `name` (guard with _TIMEIT_ON at call site)."""
    import time

    if kind and name not in _TIMES_KIND:
        _TIMES_KIND[name] = kind
    rec = _TIMES.setdefault(name, [0.0, 0])
    rec[0] += time.perf_counter() - t0
    rec[1] += 1



if _TIMEIT_ON:
    import atexit as _atexit

    _atexit.register(_dump_times)


def _is_cpu_dsv4(self) -> bool:
    if not current_platform.is_cpu():
        return False
    try:
        return bool(kcc.is_deepseek_v4(self.model_config.hf_config))
    except Exception:
        return False


def _configure_skip_cpu_fp8_guard(self, *, pre_model_load_memory):
    # Mirrors KVCacheConfigurator.configure() MINUS the generic CPU-fp8-KV guard,
    # which does not apply to DSV4's native packed MLA pool. Keep in lockstep with
    # upstream; refresh if configure() changes.
    if not self.spec_algorithm.is_none() and self.is_draft_worker:
        assert self.memory_pool_config is not None, (
            "Draft worker requires memory_pool_config"
        )
        config = self.memory_pool_config
    else:
        config = self._resolve_memory_pool_config(pre_model_load_memory)

    sizes = self._derive_pool_sizes(config=config)

    pools = self._init_pools(
        sizes=sizes,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
    )

    swa_max_total_num_tokens = sizes.swa_max_total_num_tokens
    alloc = pools.token_to_kv_pool_allocator
    if not self.is_draft_worker and kcc.is_swa_req_ring(alloc):
        swa_max_total_num_tokens = alloc.size_swa
        kcc.logger.info(
            "SWA ring: swa_max_total_num_tokens "
            f"{sizes.swa_max_total_num_tokens} -> {swa_max_total_num_tokens} "
            "(fixed per-request SWA ring capacity)."
        )

    kcc.logger.info(
        f"Memory pool end. "
        f"avail mem={kcc.get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )

    return kcc.KVCacheConfigResult(
        max_total_num_tokens=sizes.max_total_num_tokens,
        max_running_requests=sizes.max_running_requests,
        full_max_total_num_tokens=sizes.full_max_total_num_tokens,
        swa_max_total_num_tokens=swa_max_total_num_tokens,
        req_to_token_pool=pools.req_to_token_pool,
        token_to_kv_pool=pools.token_to_kv_pool,
        token_to_kv_pool_allocator=pools.token_to_kv_pool_allocator,
        memory_pool_config=config,
        unified_memory_pool=pools.unified_memory_pool,
    )


def _patched_configure(self, *, pre_model_load_memory):
    if _is_cpu_dsv4(self):
        logger.info(
            "intel_cpu_models: DSV4 on CPU — skipping the generic CPU-fp8-KV guard "
            "(native packed MLA pool + dsv4 backend)."
        )
        return _configure_skip_cpu_fp8_guard(
            self, pre_model_load_memory=pre_model_load_memory
        )
    return _orig_configure(self, pre_model_load_memory=pre_model_load_memory)


def install() -> None:
    """Idempotently install the DSV4 CPU KV-pool configurator patch."""
    global _INSTALLED
    if _INSTALLED:
        return
    kcc.KVCacheConfigurator.configure = _patched_configure
    _install_mla_cpu_tp_config_fix()
    _install_dsv4_rope_cpu_fix()
    _install_cpu_paged_allocator()
    _install_dsa_cpu_kernels()
    _install_mhc_cpu()
    _install_dsv4_core_kernels()
    _install_dsv4_attention_cpu()
    _install_moe_gate_cpu_fix()
    _install_moe_hash_cpu_fix()
    _install_fp4_expert_cpu_dequant()
    _install_dsa_cpu_wire()
    _install_dsa_profile_bypass()
    _INSTALLED = True
    logger.info("intel_cpu_models: installed CPU DSV4 KV-pool configurator patch.")


def _install_dsv4_rope_cpu_fix() -> None:
    """Let pure-SWA (compress_ratio=0) layers build their RoPE on CPU.

    DeepSeek-V4 selects RoPE per layer: C4/C128 layers use the compressed YaRN
    RoPE (rope_scaling set), while pure-SWA layers (compress_ratio=0) use the
    main *unscaled* RoPE and pass rope_scaling=None. The CPU factory
    (get_rope_cpu) only implements deepseek_yarn and asserts
    ``rope_scaling is not None``, so the unscaled layers fail at build time.
    Flash trips this immediately (layers 0-1 are compress_ratio=0); Pro's first
    C128 layer hid it. Route the unscaled CPU case to the standard ``get_rope``,
    whose plain RotaryEmbedding runs natively (forward_native) on CPU.
    """
    from sglang.srt.layers.rotary_embedding import factory as _rope_factory

    _orig_wrapper = _rope_factory.get_rope_wrapper

    def _patched_get_rope_wrapper(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style=True,
        rope_scaling=None,
        dtype=None,
        partial_rotary_factor=1.0,
        device=None,
    ):
        if device == "cpu" and rope_scaling is None:
            # Unscaled (pure-SWA) RoPE: the standard factory returns a plain
            # RotaryEmbedding that dispatches to forward_native on CPU.
            return _rope_factory.get_rope(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                None,
                dtype,
                partial_rotary_factor,
            )
        return _orig_wrapper(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
            device,
        )

    _rope_factory.get_rope_wrapper = _patched_get_rope_wrapper
    # deepseek_v4 imported the symbol by value; repoint that reference too.
    try:
        from sglang.srt.models import deepseek_v4 as _dsv4

        if getattr(_dsv4, "get_rope_wrapper", None) is not None:
            _dsv4.get_rope_wrapper = _patched_get_rope_wrapper
    except Exception:
        pass


def _cpu_fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions):
    # Torch port of dsv4 fused_q_norm_rope (elementwise.py). Reference = the AOT test
    # oracle (_ref_rmsnorm_self + _ref_rope_interleaved): RMSNorm-self over head_dim
    # (no weight), then interleaved RoPE on the LAST rope_dim, writing into q_output.
    import torch

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # [max_pos, rope_dim]
    rope_dim = freqs_real.shape[-1]
    head_dim = q_input.shape[-1]
    nope = head_dim - rope_dim

    x = q_input.float()
    xn = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if q_input.shape[0] == 0:
        q_output.copy_(xn.to(q_output.dtype))
        return

    freq = freqs_real[positions.long()].reshape(-1, rope_dim // 2, 2)  # [T, r/2, 2]
    f_real = freq[..., 0].unsqueeze(1)  # [T, 1, r/2]
    f_imag = freq[..., 1].unsqueeze(1)
    rope_part = xn[..., nope:]
    pairs = rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
    xr, xi = pairs[..., 0], pairs[..., 1]
    rot = torch.stack([xr * f_real - xi * f_imag, xr * f_imag + xi * f_real], dim=-1)
    out = xn.clone()
    out[..., nope:] = rot.reshape(rope_part.shape)
    q_output.copy_(out.to(q_output.dtype))


def _cpu_fused_rope_inplace(q, k, freqs_cis, positions, inverse=False):
    # Interleaved RoPE applied IN PLACE to full [B, heads, rope_dim] q/k tensors.
    import torch

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    freq = freqs_real[positions.long()].reshape(-1, rope_dim // 2, 2)
    fr = freq[..., 0].unsqueeze(1)
    fi = freq[..., 1].unsqueeze(1)
    if inverse:
        fi = -fi
    for t in (q, k):
        if t is None:
            continue
        pairs = t.float().reshape(t.shape[0], t.shape[1], rope_dim // 2, 2)
        xr, xi = pairs[..., 0], pairs[..., 1]
        rot = torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1)
        t.copy_(rot.reshape(t.shape).to(t.dtype))


def _install_dsv4_core_kernels() -> None:
    # Reference-first torch ports of the dsv4 MLA-core fused kernels the MODEL FORWARD
    # calls directly (CUDA-JIT, no CPU path). Each mirrors the in-tree AOT/test oracle.
    if not current_platform.is_cpu():
        return
    import sglang.srt.models.deepseek_v4 as _dv4

    _dv4.fused_q_norm_rope = _cpu_fused_q_norm_rope
    _dv4.fused_rope_inplace = _cpu_fused_rope_inplace
    _dv4._FP8_WO_A_GEMM = False  # route o_proj (wo_a) to the standard bf16 path on CPU

    # K path: fused norm+rope + fp8-pack + paged scatter-write. The paged writer already
    # has an in-tree torch impl (index_buf_accessor._set_k_and_s_torch); the pack math is
    # a clean torch port (per-64-tile max-abs -> pow2 ue8m0 scale -> fp8; rope kept bf16).
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as _pool

    _pool.fused_k_norm_rope_flashmla = _cpu_fused_k_norm_rope_flashmla
    import sglang.kernels.ops.attention.dsv4.index_buf_accessor as _iba

    _iba.SetKAndS.execute = _iba.SetKAndS.torch  # route the paged KV write to torch on CPU
    logger.info("intel_cpu_models: installed CPU dsv4 fused q/k norm_rope + KV write (torch ref).")


def _cpu_quant_to_nope_fp8_rope_bf16_pack(k_bf16):
    # Torch port of quant_to_nope_fp8_rope_bf16_pack_triton (quant_k_cache.py): split
    # [T,512] into nope(448)+rope(64); per-64-tile pow2(ue8m0) fp8 quant of nope; rope bf16.
    import torch

    from sglang.kernels.ops.attention.dsv4.index_buf_accessor import (
        NopeFp8RopeBf16Pack,
        fp8_dtype,
    )

    T = k_bf16.shape[0]
    dim_nope, dim_rope, tile = 448, 64, 64
    num_tiles = dim_nope // tile
    fi = torch.finfo(fp8_dtype)
    nope = k_bf16[:, :dim_nope].float().reshape(T, num_tiles, tile)
    rope = k_bf16[:, dim_nope:].to(torch.bfloat16).contiguous()
    max_abs = nope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    ceil_log2 = torch.ceil(torch.log2(max_abs / fi.max))
    x_fp8 = (nope / torch.exp2(ceil_log2)).clamp(fi.min, fi.max).to(fp8_dtype)
    return NopeFp8RopeBf16Pack(
        k_nope_fp8=x_fp8.reshape(T, dim_nope).contiguous(),
        k_rope_bf16=rope,
        scale_k_nope_ue8m0=(ceil_log2.to(torch.int32).reshape(T, num_tiles) + 127)
        .to(torch.uint8)
        .contiguous(),
    )


def _cpu_fused_k_norm_rope_flashmla(
    kv, kv_weight, eps, freqs_cis, positions, out_loc, kvcache, page_size
):
    # RMSNorm(nope, kv_weight) + interleaved RoPE(last rope_dim) -> fp8 pack -> paged write.
    import torch

    import sglang.kernels.ops.attention.dsv4.index_buf_accessor as _iba

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    nope_dim = kv.shape[-1] - rope_dim
    x = kv.float()
    # RMSNorm over the full head_dim (kv_weight spans nope+rope), then RoPE the last rope_dim.
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * kv_weight.float()
    normed = xn[:, :nope_dim]
    if kv.shape[0] > 0:
        freq = freqs_real[positions.long()].reshape(-1, rope_dim // 2, 2)
        rp = xn[:, nope_dim:].reshape(-1, rope_dim // 2, 2)
        xr, xi = rp[..., 0], rp[..., 1]
        fr, fi = freq[..., 0], freq[..., 1]
        roped = torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1).reshape(
            -1, rope_dim
        )
    else:
        roped = xn[:, nope_dim:]
    kv_out = torch.cat([normed, roped], dim=-1).to(torch.bfloat16)
    # Stash the exact dense bf16 keys (pre-pack) so the CPU torch MLA attention reads
    # them directly instead of unpacking the paged fp8 layout (numerically exact).
    stash = _KV_STASH.setdefault(kvcache.data_ptr(), {})
    for i, l in enumerate(out_loc.tolist()):
        if l >= 0:
            stash[int(l)] = kv_out[i].detach()
    _iba._set_k_and_s_torch(
        kvcache, out_loc, _cpu_quant_to_nope_fp8_rope_bf16_pack(kv_out), page_size
    )


# Dense bf16 KV stash keyed by (paged-buffer data_ptr -> {loc: key[512]}). Populated by
# the CPU K-write; read by the CPU torch MLA attention below.
_KV_STASH: dict = {}

# Route-2 DSA: per-layer compressed-KV stash keyed by (layer_id, compress_ratio) -> the
# RMSNorm'd compressed KV [N_compressed, head_dim]. Written by the CPU compressor forward,
# read by the CPU indexer (step 3). Avoids the paged state-pool entirely.
_COMPRESSED_STASH: dict = {}

# Route-2 DSA step 3: per-layer indexer top-k KV indices [N_q, k], written by the CPU
# forward_c4_indexer, read by the CPU sparse attention (step 4).
_TOPK_STASH: dict = {}
_HAD_CACHE: dict = {}

# Route-2 DSA step 4: the indexer publishes a per-query position-keep mask [N, S*ratio]
# here; the very next MLA attention (same layer) consumes AND clears it. A non-indexer layer
# leaves it None -> that layer's attention stays dense. Handles alternating c4/c128 layers.
_SEL_HOLDER: dict = {"pos_kept": None}


def _rmsnorm_torch(x, norm):
    # Torch RMSNorm (avoids the sgl_kernel CPU RMSNorm's strict input==weight dtype check).
    import torch

    w = norm.weight
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    xf = x.float()
    xf = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * w.float()).to(w.dtype)


def _walsh_hadamard(n: int, device) -> "torch.Tensor":
    # Normalized Walsh-Hadamard matrix (recursive, /sqrt(2) per doubling); mirrors the NPU
    # backend's _walsh_hadamard_matrix. _apply = plain matmul.
    import math

    import torch

    key = (n, str(device))
    cached = _HAD_CACHE.get(key)
    if cached is not None:
        return cached
    had = torch.ones(1, 1, dtype=torch.float32, device=device)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1), torch.cat([had, -had], 1)), 0) / math.sqrt(2)
    _HAD_CACHE[key] = had.contiguous()
    return _HAD_CACHE[key]


def _torch_flash_mla_with_kvcache(
    q,
    k_cache,
    head_dim_v,
    block_table=None,
    cache_seqlens=None,
    tile_scheduler_metadata=None,
    softmax_scale=None,
    is_fp8_kvcache=True,
    indices=None,
    topk_length=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    **_,
):
    # MLA-absorbed attention on CPU: value == key; out = softmax(q.kT.scale (+) sink).k.
    import time

    import torch

    T, _, H, D = q.shape
    out = torch.zeros(T, H, head_dim_v, dtype=torch.float32)
    swa = _KV_STASH.get(k_cache.data_ptr(), {})
    extra = _KV_STASH.get(extra_k_cache.data_ptr(), {}) if extra_k_cache is not None else {}
    # DSA step 4: consume+clear the indexer's per-query position-keep mask (None on a
    # non-indexer layer -> dense). A position beyond the compressed coverage is the recent
    # tail (always attended). locs are in sequence-position order (index p == position p).
    pos_kept = _SEL_HOLDER.get("pos_kept")
    _SEL_HOLDER["pos_kept"] = None
    for t in range(T):
        _t0 = time.perf_counter() if _TIMEIT_ON else 0.0
        locs = []
        if indices is not None and topk_length is not None:
            L = int(topk_length[t])
            locs = [int(x) for x in indices[t].reshape(-1)[:L].tolist() if x >= 0]
        if pos_kept is not None and t < pos_kept.shape[0]:
            cov = int(pos_kept.shape[1])
            row = pos_kept[t]
            locs = [l for p, l in enumerate(locs) if p >= cov or bool(row[p])]
        if _TIMEIT_ON:
            _tacc("dsa.mla.select", _t0, "framework")
            _t0 = time.perf_counter()
        keys = [swa[l] for l in locs if l in swa]
        if extra_indices_in_kvcache is not None and extra_topk_length is not None:
            EL = int(extra_topk_length[t])
            elocs = [int(x) for x in extra_indices_in_kvcache[t].reshape(-1)[:EL].tolist() if x >= 0]
            keys += [extra[l] for l in elocs if l in extra]
        if not keys:
            continue
        K = torch.stack(keys).float()  # [Kk, 512]
        if _TIMEIT_ON:
            _tacc("dsa.mla.gather", _t0, "framework")
            _t0 = time.perf_counter()
        s = (q[t, 0].float() @ K[:, :D].t()) * softmax_scale  # [H, Kk]
        if attn_sink is not None:
            s = torch.cat([s, attn_sink.reshape(-1, 1).float()], dim=-1)
            p = s.softmax(dim=-1)[:, :-1]
        else:
            p = s.softmax(dim=-1)
        out[t] = p @ K[:, :head_dim_v]
        if _TIMEIT_ON:
            _tacc("dsa.mla.attend", _t0, "torch")
    return (out.unsqueeze(1).to(q.dtype),)


def _install_moe_gate_cpu_fix() -> None:
    # Dummy-weight artifact: PackWeightMethod transposes the MoE gate weight to
    # [hidden, experts], but MoEGate.forward's float32 fast-path calls F.linear without
    # accounting for it (real bf16/fp8 weights take the AMX path instead). Transpose when
    # we detect that packed [hidden, experts] float32 layout on CPU.
    if not current_platform.is_cpu():
        return
    import torch
    import torch.nn.functional as F

    import sglang.srt.models.deepseek_v2 as _dv2

    _orig = _dv2.MoEGate.forward

    def _fwd(self, hidden_states, *a, **k):
        w = self.weight
        if (
            w.dtype == torch.float32
            and w.dim() == 2
            and w.shape[0] == hidden_states.shape[-1]
            and w.shape[1] != hidden_states.shape[-1]
        ):
            return F.linear(hidden_states.float(), w.t().contiguous())
        return _orig(self, hidden_states, *a, **k)

    _dv2.MoEGate.forward = _fwd


def _install_mla_cpu_tp_config_fix() -> None:
    # CPU TP: adjust_config_with_unaligned_cpu_tp() fires when total_kv_heads % tp != 0.
    # MLA has a single latent KV head (num_key_value_heads==1) that is replicated, not
    # TP-sharded, so 1 % tp != 0 always triggers it, and the generic GQA-oriented kv-head
    # padding rewrites num_key_value_heads>1 (e.g. 1->6) + num_attention_heads, violating
    # DeepseekV4's `num_key_value_heads == 1` invariant. For MLA, force the kv-head pad
    # size to 1 so the latent count (and derived query-head count) is preserved.
    if not current_platform.is_cpu():
        return
    import sglang.srt.configs.update_config as _uc

    if getattr(_uc, "_mla_cpu_tp_patched", False):
        return
    _orig_adjust = _uc.adjust_config_with_unaligned_cpu_tp

    def _patched_adjust(model_config, load_config, tp_size):
        hf = model_config.hf_config
        is_mla = (
            getattr(hf, "qk_rope_head_dim", None) is not None
            and model_config.get_total_num_kv_heads() == 1
        )
        if not is_mla:
            return _orig_adjust(model_config, load_config, tp_size)
        _orig_pad = _uc.get_num_heads_padding_size
        _uc.get_num_heads_padding_size = lambda *a, **k: 1
        try:
            return _orig_adjust(model_config, load_config, tp_size)
        finally:
            _uc.get_num_heads_padding_size = _orig_pad

    _uc.adjust_config_with_unaligned_cpu_tp = _patched_adjust
    # model_runner imported the symbol directly; repoint that reference too.
    try:
        import sglang.srt.model_executor.model_runner as _mr

        _mr.adjust_config_with_unaligned_cpu_tp = _patched_adjust
    except Exception:
        pass
    _uc._mla_cpu_tp_patched = True
    logger.info("intel_cpu_models: MLA CPU-TP config fix (preserve num_key_value_heads==1).")


def _install_moe_hash_cpu_fix() -> None:
    # Hash-routed layers (num_hash_layers>0) use HashTopK, whose forward needs the
    # per-token input_ids to look up tid2eid. The CPU MoE path (DeepseekV2MoE.forward_cpu)
    # only receives hidden_states and calls self.topk(hidden_states, router_logits) with no
    # input_ids -> TypeError. Thread input_ids to HashTopK on CPU by stashing it in forward
    # and binding it into the topk call within the original forward_cpu.
    if not current_platform.is_cpu():
        return
    import os

    # HashTopK's fused path JIT-compiles a CUDA kernel (nvcc); force the torch fallback.
    os.environ["SGLANG_OPT_USE_FUSED_HASH_TOPK"] = "0"
    import sglang.srt.models.deepseek_v2 as _dv2

    MoE = _dv2.DeepseekV2MoE
    if getattr(MoE, "_cpu_hash_patched", False):
        return
    _orig_forward = MoE.forward
    _orig_forward_cpu = MoE.forward_cpu

    def _patched_forward(
        self,
        hidden_states,
        forward_batch=None,
        gemm_output_zero_allocator=None,
        input_ids=None,
        input_ids_global=None,
        skip_shared_experts=False,
    ):
        # HashTopK indexes tid2eid with the global token ids (see forward_normal).
        self._cpu_hash_input_ids = (
            input_ids_global if input_ids_global is not None else input_ids
        )
        return _orig_forward(
            self,
            hidden_states,
            forward_batch,
            gemm_output_zero_allocator,
            input_ids,
            input_ids_global,
            skip_shared_experts,
        )

    def _patched_forward_cpu(self, hidden_states):
        if not getattr(self, "is_hash", False):
            return _orig_forward_cpu(self, hidden_states)
        _ids = getattr(self, "_cpu_hash_input_ids", None)
        _orig_topk_forward = self.topk.forward

        def _bound(hs, rl, *a, **k):
            k.setdefault("input_ids", _ids)
            return _orig_topk_forward(hs, rl, *a, **k)

        self.topk.forward = _bound
        try:
            return _orig_forward_cpu(self, hidden_states)
        finally:
            self.topk.forward = _orig_topk_forward

    MoE.forward = _timed("moe.forward", "parent")(_patched_forward)
    MoE.forward_cpu = _timed("moe.forward_cpu", "parent")(_patched_forward_cpu)
    MoE._cpu_hash_patched = True
    logger.info("intel_cpu_models: threaded input_ids to HashTopK on CPU (hash MoE layers).")


def _dequant_fp4_to_bf16(w_int8, scale):
    # fp4(e2m1) packed int8 [O, I//2] + block scale -> bf16 [O, I]. Reuse SGLang's lossless
    # fp4->fp8 cast (128-block ue8m0 scale), then fp8 * 2^scale per 128-block -> bf16.
    import torch

    import sglang.srt.layers.quantization.fp8 as _fp8

    w_fp8, s = _fp8.cast_e2m1fn_to_e4m3fn(w_int8, scale)  # fp8 [O,I], e8m0 [O//128, I//128]
    o, i = w_fp8.shape
    wf = w_fp8.float().reshape(o // 128, 128, i // 128, 128)
    sf = s.float().reshape(o // 128, 1, i // 128, 1)
    return (wf * sf).reshape(o, i).to(torch.bfloat16)


def _torch_fp4_moe_apply(layer, x, topk_weights, topk_ids):
    # BF16 showcase MoE: keep experts fp4 in RAM, dequant each ROUTED expert fp4->bf16 once,
    # then bf16 matmul (SwiGLU). Fits the node (fp4 storage); bf16 compute = accuracy-safe.
    import torch
    import torch.nn.functional as F

    T, hidden = x.shape
    out = torch.zeros(T, hidden, dtype=torch.float32)
    xf = x.float()
    for e in topk_ids.unique().tolist():
        if e < 0 or e >= layer.w13_weight.shape[0]:
            continue
        mask = topk_ids == e
        tok, slot = mask.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13 = _dequant_fp4_to_bf16(layer.w13_weight[e], layer.w13_weight_scale_inv[e]).float()
        w2 = _dequant_fp4_to_bf16(layer.w2_weight[e], layer.w2_weight_scale_inv[e]).float()
        gate_up = xf[tok] @ w13.t()  # [n, 2*inter]
        g, u = gate_up.chunk(2, dim=-1)
        act = F.silu(g) * u
        oe = act @ w2.t()  # [n, hidden]
        out.index_add_(0, tok, oe * topk_weights[tok, slot].float().unsqueeze(-1))
    return out.to(x.dtype)


def _install_fp4_expert_cpu_dequant() -> None:
    # Plan A: GNR AMX has no 4-bit matmul, and Fp8MoEMethod's fp4->fp8 dequant lives in the
    # NON-CPU branch of process_weights_after_loading; the _is_cpu branch hands fp4-packed
    # weights straight to AMX prepack -> fused_experts_cpu CHECK_EQ(packed_w1, packed_K) fails
    # (fp4 K/2 packing). Dequant fp4->fp8 (lossless, SGLang's cast_e2m1fn_to_e4m3fn) on CPU
    # BEFORE the prepack, so the existing fp8 W8A16 CPU path (which itself dequants fp8->bf16
    # for AMX) runs. Accuracy-parity: fp8 represents the fp4 levels exactly.
    if not current_platform.is_cpu():
        return
    import torch

    import sglang.srt.layers.quantization.fp8 as _fp8

    Method = _fp8.Fp8MoEMethod
    if getattr(Method, "_cpu_fp4_dequant_patched", False):
        return
    import os

    # BF16 showcase (large models that don't fit fp8): keep experts fp4 in RAM, dequant to bf16
    # in the MoE forward. fp8 load-dequant doubles experts (~1.58TB > node); fp4 raw ~790GB fits.
    _bf16_moe = os.environ.get("INTEL_CPU_DSV4_FP4_MOE_BF16", "0") == "1"
    _orig_pwal = Method.process_weights_after_loading
    _orig_apply = Method.apply

    def _patched_pwal(self, layer):
        if getattr(self, "is_fp4_expert", False) and _bf16_moe:
            # Keep fp4 raw (skip fp8 dequant + AMX prepack); the apply dequants per expert.
            layer._fp4_bf16_moe = True
            logger.info("intel_cpu_models: FP4 experts kept raw for BF16-in-forward MoE.")
            return
        if getattr(self, "is_fp4_expert", False):
            for weight_param, scale_param in [
                (layer.w13_weight, layer.w13_weight_scale_inv),
                (layer.w2_weight, layer.w2_weight_scale_inv),
            ]:
                new_w, new_s = [], []
                for e in range(weight_param.shape[0]):
                    w, s = _fp8.cast_e2m1fn_to_e4m3fn(
                        weight_param.data[e], scale_param.data[e]
                    )
                    new_w.append(w)
                    new_s.append(s)
                weight_param.data = torch.stack(new_w)
                scale_param.data = torch.stack(new_s).float()
                scale_param.format_ue8m0 = False
            self.is_fp4_expert = False
            logger.info("intel_cpu_models: dequantized FP4 experts -> FP8 on CPU (Plan A).")
        return _orig_pwal(self, layer)

    def _patched_apply(self, layer, dispatch_output):
        if _TIMEIT_ON and not _MOE_DIAG.get("done"):
            _MOE_DIAG["done"] = True
            try:
                import torch as _t

                x = dispatch_output.hidden_states
                _tw, ti, _ = dispatch_output.topk_output
                w13 = getattr(layer, "w13_weight", None)
                w2 = getattr(layer, "w2_weight", None)
                logger.warning(
                    "[MOE DIAG] tokens=%s xdtype=%s topk_ids=%s w13=%s/%s w2=%s/%s "
                    "threads=%d amx=%s",
                    tuple(x.shape),
                    x.dtype,
                    tuple(ti.shape),
                    tuple(w13.shape) if w13 is not None else None,
                    getattr(w13, "dtype", None),
                    tuple(w2.shape) if w2 is not None else None,
                    getattr(w2, "dtype", None),
                    _t.get_num_threads(),
                    use_intel_amx_backend(layer) if "use_intel_amx_backend" in globals() else "?",
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning("[MOE DIAG] failed: %s", _e)
        if _os.environ.get("INTEL_CPU_DSV4_MOE_ISO", "0") == "1" and not _MOE_DIAG.get("iso"):
            _MOE_DIAG["iso"] = True
            # In-situ isolation with the REAL packed weights: re-time the real expert apply
            # across thread counts. Flat -> kernel/shape bound (e.g. all-256-expert stream);
            # scaling -> thread/binding bound. Decides the runtime-config vs kernel question.
            try:
                import time as _t

                import torch as _tt

                _orig_nt = _tt.get_num_threads()
                for _nth in sorted({_orig_nt, _orig_nt * 2, 16, 8}):
                    if _nth < 1:
                        continue
                    _tt.set_num_threads(_nth)
                    _r = _orig_apply(self, layer, dispatch_output)  # warm
                    _t0 = _t.perf_counter()
                    for _ in range(3):
                        _r = _orig_apply(self, layer, dispatch_output)
                    logger.warning(
                        "[MOE ISO] threads=%d apply=%.1fms", _nth, (_t.perf_counter() - _t0) / 3 * 1e3
                    )
                _tt.set_num_threads(_orig_nt)
            except Exception as _e:  # noqa: BLE001
                logger.warning("[MOE ISO] failed: %s", _e)
        if getattr(layer, "_fp4_bf16_moe", False):
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            out = _torch_fp4_moe_apply(
                layer, dispatch_output.hidden_states, topk_weights, topk_ids
            )
            return StandardCombineInput(hidden_states=out)
        return _orig_apply(self, layer, dispatch_output)

    Method.process_weights_after_loading = _patched_pwal
    Method.apply = _timed("moe.expert_apply", "kernel")(_patched_apply)
    Method._cpu_fp4_dequant_patched = True
    logger.info("intel_cpu_models: installed CPU FP4 expert handler (Plan A fp8 / BF16-in-forward).")


def _install_dsv4_attention_cpu() -> None:
    # Replace the CUDA flash-MLA serving kernel with the CPU torch MLA attention
    # (reads the dense KV stash). The backend forward imports these at call time.
    import sys
    import types

    if not current_platform.is_cpu():
        return
    # Force the non-sparse flash-MLA path on CPU (the sparse-prefill path uses Triton
    # build_swa_token_ids); our torch attention handles the SWA + extra gather itself.
    import os

    os.environ.setdefault("SGLANG_OPT_FLASHMLA_SPARSE_PREFILL", "0")
    try:
        import sglang.srt.layers.attention.deepseek_v4_backend as _b

        _b._LARGE_INDEXER_QUERY_THRESHOLD = 10**9
    except Exception:
        pass
    try:
        import sgl_kernel.flash_mla as _fm
    except Exception:
        _fm = types.ModuleType("sgl_kernel.flash_mla")
        sys.modules["sgl_kernel.flash_mla"] = _fm
    _fm.flash_mla_with_kvcache = _timed("dsa.mla_attention", "parent")(_torch_flash_mla_with_kvcache)
    logger.info("intel_cpu_models: installed CPU torch flash-MLA attention (dense KV stash).")


def _install_mhc_cpu() -> None:
    # MHC (hash-clustering) sub-ops hc_split_sinkhorn + hc_combine are TileLang-only, but
    # in-tree TORCH references exist (_hc_split_sinkhorn_torch; combine is a trivial
    # weighted sum). Route CPU to them and select the torch hc_pre path. This is proper
    # enablement (numerically equivalent reference), not a bypass.
    import os

    if not current_platform.is_cpu():
        return
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
    import sglang.kernels.ops.layernorm.mhc as _mhc

    _mhc.hc_split_sinkhorn = _mhc._hc_split_sinkhorn_torch

    def _cpu_hc_combine(x_flat, pre, hc, out_dtype):
        # y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]
        import torch  # noqa: F401

        m = x_flat.shape[0]
        h = x_flat.shape[1] // hc
        xr = x_flat.reshape(m, hc, h).float()
        return (pre.float().unsqueeze(-1) * xr).sum(dim=1).to(out_dtype)

    _mhc.hc_combine = _cpu_hc_combine
    if hasattr(_mhc, "_mhc_post_torch"):
        # hc_post passes raw post [s,n]; _mhc_post_torch wants [s,n,1].
        _mhc.mhc_post = lambda x, residual, post, comb: _mhc._mhc_post_torch(
            x, residual, post.unsqueeze(-1), comb
        )
    try:
        import sglang.srt.models.deepseek_v4 as _dv4

        _dv4._get_mhc_ops.cache_clear()
    except Exception:
        pass
    # hc_head (final MHC head combine) local-imports a Triton fused_hc_head; route it to the
    # in-tree torch hc_head_torch on CPU (same signature).
    try:
        import sglang.kernels.ops.layernorm.mhc_head as _mhch
        import sglang.srt.models.deepseek_v4 as _dv4h

        _mhch.fused_hc_head = _dv4h.hc_head_torch
    except Exception:
        pass
    logger.info("intel_cpu_models: routed MHC sub-ops (sinkhorn, combine) to torch on CPU.")


def _cpu_forward_core_compressor(self, x, forward_batch, layer_id, compressor) -> None:
    # Route-2 CPU compressor: compute compressed KV directly over the sequence (softmax-pool
    # over windows of `ratio` tokens), RMSNorm, and stash it per (layer_id, ratio) for the CPU
    # indexer — bypassing the paged plan/state-pool. The compression math = compressor's own
    # softmax-pool (mirrors _compress_forward_c128_fallback): w=softmax(score+ape), out=sum(w*kv).
    import torch

    from intel_cpu_models.dsa_compressor_cpu import compress_softmax_pool

    if forward_batch.forward_mode.is_idle():
        return
    ratio = int(compressor.ratio)
    head_dim = int(compressor.head_dim)
    kv_score = compressor.compute_kv_score(x, forward_batch)  # [N, 2*head_dim] (kv | score)
    kv_all = kv_score[:, :head_dim]
    score_all = kv_score[:, head_dim : 2 * head_dim]
    ape = compressor.ape.view(-1, head_dim)[:ratio]  # [ratio, head_dim]

    ext = forward_batch.extend_seq_lens_cpu
    if ext is None:
        # Decode / no-extend: 1 new token per request; a window only completes every `ratio`
        # steps. Online incremental compression is a later increment — stash empty for now.
        _COMPRESSED_STASH[(layer_id, ratio)] = kv_all.new_zeros(0, head_dim)
        return

    comp_list = []
    off = 0
    for L in ext:
        L = int(L)
        nwin = L // ratio
        if nwin > 0:
            n = nwin * ratio
            kv_w = kv_all[off : off + n].reshape(nwin, ratio, head_dim)
            sc_w = score_all[off : off + n].reshape(nwin, ratio, head_dim)
            comp_list.append(compress_softmax_pool(kv_w, sc_w, ape))
        off += L
    if comp_list:
        compressed = torch.cat(comp_list, 0).to(x.dtype)
        compressed = _rmsnorm_torch(compressed, compressor.norm)  # rotate=False for c4/c128 compressor
    else:
        compressed = kv_all.new_zeros(0, head_dim)
    _COMPRESSED_STASH[(layer_id, ratio)] = compressed


def _cpu_forward_c4_indexer(self, x, q_lora, forward_batch, c4_indexer, *args, **kwargs) -> None:
    # Route-2 CPU indexer (step 3): replace forward_c4_indexer wholesale (the paged
    # fp8_paged_mqa_logits reads the paged compressed KV route-2 avoids). Mirrors the NPU torch
    # indexer + compute_q spec (rope on trailing 64 + 128-pt Hadamard, skip fp8) + my
    # dsa_indexer_cpu logits/topk over a freshly-compressed index-KV. Stashes top-k for step 4.
    # NOTE: structural (runs); exact causal/rope-phase/hadamard-norm correctness is validated on
    # REAL weights vs the dsa_attention.py oracle (dummy weights can't check accuracy).
    import torch

    from intel_cpu_models.dsa_compressor_cpu import compress_softmax_pool

    if forward_batch.forward_mode.is_idle() or x.shape[0] == 0:
        return
    positions = forward_batch.positions
    nh = int(c4_indexer.n_heads)
    hd = int(c4_indexer.head_dim)  # 128
    rope_dim = int(c4_indexer.rope_head_dim)  # 64

    # (1) compute_q: wq_b(q_lora) -> rope(trailing rope_dim) -> 128-pt Hadamard (bf16, skip fp8).
    q, _ = c4_indexer.wq_b(q_lora)
    q = q.view(-1, nh, hd)
    rp = q[..., hd - rope_dim :].contiguous()
    _cpu_fused_rope_inplace(rp, None, c4_indexer.freqs_cis, positions)
    q[..., hd - rope_dim :] = rp
    q = torch.matmul(q.float().reshape(-1, hd), _walsh_hadamard(hd, q.device)).reshape(-1, nh, hd)

    # (2) weights = weights_proj(x) * weight_scale.
    w, _ = c4_indexer.weights_proj(x)
    w = w.float() * float(c4_indexer.weight_scale)  # [N, nh]

    # (3) indexer compressed index-KV (own compressor, ratio 4): softmax-pool + RMSNorm.
    comp = c4_indexer.compressor
    ratio = int(comp.ratio)
    chd = int(comp.head_dim)
    kv_score = comp.compute_kv_score(x, forward_batch)  # [N, 2*chd]
    kv_all = kv_score[:, :chd]
    score_all = kv_score[:, chd : 2 * chd]
    ape = comp.ape.view(-1, chd)[:ratio]
    ext = forward_batch.extend_seq_lens_cpu or [int(x.shape[0])]
    comp_list, key_start_list, off = [], [], 0
    for L in ext:
        L = int(L)
        nwin = L // ratio
        if nwin > 0:
            n = nwin * ratio
            kv_w = kv_all[off : off + n].reshape(nwin, ratio, chd)
            sc_w = score_all[off : off + n].reshape(nwin, ratio, chd)
            comp_list.append(compress_softmax_pool(kv_w, sc_w, ape))
            key_start_list.append(torch.arange(nwin, device=x.device) * ratio)
        off += L
    if not comp_list:
        _TOPK_STASH[c4_indexer.layer_id] = q.new_zeros(q.shape[0], 0, dtype=torch.long)
        return
    ck = _rmsnorm_torch(torch.cat(comp_list, 0).to(x.dtype), comp.norm).float()  # [S, chd]
    key_start = torch.cat(key_start_list, 0)  # [S] window start position

    # (4) logits (relu, per-head weight, sum) + causal mask + top-k.
    scores = torch.einsum("nhd,sd->nhs", q, ck)  # [N, nh, S]
    logits = (torch.relu(scores) * w.unsqueeze(-1)).sum(dim=1)  # [N, S]
    causal = key_start.unsqueeze(0) <= positions.reshape(-1, 1)  # query sees only past windows
    logits = logits.masked_fill(~causal, float("-inf"))
    ktop = min(int(c4_indexer.index_topk), logits.shape[1])
    topk = torch.topk(logits, ktop, dim=1).indices  # [N, ktop] compressed-block indices
    _TOPK_STASH[c4_indexer.layer_id] = topk
    # Publish the per-query position-keep mask for step-4 sparse attention: block j -> original
    # positions [j*ratio, (j+1)*ratio). Consumed+cleared by the next MLA attention (same layer).
    kept = torch.zeros(logits.shape[0], ck.shape[0], dtype=torch.bool, device=logits.device)
    kept.scatter_(1, topk.clamp(min=0), True)
    _SEL_HOLDER["pos_kept"] = kept.repeat_interleave(ratio, dim=1)  # [N, S*ratio]


def _install_dsa_cpu_wire() -> None:
    # Route 2 (correctness-first DSA on CPU): compute the sparse selection over the dense KV
    # stash with the validated dsa_*_cpu kernels, AVOIDING the paged compressor plan/state-pool
    # (device-only: CUDA-JIT plan_prefill / XPU / Triton, no CPU variant). This scaffold is the
    # step-by-step ladder; opt-in via INTEL_CPU_DSV4_DSA_CPU=1 (distinct from the dense bypass).
    # Step 1: make the compressor PLAN CPU-safe (the paged plan crashes on pin_memory + nvcc JIT).
    import os

    if not current_platform.is_cpu():
        return
    if os.environ.get("INTEL_CPU_DSV4_DSA_CPU", "0") != "1":
        return
    try:
        import sglang.srt.layers.attention.deepseek_v4_backend as _dsv4b
        import sglang.srt.layers.attention.dsv4.compressor_v2 as _cv2

        def _cpu_no_paged_plan(*a, **k):
            # No paged plan on CPU; route-2 computes compression over the stash instead.
            return None

        _cv2.create_paged_compressor_data = _cpu_no_paged_plan
        _dsv4b.create_paged_compressor_data = _cpu_no_paged_plan
        _dsv4b.DeepseekV4AttnBackend.forward_core_compressor = _timed("dsa.compressor", "torch")(
            _cpu_forward_core_compressor
        )
        _dsv4b.DeepseekV4AttnBackend.forward_c4_indexer = _timed("dsa.indexer", "torch")(
            _cpu_forward_c4_indexer
        )
    except Exception:
        pass
    logger.warning(
        "intel_cpu_models: DSA CPU wire (route 2) scaffold active (INTEL_CPU_DSV4_DSA_CPU=1) — "
        "paged compressor plan disabled; sparse chain wiring in progress."
    )


def _install_dsa_profile_bypass() -> None:
    # OPT-IN profiling bypass: force need_compress=False so the DSV4 backend skips the
    # not-yet-ported DSA compressor + indexer and runs pure MLA+MoE+dense. This makes the
    # model RUN end-to-end for model-profile-hotspots BEFORE the DSA kernels are authored.
    # It disables sparse attention, so it is STRUCTURAL/PERF ONLY and NOT numerically
    # correct — never use for accuracy. Off unless INTEL_CPU_DSV4_BYPASS_DSA=1.
    import os

    if not current_platform.is_cpu():
        return
    if os.environ.get("INTEL_CPU_DSV4_BYPASS_DSA", "0") != "1":
        return
    import sglang.srt.layers.attention.deepseek_v4_backend as _dsv4b

    _orig_prefill_md = _dsv4b.DeepseekV4AttnBackend.init_forward_metadata_prefill

    def _prefill_md_no_compress(self, *args, **kwargs):
        kwargs["need_compress"] = False
        return _orig_prefill_md(self, *args, **kwargs)

    _dsv4b.DeepseekV4AttnBackend.init_forward_metadata_prefill = _prefill_md_no_compress

    # Decode path: make_forward_metadata_from_raw_decode hardcodes need_compress=True and
    # builds the compress plan (nvcc JIT -> "Failed to build sgl_kernel_jit_dpsk_v4_compress_plan"
    # on CPU). Mirror the prefill bypass: build core-attn metadata with need_compress=False and
    # no indexer/compress plan.
    if hasattr(_dsv4b.DeepseekV4AttnBackend, "make_forward_metadata_from_raw_decode"):
        _DSV4Metadata = _dsv4b.DSV4Metadata

        def _decode_md_no_compress(self, raw_metadata):
            core = self.make_core_attn_metadata(
                req_to_token=self.req_to_token,
                req_pool_indices_repeated=raw_metadata.req_pool_indices,
                seq_lens_casual=raw_metadata.seq_lens,
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                out_loc=raw_metadata.out_cache_loc,
                need_compress=False,
            )
            return _DSV4Metadata(
                core,
                None,
                c4_compress_metadata=None,
                c128_compress_metadata=None,
            )

        _dsv4b.DeepseekV4AttnBackend.make_forward_metadata_from_raw_decode = (
            _decode_md_no_compress
        )

    # need_compress=False builds no compress metadata, but the model forward still calls
    # forward_core_compressor (plan=None -> crash). No-op it under the bypass.
    if hasattr(_dsv4b.DeepseekV4AttnBackend, "forward_core_compressor"):
        _dsv4b.DeepseekV4AttnBackend.forward_core_compressor = (
            lambda self, *a, **k: None
        )
    logger.warning(
        "intel_cpu_models: DSA compressor+indexer BYPASSED (INTEL_CPU_DSV4_BYPASS_DSA=1) "
        "— non-sparse attention, PROFILING/STRUCTURAL ONLY, NOT numerically correct."
    )


# ---------------------------------------------------------------------------
# DSA (DeepSeek Sparse Attention) CPU kernels — the compressor/indexer layer.
# These have NO in-tree torch fallback (unlike the generic infra), so they are
# genuine CPU ports. Correctness-first (vectorized torch); the make-it-fast pass
# replaces the RoI-ranked hot ones with AMX kernels.
# ---------------------------------------------------------------------------
def _cpu_init_compressed_attn_metadata(
    seq_lens,
    positions,
    raw_out_loc,
    page_table=None,
    page_size: int = 0,
    compute_page_indices: bool = True,
):
    # Vectorized torch port of _init_compressed_attn_metadata_kernel
    # (sglang/kernels/ops/attention/dsv4/metadata_kernel.py): builds the c4 (ratio-4)
    # and c128 (ratio-128) compressed-attention metadata + c128 page indices.
    import torch

    num_write_tokens = raw_out_loc.shape[0]
    device = seq_lens.device
    sl = seq_lens.to(torch.int64)
    pos = positions.to(torch.int64)

    c4_positions = (pos & ~3).to(torch.int32)
    c4_seq_lens_raw = (sl // 4).to(torch.int32)
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)
    c128_positions = (pos & ~127).to(torch.int32)
    c128_seq_lens_raw = (sl // 128).to(torch.int32)
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    slw = sl[:num_write_tokens]
    rol = raw_out_loc.to(torch.int64)
    zeros = torch.zeros_like(rol)
    c4_out_loc = torch.where((slw % 4) == 0, rol // 4, zeros)
    c128_out_loc = torch.where((slw % 128) == 0, rol // 128, zeros)

    c128_page_indices = None
    if compute_page_indices:
        assert page_table is not None
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_cur_max_seq_len = c128_page_size * max_pages
        offsets = torch.arange(c128_cur_max_seq_len, device=device, dtype=torch.int64)
        page_idx = offsets // c128_page_size
        offset_in_page = offsets % c128_page_size
        gathered = page_table.to(torch.int64)[:, page_idx]  # (bs, L)
        vals = gathered * c128_page_size + offset_in_page.unsqueeze(0)
        valid = offsets.unsqueeze(0) < c128_seq_lens_raw.to(torch.int64).unsqueeze(1)
        c128_page_indices = torch.where(valid, vals, torch.full_like(vals, -1)).to(
            torch.int32
        )

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_raw,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def _install_dsa_cpu_kernels() -> None:
    if not current_platform.is_cpu():
        return
    # The indexer's fp8 paged-MQA-logits has an in-tree torch path (fp8_paged_mqa_logits_
    # torch) gated by this env; enabling it avoids the CUDA-only deep_gemm import in
    # PagedIndexerMetadata.__post_init__ and routes the indexer logits through torch on CPU.
    import os

    os.environ.setdefault("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "1")

    # init_compression_metadata() references _init_compressed_attn_metadata_triton as a
    # module global, so repointing this one attribute reroutes every caller (incl. the
    # deepseek_v4 CPU backend which imported init_compression_metadata by value).
    import sglang.kernels.ops.attention.dsv4.metadata_kernel as _mk

    _mk._init_compressed_attn_metadata_triton = _cpu_init_compressed_attn_metadata
    logger.info("intel_cpu_models: installed CPU DSA compressed-attn metadata kernel.")

    # FlashMLA metadata is a CUDA tile-scheduling artifact that already returns None on
    # sm120/xpu; CPU (which uses its own MLA attention path) is the same case.
    import sglang.srt.layers.attention.deepseek_v4_backend as _dsv4b

    _dsv4b._create_flashmla_metadata = lambda: None
    logger.info("intel_cpu_models: stubbed FlashMLA metadata (None) on CPU.")

    # topk_v2 JIT-compiles a CUDA kernel (nvcc); xpu already opts out via `not _is_xpu`
    # and the non-v2 branch is a trivial empty tensor. Opt CPU out the same way.
    import sglang.srt.layers.attention.dsa.dsa_topk_backend as _topk

    _orig_should_v2 = _topk.DSATopKBackend.should_use_topk_v2

    def _should_use_topk_v2_cpu(self):
        if current_platform.is_cpu():
            return False
        return _orig_should_v2(self)

    _topk.DSATopKBackend.should_use_topk_v2 = _should_use_topk_v2_cpu
    logger.info("intel_cpu_models: disabled DSA topk_v2 (CUDA JIT) on CPU.")


# ---------------------------------------------------------------------------
# CPU paged-allocator kernels.
#
# DSV4 forces page_size=256 (paged KV), so the allocator's alloc_extend/alloc_decode
# run the Triton `alloc_extend_kernel`/`alloc_decode_kernel`, which have NO CPU driver
# ("0 active drivers"). These torch ports replicate the Triton index bookkeeping
# (sglang/kernels/ops/memory/allocator.py) exactly, on CPU. bs is small for decode/
# extend, so a per-sequence loop is fine for make-it-work; vectorize later if the
# profile flags it.
# ---------------------------------------------------------------------------
import torch  # noqa: E402


class _CpuGridShim:
    """Mimic a Triton kernel's ``kernel[(bs,)](...)`` call form on CPU."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _grid):
        return self._fn


def _cpu_alloc_extend_kernel(
    pre_lens, seq_lens, last_loc, free_pages, out_indices, bs_upper, page_size
):
    ps = page_size
    bs = pre_lens.shape[0]
    extend_lens = seq_lens - pre_lens
    out_start = torch.cumsum(extend_lens, 0) - extend_lens
    num_new_pages = (seq_lens + ps - 1) // ps - (pre_lens + ps - 1) // ps
    new_page_start = torch.cumsum(num_new_pages, 0) - num_new_pages
    for i in range(bs):
        pre_len = int(pre_lens[i])
        seq_len = int(seq_lens[i])
        os_ = int(out_start[i])
        nps = int(new_page_start[i])
        npsl_self = int(num_new_pages[i])
        ll = int(last_loc[i])
        # Part 1: fill the old partial page.
        part1 = min(seq_len, ((pre_len + ps - 1) // ps) * ps) - pre_len
        if part1 > 0:
            idx = torch.arange(part1, dtype=out_indices.dtype)
            out_indices[os_ : os_ + part1] = ll + 1 + idx
        if pre_len + part1 == seq_len:
            continue
        # Part 2: fill new full pages.
        part2 = (seq_len // ps) * ps - ((pre_len + ps - 1) // ps) * ps
        if part2 > 0:
            off = torch.arange(part2, dtype=out_indices.dtype)
            pages = free_pages[nps + off // ps]
            out_indices[os_ + part1 : os_ + part1 + part2] = pages * ps + (off % ps)
        if pre_len + part1 + part2 == seq_len:
            continue
        # Part 3: fill the new partial page.
        part3 = seq_len - (seq_len // ps) * ps
        start_loc = int(free_pages[nps + npsl_self - 1])
        base = os_ + part1 + part2
        idx3 = torch.arange(part3, dtype=out_indices.dtype)
        out_indices[base : base + part3] = start_loc * ps + idx3


def _cpu_alloc_decode_kernel(
    seq_lens, last_loc, free_pages, out_indices, bs_upper, page_size
):
    ps = page_size
    bs = seq_lens.shape[0]
    pre_lens = seq_lens - 1
    num_new_pages = (seq_lens + ps - 1) // ps - (pre_lens + ps - 1) // ps
    new_page_start = torch.cumsum(num_new_pages, 0) - num_new_pages
    for i in range(bs):
        if int(num_new_pages[i]) == 0:
            out_indices[i] = int(last_loc[i]) + 1
        else:
            out_indices[i] = int(free_pages[int(new_page_start[i])]) * ps


def _install_cpu_paged_allocator() -> None:
    if not current_platform.is_cpu():
        return
    extend_shim = _CpuGridShim(_cpu_alloc_extend_kernel)
    decode_shim = _CpuGridShim(_cpu_alloc_decode_kernel)
    import sglang.srt.mem_cache.allocator.paged as _paged

    _paged.alloc_extend_kernel = extend_shim
    _paged.alloc_decode_kernel = decode_shim
    try:
        import sglang.srt.mem_cache.allocator.unified_sub_pool as _usp

        _usp.alloc_extend_kernel = extend_shim
        _usp.alloc_decode_kernel = decode_shim
    except Exception:
        pass
    _route_allocation_to_cpu()
    logger.info("intel_cpu_models: installed CPU paged-allocator kernels (alloc_extend/decode).")


def _route_allocation_to_cpu() -> None:
    # allocation.py already carries CPU/torch fallbacks (write_cache_indices' non-triton
    # branch via req_to_token_pool.write, and get_last_loc_torch); they just aren't picked
    # because support_triton("dsv4") is True and get_last_loc's dispatch ignores CPU. Force
    # the existing CPU paths instead of porting anything.
    import sys

    import sglang.srt.utils.common as _common

    # support_triton(backend)=backend not in [torch_native,intel_amx] -> True for dsv4, with
    # no CPU awareness. Many modules dispatch triton-vs-torch on it (allocation, forward_batch
    # _info/compute_position, ...) and each imported it BY VALUE, so patch every live reference.
    _orig_support_triton = _common.support_triton

    def _support_triton_cpu(backend):
        if current_platform.is_cpu():
            return False
        return _orig_support_triton(backend)

    patched = 0
    for _m in list(sys.modules.values()):
        if _m is None:
            continue
        if getattr(_m, "support_triton", None) is _orig_support_triton:
            try:
                _m.support_triton = _support_triton_cpu
                patched += 1
            except Exception:
                pass
    logger.info(
        "intel_cpu_models: routed support_triton -> CPU/torch path in %d modules.", patched
    )

    # get_last_loc uses its own (non-support_triton) dispatch that omits CPU; force torch.
    import sglang.srt.mem_cache.allocation as _alloc

    _orig_get_last_loc = _alloc.get_last_loc

    def _get_last_loc_cpu(req_to_token, req_pool_indices_tensor, prefix_lens_tensor):
        if current_platform.is_cpu():
            return _alloc.get_last_loc_torch(
                req_to_token, req_pool_indices_tensor, prefix_lens_tensor
            )
        return _orig_get_last_loc(
            req_to_token, req_pool_indices_tensor, prefix_lens_tensor
        )

    _alloc.get_last_loc = _get_last_loc_cpu
