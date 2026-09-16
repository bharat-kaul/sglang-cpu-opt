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
    _install_cpu_paged_allocator()
    _install_dsa_cpu_kernels()
    _install_mhc_cpu()
    _install_dsv4_core_kernels()
    _install_dsa_profile_bypass()
    _INSTALLED = True
    logger.info("intel_cpu_models: installed CPU DSV4 KV-pool configurator patch.")


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


def _install_dsv4_core_kernels() -> None:
    # Reference-first torch ports of the dsv4 MLA-core fused kernels the MODEL FORWARD
    # calls directly (CUDA-JIT, no CPU path). Each mirrors the in-tree AOT/test oracle.
    if not current_platform.is_cpu():
        return
    import sglang.srt.models.deepseek_v4 as _dv4

    _dv4.fused_q_norm_rope = _cpu_fused_q_norm_rope

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
    _iba._set_k_and_s_torch(
        kvcache, out_loc, _cpu_quant_to_nope_fp8_rope_bf16_pack(kv_out), page_size
    )


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
    try:
        import sglang.srt.models.deepseek_v4 as _dv4

        _dv4._get_mhc_ops.cache_clear()
    except Exception:
        pass
    logger.info("intel_cpu_models: routed MHC sub-ops (sinkhorn, combine) to torch on CPU.")


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
