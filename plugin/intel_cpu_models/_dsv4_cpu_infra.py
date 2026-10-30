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
# Set once the decode-wide thread cap is installed; disables the (superseded) per-op MoE cap.
_DECODE_CAP_ACTIVE = False
# Set once gloo TP all-reduce is installed; makes the per-forward decode cap tp>1-safe.
_TP_GLOO_ACTIVE = False
# Phase tag (prefill vs decode) appended to timed op names when INTEL_CPU_DSV4_TIMEIT_PHASE=1.
# Lets one run separate the prefill (compute-bound) from the decode (BW/overhead-bound) profile.
_PHASE = ""
_PHASE_ON = _os.environ.get("INTEL_CPU_DSV4_TIMEIT_PHASE", "0") == "1"


def _pkey(name: str) -> str:
    # Append the current forward phase to an op name when phase-tagging is active.
    if _PHASE_ON and _PHASE:
        return f"{name}.{_PHASE}"
    return name


def _release_freed_memory() -> None:
    """Return allocator-held freed memory to the OS (glibc malloc_trim) + run gc. Used after
    each layer's fp4->fp8 dequant so the single-rank (tp=1) load peak doesn't overflow node RAM."""
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1 << 30)
    except Exception:
        return -1.0


def _set_numa_interleave_all() -> None:
    """Re-apply an interleave-all memory policy for this process.

    numactl --interleave=all is CLOBBERED by sgl_kernel.init_cpu_threads_env (it binds threads and
    sets node-local allocation), so a single tp=1 rank concentrates all weights on node 0 and OOMs
    at that node's ~256GB even though the node has ~1.5TB. Re-setting the interleave mask via libnuma
    after that (before weight load) spreads allocations round-robin across all NUMA nodes."""
    try:
        import ctypes

        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            logger.warning("intel_cpu_models: libnuma reports NUMA unavailable; interleave not set.")
            return
        mask = ctypes.c_void_p.in_dll(libnuma, "numa_all_nodes_ptr")
        libnuma.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
        libnuma.numa_set_interleave_mask(mask)
        logger.info(
            "intel_cpu_models: re-applied NUMA interleave-all memory policy "
            "(sgl_kernel thread init had reset it to node-local)."
        )
    except Exception as e:
        logger.warning("intel_cpu_models: could not set NUMA interleave-all: %s", e)


_LIBNUMA = None
_LIBC_MBIND = None
_NUMA_NNODES = 0


def _get_libnuma():
    """Cache a libnuma handle (None if unavailable)."""
    global _LIBNUMA, _NUMA_NNODES
    if _LIBNUMA is None:
        try:
            import ctypes

            lib = ctypes.CDLL("libnuma.so.1")
            if lib.numa_available() < 0:
                _LIBNUMA = False
            else:
                lib.numa_num_configured_nodes.restype = ctypes.c_int
                _NUMA_NNODES = int(lib.numa_num_configured_nodes())
                _LIBNUMA = lib
        except Exception:
            _LIBNUMA = False
    return _LIBNUMA or None


def _get_mbind():
    """Cache a libc handle for the mbind syscall (None if unavailable).

    glibc does NOT export `mbind` as a public symbol, so we invoke it via the raw
    syscall (SYS_mbind = 237 on x86_64). We set argtypes EXPLICITLY for the fixed
    7-arg form: without argtypes, ctypes mis-marshals the variadic 7th arg (the
    mbind `flags`), silently dropping MPOL_MF_MOVE so mbind returns 0 but never
    migrates the already-faulted pages."""
    global _LIBC_MBIND
    if _LIBC_MBIND is None:
        try:
            import ctypes
            import platform

            if platform.machine() != "x86_64":
                _LIBC_MBIND = False
                return None
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.syscall.restype = ctypes.c_long
            libc.syscall.argtypes = [
                ctypes.c_long,  # syscall number
                ctypes.c_void_p,  # addr
                ctypes.c_ulong,  # len
                ctypes.c_int,  # mode
                ctypes.c_void_p,  # nodemask*
                ctypes.c_ulong,  # maxnode
                ctypes.c_uint,  # flags
            ]
            _LIBC_MBIND = libc
        except Exception:
            _LIBC_MBIND = False
    return _LIBC_MBIND or None


def _interleave_tensor_memory(t) -> None:
    """Migrate a CPU tensor's pages to interleave across all NUMA nodes.

    Setting the calling thread's interleave mask does NOT spread weights: the fp4->fp8 dequant and
    AMX-prepack pages are FAULTED by sgl_kernel worker threads bound to node 0 (node-local policy),
    so they all land on node 0 regardless of the main thread's policy (measured: 245GB/266GB on
    node 0 -> OOM at node 0's ~258GB). We call mbind(MPOL_INTERLEAVE, MPOL_MF_MOVE) DIRECTLY on the
    tensor's address range, which MIGRATES the already-faulted pages round-robin across all nodes
    (libnuma's numa_interleave_memory would NOT migrate -- it omits MPOL_MF_MOVE). Per-layer, so
    node 0 never accumulates."""
    try:
        import ctypes

        if t is None or t.numel() == 0 or not t.is_cpu:
            return
        nbytes = t.numel() * t.element_size()
        if nbytes < (1 << 20):  # skip small tensors; migration has per-call overhead
            return
        if _get_libnuma() is None:
            return
        libc = _get_mbind()
        if libc is None or _NUMA_NNODES <= 1:
            return
        SYS_mbind = 237  # x86_64
        MPOL_INTERLEAVE = 3
        MPOL_MF_MOVE = 1 << 1
        addr = t.data_ptr()
        page = 4096
        aligned = addr & ~(page - 1)  # mbind needs a page-aligned start
        size = nbytes + (addr - aligned)
        nodemask = ctypes.c_ulong((1 << _NUMA_NNODES) - 1)  # all configured nodes
        maxnode = ctypes.sizeof(nodemask) * 8  # bits available in the mask word
        ret = libc.syscall(
            ctypes.c_long(SYS_mbind),
            ctypes.c_void_p(aligned),
            ctypes.c_ulong(size),
            ctypes.c_int(MPOL_INTERLEAVE),
            ctypes.byref(nodemask),
            ctypes.c_ulong(maxnode),
            ctypes.c_uint(MPOL_MF_MOVE),
        )
        if ret != 0:
            err = ctypes.get_errno()
            if not _MOE_DIAG.get("mbind_warned"):
                _MOE_DIAG["mbind_warned"] = True
                logger.warning("intel_cpu_models: mbind(MPOL_INTERLEAVE|MOVE) errno=%d", err)
            return
        _MOE_DIAG["mbind_bytes"] = _MOE_DIAG.get("mbind_bytes", 0) + nbytes
        _MOE_DIAG["mbind_ok"] = _MOE_DIAG.get("mbind_ok", 0) + 1
    except Exception as e:
        logger.warning("intel_cpu_models: mbind interleave failed: %s", e)


def _interleave_layer_weights(layer) -> None:
    """Interleave the large packed weight/scale tensors of a just-loaded layer across NUMA nodes."""
    if not _os.environ.get("INTEL_CPU_DSV4_INTERLEAVE_MEM"):
        return
    _before = _MOE_DIAG.get("mbind_bytes", 0)
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
        "weight",
        "weight_scale_inv",
    ):
        t = getattr(layer, name, None)
        if t is not None:
            _interleave_tensor_memory(getattr(t, "data", t))
    # Diagnostic: confirm migration actually ran (first few layers only, to prove the fix).
    _nl = _MOE_DIAG.get("mbind_layers", 0)
    _MOE_DIAG["mbind_layers"] = _nl + 1
    if _nl < 3:
        moved = (_MOE_DIAG.get("mbind_bytes", 0) - _before) / (1 << 30)
        logger.info(
            "intel_cpu_models: layer mbind-migrated %.1fGB across %d nodes (ok=%d, errno_warned=%s)",
            moved, _NUMA_NNODES, _MOE_DIAG.get("mbind_ok", 0), _MOE_DIAG.get("mbind_warned", False),
        )



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
                key = _pkey(name)
                if key is not name and key not in _TIMES_KIND:
                    _TIMES_KIND[key] = kind
                rec = _TIMES.setdefault(key, [0.0, 0])
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
    # Machine-readable pivot-schema dump for time_attribution_pivot.py (INTEL_CPU_DSV4_TIMEIT_JSON=<path>).
    _json_path = _os.environ.get("INTEL_CPU_DSV4_TIMEIT_JSON", "")
    if _json_path:
        _dump_times_json(_json_path)


def _dump_times_json(path: str) -> None:
    """Emit accumulated TIMEIT counters in the time_attribution_pivot.py schema.
    Phase from the `.pf`/`.dec` key suffix; `wall_s` = the outermost `parent` op
    (layer.total) so the chart's OTHER slice = wall − sum(leaf ops) stays honest."""
    import json as _json
    _suf2phase = {"pf": "prefill", "dec": "decode"}
    out = {"prefill": {"wall_s": None, "ops": []}, "decode": {"wall_s": None, "ops": []}}
    nophase = {"wall_s": None, "ops": []}
    for key, (t, _n) in _TIMES.items():
        base, _, suf = key.rpartition(".")
        ph = out[_suf2phase[suf]] if suf in _suf2phase else nophase
        opname = base if suf in _suf2phase else key
        kind = _TIMES_KIND.get(key) or _TIMES_KIND.get(base) or ""
        if kind == "parent":
            if ph["wall_s"] is None or t > ph["wall_s"]:
                ph["wall_s"] = t  # outermost wrapper = phase wall
            continue
        if kind in ("kernel", "torch", "framework"):
            ph["ops"].append({"op": opname, "s": round(t, 4), "bucket": kind})
    if nophase["ops"] and not out["prefill"]["ops"]:  # phase tagging off -> fold into prefill
        out["prefill"] = nophase
    data = {
        "model": _os.environ.get("TIMEIT_MODEL", "DeepSeek-V4-Flash"),
        "node": _os.environ.get("TIMEIT_NODE", "EMR"),
        "precision": _os.environ.get("TIMEIT_PRECISION", "MXFP4 W4A16 experts, bf16 compute"),
        "batch": int(_os.environ.get("TIMEIT_BATCH", "1")),
        "phases": {p: out[p] for p in ("prefill", "decode") if out[p]["ops"]},
    }
    try:
        with open(path, "w") as _f:
            _json.dump(data, _f, indent=2)
        logger.warning(f"[DSV4 TIMEIT] wrote pivot JSON -> {path}")
    except Exception as _e:
        logger.warning(f"[DSV4 TIMEIT] pivot JSON write failed ({path}): {_e}")


def _tacc(name: str, t0: float, kind: str = "") -> None:
    """Accumulate an elapsed interval under `name` (guard with _TIMEIT_ON at call site)."""
    import time

    key = _pkey(name)
    if kind and key not in _TIMES_KIND:
        _TIMES_KIND[key] = kind
    rec = _TIMES.setdefault(key, [0.0, 0])
    rec[0] += time.perf_counter() - t0
    rec[1] += 1


def _resolve_moe_thread_cap(orig_apply, self, layer, dispatch_output):
    """Thread count to cap the native MoE apply at (runtime-config fix), or None.

    `fused_experts_cpu` scales INVERSELY with threads on this MoE shape (isolation:
    ~0.1ms@8thr vs ~2432ms@60thr) -- a threading pathology, not a bad kernel. Set
    INTEL_CPU_DSV4_MOE_THREADS=N for a fixed cap, or =auto to self-tune once on the real
    node/shape (first-call micro-sweep picks the fastest). Cached after the first resolve.

    SUPERSEDED by the decode-wide cap: when INTEL_CPU_DSV4_DECODE_THREADS is active it already
    caps the whole M=1 decode forward (MoE included), and a per-op MoE cap on top CONFLICTS
    (measured: composing the two regresses decode ~200x back to the cliff). So this per-op cap
    is disabled whenever the decode cap is installed; use the decode cap alone.
    """
    spec = _os.environ.get("INTEL_CPU_DSV4_MOE_THREADS", "")
    if not spec or _DECODE_CAP_ACTIVE:
        return None
    if "cap" in _MOE_DIAG:
        return _MOE_DIAG["cap"]
    import torch as _tt

    cur = _tt.get_num_threads()
    if spec.isdigit():
        _MOE_DIAG["cap"] = max(1, min(int(spec), cur))
        logger.warning("[MOE THREADCAP] fixed threads=%d (was %d)", _MOE_DIAG["cap"], cur)
        return _MOE_DIAG["cap"]
    if spec == "auto":
        import time as _t

        cands = sorted({c for c in (4, 8, 16, 24, 32, cur) if 1 <= c <= cur})
        best, best_t = cur, float("inf")
        for c in cands:
            _tt.set_num_threads(c)
            orig_apply(self, layer, dispatch_output)  # warm at this count
            _t0 = _t.perf_counter()
            for _ in range(2):
                orig_apply(self, layer, dispatch_output)
            dt = (_t.perf_counter() - _t0) / 2
            if dt < best_t:
                best_t, best = dt, c
        _tt.set_num_threads(cur)
        _MOE_DIAG["cap"] = best
        logger.warning(
            "[MOE THREADCAP] auto-picked threads=%d (%.1fms) from %s (was %d)",
            best, best_t * 1e3, cands, cur,
        )
        return best
    return None



def _install_phase_tag() -> None:
    # Env-gated (INTEL_CPU_DSV4_TIMEIT_PHASE=1): tag every timed op with the forward phase
    # (pf/dec) by setting a module global around the model's top-level forward. Unlocks a
    # separate decode per-op breakdown from the same run. Off by default; no-op otherwise.
    if not _TIMEIT_ON or not _PHASE_ON or not current_platform.is_cpu():
        return
    import sglang.srt.models.deepseek_v4 as _dv4

    cls = _dv4.DeepseekV4ForCausalLM
    if getattr(cls, "_phase_tagged", False):
        return
    _orig_fwd = cls.forward

    def _fwd(self, *a, **k):
        global _PHASE
        fb = k.get("forward_batch")
        if fb is None:
            for _x in a:
                if hasattr(_x, "forward_mode"):
                    fb = _x
                    break
        try:
            _PHASE = "dec" if fb.forward_mode.is_decode_or_idle() else "pf"
        except Exception:
            _PHASE = ""
        try:
            return _orig_fwd(self, *a, **k)
        finally:
            _PHASE = ""
            # Flush TIMEIT after EACH model forward: the scheduler runs in a subprocess that is
            # hard-killed on shutdown, so the atexit dump never fires. Each flush overwrites with the
            # cumulative _TIMES, so the final forward's write is complete and survives the kill.
            try:
                _dump_times()
            except Exception:  # noqa: BLE001
                pass

    cls.forward = _fwd
    cls._phase_tagged = True
    logger.warning("[DSV4 TIMEIT] phase-tagging ON (prefill/decode split).")


def _install_hc_pre_cpu() -> None:
    # TIMEIT sub-timer only. (OPT #2e, the fused no-temp einsum rmsnorm, was measured a NON-WIN on
    # the clean passive-wait path -- fused 0.115s vs default 0.043s -- and removed; hc_pre is ~1.5%,
    # a non-target.) Reproduces ONLY the CPU torch `else` branch of deepseek_v4.hc_pre to sub-time
    # upcast/rmsnorm/gemm when INTEL_CPU_DSV4_TIMEIT_HCPRE=1 (with TIMEIT); delegates every other
    # path to the original. Off by default; must stay in lockstep with deepseek_v4.hc_pre.
    _time_it = _TIMEIT_ON and _os.environ.get("INTEL_CPU_DSV4_TIMEIT_HCPRE") == "1"
    if not _time_it or not current_platform.is_cpu():
        return
    import time

    import torch
    import torch.nn.functional as F

    import sglang.srt.models.deepseek_v4 as _dv4

    Layer = _dv4.DeepseekV4DecoderLayer
    if getattr(Layer, "_hc_pre_cpu_patched", False):
        return
    _orig_hc_pre = Layer.hc_pre

    def _hc_pre_cpu(self, x, hc_fn, hc_scale, hc_base, norm=None, forward_batch=None):
        cpu_torch = (
            x.shape[0] > 0
            and not _dv4._is_npu
            and not _dv4._is_xpu
            and not _dv4._is_hip
            and not _dv4.envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()
            and not _dv4.envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
            and not (
                _dv4.envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
                and x.shape[0] >= _dv4._HC_PRENORM_DEEPGEMM_MIN_TOKENS
            )
        )
        if not cpu_torch:
            return _orig_hc_pre(self, x, hc_fn, hc_scale, hc_base, norm, forward_batch)
        _t = time.perf_counter()
        x_flat = x.flatten(1).float()
        _tacc("mhc.hc_pre.upcast", _t, "torch")
        _t = time.perf_counter()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        _tacc("mhc.hc_pre.rmsnorm", _t, "torch")
        _t = time.perf_counter()
        gemm = F.linear(x_flat, hc_fn)
        _tacc("mhc.hc_pre.gemm", _t, "torch")
        mixes = (gemm * rsqrt).unsqueeze(1)
        pre, post, comb = _dv4._get_mhc_ops().hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        from sglang.kernels.ops.layernorm.mhc import hc_combine

        with _dv4.use_symmetric_memory(
            _dv4.get_tp_group(), disabled=not _dv4.is_allocation_symmetric()
        ):
            # Monolith increment 4: fuse the final hc_combine + the caller's post-norm into ONE cooperative
            # region (removes one fork-join per hc_pre call). Gated; eager fallback + optional parity assert.
            _coop = _os.environ.get("INTEL_CPU_DSV4_COOP_MHC") == "1" and norm is not None
            if _coop:
                try:
                    _preS = pre.squeeze(1)
                    _w = getattr(norm, "weight", None)
                    _eps = getattr(norm, "variance_epsilon", self.rms_norm_eps)
                    _wt = _w.data if hasattr(_w, "data") else _w
                    y_fused = torch.ops.sgl_kernel.coop_hc_combine_rmsnorm_f32_cpu(
                        x_flat.contiguous(), _preS.float().contiguous(), _wt.contiguous(),
                        self.hc_mult, float(_eps),
                    )
                    if _os.environ.get("INTEL_CPU_DSV4_DSA_CKERNEL_CHECK") == "1":
                        y_ref = norm(hc_combine(x_flat, _preS, self.hc_mult, x.dtype))
                        _cos = torch.nn.functional.cosine_similarity(
                            y_fused.float().reshape(-1), y_ref.float().reshape(-1), dim=0, eps=1e-12
                        ).item()
                        assert _cos >= 0.9995, f"coop hc_combine+norm parity cos={_cos:.6f}"
                    return y_fused, post.squeeze(1), comb.squeeze(1), True  # norm FUSED -> caller skips it
                except Exception as _e:  # noqa: BLE001
                    logger.warning("[DSV4] coop hc_combine+norm fell back to eager: %r", _e)
            y = hc_combine(x_flat, pre.squeeze(1), self.hc_mult, x.dtype)
        return y, post.squeeze(1), comb.squeeze(1), False

    Layer.hc_pre = _hc_pre_cpu
    Layer._hc_pre_cpu_patched = True
    logger.warning("[DSV4] hc_pre CPU sub-timer installed (TIMEIT_HCPRE).")


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


def _check_omp_wait_policy() -> None:
    """libgomp (torch's OpenMP) reads OMP_WAIT_POLICY once at LOAD -- before this plugin imports --
    so it CANNOT be fixed in-process. Warn loudly if idle threads will busy-wait: that inter-region
    spin contention inflates every forward after the first ~63x (prefill) / ~10x (decode), measured.
    The only fix is the LAUNCH env."""
    if not current_platform.is_cpu():
        return
    if _os.environ.get("OMP_WAIT_POLICY", "").lower() != "passive":
        logger.warning(
            "[DSV4 PERF] OMP_WAIT_POLICY=%r (expected 'passive'): OpenMP idle threads will SPIN and "
            "thrash each forward after the first (~63x prefill / ~10x decode slowdown, measured). "
            "libgomp reads this at load so it CANNOT be set in-process -- export it in the LAUNCH env "
            "BEFORE starting the server/engine:  export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0",
            _os.environ.get("OMP_WAIT_POLICY", "<unset>"),
        )


def install() -> None:
    """Idempotently install the DSV4 CPU KV-pool configurator patch."""
    global _INSTALLED
    if _INSTALLED:
        return
    _check_omp_wait_policy()
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
    _install_phase_tag()
    _install_hc_pre_cpu()
    _install_gemm_timeit()
    _install_layer_timeit()
    _install_decode_thread_cap()
    _install_cpu_interleave_mem()
    _install_cpu_gloo_allreduce()
    _install_hidden_debug()
    _install_fp8_proj_parity()
    _install_fp8_force_bf16()
    _install_wo_a_logical_unpack()
    _install_real_router_injection()
    _install_full_capture()
    _INSTALLED = True
    logger.info("intel_cpu_models: installed CPU DSV4 KV-pool configurator patch.")


def _install_real_router_injection() -> None:
    # PERF-PROXY mode-collapse guard: under --load-format dummy the MoE router/gate weights are RANDOM,
    # so routing is uniform-random (~135 distinct experts / 32-token batch) vs the real model's learned
    # ~59 — a 2.3x over-count that biases decode (MoE weight-streaming-bound) tok/s. Inject the REAL
    # (tiny) gate weights + correction bias (score layers) and tid2eid (hash layers 0-2) from the
    # checkpoint so the dummy-weight proxy routes like the real model. Env-gated INTEL_CPU_DSV4_REAL_ROUTER=1
    # (only meaningful with dummy expert weights; harmless with real weights since it re-copies the same).
    import os as _o

    if not current_platform.is_cpu():
        return
    if _o.environ.get("INTEL_CPU_DSV4_REAL_ROUTER") != "1":
        return
    import json as _json
    import re as _re

    import torch

    ckpt = _o.environ.get("INTEL_CPU_DSV4_CKPT", "/scratch/bkaul/models/DeepSeek-V4-Flash")
    try:
        from safetensors import safe_open
        _idx = _json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[REAL_ROUTER] no checkpoint index (%r); skipping injection", e)
        return

    def _load(name):
        shard = _idx.get(name)
        if shard is None:
            return None
        try:
            with safe_open(f"{ckpt}/{shard}", "pt") as t:
                return t.get_tensor(name)
        except Exception:  # noqa: BLE001
            return None

    def _layer_of(prefix):
        m = _re.search(r"layers\.(\d+)\.", prefix or "")
        return int(m.group(1)) if m else None

    import sglang.srt.models.deepseek_v2 as _dv2

    MoEGate = _dv2.MoEGate
    if not getattr(MoEGate, "_dsv4_realrouter_tag", False):
        _oinit = MoEGate.__init__

        def _ginit(self, config, quant_config, prefix="", *a, **k):
            _oinit(self, config, quant_config, prefix, *a, **k)
            self._dsv4_gate_layer = _layer_of(prefix)

        MoEGate.__init__ = _ginit
        MoEGate._dsv4_realrouter_tag = True

    # Score-layer gate weight + correction bias: inject AFTER the dummy fill, BEFORE the VNNI pack.
    from sglang.srt.layers.amx_utils import PackWeightMethod

    if not getattr(PackWeightMethod, "_dsv4_realrouter_patched", False):
        _opack = PackWeightMethod.process_weights_after_loading

        def _pack(self, module):
            lid = getattr(module, "_dsv4_gate_layer", None)
            if lid is not None and hasattr(module, "weight"):
                real = _load(f"layers.{lid}.ffn.gate.weight")
                if real is not None and tuple(real.shape) == tuple(module.weight.shape):
                    module.weight.data.copy_(real.to(module.weight.dtype))
                    logger.warning("[REAL_ROUTER] injected real gate weight L%d %s", lid, tuple(real.shape))
                bias = getattr(module, "e_score_correction_bias", None)
                if bias is not None:
                    rb = _load(f"layers.{lid}.ffn.gate.bias")
                    if rb is not None and tuple(rb.shape) == tuple(bias.shape):
                        bias.data.copy_(rb.to(bias.dtype))
            return _opack(self, module)

        PackWeightMethod.process_weights_after_loading = _pack
        PackWeightMethod._dsv4_realrouter_patched = True

    # Hash-layer (0-2) tid2eid: inject at HashTopK construction.
    try:
        from sglang.srt.layers.moe.hash_topk import HashTopK

        if not getattr(HashTopK, "_dsv4_realrouter_patched", False):
            _ohinit = HashTopK.__init__

            def _hinit(self, *a, **k):
                _ohinit(self, *a, **k)
                lid = getattr(self, "layer_id", None)
                if lid is not None and getattr(self, "tid2eid", None) is not None:
                    real = _load(f"layers.{lid}.ffn.gate.tid2eid")
                    if real is not None and tuple(real.shape) == tuple(self.tid2eid.shape):
                        self.tid2eid.data.copy_(real.to(self.tid2eid.dtype))
                        logger.warning("[REAL_ROUTER] injected real tid2eid L%d %s", lid, tuple(real.shape))

            HashTopK.__init__ = _hinit
            HashTopK._dsv4_realrouter_patched = True
    except Exception as e:  # noqa: BLE001
        logger.warning("[REAL_ROUTER] HashTopK hook failed: %r", e)

    logger.warning("[REAL_ROUTER] installed — real gate weights + tid2eid injected under dummy load.")
    _install_expert_hist_diag()


def _install_expert_hist_diag() -> None:
    # Confirms the real-router injection worked: logs mean DISTINCT experts selected per forward call
    # (should drop from dummy-random ~135 toward the real ~59 for B=32). Env-gated
    # INTEL_CPU_DSV4_EXPERT_HIST=1 — OFF during timed perf runs (adds a tiny per-call reduction).
    import os as _o

    if _o.environ.get("INTEL_CPU_DSV4_EXPERT_HIST") != "1":
        return
    import torch

    _hist = {"calls": 0, "sum_distinct": 0, "sum_tokens": 0}

    def _count(out):
        try:
            tids = getattr(out, "topk_ids", None)
            if tids is not None and tids.numel() > 0:
                _hist["calls"] += 1
                _hist["sum_distinct"] += int(torch.unique(tids[tids >= 0]).numel())
                _hist["sum_tokens"] += int(tids.shape[0])
                if _hist["calls"] % 20 == 0:
                    msg = ("[EXPERT_HIST] calls=%d mean_distinct/call=%.1f mean_tokens/call=%.1f"
                           % (_hist["calls"], _hist["sum_distinct"] / _hist["calls"],
                              _hist["sum_tokens"] / _hist["calls"]))
                    logger.warning(msg)
                    try:
                        with open(_o.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_expert_hist.txt"), "a") as f:
                            f.write(msg + "\n")
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass

    try:
        from sglang.srt.layers.moe.topk import TopK
        classes = [TopK]
        try:
            from sglang.srt.layers.moe.hash_topk import HashTopK
            classes.append(HashTopK)
        except Exception:  # noqa: BLE001
            pass
        for Cls in classes:
            if getattr(Cls, "_dsv4_hist_patched", False):
                continue
            _ofwd = Cls.forward

            def _mk(orig):
                def _f(self, *a, **k):
                    out = orig(self, *a, **k)
                    _count(out)
                    return out
                return _f

            Cls.forward = _mk(_ofwd)
            Cls._dsv4_hist_patched = True
        logger.warning("[EXPERT_HIST] installed (distinct-expert counter).")
    except Exception as e:  # noqa: BLE001
        logger.warning("[EXPERT_HIST] install failed: %r", e)


def _install_wo_a_logical_unpack() -> None:
    # wo_a is a quantized ColumnParallelLinear that is NEVER applied through its quant method:
    # the o_proj absorb GEMM reads self.wo_a.weight directly and runs a plain batched einsum
    # (_apply_wo_a_bf16_matmul: "tgd,grd->tgr") that wants the ROW-MAJOR [G, R, D] weight. The
    # native model opts wo_a out of ROCm's aiter B-preshuffle (skip_aiter_bpreshuffle=True) for
    # exactly this reason, but the CPU path unconditionally VNNI-packs every weight via
    # _amx_process_weight_after_loading -- which silently permutes wo_a and makes the einsum return
    # noise (the whole-model garbage-output bug). Mirror the ROCm opt-out on CPU: keep wo_a's weight
    # a plain row-major bf16, skipping the AMX pack (wo_a is only consumed by the einsum). On CPU
    # _FP8_WO_A_GEMM is forced False, so wo_a is an UNQUANTIZED bf16 linear (already dequantized by
    # _prepare_deepseek_v4_weights) -> the pack happens in UnquantizedLinearMethod, not Fp8LinearMethod.
    # PERF TODO (roofline): row-major einsum is NOT AMX-accelerated; for perf, route wo_a's grouped
    # absorb GEMM through the AMX weight_packed_linear (is_vnni) per group instead of un-packing.
    if not current_platform.is_cpu():
        return
    import torch

    import sglang.srt.models.deepseek_v4 as _dv4

    # Tag wo_a at construction so the weight processor can identify it (it carries no quant marker
    # on CPU because it is created unquantized bf16).
    _Attn = getattr(_dv4, "MQALayer", None)
    if _Attn is not None and not getattr(_Attn, "_wo_a_tag_patched", False):
        _oinit = _Attn.__init__

        def _init(self, *a, **k):
            _oinit(self, *a, **k)
            try:
                if getattr(self, "wo_a", None) is not None:
                    self.wo_a._dsv4_is_wo_a = True
            except Exception:  # noqa: BLE001
                pass

        _Attn.__init__ = _init
        _Attn._wo_a_tag_patched = True

    # Same AMX-VNNI-pack-vs-plain-matmul bug for the DSA compressor's wkv_gate: it is a bf16
    # ReplicatedLinear consumed by compute_kv_score -> linear_bf16_fp32 (a PLAIN x@wT matmul), but
    # the CPU weight processor VNNI-packs it -> permuted weight -> scrambled kv_score -> garbage
    # compressed KV. Tag both the main-attention and indexer compressors' wkv_gate to keep row-major.
    try:
        import sglang.srt.layers.attention.dsv4.compressor as _cmp

        _Comp = getattr(_cmp, "Compressor", None)
        if _Comp is not None and not getattr(_Comp, "_wkvgate_tag_patched", False):
            _cinit = _Comp.__init__

            def _cinit_tag(self, *a, **k):
                _cinit(self, *a, **k)
                try:
                    if getattr(self, "wkv_gate", None) is not None:
                        self.wkv_gate._dsv4_keep_rowmajor = True
                except Exception:  # noqa: BLE001
                    pass

            _Comp.__init__ = _cinit_tag
            _Comp._wkvgate_tag_patched = True
    except Exception:  # noqa: BLE001
        pass

    def _blk_dequant(w_fp8, s):
        wf = w_fp8.float()
        O, I = wf.shape
        sf = s.float()
        if sf.dim() == 2 and sf.shape[0] > 0 and sf.shape[1] > 0:
            bo = max(1, O // sf.shape[0])
            bi = max(1, I // sf.shape[1])
            sfull = sf.repeat_interleave(bo, 0).repeat_interleave(bi, 1)[:O, :I]
            return wf * sfull
        return wf * sf

    def _keep_wo_a_rowmajor(layer):
        # Return True if we handled the layer (skip the caller's AMX pack). Covers wo_a AND the DSA
        # compressor wkv_gate — both are plain-matmul-consumed bf16/fp8 linears the VNNI pack corrupts.
        if not (getattr(layer, "_dsv4_is_wo_a", False) or getattr(layer, "_dsv4_keep_rowmajor", False)):
            return False
        _tag = "wo_a" if getattr(layer, "_dsv4_is_wo_a", False) else "wkv_gate"
        w = getattr(layer, "weight", None)
        s = getattr(layer, "weight_scale_inv", getattr(layer, "weight_scale", None))
        try:
            if w is not None and w.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8) and s is not None:
                wv = w.view(torch.float8_e4m3fn) if w.dtype != torch.float8_e4m3fn else w
                logical = _blk_dequant(wv, s).to(torch.bfloat16).contiguous()
                layer.weight = torch.nn.Parameter(logical, requires_grad=False)
            elif torch.is_tensor(w):
                # Already bf16-logical (dequantized by _prepare_deepseek_v4_weights); just don't pack.
                layer.weight = torch.nn.Parameter(w.detach().to(torch.bfloat16).contiguous(), requires_grad=False)
            layer._dsv4_wo_a_logical_bf16 = True
            logger.info(
                "intel_cpu_models: %s kept row-major bf16 %s (skipped AMX VNNI pack).",
                _tag, tuple(layer.weight.shape),
            )
            return True
        except Exception as _e:  # noqa: BLE001
            logger.warning("intel_cpu_models: %s logical unpack failed: %s", _tag, _e)
            return False

    def _wrap(Method):
        if getattr(Method, "_wo_a_unpack_patched", False):
            return
        _orig = Method.process_weights_after_loading

        def _pwal(self, layer):
            if _keep_wo_a_rowmajor(layer):
                return
            return _orig(self, layer)

        Method.process_weights_after_loading = _pwal
        Method._wo_a_unpack_patched = True

    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        _wrap(UnquantizedLinearMethod)
    except Exception:  # noqa: BLE001
        pass
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
        _wrap(Fp8LinearMethod)
    except Exception:  # noqa: BLE001
        pass


def _install_full_capture() -> None:
    # Comprehensive full-tensor per-layer capture (prefill AND decode) for offline cosine-per-layer
    # vs the GPU oracle. Saves each layer's OUTPUT hidden + logits as full fp32 tensors to FULLCAP_FILE.
    if _os.environ.get("INTEL_CPU_DSV4_FULLCAP") != "1":
        return
    import torch

    import sglang.srt.models.deepseek_v4 as _dv4

    Layer = _dv4.DeepseekV4DecoderLayer
    if getattr(Layer, "_fullcap_patched", False):
        return
    _F = _os.environ.get("FULLCAP_FILE", "/scratch/bkaul/dsv4_fullcap.pt")
    _CAP: dict = {}
    _ST = {"seen_pf": False, "dc": 0}

    def _rk():
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank as _g
            return _g()
        except Exception:  # noqa: BLE001
            return -1

    _ofwd = Layer.forward

    def _fwd(self, *a, **k):
        lid = getattr(self, "layer_id", getattr(self, "layer_idx", -1))
        out = _ofwd(self, *a, **k)
        try:
            # Identity from forward_batch.forward_mode (review fix): T-heuristics missed B=32 decode
            # (T=batch=32, never in {2..16} nor ==1). Scan args/kwargs for the ForwardBatch.
            fb = None
            for _v in list(a) + list(k.values()):
                if hasattr(_v, "forward_mode"):
                    fb = _v
                    break
            h = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(h) and fb is not None:
                fm = fb.forward_mode
                is_dec = bool(getattr(fm, "is_decode", lambda: False)())
                if is_dec:
                    _ST["seen_pf"] = True  # decode implies prefill already happened
                    key = f"dc{_ST['dc'] // 43}.L{lid}.r{_rk()}"
                    if key not in _CAP and _ST["dc"] // 43 < 4:  # cap at 4 decode steps
                        _CAP[key] = h.detach().float().cpu()
                    _ST["dc"] += 1
                else:
                    _ST["seen_pf"] = True
                    key = f"pf.L{lid}.r{_rk()}"
                    if key not in _CAP:
                        _CAP[key] = h.detach().float().cpu()
            elif torch.is_tensor(h):
                # Fallback (no forward_batch reachable): keep old T-heuristic for prefill only.
                T = h.shape[0]
                if 2 <= T <= 16:
                    _ST["seen_pf"] = True
                    key = f"pf.L{lid}.r{_rk()}"
                    if key not in _CAP:
                        _CAP[key] = h.detach().float().cpu()
        except Exception:  # noqa: BLE001
            pass
        return out

    Layer.forward = _fwd
    Layer._fullcap_patched = True
    # Also capture the attention-block output (MQALayer) + MoE output per layer to split attn vs FFN/MHC.
    _Attn = getattr(_dv4, "MQALayer", None)
    if _Attn is not None and not getattr(_Attn, "_fullcap_patched", False):
        _oatt = _Attn.forward

        def _attw(self, *a, **k):
            r = _oatt(self, *a, **k)
            try:
                o = r[0] if isinstance(r, (tuple, list)) else r
                if torch.is_tensor(o):
                    T = o.shape[0]
                    tag = "pf" if 2 <= T <= 16 else ("dc0" if T == 1 and _ST["seen_pf"] else None)
                    if tag:
                        lid = getattr(self, "layer_id", -1)
                        key = f"{tag}.attn.L{lid}.r{_rk()}"
                        if key not in _CAP:
                            _CAP[key] = o.detach().float().cpu()
            except Exception:  # noqa: BLE001
                pass
            return r

        _Attn.forward = _attw
        _Attn._fullcap_patched = True
    _mrun = getattr(Layer, "_run_moe_ffn_dp_sync", None)
    if _mrun is not None and not getattr(Layer, "_fullcap_moe", False):
        def _moew(self, hidden_states, *a, **k):
            r = _mrun(self, hidden_states, *a, **k)
            try:
                if torch.is_tensor(r):
                    T = r.shape[0]
                    tag = "pf" if 2 <= T <= 16 else ("dc0" if T == 1 and _ST["seen_pf"] else None)
                    if tag:
                        lid = getattr(self, "layer_id", -1)
                        key = f"{tag}.moe.L{lid}.r{_rk()}"
                        if key not in _CAP:
                            _CAP[key] = r.detach().float().cpu()
            except Exception:  # noqa: BLE001
                pass
            return r
        Layer._run_moe_ffn_dp_sync = _moew
        Layer._fullcap_moe = True
    # Attention-core output (pre o_proj) + q input, to split attn-core (K latent) vs o_proj.
    try:
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend as _B,
        )

        if not getattr(_B, "_fullcap_patched", False):
            _obf = _B.forward

            def _bf(self, *a, **k):
                o = _obf(self, *a, **k)
                try:
                    q = k.get("q", a[0] if a else None)
                    layer = k.get("layer", a[3] if len(a) > 3 else None)
                    lid = getattr(layer, "layer_id", -1)
                    oo = o[0] if isinstance(o, (tuple, list)) else o
                    if torch.is_tensor(oo):
                        T = oo.shape[0]
                        tag = "pf" if 2 <= T <= 16 else ("dc0" if T == 1 and _ST["seen_pf"] else None)
                        if tag:
                            key = f"{tag}.acore.L{lid}.r{_rk()}"
                            if key not in _CAP:
                                _CAP[key] = oo.detach().float().cpu()
                            if torch.is_tensor(q):
                                kq = f"{tag}.qcore.L{lid}.r{_rk()}"
                                if kq not in _CAP:
                                    _CAP[kq] = q.detach().float().cpu()
                except Exception:  # noqa: BLE001
                    pass
                return o

            _B.forward = _bf
            _B._fullcap_patched = True
    except Exception:  # noqa: BLE001
        pass
    # Post-inverse-RoPE o[...,-64:] per layer (both CPU+GPU call fused_rope_inplace inverse=True).
    _ofr = getattr(_dv4, "fused_rope_inplace", None)
    if _ofr is not None and not getattr(_ofr, "_fullcap_wrapped", False):
        _irc = {"pf": 0, "dc0": 0}

        def _frw(q, k=None, *a, **kw):
            r = _ofr(q, k, *a, **kw)
            try:
                if kw.get("inverse", False) and torch.is_tensor(q):
                    T = q.shape[0]
                    tag = "pf" if 2 <= T <= 16 else ("dc0" if T == 1 and _ST["seen_pf"] else None)
                    if tag is not None:
                        lid = _irc[tag]
                        _irc[tag] += 1
                        key = f"{tag}.invrope.L{lid}.r{_rk()}"
                        if key not in _CAP:
                            _CAP[key] = q.detach().float().cpu()
            except Exception:  # noqa: BLE001
                pass
            return r

        _frw._fullcap_wrapped = True
        _dv4.fused_rope_inplace = _frw
    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor

        if not getattr(LogitsProcessor, "_fullcap_patched", False):
            _olp = LogitsProcessor.forward

            def _lp(self, *a, **k):
                r = _olp(self, *a, **k)
                try:
                    lg = getattr(r, "next_token_logits", None)
                    if lg is not None:
                        n = sum(1 for kk in _CAP if kk.startswith("logits"))
                        _CAP[f"logits.{n}.r{_rk()}"] = lg.detach().float().cpu()
                        torch.save(_CAP, _F)
                except Exception:  # noqa: BLE001
                    pass
                return r

            LogitsProcessor.forward = _lp
            LogitsProcessor._fullcap_patched = True
    except Exception:  # noqa: BLE001
        pass


def _install_cpu_gloo_allreduce() -> None:
    """Route the CPU TP all-reduce through torch.distributed (gloo) instead of SGLang's custom
    shared-memory all-reduce.

    parallel_state._all_reduce_in_place picks `sgl_kernel.shm_allreduce` when is_shm_available(),
    else `torch.distributed.all_reduce`. On this node the shm path is pathological: in-model
    ~3s/op under the bench (desynced spin barrier), and even the synced case ~58ms/op -- vs gloo
    ~4ms/op measured standalone (a generic collective). Our per-token TP comm is tiny (8KB x ~2/
    layer), so latency, not bandwidth, dominates; gloo is far better than the broken shm and kills
    the ~750x pathology. Gate: INTEL_CPU_DSV4_TP_GLOO=1. (A tuned oneCCL collective would be ~us,
    but it is not installed here.)
    """
    if not _os.environ.get("INTEL_CPU_DSV4_TP_GLOO"):
        return
    try:
        import sglang.srt.distributed.parallel_state as _ps

        _ps.is_shm_available = lambda *a, **k: False
        global _TP_GLOO_ACTIVE
        _TP_GLOO_ACTIVE = True
        logger.info(
            "intel_cpu_models: CPU TP all-reduce forced through torch.distributed (gloo); "
            "bypassing the pathological sgl_kernel shm all-reduce."
        )
    except Exception as e:
        logger.warning("intel_cpu_models: could not force gloo TP all-reduce: %s", e)


def _install_cpu_interleave_mem() -> None:
    """Report FULL-NODE free memory on CPU so a single tp=1 rank can hold a model larger than
    one NUMA domain (interleaved across the node with `numactl --interleave=all`).

    SGLang sizes per-rank CPU memory as total_free / n_numa (get_available_gpu_memory), so tp=1
    on a 6-domain node sees only ~1/6 of RAM and a model that exceeds one domain won't fit. For
    the tp=1-interleaved decode-roofline path (which avoids the TP shm AllReduce entirely), report
    the full node's free memory instead. Gate: INTEL_CPU_DSV4_INTERLEAVE_MEM=1. Use ONLY at tp=1.
    """
    if not _os.environ.get("INTEL_CPU_DSV4_INTERLEAVE_MEM"):
        return
    _set_numa_interleave_all()
    import sglang.srt.utils.common as _c

    _orig = _c.get_available_gpu_memory

    def _patched(device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
        if device == "cpu":
            import psutil

            return round(psutil.virtual_memory().available / (1 << 30), 3)
        return _orig(
            device, gpu_id, distributed=distributed, empty_cache=empty_cache, cpu_group=cpu_group
        )

    _c.get_available_gpu_memory = _patched
    for _mod in ("sglang.srt.model_executor.model_runner",):
        try:
            import importlib

            m = importlib.import_module(_mod)
            if hasattr(m, "get_available_gpu_memory"):
                m.get_available_gpu_memory = _patched
        except Exception:
            pass
    logger.info(
        "intel_cpu_models: CPU interleaved-memory mode -- reporting full-node free memory (tp=1)."
    )


_OPS_DUMPED = False
_PROFILED_ONCE = False


def _dump_sgl_ops_once() -> None:
    """Probe (c): authoritative in-worker enumeration of registered CPU ops.

    Runs inside the scheduler/worker process (where the model forward executes and all kernel
    shared-libs are loaded), gated by INTEL_CPU_DSV4_DUMP_OPS=<json path>. Resolves the
    'wire vs author' fork for D2: lists every dispatcher-registered sgl_kernel:: op (via
    torch._C._dispatch_get_all_op_names, which sees C++ registrations regardless of the lazy
    python wrapper), flags attention/mla/flash/index/gemm/sparse-relevant names, and records
    whether sgl_kernel.flash_mla is a REAL module (a CPU MLA serving kernel to wire) or the
    torch stub this plugin fabricates (=> authoring, not wiring).
    """
    global _OPS_DUMPED
    if _OPS_DUMPED:
        return
    out = _os.environ.get("INTEL_CPU_DSV4_DUMP_OPS", "")
    if not out:
        return
    _OPS_DUMPED = True
    import torch as _tt

    info: dict = {}
    try:
        names = list(_tt._C._dispatch_get_all_op_names())
    except Exception as e:  # noqa: BLE001
        names = []
        info["dispatch_enum_error"] = repr(e)
    sgl = sorted(n for n in names if n.startswith("sgl_kernel::"))
    kw = ("attn", "mla", "flash", "index", "gemm", "sparse", "softmax", "sdpa", "paged", "kv")
    info["sgl_kernel_op_count"] = len(sgl)
    info["sgl_kernel_ops"] = sgl
    info["attention_relevant"] = sorted(n for n in sgl if any(k in n.lower() for k in kw))
    # Is there a REAL sgl_kernel.flash_mla CPU module, or does the plugin stub it?
    fm = {}
    try:
        import sgl_kernel.flash_mla as _fm  # noqa: F401
        fm["importable"] = True
        fm["file"] = getattr(_fm, "__file__", None)
        fm["is_torch_stub"] = getattr(_fm, "__file__", None) is None
        fm["symbols"] = sorted(a for a in dir(_fm) if not a.startswith("__"))
    except Exception as e:  # noqa: BLE001
        fm["importable"] = False
        fm["error"] = repr(e)
    info["sgl_kernel.flash_mla"] = fm
    # Stock intel_amx CPU attention backend availability (the MLA-rejecting donor path).
    amx = {}
    try:
        import sgl_kernel as _sk
        amx["sgl_kernel_file"] = getattr(_sk, "__file__", None)
        amx["has_flash_attn_varlen"] = hasattr(getattr(_tt.ops, "sgl_kernel", object()), "flash_attn_varlen_func") \
            if hasattr(_tt.ops, "sgl_kernel") else False
    except Exception as e:  # noqa: BLE001
        amx["error"] = repr(e)
    info["sgl_kernel_pkg"] = amx
    try:
        _atomic_dump = __import__("json").dumps(info, indent=2)
        with open(out, "w") as _f:
            _f.write(_atomic_dump)
        logger.warning("[DUMP_OPS] wrote %d sgl_kernel ops -> %s (flash_mla real=%s)",
                       len(sgl), out, fm.get("importable") and not fm.get("is_torch_stub"))
    except Exception as e:  # noqa: BLE001
        logger.warning("[DUMP_OPS] write failed: %r", e)


def _install_decode_thread_cap() -> None:
    """Cap threads for the whole M=1 decode forward (tp=1 only; tp>1 uses core-binding headroom).

    At M=1 decode every kernel is trivially small, so the cost is the parallel-for BARRIER, not
    compute. SGLang binds the full NUMA-domain thread count (~42-43, ignoring OMP_NUM_THREADS);
    using the FULL bound count starves the main/framework/OS thread and the barrier stalls -- a
    fixed-cap sweep (proxy, 4-layer, tp=1) is monotonic up to bound-2 then cliffs at the bound:
    32->0.182s, 40(bound-2)->0.062s, 42(bound)->12.8s (~200x). Capping the decode forward to
    bound-2 gives the 200x win. `INTEL_CPU_DSV4_DECODE_THREADS=N` fixes the count; `=auto` uses
    bound - INTEL_CPU_DSV4_DECODE_HEADROOM (default 2).

    tp>1 CAVEAT: SGLang's TP AllReduce is a shared-memory SPIN BARRIER (sgl_kernel.initialize)
    sized to the thread count at init. Calling set_num_threads() per-forward changes the count
    after init and DEADLOCKS the barrier at tp>1. So the per-forward cap is applied ONLY at
    world_size==1. For tp>1, leave the headroom via `SGLANG_CPU_OMP_THREADS_BIND` (bind each rank
    to its domain cores MINUS ~2) -- set before launch, so the count is fixed once at init and the
    shm barrier is sized correctly (tp-safe, no hang).
    """
    spec = _os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS", "")
    ctl_file = _os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS_FILE", "")
    # Accept: digit (fixed), "auto" (bound-headroom), an M-ADAPTIVE map "M1:T1,M2:T2,..." (pick the
    # threads for the largest breakpoint M<=current decode batch), or a dynamic control FILE (one int,
    # re-read each forward) that lets a PERSISTENT process sweep threads on the RESIDENT model.
    is_adaptive = ":" in spec
    if not (spec.isdigit() or spec == "auto" or is_adaptive or ctl_file):
        return
    import torch as _tt

    import sglang.srt.models.deepseek_v4 as _dv4

    cls = _dv4.DeepseekV4ForCausalLM
    if getattr(cls, "_decode_thread_capped", False):
        return
    _orig_fwd = cls.forward
    fixed = int(spec) if spec.isdigit() else None
    headroom = int(_os.environ.get("INTEL_CPU_DSV4_DECODE_HEADROOM", "2"))
    _sticky = _os.environ.get("INTEL_CPU_DSV4_DECODE_THREADS_STICKY", "0") == "1"
    adaptive = None
    if is_adaptive:
        try:
            adaptive = sorted((int(a), int(b)) for a, b in (p.split(":") for p in spec.split(",")))
        except Exception:  # noqa: BLE001
            adaptive = None
    st = {}

    def _fwd(self, input_ids, positions, forward_batch, *a, **k):
        _dump_sgl_ops_once()
        if "world" not in st:
            try:
                import torch.distributed as _d

                st["world"] = _d.get_world_size() if (_d.is_available() and _d.is_initialized()) else 1
            except Exception:
                st["world"] = 1
        world = st["world"]
        # tp>1: DO NOT touch the thread count. The sgl_kernel shm spin barriers (MoE dispatch,
        # all-reduce) are sized at init_cpu_threads_env and any post-init set_num_threads() desyncs
        # them -> DEADLOCK. Verified both ways: per-forward set/restore hung, and a permanent set
        # hung too (also ranks get UNEQUAL counts since domains differ 42 vs 43 cores). To cap M=1
        # decode threads at tp>1 you must reduce the count AT INIT (before the barriers size), which
        # this hook cannot do. So tp>1 runs uncapped here (see the SNC/domain-size analysis: tp>1
        # M=1 decode is ~330x slower than tp=1 on this stack regardless).
        if world > 1:
            if "n" not in st:
                st["n"] = 1
                logger.warning(
                    "[DECODE THREADCAP] tp=%d -> cap SKIPPED (post-init set_num_threads desyncs the "
                    "shm barriers -> deadlock; count must be reduced at init, not here).",
                    world,
                )
            return _orig_fwd(self, input_ids, positions, forward_batch, *a, **k)
        # tp=1: safe to cap per-decode-forward and restore.
        is_decode = False
        try:
            is_decode = forward_batch.forward_mode.is_decode_or_idle()
        except Exception:
            pass
        if not is_decode:
            return _orig_fwd(self, input_ids, positions, forward_batch, *a, **k)
        prev = _tt.get_num_threads()
        try:
            M = int(input_ids.shape[0])
        except Exception:  # noqa: BLE001
            M = 1
        # dynamic control FILE (one int) wins — lets a persistent process sweep threads per experiment
        want = None
        if ctl_file:
            try:
                with open(ctl_file) as _cf:
                    want = int(_cf.read().strip())
            except Exception:  # noqa: BLE001
                want = None
        if want is None and adaptive:  # M-adaptive: largest breakpoint M <= current decode batch
            want = adaptive[0][1]
            for mb, tb in adaptive:
                if M >= mb:
                    want = tb
        if want is None:
            want = fixed if fixed is not None else max(1, prev - headroom)
        cap = min(max(1, want), prev)
        if st.get("last_cap") != cap or st.get("last_M") != M:
            st["last_cap"], st["last_M"] = cap, M
            logger.warning("[DECODE THREADCAP] M=%d threads %d -> %d%s", M, prev, cap,
                           " (sticky)" if _sticky else "")
        _tt.set_num_threads(cap)
        try:
            _prof_out = _os.environ.get("INTEL_CPU_DSV4_PROFILE_OUT")
            global _PROFILED_ONCE
            if _prof_out and is_decode and not _PROFILED_ONCE:
                _PROFILED_ONCE = True
                import json as _json

                import torch.profiler as _prof
                with _prof.profile(activities=[_prof.ProfilerActivity.CPU]) as _p:
                    _r = _orig_fwd(self, input_ids, positions, forward_batch, *a, **k)
                # independent cross-check of the provenance COUNTER: the actual op-level self-CPU time.
                # Confirm C kernels (dsv4ck*/sgl_kernel) dominate and no unexpected eager aten op is large.
                _evs = sorted(_p.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:30]
                _top = [{"op": e.key, "self_cpu_ms": round(e.self_cpu_time_total / 1000.0, 3),
                         "calls": e.count} for e in _evs]
                try:
                    with open(_prof_out, "w") as _f:
                        _json.dump({"M": M, "top_ops_by_self_cpu_ms": _top}, _f, indent=2)
                    logger.warning("[PROFILE] wrote one-decode-step op profile -> %s", _prof_out)
                except Exception:  # noqa: BLE001
                    pass
                return _r
            return _orig_fwd(self, input_ids, positions, forward_batch, *a, **k)
        finally:
            # STICKY (measurement): leave threads at `cap` so there is NO per-forward 58->T->58 resize
            # churn (a confound in the resident sweep). Default stays restore-prev (prefill gets full cores).
            if not _sticky:
                _tt.set_num_threads(prev)

    cls.forward = _fwd
    cls._decode_thread_capped = True
    global _DECODE_CAP_ACTIVE
    _DECODE_CAP_ACTIVE = True
    logger.info(
        "intel_cpu_models: decode thread cap = %s (headroom=%d; tp=1 per-forward, tp>1 permanent)",
        spec, headroom,
    )


def _install_layer_timeit() -> None:
    # TIMEIT-only: quantify the UNTIMED per-layer framework/wiring glue (MHC hc_pre/hc_post, norms,
    # rope, residual, Python dispatch) that scales ~43x on the full net. layer.total minus the timed
    # child kernels (moe/attn/dense/mhc) = the glue remainder. Reach for this before extrapolating a
    # truncated-model profile to 43 layers.
    if not _TIMEIT_ON or not current_platform.is_cpu():
        return
    import sglang.srt.models.deepseek_v4 as _dv4

    Layer = _dv4.DeepseekV4DecoderLayer
    if getattr(Layer, "_layer_timeit", False):
        return
    Layer.forward = _timed("layer.total", "parent")(Layer.forward)
    for _m, _tag in (("hc_pre", "mhc.hc_pre"), ("hc_post", "mhc.hc_post")):
        _fn = getattr(Layer, _m, None)
        if callable(_fn):
            setattr(Layer, _m, _timed(_tag, "torch")(_fn))
    Layer._layer_timeit = True
    # lm_head (LogitsProcessor) + the whole-model forward are PER-STEP fixed ops OUTSIDE layer.total:
    # time them so the layer-scaling counterfactual can split per-step-fixed (lm_head/embed/sample,
    # NOT recoverable by per-layer fusion) from per-layer (recoverable). Tagged 'fixed'.
    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor as _LP
        if not getattr(_LP, "_lmhead_timeit", False):
            _LP.forward = _timed("step.lm_head", "fixed")(_LP.forward)
            _LP._lmhead_timeit = True
    except Exception:  # noqa: BLE001
        pass
    try:
        _MF = _dv4.DeepseekV4Model
        if not getattr(_MF, "_model_timeit", False):
            _MF.forward = _timed("step.model_total", "stepparent")(_MF.forward)
            _MF._model_timeit = True
    except Exception:  # noqa: BLE001
        pass


def _install_gemm_timeit() -> None:
    """TIMEIT-only: wall-time the dense linear GEMMs in-model (they are ~0.02ms isolated but
    the cProfile decode profile attributed ~0.25s/call — measure the true in-model wall time
    and the actual thread count, without the profiler perturbing it).

    Also supports an experimental per-call thread cap (INTEL_CPU_DSV4_DENSE_THREADS=N): the
    dense M=1 GEMM runs at torch's default 42 threads in-model (SGLang binds the NUMA-domain
    core count and ignores OMP_NUM_THREADS); capping tests whether that 42-thread contended
    barrier is the ~0.25s/call pathology."""
    dense_threads = _os.environ.get("INTEL_CPU_DSV4_DENSE_THREADS", "")
    dense_cap = int(dense_threads) if dense_threads.isdigit() else None
    if not _TIMEIT_ON and dense_cap is None:
        return
    import time

    import torch as _tt

    def _wrap(cls, label):
        if getattr(cls, "_gemm_timed", False):
            return
        orig = cls.apply

        def _apply(self, *a, **k):
            if "gemm.threads" not in _MOE_DIAG:
                _MOE_DIAG["gemm.threads"] = _tt.get_num_threads()
                logger.warning(
                    "[GEMM TIMEIT] torch.get_num_threads()=%d dense_cap=%s",
                    _MOE_DIAG["gemm.threads"], dense_cap,
                )
            t0 = time.perf_counter() if _TIMEIT_ON else 0.0
            prev = None
            if dense_cap is not None:
                prev = _tt.get_num_threads()
                _tt.set_num_threads(dense_cap)
            try:
                return orig(self, *a, **k)
            finally:
                if prev is not None:
                    _tt.set_num_threads(prev)
                if _TIMEIT_ON:
                    _tacc(label, t0, "kernel")

        cls.apply = _apply
        cls._gemm_timed = True

    try:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        _wrap(Fp8LinearMethod, "dense.fp8_linear")
    except Exception:
        pass
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        _wrap(UnquantizedLinearMethod, "dense.bf16_linear")
    except Exception:
        pass


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


def _fp8_roundtrip_nope_bf16(k_bf16):
    # QAT KV match: fused fp8 quant+DEQUANT of the NOPE dims (rope dims stay bf16), mirroring the
    # reference `act_quant(kv[...,:-rope], 64, ue8m0, inplace=True)` (inference/model.py). Returns a
    # bf16 tensor whose nope dims are fp8(e4m3, pow2/ue8m0 per-64-tile)-ROUND-TRIPPED -> the KEYS the
    # QAT-trained model expects. rope dims unchanged (bf16).
    import torch

    from sglang.kernels.ops.attention.dsv4.index_buf_accessor import fp8_dtype

    T = k_bf16.shape[0]
    if T == 0:
        return k_bf16
    dim_nope, tile = 448, 64
    num_tiles = dim_nope // tile
    fi = torch.finfo(fp8_dtype)
    nope = k_bf16[:, :dim_nope].float().reshape(T, num_tiles, tile)
    max_abs = nope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    ceil_log2 = torch.ceil(torch.log2(max_abs / fi.max))
    scale = torch.exp2(ceil_log2)
    x_fp8 = (nope / scale).clamp(fi.min, fi.max).to(fp8_dtype)
    nope_rt = (x_fp8.float() * scale).reshape(T, dim_nope)  # dequant back = fp8-rounded nope
    return torch.cat([nope_rt, k_bf16[:, dim_nope:].float()], dim=-1).to(torch.bfloat16)


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
    # K-path shadow: independent complex-multiply rope + nope-only-vs-full RMS scope check.
    if _os.environ.get("INTEL_CPU_DSV4_KROPE_SHADOW") == "1" and kv.shape[0] > 0 and _KROPE_DBG["n"] < 4:
        _KROPE_DBG["n"] += 1
        try:
            fc = freqs_cis[positions.long()]  # complex [T, rope_dim//2]
            # Compare on the kernel's ACTUAL rope input xn[rope] (normed+weighted): interleaved
            # stack (kernel `roped`) vs independent complex-multiply. No weight-stripping (invalid:
            # per-element weight lives inside the rotation). cos<1 => real kernel-rope impl bug.
            xn_rope = xn[:, nope_dim:].reshape(-1, rope_dim // 2, 2).contiguous()
            roped_ref = torch.view_as_real(torch.view_as_complex(xn_rope) * fc).reshape(-1, rope_dim)
            cos = torch.nn.functional.cosine_similarity(
                roped.float().reshape(1, -1), roped_ref.reshape(1, -1)
            ).item()
            err = (roped.float() - roped_ref).abs().max().item()
            with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_krope.txt"), "a") as fh:
                fh.write(f"[KROPE#{_KROPE_DBG['n']}] rope_cos={cos:.6f} rope_maxerr={err:.3e} (cos<1 => kernel rope impl bug)\n")
        except Exception as _e:  # noqa: BLE001
            try:
                with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_krope.txt"), "a") as fh:
                    fh.write(f"[KROPE] shadow failed: {_e}\n")
            except Exception:  # noqa: BLE001
                pass
    # Stash the dense keys for the CPU torch MLA attention. QAT FIX: the authoritative reference
    # (inference/model.py Attention.forward) does `act_quant(kv[...,:-rope], 64, ue8m0, inplace=True)`
    # -> the NOPE dims are fp8-ROUND-TRIPPED (the model is QAT-trained for fp8 KV). Stashing the raw
    # bf16 kv_out (the old "numerically exact" path) feeds the model KEYS MORE PRECISE than it was
    # trained for -> systematic attention divergence from layer 0. Match the reference: fp8-round the
    # nope dims (rope stays bf16). INTEL_CPU_DSV4_KV_BF16_EXACT=1 reverts to the old bf16 path (A/B).
    if _os.environ.get("INTEL_CPU_DSV4_KV_BF16_EXACT") == "1":
        kv_stash = kv_out
    else:
        kv_stash = _fp8_roundtrip_nope_bf16(kv_out)
    _stash_kv_write(kvcache.data_ptr(), out_loc, kv_stash)
    _iba._set_k_and_s_torch(
        kvcache, out_loc, _cpu_quant_to_nope_fp8_rope_bf16_pack(kv_out), page_size
    )


# Dense bf16 KV stash as a CONTIGUOUS buffer per paged-buffer data_ptr: _KV_BUF[dp] holds
# keys[pool, 512], _KV_VALID[dp] marks written slots. The CPU MLA attention gathers with a
# single index_select instead of a Python dict+stack (the per-token loop was ~46s of decode
# dispatch overhead). Populated by the CPU K-write below; read by the MLA attention.
_KV_BUF: dict = {}
_KV_VALID: dict = {}


def _stash_kv_write(dp: int, out_loc, kv_out) -> None:
    import torch

    ol = out_loc.long()
    m = ol >= 0
    if not bool(m.any()):
        return
    olv = ol[m]
    kvv = kv_out[m].detach()
    hd = kvv.shape[-1]
    need = int(olv.max().item()) + 1
    buf = _KV_BUF.get(dp)
    if buf is None:
        cap = max(need, 1024)
        buf = torch.zeros(cap, hd, dtype=kvv.dtype)
        _KV_BUF[dp] = buf
        _KV_VALID[dp] = torch.zeros(cap, dtype=torch.bool)
    elif need > buf.shape[0]:
        cap = max(need, buf.shape[0] * 2)
        nb = torch.zeros(cap, hd, dtype=buf.dtype)
        nb[: buf.shape[0]] = buf
        nv = torch.zeros(cap, dtype=torch.bool)
        nv[: buf.shape[0]] = _KV_VALID[dp]
        _KV_BUF[dp] = buf = nb
        _KV_VALID[dp] = nv
    buf[olv] = kvv.to(buf.dtype)
    _KV_VALID[dp][olv] = True


def _stash_kv_gather(dp: int, idx, out_dtype=None):
    # Filter idx to written slots and gather rows in one index_select. Returns None if empty.
    # out_dtype=bfloat16 keeps the gathered KV in the buffer's native bf16 (the attention GEMMs
    # consume bf16 anyway) -> skips the bf16->f32 upcast here + the f32->bf16 cvt downstream.
    # bf16->f32->bf16 is lossless, so bf16 out is bit-exact vs the old .float() path.
    import torch

    buf = _KV_BUF.get(dp)
    if buf is None or idx.numel() == 0:
        return None
    valid = _KV_VALID[dp]
    idx = idx[idx < buf.shape[0]]
    idx = idx[valid[idx]]
    if idx.numel() == 0:
        return None
    g = buf.index_select(0, idx)
    tgt = torch.float32 if out_dtype is None else out_dtype
    return g if g.dtype == tgt else g.to(tgt)

# Route-2 DSA: per-layer compressed-KV stash keyed by (layer_id, compress_ratio) -> the
# RMSNorm'd compressed KV [N_compressed, head_dim]. Written by the CPU compressor forward,
# read by the CPU indexer (step 3). Avoids the paged state-pool entirely.
_COMPRESSED_STASH: dict = {}

# Route-2-DIRECT: the corrected compressor publishes the main-attention compressed KV here,
# keyed by extra_k_cache.data_ptr() (shared buffer between compressor + attention). Value =
# (comp_kv_bf16 [S, head_dim], block_start_abs_pos [S], ratio, query_abs_pos [T]). Consumed
# (and cleared) by _batched_mla_attend of the SAME layer with a causal coverage mask. Bypasses
# the CPU-disabled paged slot plan (out_loc / c4_sparse_page_indices are uninitialized).
_COMPRESSED_HANDOFF: dict = {}

# Decode persistence: prefill stores each request's compressed KV here keyed by (layer_id,
# req_pool_index) so decode steps (which produce no new full blocks for short gens) can re-publish it.
_COMPRESSED_PERSIST: dict = {}

# Route-2 DSA step 3: per-layer indexer top-k KV indices [N_q, k], written by the CPU
# forward_c4_indexer, read by the CPU sparse attention (step 4).
_TOPK_STASH: dict = {}
_HAD_CACHE: dict = {}

# Route-2 DSA step 4: the indexer publishes a per-query position-keep mask [N, S*ratio]
# here; the very next MLA attention (same layer) consumes AND clears it. A non-indexer layer
# leaves it None -> that layer's attention stays dense. Handles alternating c4/c128 layers.
_SEL_HOLDER: dict = {"pos_kept": None}
# Sliding-window size for the pure-SWA (indices=None) attention mask. Default 128 (DSV4
# config.sliding_window); env-overridable. Set once from config at attention install time.
_SLIDING_WINDOW = int(_os.environ.get("INTEL_CPU_DSV4_SLIDING_WINDOW", "128"))

# One-shot attention diagnostic (gated by INTEL_CPU_DSV4_ATTN_DEBUG=1): file-based so it survives
# the Engine scheduler subprocess whose stdout the sbatch log does not capture.
_ATTN_DBG: dict = {"n": 0}
_KROPE_DBG: dict = {"n": 0}
_DSA_DIAG: dict = {"n": 0}
_BOUNDARY_CAP: dict = {}
_COMP_INPUT_CAP: dict = {}
_MLA_PATH_DBG: dict = {"n": 0}

# Per-layer hidden-state diagnostic (gated by INTEL_CPU_DSV4_HID_DEBUG=1): logs each decoder
# layer's input/output activation stats to a file to localize where the forward goes bad.
_HID_DBG: dict = {"n": 0}


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


def _build_coverage_mask_vectorized(T, n, vslots, indices, topk_length, pos_kept):
    """Vectorized DSA prefill coverage mask — bit-identical to the per-token loop it replaces.

    Proven equivalent over 4000 fuzzed cases (plugin/validate/test_vectorized_mask.py), so the
    attention output (and thus accuracy) is unchanged; this removes the ~19%-of-prefill Python-loop
    cost (`mla.b.mask`, job 393238). INTEL_CPU_DSV4_MASK_LOOP=1 reverts to the loop for A/B.
    """
    import torch

    inv = torch.full((int(vslots.max()) + 1,), -1, dtype=torch.long)
    inv[vslots] = torch.arange(n)
    idxC = indices.reshape(T, -1).long()                      # [T, C] per-token candidate slots
    C = idxC.shape[1]
    col = torch.arange(C)
    within_L = col.unsqueeze(0) < topk_length.reshape(T, 1).long()
    valid = within_L & (idxC >= 0)                            # loop's idx[:L][idx>=0]
    if pos_kept is not None:
        cov = int(pos_kept.shape[1])
        Tp = int(pos_kept.shape[0])
        rank = torch.cumsum(valid.long(), dim=1) - 1          # rank among valid entries (compacted idx pos)
        rank_cl = rank.clamp(min=0, max=cov - 1)
        pk = pos_kept.bool()
        pk_gather = pk[torch.arange(Tp).reshape(Tp, 1), rank_cl[:Tp]]   # pos_kept[t, rank]
        pk_full = (rank >= cov) | torch.cat(
            [pk_gather, torch.zeros(T - Tp, C, dtype=torch.bool)], dim=0
        )
        has_valid = valid.any(dim=1)
        apply = torch.zeros(T, dtype=torch.bool)
        apply[:Tp] = has_valid[:Tp]                           # loop guards t<Tp AND idx nonempty
        keep = torch.where(apply.reshape(T, 1), valid & pk_full, valid)
    else:
        keep = valid
    mask_main = torch.zeros(T, n, dtype=torch.bool)
    kept_t, kept_c = keep.nonzero(as_tuple=True)
    if kept_t.numel() > 0:
        pos = inv[idxC[kept_t, kept_c]]
        ok = pos >= 0
        mask_main[kept_t[ok], pos[ok]] = True
    # empty-fallback: tokens with no kept key get a causal prefix mask [:cl], cl=clamp(n-(T-1)+t,0,n)
    empties = (keep.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if empties.numel() > 0:
        tt = torch.arange(T)
        cl = (n - (T - 1) + tt).clamp(min=0, max=n)
        fill = torch.arange(n).unsqueeze(0) < cl.reshape(T, 1)
        mask_main[empties] = fill[empties]
    return mask_main


def _batched_mla_attend(
    q, dp, edp, D, head_dim_v, softmax_scale, indices, topk_length,
    extra_indices, extra_topk, attn_sink, pos_kept, force_dense
):
    # Batched bf16 MLA attention: the per-token softmax-attention loop collapsed into two AMX
    # GEMMs (scores=QK^T, out=PV) over the FULL valid KV plus a per-token key mask. K==V (MLA),
    # shared across query tokens, so both GEMMs are a single [T*H,*] matmul. Masked keys get
    # p=0 -> identical to the loop (softmax-weighted sum is order-invariant). Two-source KV
    # (main compressed + extra SWA/compressor) is concatenated into one K_all with a joint mask.
    # Returns out [T,H,hdv] fp32, or None to fall back to the reference loop.
    import os
    import time

    import torch

    _ts = time.perf_counter() if _TIMEIT_ON else 0.0
    T, _, H, _Dq = q.shape
    valid = _KV_VALID.get(dp)
    if valid is None:
        return torch.zeros(T, H, head_dim_v, dtype=torch.float32)
    vslots = valid.nonzero(as_tuple=False).reshape(-1)
    n = int(vslots.numel())
    if n == 0:
        return torch.zeros(T, H, head_dim_v, dtype=torch.float32)
    K_main = _stash_kv_gather(dp, vslots, out_dtype=torch.bfloat16)
    if K_main is None:
        return None
    if _TIMEIT_ON:
        _tacc("mla.b.gather", _ts, "framework")
        _ts = time.perf_counter()
    ar = torch.arange(n)
    # ABLATION (default-inert): INTEL_CPU_DSV4_ABLATE_MASK=1 forces the CHEAP vectorized dense-causal
    # mask branch instead of the expensive per-token Python loop (and skips the compressed-KV coverage
    # mask below). The prefill throughput delta vs baseline = the UPPER BOUND of fusing/vectorizing the
    # mask construction (the measured ~19% `mla.b.mask` glue). Timing-only: attention output is a
    # (valid but non-sparse) dense-causal approximation, so pair with the moe.expert_apply TIMEIT
    # stability check to rule out a routing-shift confound.
    _ablate_mask = _os.environ.get("INTEL_CPU_DSV4_ABLATE_MASK") == "1"
    if indices is None or topk_length is None or force_dense or _ablate_mask:
        # SLIDING-WINDOW causal (reference get_window_topk_idxs, window_size=128): query at absolute
        # position `base+t` attends ONLY to keys in [base+t-(W-1), base+t], NOT all prior keys. Our
        # old dense-causal attended to ALL prior -> softmax spread over ~W/n too many keys (the "2x
        # flatter" symptom) -> divergence. INTEL_CPU_DSV4_NO_SWA_MASK=1 reverts to dense (A/B).
        _W = _SLIDING_WINDOW
        hi = (n - (T - 1)) + torch.arange(T) - 1            # absolute causal upper bound per query
        mask_main = (ar.unsqueeze(0) <= hi.clamp(min=-1, max=n - 1).unsqueeze(1))
        if _W and _W > 0 and _os.environ.get("INTEL_CPU_DSV4_NO_SWA_MASK") != "1":
            mask_main = mask_main & (ar.unsqueeze(0) > (hi - _W).unsqueeze(1))  # lower window bound
    else:
        if _os.environ.get("INTEL_CPU_DSV4_MASK_LOOP") == "1":
            # Legacy per-token Python loop (A/B escape hatch). The default below is a VECTORIZED
            # equivalent, proven BIT-IDENTICAL over 4000 fuzzed cases (plugin/validate/test_vectorized_mask.py)
            # so accuracy is unchanged; it removes the ~19%-of-prefill `mla.b.mask` Python-loop cost.
            inv = torch.full((int(vslots.max()) + 1,), -1, dtype=torch.long)
            inv[vslots] = torch.arange(n)
            mask_main = torch.zeros(T, n, dtype=torch.bool)
            for t in range(T):
                L = int(topk_length[t])
                idx = indices[t].reshape(-1)[:L].long()
                idx = idx[idx >= 0]
                if pos_kept is not None and t < pos_kept.shape[0] and idx.numel() > 0:
                    cov = int(pos_kept.shape[1])
                    m = min(cov, idx.shape[0])
                    keep = torch.ones(idx.shape[0], dtype=torch.bool)
                    if m > 0:
                        keep[:m] = pos_kept[t, :m].bool()
                    idx = idx[keep]
                if idx.numel() == 0:
                    cl = max(0, min(n, n - (T - 1) + t))
                    mask_main[t, :cl] = True
                else:
                    pos = inv[idx]
                    mask_main[t, pos[pos >= 0]] = True
        else:
            mask_main = _build_coverage_mask_vectorized(T, n, vslots, indices, topk_length, pos_kept)
            if _os.environ.get("INTEL_CPU_DSV4_MASK_ASSERT") == "1":
                # In-situ bit-exact guard: recompute via the loop on the REAL operands and assert
                # equality (proves the vectorization holds for the actual indices/topk_length shapes,
                # not just the offline fuzz). Expensive — validation runs only.
                _m_ref = torch.zeros(T, n, dtype=torch.bool)
                _invr = torch.full((int(vslots.max()) + 1,), -1, dtype=torch.long)
                _invr[vslots] = torch.arange(n)
                for _t in range(T):
                    _L = int(topk_length[_t]); _idx = indices[_t].reshape(-1)[:_L].long(); _idx = _idx[_idx >= 0]
                    if pos_kept is not None and _t < pos_kept.shape[0] and _idx.numel() > 0:
                        _cov = int(pos_kept.shape[1]); _m = min(_cov, _idx.shape[0])
                        _keep = torch.ones(_idx.shape[0], dtype=torch.bool)
                        if _m > 0:
                            _keep[:_m] = pos_kept[_t, :_m].bool()
                        _idx = _idx[_keep]
                    if _idx.numel() == 0:
                        _cl = max(0, min(n, n - (T - 1) + _t)); _m_ref[_t, :_cl] = True
                    else:
                        _p = _invr[_idx]; _m_ref[_t, _p[_p >= 0]] = True
                if not torch.equal(mask_main, _m_ref):
                    raise AssertionError(
                        f"[DSV4] vectorized coverage mask != loop (T={T},n={n}): "
                        f"{int((mask_main ^ _m_ref).sum())} cells differ")
    K_parts, mask_parts = [K_main], [mask_main]
    # Route-2-DIRECT compressed KV: consume the corrected compressor's per-edp handoff (bypasses the
    # CPU-disabled paged slot plan whose extra_indices/out_loc are uninitialized). Append the
    # compressed blocks as a 2nd KV source with a causal COVERAGE mask: block j (start abs pos
    # bpos[j]) is visible to query at abs pos p iff (j+1)*ratio <= p+1  <=>  p >= bpos[j]+ratio-1.
    _handoff = _COMPRESSED_HANDOFF.pop(edp, None) if edp is not None else None
    if _handoff is not None and not _ablate_mask and _os.environ.get("INTEL_CPU_DSV4_COMPRESSOR_FIX", "1") == "1":
        comp_kv = _handoff["comp_kv"]
        block_pos = _handoff["block_pos"]
        block_reqid = _handoff["block_reqid"]
        query_pos = _handoff["query_pos"]
        query_reqid = _handoff["query_reqid"]
        cratio = _handoff["ratio"]
        S = comp_kv.shape[0]
        if S > 0:
            qp = query_pos.reshape(-1)[-T:].to(torch.long)      # this call's query positions
            qr = query_reqid.reshape(-1)[-T:].to(torch.long)    # this call's query request ids
            bp = block_pos.reshape(1, -1).to(torch.long)
            br = block_reqid.reshape(1, -1).to(torch.long)
            # same-request AND causal coverage (block j visible iff (j+1)*ratio <= p+1)
            vis = (qr.unsqueeze(1) == br) & (qp.unsqueeze(1) >= bp + int(cratio) - 1)  # [T,S]
            K_parts.append(comp_kv[:, :K_main.shape[1]].to(torch.bfloat16))
            mask_parts.append(vis)
    elif edp is not None and extra_indices is not None and extra_topk is not None:
        # Legacy paged-index path (used only when the handoff is absent, e.g. COMPRESSOR_FIX=0).
        per_tok = []
        for t in range(T):
            EL = int(extra_topk[t])
            e = extra_indices[t].reshape(-1)[:EL].long()
            per_tok.append(e[e >= 0])
        cat_e = torch.cat(per_tok) if per_tok else torch.empty(0, dtype=torch.long)
        if cat_e.numel() > 0:
            all_e = torch.unique(cat_e)
            K_extra = _stash_kv_gather(edp, all_e, out_dtype=torch.bfloat16)
            if K_extra is not None:
                ne = int(all_e.numel())
                inv_e = torch.full((int(all_e.max()) + 1,), -1, dtype=torch.long)
                inv_e[all_e] = torch.arange(ne)
                mask_e = torch.zeros(T, ne, dtype=torch.bool)
                for t in range(T):
                    if per_tok[t].numel() > 0:
                        pe = inv_e[per_tok[t]]
                        mask_e[t, pe[pe >= 0]] = True
                K_parts.append(K_extra)
                mask_parts.append(mask_e)
    if _os.environ.get("INTEL_CPU_DSV4_DSA_DIAG") == "1" and _DSA_DIAG["n"] < 6:
        _DSA_DIAG["n"] += 1
        try:
            _kx = K_parts[1] if len(K_parts) > 1 else None
            _msg = (
                "[DSA DIAG #%d] T=%d K_main(window)=%d parts=%d extra_present=%s "
                "extra_K=%s handoff=%s edp_set=%s | K_main last-row rope[-8:]=%s"
            ) % (
                _DSA_DIAG["n"], T, n, len(K_parts), len(K_parts) > 1,
                (tuple(_kx.shape) if _kx is not None else None),
                _handoff is not None,
                edp is not None,
                [round(float(z), 3) for z in K_main[-1, -8:].tolist()],
            )
            with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_dsa_diag.txt"), "a") as _f:
                _f.write(_msg + "\n")
        except Exception:  # noqa: BLE001
            pass
    K_all = torch.cat(K_parts, 0) if len(K_parts) > 1 else K_parts[0]
    mask = torch.cat(mask_parts, 1) if len(mask_parts) > 1 else mask_parts[0]
    if _TIMEIT_ON:
        _tacc("mla.b.mask", _ts, "framework")
        _ts = time.perf_counter()
    nk = K_all.shape[0]
    qb = q[:, 0].to(torch.bfloat16).reshape(T * H, _Dq)[:, :D]
    # K_all is already bf16 from the gather -> Kb/Vb are plain slices (contiguous for the AMX
    # GEMMs); the old .to(bfloat16) round-trip is gone.
    Kb = K_all[:, :D].contiguous()
    Vb = K_all[:, :head_dim_v].contiguous()
    _fused = os.environ.get("INTEL_CPU_DSV4_ATTN_FUSED") == "1"
    if _fused:
        qb = qb.contiguous()  # [:, :D] slice is strided -> matmul would copy each call
    if _TIMEIT_ON:
        _tacc("mla.b.cvt", _ts, "framework")
        _ts = time.perf_counter()
    # FUSED C/C++ sparse attend (serving-integration, graph-loop): shared kv + per-query mask + per-head
    # sink, online-softmax, no score/p DRAM materialization. Gated by DSA_CKERNEL; the eager block below
    # is the A/B oracle + fallback. Requires the absorbed-MLA form k==v (D == head_dim_v) and a sink.
    _use_ck_attn = False
    try:
        from . import _dsa_ckernels as _ck
        _use_ck_attn = _ck.enabled("sparse_attend") and head_dim_v == D and attn_sink is not None
    except Exception:  # noqa: BLE001
        _use_ck_attn = False
    if _use_ck_attn:
        try:
            q_in = q[:, 0, :, :D].to(torch.float32).contiguous()   # [T,H,D]
            kv_sh = K_all[:, :D].to(torch.float32).contiguous()    # [nk,D] shared latent (k==v)

            def _oracle():
                sc = torch.matmul(q_in.reshape(T * H, D), kv_sh.t()).reshape(T, H, nk) * softmax_scale
                sc = sc.masked_fill(~mask.unsqueeze(1), float("-inf"))
                snk = attn_sink.reshape(1, H, 1).float().expand(T, H, 1)
                pp = torch.nan_to_num(torch.cat([sc, snk], dim=-1).softmax(dim=-1)[..., :-1], nan=0.0)
                return torch.matmul(pp.reshape(T * H, nk), kv_sh).reshape(T, H, D)

            out = _ck.sparse_attend_masked(q_in, kv_sh, mask, attn_sink, float(softmax_scale), _oracle)
            if _TIMEIT_ON:
                _tacc("mla.b.pv", _ts, "kernel")
            _maybe_boundary_capture(q, K_all, mask, softmax_scale, attn_sink, out, _handoff, _KV_VALID.get(dp), n, T, H, D, head_dim_v)
            return out
        except Exception as _e:  # noqa: BLE001 — parity fail / bad shape -> eager oracle fallback
            if _ck.strict():
                raise
            logger.warning("[DSA] sparse_attend C kernel fell back to eager: %r", _e)
    scores = torch.matmul(qb.float(), Kb.float().t()).reshape(T, H, nk) * softmax_scale  # fp32 scores (match GPU FlashMLA)
    try:
        from . import _dsa_ckernels as _ckm
        _ckm.mark("sparse_attend", "eager")
    except Exception:  # noqa: BLE001
        pass
    if _TIMEIT_ON:
        _tacc("mla.b.qk", _ts, "kernel")
        _ts = time.perf_counter()
    scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))
    if _fused:
        # Manual stable softmax: fold the attention sink into the normalizer instead of
        # cat([scores,sink])+softmax+slice, and drop nan_to_num. Fewer eager passes/materializations
        # (the 5.8s softmax is dispatch/pass-bound, not exp-bound). Equivalent to the cat path.
        if attn_sink is not None:
            sink = attn_sink.reshape(1, H, 1).float()
            m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
            e = torch.exp(scores - m)
            # empty row (all keys masked): scores=-inf, m=sink -> e=0, denom=exp(0)=1 -> p=0 (matches loop)
            p = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sink - m))
        else:
            m = scores.amax(dim=-1, keepdim=True)
            e = torch.exp(scores - m)
            p = torch.nan_to_num(e / e.sum(dim=-1, keepdim=True), nan=0.0)
    elif attn_sink is not None:
        sink = attn_sink.reshape(1, H, 1).float().expand(T, H, 1)
        p = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
        p = torch.nan_to_num(p, nan=0.0)
    else:
        p = scores.softmax(dim=-1)
        p = torch.nan_to_num(p, nan=0.0)  # tokens with no keys -> zero row (matches the loop's out[t]=0)
    if _TIMEIT_ON:
        _tacc("mla.b.softmax", _ts, "kernel")
        _ts = time.perf_counter()
    out = torch.matmul(p.reshape(T * H, nk).float(), Vb.float()).reshape(T, H, head_dim_v)  # fp32 PV (no bf16 round of p; match GPU)
    if _TIMEIT_ON:
        _tacc("mla.b.pv", _ts, "kernel")
    _maybe_boundary_capture(q, K_all, mask, softmax_scale, attn_sink, out, _handoff, _KV_VALID.get(dp), n, T, H, D, head_dim_v)
    return out


def _maybe_boundary_capture(q, K_all, mask, softmax_scale, attn_sink, out, handoff, kv_valid, n, T, H, D, head_dim_v):
    # LAYER-2 BOUNDARY CAPTURE (reviewer-recommended): on the FIRST compressed-layer attention call
    # (handoff consumed), dump all attention operands + output so an independent reference can be
    # replayed OFFLINE on identical operands to localize the first failing boundary. One-shot.
    if handoff is None or not _os.environ.get("INTEL_CPU_DSV4_BOUNDARY_CAPTURE") or _BOUNDARY_CAP.get("done"):
        return
    _BOUNDARY_CAP["done"] = True
    try:
        import torch as _t

        comp_kv_h = handoff["comp_kv"]
        bpos_h = handoff["block_pos"]
        cratio_h = handoff["ratio"]
        qpos_h = handoff["query_pos"]
        n_win = int(kv_valid.sum()) if kv_valid is not None else int(n)
        snap = {
            "q": q.detach().float().cpu(),
            "K_all": K_all.detach().float().cpu(),
            "mask": mask.detach().cpu(),
            "softmax_scale": float(softmax_scale),
            "attn_sink": (attn_sink.detach().float().cpu() if attn_sink is not None else None),
            "out": out.detach().float().cpu(),
            "n_window": int(n),
            "S_compressed": int(comp_kv_h.shape[0]),
            "comp_kv": comp_kv_h.detach().float().cpu(),
            "bpos": bpos_h.detach().cpu(),
            "qpos": qpos_h.detach().cpu(),
            "ratio": int(cratio_h), "T": int(T), "H": int(H), "D": int(D), "hdv": int(head_dim_v),
        }
        _dst = _os.environ["INTEL_CPU_DSV4_BOUNDARY_CAPTURE"]
        _t.save(snap, _dst)
        logger.warning("[BOUNDARY] layer-2 attention snapshot -> %s (T=%d nk=%d S=%d)",
                       _dst, T, K_all.shape[0], comp_kv_h.shape[0])
    except Exception as _e:  # noqa: BLE001
        logger.warning("[BOUNDARY] capture failed: %r", _e)


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
    dp = k_cache.data_ptr()
    edp = extra_k_cache.data_ptr() if extra_k_cache is not None else None
    # DSA step 4: consume+clear the indexer's per-query position-keep mask (None on a
    # non-indexer layer -> dense). A position beyond the compressed coverage is the recent
    # tail (always attended). locs are in sequence-position order (index p == position p).
    pos_kept = _SEL_HOLDER.get("pos_kept")
    _SEL_HOLDER["pos_kept"] = None
    _dbg = _os.environ.get("INTEL_CPU_DSV4_ATTN_DEBUG") == "1"
    # Isolation switch: ignore the DSA indexer selection and attend dense/causal over all valid KV.
    # At short context dense == full attention, so coherent output here => the bug is the indexer.
    _force_dense = _os.environ.get("INTEL_CPU_DSV4_FORCE_DENSE") == "1"
    if _dbg and _ATTN_DBG["n"] < 8:
        _v = _KV_VALID.get(dp)
        try:
            with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_attn_debug.txt"), "a") as _f:
                _f.write(
                    f"ATTN#{_ATTN_DBG['n']}: T={T} H={H} D={D} hdv={head_dim_v} dp={dp} "
                    f"kv_valid={_v is not None} n_valid={int(_v.sum()) if _v is not None else -1} "
                    f"idx_from_indices={indices is not None and topk_length is not None} "
                    f"pos_kept={pos_kept is not None} nbufs={len(_KV_BUF)}\n"
                )
        except Exception:  # noqa: BLE001
            pass
    # Batched AMX bf16 path (default): 2 GEMMs + joint key mask over the two-source KV instead of
    # the per-token loop. INTEL_CPU_DSV4_ATTN_LOOP=1 forces the fp8/fp32 reference loop (A/B).
    # INTEL_CPU_DSV4_ATTN_SHADOW=1 = T1 equivalence check: run BOTH on the same input, log cos/mag/relerr.
    _shadow = _os.environ.get("INTEL_CPU_DSV4_ATTN_SHADOW") == "1"
    _ob = None
    if _os.environ.get("INTEL_CPU_DSV4_ATTN_LOOP") != "1":
        _t0 = time.perf_counter() if _TIMEIT_ON else 0.0
        _ob = _batched_mla_attend(
            q, dp, edp, D, head_dim_v, softmax_scale, indices, topk_length,
            extra_indices_in_kvcache, extra_topk_length, attn_sink, pos_kept, _force_dense,
        )
        if _ob is not None and not _shadow:
            if _TIMEIT_ON:
                _tacc("dsa.mla.attend", _t0, "torch")
            out = _ob
            if _dbg and _ATTN_DBG["n"] < 8:
                try:
                    _z = int((out.abs().sum(dim=(1, 2)) == 0).sum())
                    with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_attn_debug.txt"), "a") as _f:
                        _f.write(f"ATTN#{_ATTN_DBG['n']} OUT[batched]: sum_abs={float(out.abs().sum()):.3e} zero_rows={_z}/{T}\n")
                    _ATTN_DBG["n"] += 1
                except Exception:  # noqa: BLE001
                    pass
            return (out.unsqueeze(1).to(q.dtype),)
    for t in range(T):
        _t0 = time.perf_counter() if _TIMEIT_ON else 0.0
        if indices is not None and topk_length is not None:
            L = int(topk_length[t])
            idx = indices[t].reshape(-1)[:L].long()
            idx = idx[idx >= 0]
        else:
            idx = torch.empty(0, dtype=torch.long)
        if _force_dense:
            idx = torch.empty(0, dtype=torch.long)  # -> dense fallback below (ignore indexer)
        # pos_kept positional filter (over the first `cov` selected locs; tail always kept).
        if pos_kept is not None and t < pos_kept.shape[0] and idx.numel() > 0:
            cov = int(pos_kept.shape[1])
            n = idx.shape[0]
            keep = torch.ones(n, dtype=torch.bool)
            m = min(cov, n)
            if m > 0:
                keep[:m] = pos_kept[t, :m].bool()
            idx = idx[keep]
        # DENSE FALLBACK: when no sparse selection survives -- a non-indexer layer (indices=None),
        # or decode where the incremental indexer is not yet wired -- attend CAUSALLY over all
        # valid KV up to this query's position (else out[t] stays 0 => attention is a no-op and
        # the model emits garbage). valid slots are written in position order; query t (0-indexed
        # in the batch of T) is at position base+t with base = n_valid - T, so it sees [0, base+t].
        if idx.numel() == 0:
            _valid = _KV_VALID.get(dp)
            if _valid is not None:
                _vslots = _valid.nonzero(as_tuple=False).reshape(-1)
                _clen = max(0, min(_vslots.numel(), _vslots.numel() - (T - 1) + t))
                idx = _vslots[:_clen]
        if _TIMEIT_ON:
            _tacc("dsa.mla.select", _t0, "framework")
            _t0 = time.perf_counter()
        # Gather selected keys in one index_select (attention is a softmax-weighted sum over
        # keys => order-invariant, so the set of keys is all that matters).
        parts = []
        kk = _stash_kv_gather(dp, idx)
        if kk is not None:
            parts.append(kk)
        if extra_indices_in_kvcache is not None and extra_topk_length is not None and edp is not None:
            EL = int(extra_topk_length[t])
            eidx = extra_indices_in_kvcache[t].reshape(-1)[:EL].long()
            eidx = eidx[eidx >= 0]
            ek = _stash_kv_gather(edp, eidx)
            if ek is not None:
                parts.append(ek)
        if not parts:
            continue
        K = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)  # [Kk, 512]
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
        if _dbg and _ATTN_DBG["n"] < 2 and t < 5:
            try:
                with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_attn_debug.txt"), "a") as _f:
                    _f.write(
                        f"TOK t={t} nkeys={K.shape[0]} idx={idx[:6].tolist()} scale={float(softmax_scale):.5f} "
                        f"K0={[round(v,4) for v in K[0,:4].tolist()]} "
                        f"qK0={[round(v,4) for v in s[0,:min(4,s.shape[1])].tolist()]} "
                        f"p0={[round(v,4) for v in p[0,:min(4,p.shape[1])].tolist()]} "
                        f"out={[round(v,4) for v in out[t,0,:4].tolist()]}\n"
                    )
            except Exception:  # noqa: BLE001
                pass
        if _TIMEIT_ON:
            _tacc("dsa.mla.attend", _t0, "torch")
    if _shadow and _ob is not None:
        # T1 in-situ A/B: batched (optimized) vs the reference loop on the SAME input, this forward.
        try:
            _a = _ob.reshape(-1).float()
            _b = out.reshape(-1).float()
            _cos = torch.nn.functional.cosine_similarity(_a.unsqueeze(0), _b.unsqueeze(0)).item()
            _mag = (_a.norm() / _b.norm().clamp_min(1e-9)).item()
            _rel = ((_a - _b).abs().max() / _b.abs().max().clamp_min(1e-9)).item()
            logger.info(
                "intel_cpu_models: [ATTN SHADOW] cos=%.6f mag=%.4f relerr=%.3e T=%d (batched vs reference loop)",
                _cos, _mag, _rel, T,
            )
        except Exception:  # noqa: BLE001
            pass
    if _dbg and _ATTN_DBG["n"] < 8:
        try:
            _z = int((out.abs().sum(dim=(1, 2)) == 0).sum())
            with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_attn_debug.txt"), "a") as _f:
                _f.write(f"ATTN#{_ATTN_DBG['n']} OUT: sum_abs={float(out.abs().sum()):.3e} zero_rows={_z}/{T}\n")
        except Exception:  # noqa: BLE001
            pass
        _ATTN_DBG["n"] += 1
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


_SWIGLU_DIAG: dict = {}


def _tensor_scalar(v):
    try:
        import torch
        if torch.is_tensor(v):
            return float(v.reshape(-1)[0].item()) if v.numel() else None
    except Exception:  # noqa: BLE001
        pass
    return v


def _swiglu_clamp_limit(layer):
    # DeepSeek-V4 SwiGLU activation CLAMP (config swiglu_limit, e.g. 10.0). The reference
    # (inference/model.py Expert.forward) does, for BOTH routed + shared experts:
    #   up   = clamp(up,  -L, +L)
    #   gate = clamp(gate,     +L)   # asymmetric: gate clamped on MAX only
    #   out  = silu(gate) * up       # bias=0, alpha=1 (NOT the gpt-oss +1/alpha=1.702 variant)
    # sglang's GPU path applies this via rc.swiglu_limit; the CPU plugin previously read
    # rc.gemm1_clamp_limit (a DIFFERENT, unset field) -> clamp silently skipped -> every expert
    # output wrong from token 1. Returns the active limit (>0) or None. Gate-off with
    # INTEL_CPU_DSV4_SWIGLU_CLAMP=0 for A/B.
    import os as _o

    if _o.environ.get("INTEL_CPU_DSV4_SWIGLU_CLAMP", "1") != "1":
        return None
    rc = getattr(layer, "moe_runner_config", None)
    for attr in ("swiglu_limit", "gemm1_clamp_limit"):
        v = getattr(rc, attr, None) if rc is not None else None
        if v is not None and float(v) > 0:
            return float(v)
    cfg = getattr(layer, "config", None)
    v = getattr(cfg, "swiglu_limit", None) if cfg is not None else None
    return float(v) if (v is not None and float(v) > 0) else None


def _swiglu_clamp(g, u, limit):
    # DeepSeek clamp: gate on max, up on both sides (in-place-safe copies).
    if limit is None:
        return g, u
    return g.clamp(max=limit), u.clamp(-limit, limit)


def _torch_fp4_moe_apply(layer, x, topk_weights, topk_ids):
    # BF16 showcase MoE: keep experts fp4 in RAM, dequant each ROUTED expert fp4->bf16 once,
    # then bf16 matmul (SwiGLU). Fits the node (fp4 storage); bf16 compute = accuracy-safe.
    import torch
    import torch.nn.functional as F

    T, hidden = x.shape
    out = torch.zeros(T, hidden, dtype=torch.float32)
    xf = x.float()
    _lim = _swiglu_clamp_limit(layer)
    _diag = _os.environ.get("INTEL_CPU_DSV4_SWIGLU_DIAG", "0") == "1" and not _SWIGLU_DIAG.get("done")
    if _diag:
        rc = getattr(layer, "moe_runner_config", None)
        _SWIGLU_DIAG["done"] = True
        _msg = (
            "[SWIGLU DIAG] resolved_limit=%s | rc.activation=%s rc.swiglu_limit=%s rc.gemm1_clamp_limit=%s "
            "rc.gemm1_alpha=%s | layer.gemm1_beta=%s layer.gemm1_clamp_limit=%s layer.swiglu_limit=%s"
        ) % (
            _lim,
            getattr(rc, "activation", None), getattr(rc, "swiglu_limit", None),
            getattr(rc, "gemm1_clamp_limit", None), getattr(rc, "gemm1_alpha", None),
            _tensor_scalar(getattr(layer, "gemm1_beta", None)),
            _tensor_scalar(getattr(layer, "gemm1_clamp_limit", None)),
            _tensor_scalar(getattr(layer, "swiglu_limit", None)),
        )
        logger.warning(_msg)
        try:
            with open(_os.environ.get("INTEL_CPU_DSV4_SWIGLU_DIAG_OUT", "/scratch/bkaul/dsv4_swiglu_diag.txt"), "a") as _f:
                _f.write(_msg + "\n")
        except Exception:  # noqa: BLE001
            pass
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
        if _diag and not _SWIGLU_DIAG.get("stats"):
            _SWIGLU_DIAG["stats"] = True
            L = _lim if _lim else 10.0
            _msg2 = (
                "[SWIGLU DIAG] gate: max|%.3f| frac>L=%.4f  up: max|%.3f| frac>L=%.4f  (L=%.1f)"
            ) % (
                g.abs().max().item(), (g.abs() > L).float().mean().item(),
                u.abs().max().item(), (u.abs() > L).float().mean().item(), L,
            )
            logger.warning(_msg2)
            try:
                with open(_os.environ.get("INTEL_CPU_DSV4_SWIGLU_DIAG_OUT", "/scratch/bkaul/dsv4_swiglu_diag.txt"), "a") as _f:
                    _f.write(_msg2 + "\n")
            except Exception:  # noqa: BLE001
                pass
        g, u = _swiglu_clamp(g, u, _lim)
        act = F.silu(g) * u
        oe = act @ w2.t()  # [n, hidden]
        out.index_add_(0, tok, oe * topk_weights[tok, slot].float().unsqueeze(-1))
    return out.to(x.dtype)


# Standard OCP e2m1: nibble -> value (matches csrc/cpu/vec.h cvt_mxfp4_e2m1_bf16, low-nibble-first).
_MXFP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _mxfp4_dequant_raw(packed, scale_f32, group=32):
    # packed fp4 [O, K//2] (int8/uint8 bits) + float32 pow2 group-32 scale [O, K//group] -> fp32 [O,K].
    import torch

    lut = torch.tensor(_MXFP4_E2M1, dtype=torch.float32)
    pu = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    low = (pu & 0xF).long()
    high = ((pu >> 4) & 0xF).long()
    O, khalf = pu.shape
    nib = torch.empty(O, khalf * 2, dtype=torch.long)
    nib[:, 0::2] = low
    nib[:, 1::2] = high
    vals = lut[nib]
    s = scale_f32.float().repeat_interleave(group, dim=-1)
    return vals * s


def _mxfp4_insitu_parity(layer, x0, topk_w0, topk_id0, out_kernel, rc):
    # Real-checkpoint correctness probe: compare the MXFP4 kernel output to a torch dequant+SwiGLU
    # oracle built from the SAME raw fp4 weights + scales, on the SAME real activations. Runs once.
    import torch
    import torch.nn.functional as F

    ref = layer._mxfp4_ref
    try:
        if getattr(rc, "apply_router_weight_on_input", False):
            logger.warning("[MXFP4 PARITY] apply_router_weight_on_input=True; oracle uses output weighting")
        E = ref["w13"].shape[0]
        T, hidden = x0.shape
        xf = x0.float()
        w13 = torch.stack([_mxfp4_dequant_raw(ref["w13"][e], ref["w13s"][e]) for e in range(E)])
        w2 = torch.stack([_mxfp4_dequant_raw(ref["w2"][e], ref["w2s"][e]) for e in range(E)])
        oracle = torch.zeros(T, hidden, dtype=torch.float32)
        _lim = _swiglu_clamp_limit(layer)
        for e in range(E):
            mask = topk_id0 == e
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            g, u = (xf[tok] @ w13[e].t()).chunk(2, dim=-1)
            g, u = _swiglu_clamp(g, u, _lim)
            oe = (F.silu(g) * u) @ w2[e].t()
            oracle.index_add_(0, tok, oe * topk_w0[tok, slot].float().unsqueeze(-1))
        got = out_kernel.float()
        diff = (got - oracle).abs()
        sc = oracle.abs().max().clamp_min(1e-6)
        max_abs = diff.max().item()
        rel = (max_abs / sc).item()
        cos = F.cosine_similarity(got.reshape(1, -1), oracle.reshape(1, -1)).item()
        verdict = "PASS" if (cos >= 0.99 and rel <= 0.08) else "FAIL"
        logger.warning(
            "[MXFP4 PARITY] %s cosine=%.6f rel=%.3e max_abs=%.3e (kernel vs torch fp4 oracle, real ckpt)",
            verdict, cos, rel, max_abs,
        )
        try:  # file-based too: the Engine scheduler subprocess stdout is not captured in the sbatch log
            with open(_os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_mxfp4_parity_insitu.txt"), "a") as _f:
                _f.write(f"[MXFP4 PARITY] {verdict} cosine={cos:.6f} rel={rel:.3e} max_abs={max_abs:.3e}\n")
        except Exception:  # noqa: BLE001
            pass
    except Exception as _e:  # noqa: BLE001
        logger.warning("[MXFP4 PARITY] probe failed: %s", _e)
    finally:
        layer._mxfp4_ref = None  # free the stashed raw fp4 weights


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
    # NATIVE MXFP4: keep experts as packed fp4 + e8m0 scales and run the CPU AMX MXFP4 W4A16 kernel
    # (fused fp4->bf16 dequant inside the GEMM). Experts stay ~137GB (vs 275GB after fp8 dequant), so
    # the full model FITS ONE SNC domain -> tp=1 -> the decode thread-cap fast path applies. This
    # mirrors sglang's Mxfp4MoEMethod CPU branch exactly (the checkpoint is already native MXFP4).
    _mxfp4_moe = os.environ.get("INTEL_CPU_DSV4_MXFP4_MOE", "0") == "1"
    _orig_pwal = Method.process_weights_after_loading
    _orig_apply = Method.apply

    def _patched_pwal(self, layer):
        if getattr(self, "is_fp4_expert", False) and _bf16_moe:
            # Keep fp4 raw (skip fp8 dequant + AMX prepack); the apply dequants per expert.
            layer._fp4_bf16_moe = True
            logger.info("intel_cpu_models: FP4 experts kept raw for BF16-in-forward MoE.")
            return
        if getattr(self, "is_fp4_expert", False) and _mxfp4_moe:
            # NATIVE MXFP4 W4A16: VNNI-prepack the packed-fp4 weights + pack the e8m0 scales, then
            # the apply routes to fused_experts_cpu(MXFP4). Experts stay 4-bit -> fit one domain.
            import torch as _t

            _rss0 = _rss_gb()
            # Optional in-situ correctness probe: stash the FIRST layer's raw fp4 weights + scales
            # (before they are replaced by the packed versions) so _patched_apply can compare the
            # kernel output against a torch dequant+SwiGLU oracle on the real checkpoint + real
            # activations. Gated + first-layer-only so it costs ~one layer of fp4 RAM, freed after.
            if (
                _os.environ.get("INTEL_CPU_DSV4_MOE_PARITY", "0") == "1"
                and _MOE_DIAG.get("parity_ref") is None
            ):
                _MOE_DIAG["parity_ref"] = True
                layer._mxfp4_ref = {
                    "w13": layer.w13_weight.detach().clone(),
                    "w2": layer.w2_weight.detach().clone(),
                    "w13s": (getattr(layer, "w13_weight_scale_inv", None)
                             if getattr(layer, "w13_weight_scale_inv", None) is not None
                             else layer.w13_weight_scale).detach().clone(),
                    "w2s": (getattr(layer, "w2_weight_scale_inv", None)
                            if getattr(layer, "w2_weight_scale_inv", None) is not None
                            else layer.w2_weight_scale).detach().clone(),
                }
            # The loader delivers the group-32 scale as FLOAT32 power-of-2 values in
            # w13_weight_scale_inv ([E, OC, K/32]); the MXFP4 kernel wants the e8m0 EXPONENT byte
            # (uint8, scale = 2^(byte-127)) = the float32 biased-exponent field (bits>>23)&0xFF.
            # (An earlier .view(uint8) wrongly split each 4-byte float32 into 4 "groups" -> [.,512];
            # the non-replicated head [0,0,0,60,..] is LE float32 2^-7.)
            def _scale_to_e8m0(s):
                if s.dtype == _t.uint8:
                    return s.contiguous()
                if s.dtype == _t.int8:
                    return s.view(_t.uint8).contiguous()
                if s.dtype == _t.float32:
                    return ((s.contiguous().view(_t.int32) >> 23) & 0xFF).to(_t.uint8).contiguous()
                # bf16/fp16/float8_e8m0 power-of-2 -> round(log2)+127
                e = _t.clamp(_t.round(_t.log2(s.float().clamp_min(1.1754944e-38))) + 127, 0, 255)
                return e.to(_t.uint8).contiguous()

            _w13s_raw = getattr(layer, "w13_weight_scale_inv", None)
            if _w13s_raw is None:
                _w13s_raw = layer.w13_weight_scale
            _w2s_raw = getattr(layer, "w2_weight_scale_inv", None)
            if _w2s_raw is None:
                _w2s_raw = layer.w2_weight_scale
            if not _MOE_DIAG.get("mx_shape_logged"):
                _MOE_DIAG["mx_shape_logged"] = True
                logger.warning(
                    "[MXFP4 SHAPE] w13=%s/%s w2=%s/%s w13_scale_inv=%s/%s w2_scale_inv=%s/%s",
                    tuple(layer.w13_weight.shape), layer.w13_weight.dtype,
                    tuple(layer.w2_weight.shape), layer.w2_weight.dtype,
                    tuple(_w13s_raw.shape), _w13s_raw.dtype,
                    tuple(_w2s_raw.shape), _w2s_raw.dtype,
                )
            w13s = _scale_to_e8m0(_w13s_raw)  # [E, 2N, K/32] uint8 e8m0
            w2s = _scale_to_e8m0(_w2s_raw)  # [E, K, N/32] uint8 e8m0

            # Use convert_weight_packed (exactly what fused_experts_cpu(MXFP4) calls internally when
            # is_vnni=False) so is_vnni=True gets the identical layout. CRITICAL: view the weight as
            # UINT8 first -- it loads as int8 (kChar), and convert_weight_packed dispatches on dtype:
            # int8 -> W8A8 packing (row = IC + 4-byte compensation -> 2052, kernel rejects); uint8
            # (kByte) -> the mxfp4/int4 packing (row = IC*2 >> 1 = IC = 2048). Same bytes, free view.
            _w13 = layer.w13_weight
            _w2 = layer.w2_weight
            _w13 = _w13.view(_t.uint8) if _w13.dtype != _t.uint8 else _w13
            _w2 = _w2.view(_t.uint8) if _w2.dtype != _t.uint8 else _w2
            layer.w13_weight = _t.nn.Parameter(
                _t.ops.sgl_kernel.convert_weight_packed(_w13.contiguous()), requires_grad=False
            )
            layer.w2_weight = _t.nn.Parameter(
                _t.ops.sgl_kernel.convert_weight_packed(_w2.contiguous()), requires_grad=False
            )
            layer.use_intel_amx_backend = True
            layer.w13_weight_scale = _t.nn.Parameter(
                _t.ops.sgl_kernel.convert_scale_packed(w13s.contiguous()), requires_grad=False
            )
            layer.w2_weight_scale = _t.nn.Parameter(
                _t.ops.sgl_kernel.convert_scale_packed(w2s.contiguous()), requires_grad=False
            )
            layer._mxfp4_moe = True
            self.is_fp4_expert = False
            _release_freed_memory()
            _di = _MOE_DIAG.setdefault("mx_layer", 0)
            _MOE_DIAG["mx_layer"] = _di + 1
            logger.info(
                "intel_cpu_models: MXFP4 MoE layer %d prepacked (W4A16, experts stay 4-bit) "
                "RSS %.1f->%.1fGB",
                _di, _rss0, _rss_gb(),
            )
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
        if getattr(self, "is_fp4_expert", False):
            _rss0 = _rss_gb()
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
                del new_w, new_s
            self.is_fp4_expert = False
            _rss1 = _rss_gb()
            out = _orig_pwal(self, layer)
            _interleave_layer_weights(layer)  # migrate packed weights off node0 -> spread across nodes
            _release_freed_memory()
            _di = _MOE_DIAG.setdefault("dq_layer", 0)
            _MOE_DIAG["dq_layer"] = _di + 1
            logger.info(
                "intel_cpu_models: dequantized FP4->FP8 layer %d (Plan A) RSS %.1f->%.1f->%.1fGB "
                "(pre-dq / post-dq / post-prepack+trim)",
                _di, _rss0, _rss1, _rss_gb(),
            )
            return out
        out = _orig_pwal(self, layer)
        # Return freed fp4/intermediate memory to the OS. At tp=1 all 43 layers' experts are
        # dequanted on one rank; without this the CPU allocator holds every layer's freed fp4 +
        # AMX-prepack scratch and the load peak overflows node RAM (single-rank ~275GB fp8, but
        # transient peak was >1.25TB -> OOM). glibc malloc_trim gives it back after each layer.
        _interleave_layer_weights(layer)
        _release_freed_memory()
        return out

    def _mxfp4_apply_raw(layer, dispatch_output, with_parity=True):
        # The raw native MXFP4 W4A16 CPU AMX kernel call (mirrors sglang Mxfp4MoEMethod CPU apply),
        # returning the output tensor. Shared by the live path and the [MOE ISO] thread sweep so the
        # sweep times the ACTUAL packed-MXFP4 kernel (not _orig_apply, which rejects the packed layout).
        import torch as _tt

        from sglang.srt.layers.amx_utils import CPUQuantMethod
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        rc = layer.moe_runner_config
        _ref = getattr(layer, "_mxfp4_ref", None) if with_parity else None
        _x0 = x if _ref is None else x.detach().clone()
        _tw0 = None if _ref is None else topk_weights.detach().clone()
        _ti0 = None if _ref is None else topk_ids.detach().clone()
        x, topk_weights = apply_topk_weights_cpu(rc.apply_router_weight_on_input, topk_weights, x)
        if with_parity:
            try:
                from . import _dsa_ckernels as _ckmoe
                _ckmoe.mark("moe", "c_kernel")
            except Exception:  # noqa: BLE001
                pass
        out = _tt.ops.sgl_kernel.fused_experts_cpu(
            x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids, False,
            CPUQuantMethod.MXFP4, layer.w13_weight_scale, layer.w2_weight_scale,
            None, None, None,
            getattr(layer, "w13_weight_bias", None), getattr(layer, "w2_weight_bias", None),
            getattr(rc, "gemm1_alpha", None), getattr(rc, "gemm1_clamp_limit", None),
            True, rc.activation,
        )
        if _ref is not None:
            _mxfp4_insitu_parity(layer, _x0, _tw0, _ti0, out, rc)
        return out

    def _patched_apply(self, layer, dispatch_output):
        # MoE occupancy + branch probe (review §3/§6): record, per token-count M, the EXECUTED path
        # (use_brgemm = avg_M>4, avg_M=M*topk//E -> source-verified gate in moe.cpp) and the real per-expert
        # row-count distribution across ALL layers/steps. Confirms the vector-vs-BRGEMM branch + occupancy.
        if _os.environ.get("INTEL_CPU_DSV4_MOE_OCC", "0") == "1":
            try:
                import json as _json

                import torch as _t

                x = dispatch_output.hidden_states
                _tw, ti, _ = dispatch_output.topk_output
                M = int(x.shape[0]); topk = int(ti.shape[1])
                w13 = getattr(layer, "w13_weight", None)
                E = int(w13.shape[0]) if w13 is not None else 256
                avg_M = max(1, (M * topk) // E)
                use_brgemm = avg_M > 4
                _uniq, _cnts = _t.unique(ti.reshape(-1), return_counts=True)
                occ = _MOE_DIAG.setdefault("occ", {})
                k = str(M)
                e = occ.setdefault(k, {"M": M, "topk": topk, "E": E, "avg_M": avg_M,
                                       "use_brgemm": use_brgemm, "calls": 0, "active_sum": 0,
                                       "rows_min": 1 << 30, "rows_max": 0, "rows_sum": 0, "rows_n": 0,
                                       "hist": {"1": 0, "2": 0, "3-4": 0, "5-8": 0, "9+": 0}})
                e["calls"] += 1
                e["active_sum"] += int(_uniq.numel())
                for c in _cnts.tolist():
                    e["rows_min"] = min(e["rows_min"], c); e["rows_max"] = max(e["rows_max"], c)
                    e["rows_sum"] += c; e["rows_n"] += 1
                    b = "1" if c == 1 else "2" if c == 2 else "3-4" if c <= 4 else "5-8" if c <= 8 else "9+"
                    e["hist"][b] += 1
                _f = _os.environ.get("INTEL_CPU_DSV4_MOE_OCC_FILE")
                if _f and e["calls"] % 43 == 0:  # ~once per decode step (43 layers); cheap
                    with open(_f, "w") as _w:
                        _json.dump(occ, _w, indent=2)
            except Exception as _e:  # noqa: BLE001
                logger.warning("[MOE OCC] failed: %r", _e)
        # MoE ROOFLINE probe (review §4): once per distinct decode M, roofline the EXECUTED apply vs the
        # config-matched weight-streaming floor on THIS node. Separates 'memory-bound at floor' from
        # 'overhead above floor'. Streams the active experts' packed 4-bit weight bytes as the floor.
        if _os.environ.get("INTEL_CPU_DSV4_MOE_ROOFLINE", "0") == "1":
            try:
                import time as _tm

                import torch as _t

                x = dispatch_output.hidden_states
                _tw, ti, _ = dispatch_output.topk_output
                M = int(x.shape[0])
                rl = _MOE_DIAG.setdefault("rl_m", set())
                if M not in rl:
                    rl.add(M)
                    w13 = layer.w13_weight; w2 = layer.w2_weight  # packed uint8 (4-bit), [E, ...]
                    active = int(_t.unique(ti.reshape(-1)).numel())
                    E = int(w13.shape[0])
                    bytes_per_expert = (w13[0].numel() * w13.element_size()
                                        + w2[0].numel() * w2.element_size())
                    gb = active * bytes_per_expert / 1e9
                    nth = _t.get_num_threads()
                    # apply time (executed path), median of reps
                    reps = 7
                    _ts = []
                    for _ in range(reps):
                        _t0 = _tm.perf_counter(); _mxfp4_apply_raw(layer, dispatch_output, with_parity=False)
                        _ts.append(_tm.perf_counter() - _t0)
                    _ts.sort(); apply_ms = _ts[len(_ts) // 2] * 1e3
                    # weight-stream floor (VALID): achievable SEQUENTIAL read BW on-node at this thread count,
                    # measured once over a large contiguous buffer, then floor_ms = gb / seq_BW. (A strided
                    # index_select gather is NOT a floor -- read+write + prefetch-defeating -> slower than the
                    # fused kernel, which is physically impossible for a lower bound.)
                    seq_bw = _MOE_DIAG.get("seq_bw_gbs")
                    if seq_bw is None:
                        buf = _t.empty(int(1.5e9), dtype=_t.uint8)  # 1.5 GB contiguous
                        buf.fill_(1)
                        _bs = []
                        for _ in range(3):
                            _t0 = _tm.perf_counter()
                            _ = buf.sum(dtype=_t.int64)  # sequential read of all bytes, multi-threaded
                            _bs.append(_tm.perf_counter() - _t0)
                        _bs.sort()
                        seq_bw = (buf.numel() / 1e9) / _bs[len(_bs) // 2]
                        _MOE_DIAG["seq_bw_gbs"] = seq_bw
                        del buf
                    floor_ms = gb / seq_bw * 1e3
                    logger.warning(
                        "[MOE ROOFLINE] M=%d threads=%d active=%d/%d | stream=%.3fGB  apply=%.2fms (%.0f GB/s) "
                        "floor=%.2fms (%.0f GB/s)  apply/floor=%.1fx",
                        M, nth, active, E, gb, apply_ms, gb / (apply_ms / 1e3),
                        floor_ms, gb / (floor_ms / 1e3), apply_ms / max(1e-6, floor_ms),
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning("[MOE ROOFLINE] failed: %r", _e)
        if _TIMEIT_ON:
            try:
                import torch as _t

                x = dispatch_output.hidden_states
                _dm = int(x.shape[0])
                if _dm not in _MOE_DIAG.setdefault("diag_m", set()):
                    _MOE_DIAG["diag_m"].add(_dm)
                    _tw, ti, _ = dispatch_output.topk_output
                    w13 = getattr(layer, "w13_weight", None)
                    w2 = getattr(layer, "w2_weight", None)
                    # active-group stats: distinct experts hit + rows/expert distribution (MoE regime).
                    _ne = int(w13.shape[0]) if w13 is not None else -1
                    _flat = ti.reshape(-1)
                    _uniq, _cnts = _t.unique(_flat, return_counts=True)
                    _rows = _cnts.float()
                    logger.warning(
                        "[MOE DIAG] tokens=%s xdtype=%s topk_ids=%s w13=%s/%s w2=%s/%s "
                        "threads=%d | experts_total=%d active=%d rows/expert[min/mean/max]=%d/%.1f/%d",
                        tuple(x.shape),
                        x.dtype,
                        tuple(ti.shape),
                        tuple(w13.shape) if w13 is not None else None,
                        getattr(w13, "dtype", None),
                        tuple(w2.shape) if w2 is not None else None,
                        getattr(w2, "dtype", None),
                        _t.get_num_threads(),
                        _ne, int(_uniq.numel()),
                        int(_rows.min()), float(_rows.mean()), int(_rows.max()),
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning("[MOE DIAG] failed: %s", _e)
        # [MOE ISO]: isolation thread-sweep, run ONCE PER DISTINCT token-count M (so one job covers
        # B=1 AND B=32). Physical-core-only candidates (no SMT per the single-socket-no-HT operating point).
        _iso_m = None
        if _os.environ.get("INTEL_CPU_DSV4_MOE_ISO", "0") == "1":
            try:
                _iso_m = int(dispatch_output.hidden_states.shape[0])
            except Exception:  # noqa: BLE001
                _iso_m = None
        if _iso_m is not None and _iso_m not in _MOE_DIAG.setdefault("iso_m", set()):
            _MOE_DIAG["iso_m"].add(_iso_m)
            # In-situ isolation with the REAL packed weights: re-time the real expert apply
            # across thread counts. Flat -> kernel/shape bound (e.g. all-256-expert stream);
            # scaling -> thread/binding bound. Decides the runtime-config vs kernel question.
            try:
                import time as _t

                import torch as _tt

                # Call the SAME kernel the live path uses (MXFP4 raw call, else _orig_apply).
                if getattr(layer, "_mxfp4_moe", False):
                    _iso_call = lambda: _mxfp4_apply_raw(layer, dispatch_output, with_parity=False)
                else:
                    _iso_call = lambda: _orig_apply(self, layer, dispatch_output)
                _orig_nt = _tt.get_num_threads()
                _cands = sorted({c for c in (4, 8, 16, 32, 58) if 1 <= c <= max(_orig_nt, 58)})
                for _nth in _cands:
                    if _nth < 1:
                        continue
                    _tt.set_num_threads(_nth)
                    _iso_call()  # warm
                    _t0 = _t.perf_counter()
                    for _ in range(3):
                        _iso_call()
                    logger.warning(
                        "[MOE ISO] M=%d threads=%d apply=%.2fms", _iso_m, _nth,
                        (_t.perf_counter() - _t0) / 3 * 1e3
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
        if getattr(layer, "_mxfp4_moe", False):
            # Native MXFP4 W4A16 CPU AMX kernel (mirrors sglang Mxfp4MoEMethod CPU apply).
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
            out = _mxfp4_apply_raw(layer, dispatch_output, with_parity=True)
            return StandardCombineInput(hidden_states=out)
        # runtime-config fix: cap threads around the native MoE apply (fused_experts scales
        # inversely with threads on this shape); explicit N or self-tuned via =auto.
        _cap = _resolve_moe_thread_cap(_orig_apply, self, layer, dispatch_output)
        if _cap is None:
            return _orig_apply(self, layer, dispatch_output)
        import torch as _tt

        _prev = _tt.get_num_threads()
        _tt.set_num_threads(_cap)
        try:
            return _orig_apply(self, layer, dispatch_output)
        finally:
            _tt.set_num_threads(_prev)

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


def _install_fp8_proj_parity() -> None:
    # In-situ parity probe for the fp8 W8A16 dense projections (q_a/q_b/kv_a/kv_b/o) — the last
    # unverified op in the Q/K content path. Stash the FIRST layer's raw fp8 weight + block scale
    # before the CPU prepack, then on the first apply compare the kernel output to a torch
    # block-dequant (fp8->bf16, block-128) matmul. Gated + one-shot, file-based.
    if _os.environ.get("INTEL_CPU_DSV4_FP8_PARITY") != "1":
        return
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    except Exception:  # noqa: BLE001
        return
    if getattr(Fp8LinearMethod, "_fp8_parity_patched", False):
        return
    import torch

    _f = _os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_fp8_parity.txt")
    _st: dict = {"ref": None, "done": False}
    _orig_pwal = Fp8LinearMethod.process_weights_after_loading
    _orig_apply = Fp8LinearMethod.apply

    def _pwal(self, layer):
        if _st["ref"] is None:
            w = getattr(layer, "weight", None)
            s = getattr(layer, "weight_scale_inv", getattr(layer, "weight_scale", None))
            if w is not None and s is not None and w.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8):
                _st["ref"] = {"w": w.detach().clone(), "s": s.detach().clone()}
        return _orig_pwal(self, layer)

    def _blk_dequant(w_fp8, s):
        wf = w_fp8.float()
        O, I = wf.shape
        sf = s.float()
        if sf.dim() == 2 and sf.shape[0] > 0 and sf.shape[1] > 0:
            bo = max(1, O // sf.shape[0])
            bi = max(1, I // sf.shape[1])
            sfull = sf.repeat_interleave(bo, 0).repeat_interleave(bi, 1)[:O, :I]
            return wf * sfull
        return wf * sf

    def _apply(self, layer, x, bias=None):
        out = _orig_apply(self, layer, x, bias)
        if not _st["done"] and _st["ref"] is not None:
            _st["done"] = True
            try:
                ref = _st["ref"]
                w = _blk_dequant(ref["w"], ref["s"])
                y = x.detach().float() @ w.t()
                if bias is not None:
                    y = y + bias.float()
                g = out.detach().float()
                if g.shape == y.shape:
                    d = (g - y).abs()
                    rel = (d.max() / y.abs().max().clamp_min(1e-6)).item()
                    cos = torch.nn.functional.cosine_similarity(g.reshape(1, -1), y.reshape(1, -1)).item()
                    verdict = "PASS" if (cos >= 0.99 and rel <= 0.1) else "FAIL"
                    with open(_f, "a") as fh:
                        fh.write(f"[FP8 PARITY] {verdict} cos={cos:.6f} rel={rel:.3e} shape={tuple(g.shape)} w={tuple(ref['w'].shape)}\n")
                else:
                    with open(_f, "a") as fh:
                        fh.write(f"[FP8 PARITY] SHAPE-MISMATCH out={tuple(g.shape)} ref={tuple(y.shape)}\n")
            except Exception as _e:  # noqa: BLE001
                try:
                    with open(_f, "a") as fh:
                        fh.write(f"[FP8 PARITY] probe failed: {_e}\n")
                except Exception:  # noqa: BLE001
                    pass
        return out

    Fp8LinearMethod.process_weights_after_loading = _pwal
    Fp8LinearMethod.apply = _apply
    Fp8LinearMethod._fp8_parity_patched = True


def _install_fp8_force_bf16() -> None:
    # Decisive precision test: replace every fp8 W8A16 dense projection with a bf16 block-dequant
    # matmul (fp8->bf16, block-128). Coherent output => the CPU fp8 path's accumulated precision
    # drift (vs GPU native fp8) is the garbage cause; still garbage => a logic bug remains.
    if _os.environ.get("INTEL_CPU_DSV4_FP8_BF16") != "1":
        return
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    except Exception:  # noqa: BLE001
        return
    if getattr(Fp8LinearMethod, "_fp8_bf16_patched", False):
        return
    import torch

    _store: dict = {}
    _orig_pwal = Fp8LinearMethod.process_weights_after_loading
    _orig_apply = Fp8LinearMethod.apply
    _dbgf = _os.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_fp8bf16_engage.txt")

    def _blk_dequant(w_fp8, s):
        wf = w_fp8.float()
        O, I = wf.shape
        sf = s.float()
        if sf.dim() == 2 and sf.shape[0] > 0 and sf.shape[1] > 0:
            bo = max(1, O // sf.shape[0])
            bi = max(1, I // sf.shape[1])
            sfull = sf.repeat_interleave(bo, 0).repeat_interleave(bi, 1)[:O, :I]
            return wf * sfull
        return wf * sf

    def _pwal(self, layer):
        w = getattr(layer, "weight", None)
        s = getattr(layer, "weight_scale_inv", getattr(layer, "weight_scale", None))
        # Match the parity probe's dtype set: fp8 may be stored as uint8/int8, not only float8_e4m3fn.
        if w is not None and s is not None and w.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8):
            try:
                wv = w.view(torch.float8_e4m3fn) if w.dtype != torch.float8_e4m3fn else w
                _store[id(layer)] = _blk_dequant(wv, s).to(torch.bfloat16)
                with open(_dbgf, "a") as fh:
                    fh.write(f"[FP8->BF16 ENGAGE] layer#{len(_store)} w={tuple(w.shape)} dt={w.dtype}\n")
            except Exception as _e:  # noqa: BLE001
                with open(_dbgf, "a") as fh:
                    fh.write(f"[FP8->BF16 ENGAGE] SKIP dt={w.dtype}: {_e}\n")
        return _orig_pwal(self, layer)

    def _apply(self, layer, x, bias=None):
        wbf = _store.get(id(layer))
        if wbf is not None:
            y = x.float() @ wbf.float().t()
            if bias is not None:
                y = y + bias.float()
            return y.to(x.dtype)
        return _orig_apply(self, layer, x, bias)

    Fp8LinearMethod.process_weights_after_loading = _pwal
    Fp8LinearMethod.apply = _apply
    Fp8LinearMethod._fp8_bf16_patched = True
    logger.info("intel_cpu_models: FORCED fp8 dense projections to bf16 block-dequant (precision test).")
    logger.info("intel_cpu_models: installed fp8 projection in-situ parity probe.")


def _install_hidden_debug() -> None:
    # Localizer: wrap each decoder layer's forward to record input/output activation stats
    # (abs-mean, abs-max, NaN) to a file, for the first forward pass. Finds where the model
    # goes bad (NaN/explode/collapse) when the output is garbage. Gated + one-shot.
    if _os.environ.get("INTEL_CPU_DSV4_HID_DEBUG") != "1":
        return
    import torch

    import sglang.srt.models.deepseek_v4 as _dv4

    Layer = _dv4.DeepseekV4DecoderLayer
    if getattr(Layer, "_hid_dbg_patched", False):
        return
    _orig = Layer.forward
    _f = _os.environ.get("HID_DBG_FILE", "/scratch/bkaul/dsv4_hid_debug.txt")

    def _stat(t):
        try:
            tf = t.float()
            fp = tf.reshape(-1)[:4].tolist()
            return f"mean={tf.abs().mean().item():.3e} max={tf.abs().max().item():.3e} nan={int(torch.isnan(tf).any())} fp={[round(x, 4) for x in fp]}"
        except Exception:  # noqa: BLE001
            return "NA"

    def _find_hid(a, k):
        for x in list(a) + list(k.values()):
            if torch.is_tensor(x) and x.is_floating_point() and x.dim() >= 2 and x.shape[-1] in (4096, 7168):
                return x
        return None

    def _fwd(self, *a, **k):
        lid = getattr(self, "layer_id", getattr(self, "layer_idx", -1))
        cap = _HID_DBG["n"] < 48
        if cap:
            inp = _find_hid(a, k)
            try:
                with open(_f, "a") as fh:
                    fh.write(f"L{lid} IN  {_stat(inp) if inp is not None else 'NA'}\n")
            except Exception:  # noqa: BLE001
                pass
        out = _orig(self, *a, **k)
        if cap:
            try:
                with open(_f, "a") as fh:
                    if isinstance(out, (tuple, list)):
                        for j, o in enumerate(out[:2]):
                            if torch.is_tensor(o):
                                fh.write(f"L{lid} OUT{j} {_stat(o)}\n")
                    else:
                        fh.write(f"L{lid} OUT0 {_stat(out)}\n")
            except Exception:  # noqa: BLE001
                pass
            _HID_DBG["n"] += 1
        return out

    Layer.forward = _fwd
    Layer._hid_dbg_patched = True
    # Tap the o_proj wo_a grouped matmul: log o.shape (regroup 64 heads -> o_groups) + values.
    _owoa_fn = getattr(_dv4, "_apply_wo_a_bf16_matmul", None)
    if _owoa_fn is not None and not getattr(_dv4, "_woa_fn_tapped", False):

        def _woa_fn(o, wo_a, is_decode):
            r = _owoa_fn(o, wo_a, is_decode)
            if torch.is_tensor(o) and o.shape[0] < 32 and _HID_DBG.get("woafn", 0) < 2:
                _HID_DBG["woafn"] = _HID_DBG.get("woafn", 0) + 1
                if _HID_DBG["woafn"] == 1:
                    try:
                        torch.save({
                            "o": o.detach().cpu(),
                            "wo_a": wo_a.detach().cpu(),
                            "wo_a_dtype": str(wo_a.dtype),
                            "out": r.detach().cpu(),
                        }, _f + ".woa.pt")
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    nt = min(5, o.shape[0])
                    with open(_f, "a") as fh:
                        fh.write(f"WOAFN#{_HID_DBG['woafn']} o={tuple(o.shape)} wo_a={tuple(wo_a.shape)} out={tuple(r.shape)}\n")
                        for t in range(nt):
                            fh.write(f"WOAFN#{_HID_DBG['woafn']} o[t{t},g0,:4]={[round(v,4) for v in o[t,0,:4].float().tolist()]}\n")
                        for g in range(o.shape[1]):
                            fh.write(f"WOAFN#{_HID_DBG['woafn']} o[t0,g{g},:4]={[round(v,4) for v in o[0,g,:4].float().tolist()]}\n")
                except Exception:  # noqa: BLE001
                    pass
            return r

        _dv4._apply_wo_a_bf16_matmul = _woa_fn
        _dv4._woa_fn_tapped = True
    # Tap wo_b (final o_proj RowParallel): input dim = n_groups*o_lora_rank (8192 tp1 / 2048 tp4).
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod as _Fp8LM
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod as _UnqLM

        for _lm in (_Fp8LM, _UnqLM):
            if getattr(_lm, "_wob_dbg_tapped", False):
                continue
            _oap = _lm.apply

            def _mk_ap(oap):
                def _ap(self, layer, x, bias=None):
                    r = oap(self, layer, x, bias)
                    try:
                        if torch.is_tensor(x) and x.shape[-1] in (2048, 8192) and x.shape[0] < 64 and _HID_DBG.get("wob", 0) < 3:
                            _HID_DBG["wob"] = _HID_DBG.get("wob", 0) + 1
                            w = getattr(layer, "weight", None)
                            xf = x.reshape(-1, x.shape[-1]); rf = r.reshape(-1, r.shape[-1])
                            # Save full x/weight/scale/out once so the wo_b GEMM (x@w.T?) can be
                            # verified offline vs a torch dequant+matmul reference.
                            if _HID_DBG["wob"] == 1:
                                try:
                                    _sc = getattr(layer, "weight_scale_inv", getattr(layer, "weight_scale", None))
                                    torch.save({
                                        "x": x.detach().cpu(),
                                        "w": w.detach().cpu() if torch.is_tensor(w) else None,
                                        "w_dtype": str(w.dtype) if torch.is_tensor(w) else None,
                                        "scale": _sc.detach().cpu() if torch.is_tensor(_sc) else None,
                                        "scale_dtype": str(_sc.dtype) if torch.is_tensor(_sc) else None,
                                        "out": r.detach().cpu(),
                                    }, _f + ".wob.pt")
                                except Exception:  # noqa: BLE001
                                    pass
                            with open(_f, "a") as fh:
                                fh.write(
                                    f"WOB#{_HID_DBG['wob']} x={tuple(x.shape)} w={tuple(w.shape) if torch.is_tensor(w) else None} "
                                    f"wdt={w.dtype if torch.is_tensor(w) else None} out={tuple(r.shape)} "
                                    f"x[0,:4]={[round(v,4) for v in xf[0,:4].float().tolist()]} "
                                    f"out[0,:4]={[round(v,4) for v in rf[0,:4].float().tolist()]}\n"
                                )
                    except Exception:  # noqa: BLE001
                        pass
                    return r
                return _ap

            _lm.apply = _mk_ap(_oap)
            _lm._wob_dbg_tapped = True
    except Exception:  # noqa: BLE001
        pass
    # Also capture the MHC boundary intermediates: hc_pre output y + hc_post output (residual).
    for _meth, _tag in (("hc_pre", "HCPRE"), ("hc_post", "HCPOST")):
        _orig_m = getattr(Layer, _meth, None)
        if _orig_m is None or getattr(Layer, "_" + _meth + "_hid_hooked", False):
            continue

        def _mk(orig, tag):
            key = tag.lower()

            def _w(self, *a, **k):
                r = orig(self, *a, **k)
                if tag == "HCPRE" and _HID_DBG.get("wdump", 0) < 2:
                    _HID_DBG["wdump"] = _HID_DBG.get("wdump", 0) + 1
                    try:
                        if len(a) > 0 and torch.is_tensor(a[0]) and _HID_DBG["wdump"] == 1:
                            torch.save(a[0].detach().float().cpu(), _f + ".x0.pt")
                        with open(_f, "a") as fh:
                            if len(a) > 0 and torch.is_tensor(a[0]):
                                fh.write(f"XIN shape={tuple(a[0].shape)} {_stat(a[0])}\n")
                            for wn in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_scale", "hc_attn_base"):
                                w = getattr(self, wn, None)
                                if torch.is_tensor(w):
                                    fh.write(f"WDUMP {wn} shape={tuple(w.shape)} dtype={w.dtype} {_stat(w)}\n")
                    except Exception:  # noqa: BLE001
                        pass
                if _HID_DBG.get(key, 0) < 4:
                    _HID_DBG[key] = _HID_DBG.get(key, 0) + 1
                    try:
                        y = r[0] if isinstance(r, (tuple, list)) else r
                        with open(_f, "a") as fh:
                            fh.write(f"{tag}#{_HID_DBG[key]} out {_stat(y)}\n")
                            if tag == "HCPOST":
                                # hc_post(self, x_branch, residual, post, comb)
                                names = ("x", "resid", "post", "comb")
                                for idx, nm in enumerate(names):
                                    if idx < len(a) and torch.is_tensor(a[idx]):
                                        fh.write(f"{tag}#{_HID_DBG[key]} in.{nm} {_stat(a[idx])}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            return _w

        setattr(Layer, _meth, _mk(_orig_m, _tag))
        setattr(Layer, "_" + _meth + "_hid_hooked", True)

    # Capture self_attn (MQALayer) input x + output to localize attention-internal divergence.
    _Attn = getattr(_dv4, "MQALayer", None)
    if _Attn is not None and not getattr(_Attn, "_hid_attn_hooked", False):
        _oatt = _Attn.forward

        def _attw(self, *a, **k):
            xin = k.get("x")
            if xin is None and len(a) > 0 and torch.is_tensor(a[0]):
                xin = a[0]
            # One-time tap of wo_a input = the pre-o_proj MLA attention-core output.
            _woa = getattr(self, "wo_a", None)
            if _woa is not None and not getattr(_woa, "_hid_woa_tapped", False):
                _owoa = _woa.forward

                def _woaw(x, *aa, **kk):
                    if torch.is_tensor(x) and x.shape[0] < 32 and _HID_DBG.get("woa", 0) < 6:
                        _HID_DBG["woa"] = _HID_DBG.get("woa", 0) + 1
                        try:
                            with open(_f, "a") as fh:
                                fh.write(f"WOA#{_HID_DBG['woa']} preO {_stat(x)}\n")
                        except Exception:  # noqa: BLE001
                            pass
                    return _owoa(x, *aa, **kk)

                _woa.forward = _woaw
                _woa._hid_woa_tapped = True
            out = _oatt(self, *a, **k)
            if xin is not None and torch.is_tensor(xin) and xin.shape[0] < 32 and _HID_DBG.get("attn", 0) < 4:
                _HID_DBG["attn"] = _HID_DBG.get("attn", 0) + 1
                try:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    with open(_f, "a") as fh:
                        fh.write(f"ATTN#{_HID_DBG['attn']} in {_stat(xin)}\n")
                        if torch.is_tensor(o):
                            fh.write(f"ATTN#{_HID_DBG['attn']} out {_stat(o)}\n")
                except Exception:  # noqa: BLE001
                    pass
            return out

        _Attn.forward = _attw
        _Attn._hid_attn_hooked = True

        # Split attention internals: final q (post norm+rope) and the pre-norm compressed kv (wkv).
        if hasattr(_Attn, "_compute_q_b") and hasattr(_Attn, "_compute_kv_to_cache"):
            _oqb = _Attn._compute_q_b
            _okv = _Attn._compute_kv_to_cache

            def _qbw(self, q, positions, q_out=None):
                r = _oqb(self, q, positions, q_out)
                if torch.is_tensor(r) and r.shape[0] < 32 and _HID_DBG.get("qb", 0) < 3:
                    _HID_DBG["qb"] = _HID_DBG.get("qb", 0) + 1
                    try:
                        if _HID_DBG["qb"] == 1:
                            torch.save({"q_final": r.detach().cpu(), "q_lora": q.detach().cpu()}, _f + ".qcmp.pt")
                        with open(_f, "a") as fh:
                            fh.write(f"QB#{_HID_DBG['qb']} qlora {_stat(q)}\n")
                            fh.write(f"QB#{_HID_DBG['qb']} q_final {_stat(r)}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            def _kvw(self, x, positions, forward_batch, attn_backend, qkv_a=None):
                if torch.is_tensor(x) and x.shape[0] < 32 and _HID_DBG.get("kv", 0) < 3:
                    _HID_DBG["kv"] = _HID_DBG.get("kv", 0) + 1
                    try:
                        kv = qkv_a[..., self.q_lora_rank :] if qkv_a is not None else self.wkv(x)[0]
                        with open(_f, "a") as fh:
                            fh.write(f"KV#{_HID_DBG['kv']} wkv_out {_stat(kv)}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return _okv(self, x, positions, forward_batch, attn_backend, qkv_a)

            _Attn._compute_q_b = _qbw
            _Attn._compute_kv_to_cache = _kvw

    # Tap fused_q_norm_rope: input (wq_b out) + output, nope[0:4] AND rope[-4:] dims.
    _ofqr = getattr(_dv4, "fused_q_norm_rope", None)
    if _ofqr is not None and not getattr(_dv4, "_fqr_cpu_tapped", False):

        def _fqr(qi, qo, *aa, **kk):
            r = _ofqr(qi, qo, *aa, **kk)
            try:
                if torch.is_tensor(qi) and qi.shape[0] < 32 and _HID_DBG.get("fqr", 0) < 6:
                    _HID_DBG["fqr"] = _HID_DBG.get("fqr", 0) + 1
                    qif = qi.reshape(qi.shape[0], -1)
                    qof = qo.reshape(qo.shape[0], -1)
                    with open(_f, "a") as fh:
                        fh.write(
                            f"FQR#{_HID_DBG['fqr']} "
                            f"in_nope={[round(v,4) for v in qif[0,:4].float().tolist()]} "
                            f"in_rope={[round(v,4) for v in qif[0,448:452].float().tolist()]} "
                            f"out_nope={[round(v,4) for v in qof[0,:4].float().tolist()]} "
                            f"out_rope={[round(v,4) for v in qof[0,448:452].float().tolist()]}\n"
                        )
            except Exception:  # noqa: BLE001
                pass
            return r

        _dv4.fused_q_norm_rope = _fqr
        _dv4._fqr_cpu_tapped = True

    try:
        import sglang.kernels.ops.layernorm.mhc as _mhc_dbg

        if not getattr(_mhc_dbg, "_sink_dbg_cpu", False):
            _osink = _mhc_dbg.hc_split_sinkhorn

            def _sink(mixes, *a, **k):
                r = _osink(mixes, *a, **k)
                if _HID_DBG.get("sink", 0) < 3:
                    _HID_DBG["sink"] = _HID_DBG.get("sink", 0) + 1
                    try:
                        last = mixes.shape[-1]
                        hc = last // 6
                        flat = mixes.reshape(-1, last)
                        post_slice = flat[:, hc : 2 * hc]
                        pre_o, post_o, comb_o = r
                        scale = a[0] if len(a) > 0 else k.get("hc_scale")
                        base = a[1] if len(a) > 1 else k.get("hc_base")
                        with open(_f, "a") as fh:
                            fh.write(f"SINK#{_HID_DBG['sink']} mixes {_stat(mixes)}\n")
                            fh.write(f"SINK#{_HID_DBG['sink']} mix_postslice {_stat(post_slice)}\n")
                            fh.write(f"SINK#{_HID_DBG['sink']} post_out {_stat(post_o)}\n")
                            if torch.is_tensor(scale):
                                fh.write(f"SINK#{_HID_DBG['sink']} scale {[round(x,4) for x in scale.float().reshape(-1).tolist()]}\n")
                            if torch.is_tensor(base):
                                fh.write(f"SINK#{_HID_DBG['sink']} base8 {[round(x,4) for x in base.float().reshape(-1).tolist()[:8]]}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            _mhc_dbg.hc_split_sinkhorn = _sink
            _mhc_dbg._sink_dbg_cpu = True
            try:
                _dv4._get_mhc_ops.cache_clear()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # Also capture the final next-token logits (top-5 ids/vals) — a fixed argmax regardless of the
    # (healthy) hidden state points at the lm_head / final-norm / logits head, not the transformer.
    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor

        if not getattr(LogitsProcessor, "_hid_dbg_patched", False):
            _olp = LogitsProcessor.forward

            def _lp_fwd(self, *a, **k):
                r = _olp(self, *a, **k)
                try:
                    lg = getattr(r, "next_token_logits", None)
                    if lg is not None and _HID_DBG.get("lp", 0) < 16:
                        _HID_DBG["lp"] = _HID_DBG.get("lp", 0) + 1
                        v, i = torch.topk(lg[-1].float(), 5)
                        with open(_f, "a") as fh:
                            fh.write(
                                f"LOGITS top5 ids={i.tolist()} vals={[round(x, 3) for x in v.tolist()]} "
                                f"lg_mean={lg.float().abs().mean().item():.3e} lg_nan={int(torch.isnan(lg).any())}\n"
                            )
                except Exception:  # noqa: BLE001
                    pass
                return r

            LogitsProcessor.forward = _lp_fwd
            LogitsProcessor._hid_dbg_patched = True
    except Exception:  # noqa: BLE001
        pass
    logger.info("intel_cpu_models: installed per-layer hidden-state debug hook.")


def _install_mhc_cpu() -> None:
    # MHC (hash-clustering) sub-ops hc_split_sinkhorn + hc_combine are TileLang-only, but
    # in-tree TORCH references exist (_hc_split_sinkhorn_torch; combine is a trivial
    # weighted sum). Route CPU to them and select the torch hc_pre path. This is proper
    # enablement (numerically equivalent reference), not a bypass.
    import os

    if not current_platform.is_cpu():
        return
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
    # hc_pre dispatches to deep_gemm.tf32_hc_prenorm_gemm (CUDA-only) once x.shape[0] >=
    # _HC_PRENORM_DEEPGEMM_MIN_TOKENS (1024) -- long prompts crash on CPU with NameError:
    # deep_gemm. Force the torch hc_pre path regardless of token count (deep_gemm is absent on CPU).
    os.environ["SGLANG_OPT_DEEPGEMM_HC_PRENORM"] = "0"
    import sglang.kernels.ops.layernorm.mhc as _mhc

    _sink_torch = _mhc._hc_split_sinkhorn_torch

    _MHC_PROV = {"c": 0, "eager": 0}

    def _mhc_ck_on():
        # WORKER-SIDE control (review fix): a control file re-read from disk EACH call crosses the
        # sgl.Engine subprocess boundary, unlike client os.environ (which the already-spawned worker
        # never sees). INTEL_CPU_DSV4_MHC_CK_FILE = "0" (eager) / "1" (C). Falls back to enabled().
        # Also records a branch counter to INTEL_CPU_DSV4_MHC_PROV_FILE so the A/B can PROVE the worker
        # actually switched arms (reviewer requirement; reject fallback-contaminated runs).
        on = False
        _f = _os.environ.get("INTEL_CPU_DSV4_MHC_CK_FILE")
        if _f:
            try:
                with open(_f) as _cf:
                    on = _cf.read().strip() == "1"
            except Exception:  # noqa: BLE001
                on = False
        else:
            try:
                from . import _dsa_ckernels as _ck
                on = _ck.enabled("sinkhorn")
            except Exception:  # noqa: BLE001
                on = False
        _MHC_PROV["c" if on else "eager"] += 1
        _pf = _os.environ.get("INTEL_CPU_DSV4_MHC_PROV_FILE")
        if _pf:
            try:
                with open(_pf, "w") as _pw:
                    _pw.write(f"c={_MHC_PROV['c']} eager={_MHC_PROV['eager']}")
            except Exception:  # noqa: BLE001
                pass
        return on

    def _sink_dispatch(mixes, hc_scale, hc_base, hc_mult, iters, eps):
        # Increment-1 (fused-layer branch): route the 20-iter sinkhorn recurrence to the
        # parity-validated C kernel (sinkhorn.cpp, ~23x vs torch) when MHC-CK is on; the C wrapper
        # asserts cos>=threshold vs this torch oracle (CHECK mode), and we fall back to the torch ref
        # on ANY exception (shape/parity) so correctness is never at risk.
        _tf = lambda: _sink_torch(mixes, hc_scale, hc_base, hc_mult, iters, eps)
        if _mhc_ck_on():
            try:
                from . import _dsa_ckernels as _ck
                return _ck.mhc_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps, _tf)
            except Exception as _e:  # noqa: BLE001
                try:
                    from . import _dsa_ckernels as _ck
                    if _ck.strict():
                        raise
                except Exception:
                    pass
                logger.warning("[MHC] sinkhorn C kernel fell back to torch: %r", _e)
        return _tf()

    _mhc.hc_split_sinkhorn = _timed("mhc.sinkhorn", "torch")(_sink_dispatch)

    def _cpu_hc_combine(x_flat, pre, hc, out_dtype):
        # y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]. einsum streams the contraction
        # instead of materializing the [m, hc, h] broadcast product then summing it.
        import torch  # noqa: F401

        m = x_flat.shape[0]
        h = x_flat.shape[1] // hc
        xr = x_flat.reshape(m, hc, h).float()
        return torch.einsum("mk,mkh->mh", pre.float(), xr).to(out_dtype)

    def _combine_dispatch(x_flat, pre, hc, out_dtype):
        # Increment-1: route the MHC weighted-combine to combine.cpp (parity-gated) when MHC-CK is on,
        # eager fallback on any exception. Worker-side control file (crosses the subprocess boundary).
        _tf = lambda: _cpu_hc_combine(x_flat, pre, hc, out_dtype)
        if _mhc_ck_on():
            try:
                from . import _dsa_ckernels as _ck
                return _ck.mhc_combine(x_flat, pre, hc, out_dtype, _tf)
            except Exception as _e:  # noqa: BLE001
                try:
                    from . import _dsa_ckernels as _ck
                    if _ck.strict():
                        raise
                except Exception:
                    pass
                logger.warning("[MHC] combine C kernel fell back to torch: %r", _e)
        return _tf()

    _mhc.hc_combine = _timed("mhc.combine", "torch")(_combine_dispatch)

    # NOTE: C/C++ kernel routing for sinkhorn/combine is handled INSIDE _sink_dispatch/_combine_dispatch
    # via _mhc_ck_on() (worker-side control file INTEL_CPU_DSV4_MHC_CK_FILE re-read per forward, else the
    # INTEL_CPU_DSV4_DSA_CKERNEL master). The former always-on override here bypassed that control file
    # (and its provenance counter), making the A/B impossible — removed so the dispatch wrappers are the
    # single source of truth (control-file toggle -> C-kernel/eager -> parity -> provenance).
    if hasattr(_mhc, "_mhc_post_torch"):
        import torch

        def _mhc_post_cpu(x, residual, post_layer_mix, comb_res_mix):
            # term2[s,k,h] = sum_j comb[s,j,k]*residual[s,j,h] = bmm(comb^T, residual).
            # fp32 einsum is the DEFAULT (exact, matches CPU baseline). The bf16 bmm alt
            # (INTEL_CPU_DSV4_MHC_POST_BMM=1) microbenched 11x offline but delivered NO in-model
            # speedup (16.9s vs 15.9s, job 380632) while adding bf16 drift (relerr 4e-3) -> kept
            # gated-off per the one-at-a-time rule; default stays on the exact reference.
            if os.environ.get("INTEL_CPU_DSV4_MHC_POST_BMM") == "1":
                term2 = torch.bmm(
                    comb_res_mix.transpose(1, 2).to(torch.bfloat16), residual.to(torch.bfloat16)
                )
                return (post_layer_mix * x.unsqueeze(1) + term2.to(x.dtype)).type_as(x)
            term2 = torch.einsum("sjk,sjh->skh", comb_res_mix.float(), residual.float())
            return (post_layer_mix * x.unsqueeze(1) + term2).type_as(x)

        # hc_post passes raw post [s,n]; the mix wants [s,n,1].
        _mhc.mhc_post = _timed("mhc.post", "torch")(
            lambda x, residual, post, comb: _mhc_post_cpu(x, residual, post.unsqueeze(-1), comb)
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


def _overlap_transform(tensor, ratio, d, value):
    # Reference Compressor.overlap_transform: [b,s,ratio,2d] -> [b,s,2*ratio,d].
    # positions [ratio:] = current block's normal half (dims d:2d); positions [:ratio] for
    # windows>=1 = previous block's overlap half (dims :d); window 0 [:ratio] = value.
    import torch

    b, s, _, _ = tensor.size()
    new = tensor.new_full((b, s, 2 * ratio, d), value)
    new[:, :, ratio:] = tensor[:, :, :, d:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new


def _compress_prefill_core(
    kv_all, score_all, ape, norm_weight, norm_eps, freqs_cis, positions, ext,
    ratio, head_dim, rope_dim, overlap, do_fp8, out_dtype,
):
    # PURE producer compute shared by the production compressor AND the offline test, so the test
    # certifies the REAL path (not a transcription). Inputs: kv_all/score_all [N, coff*head_dim] fp32;
    # ape [ratio, coff*head_dim] fp32 (RAW, hotfix already undone); norm_weight [head_dim]; freqs_cis
    # complex [max_pos, rope_dim//2]; positions [N]; ext list of extend lengths. Returns
    # (comp_bf16 [S, head_dim], bpos [S]). Mirrors reference Compressor.forward prefill.
    import torch

    cw = (2 if overlap else 1) * head_dim
    comp_rows, blk_pos = [], []
    off = 0
    for L in ext:
        L = int(L)
        nwin = L // ratio
        if nwin > 0:
            n = nwin * ratio
            kv_w = kv_all[off : off + n].reshape(1, nwin, ratio, cw)
            sc_w = score_all[off : off + n].reshape(1, nwin, ratio, cw) + ape
            if overlap:
                kv_w = _overlap_transform(kv_w, ratio, head_dim, 0.0)
                sc_w = _overlap_transform(sc_w, ratio, head_dim, float("-inf"))
            pooled = (kv_w * sc_w.softmax(dim=2)).sum(dim=2).squeeze(0)  # [nwin, head_dim]
            comp_rows.append(pooled)
            blk_pos.append(positions[off : off + n : ratio])
        off += L
    if not comp_rows:
        return None, None

    compressed = torch.cat(comp_rows, 0)  # [S, head_dim] fp32
    # RMSNorm over head_dim (reference casts back to activation dtype BEFORE rope; align to out_dtype).
    cn = compressed.to(out_dtype).float()
    cn = cn * torch.rsqrt(cn.square().mean(-1, keepdim=True) + norm_eps) * norm_weight.float()
    compressed = cn.to(out_dtype).float()
    bpos = torch.cat(blk_pos, 0).long()

    if freqs_cis is not None and rope_dim > 0:
        fr = torch.view_as_real(freqs_cis[bpos]).flatten(-2)  # [S, rope_dim]
        rp = compressed[:, head_dim - rope_dim :].reshape(-1, rope_dim // 2, 2)
        xr, xi = rp[..., 0], rp[..., 1]
        f = fr.reshape(-1, rope_dim // 2, 2)
        cr, ci = f[..., 0], f[..., 1]
        roped = torch.stack([xr * cr - xi * ci, xr * ci + xi * cr], dim=-1).reshape(-1, rope_dim)
        compressed[:, head_dim - rope_dim :] = roped

    comp_bf16 = compressed.to(torch.bfloat16)
    if do_fp8:
        comp_bf16 = _fp8_roundtrip_nope_bf16(comp_bf16)
    return comp_bf16, bpos


def _undo_ape_hotfix(ape_param, ratio, head_dim, overlap, converted):
    # sglang _apply_ape_hotfix (overlap): raw[ratio,coff*d] -> cat([ov_halves,no_halves],0).view(ratio,-1).
    # Undo to recover the raw [ratio, coff*head_dim] layout the reference adds pre-overlap_transform.
    import torch

    ap = ape_param.float()
    cw = (2 if overlap else 1) * head_dim
    if overlap and converted:
        a = ap.reshape(2 * ratio, head_dim)
        ap = torch.cat([a[:ratio], a[ratio:]], dim=-1)
    return ap.reshape(ratio, cw)



    # Pre-fix behavior (kept for A/B via INTEL_CPU_DSV4_COMPRESSOR_FIX=0). KNOWN-BUGGY for ratio4:
    # slices [:D]/[D:2D] of a coff=2 projection (= both KV halves) and stashes to _COMPRESSED_STASH
    # which NOTHING reads -> compressed branch absent from attention (SWA-only on compressed layers).
    import torch

    from intel_cpu_models.dsa_compressor_cpu import compress_softmax_pool

    if forward_batch.forward_mode.is_idle():
        return
    ratio = int(compressor.ratio)
    head_dim = int(compressor.head_dim)
    kv_score = compressor.compute_kv_score(x, forward_batch)
    kv_all = kv_score[:, :head_dim]
    score_all = kv_score[:, head_dim : 2 * head_dim]
    ape = compressor.ape.view(-1, head_dim)[:ratio]
    ext = forward_batch.extend_seq_lens_cpu
    if ext is None:
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
        compressed = _rmsnorm_torch(compressed, compressor.norm)
    else:
        compressed = kv_all.new_zeros(0, head_dim)
    _COMPRESSED_STASH[(layer_id, ratio)] = compressed


def _cpu_forward_core_compressor(self, x, forward_batch, layer_id, compressor) -> None:
    # CORRECTED CPU compressor (reviewer P1+P2): produce the compressed KV with the FAITHFUL C4/C128
    # pooling (coff-aware channel split + overlap_transform + compressed-key RoPE + fp8-nope-roundtrip)
    # and WRITE it into the backend's extra-key cache (_KV_BUF[extra_k_cache.data_ptr()]) at the
    # metadata out_loc slots, so _batched_mla_attend's extra gather actually finds the compressed keys
    # (previously it returned None -> SWA-only attention on compressed layers -> layer-2 divergence).
    # Validated offline vs reference model.py Compressor.forward (test_compressor_offline.py).
    # INTEL_CPU_DSV4_COMPRESSOR_FIX=0 reverts to the (buggy) legacy stash path for A/B.
    import os as _o

    import torch

    if _o.environ.get("INTEL_CPU_DSV4_COMPRESSOR_FIX", "1") != "1":
        return _cpu_forward_core_compressor_legacy(self, x, forward_batch, layer_id, compressor)
    if forward_batch.forward_mode.is_idle() or x.shape[0] == 0:
        return

    ratio = int(compressor.ratio)
    head_dim = int(compressor.head_dim)
    rope_dim = int(getattr(compressor, "rope_head_dim", 64))
    overlap = ratio == 4
    coff = 2 if overlap else 1
    _diag = _o.environ.get("INTEL_CPU_DSV4_COMPRESSOR_DIAG") == "1"

    kv_score = compressor.compute_kv_score(x, forward_batch)  # [N, 2*coff*head_dim]
    cw = coff * head_dim
    kv_all = kv_score[:, :cw].float()
    score_all = kv_score[:, cw : 2 * cw].float()
    # ape: undo the sglang load-time HOTFIX (overlap) to recover raw [ratio, coff*head_dim].
    ape = _undo_ape_hotfix(compressor.ape, ratio, head_dim, overlap, getattr(compressor, "ape_converted", False))
    positions = forward_batch.positions

    ext = forward_batch.extend_seq_lens_cpu
    if ext is None:
        # DECODE: no new full windows for short gens; re-publish each request's PERSISTED prefill
        # compressed KV so decode attention attends the long-range compressed context (not SWA-only).
        _decode_republish_compressed(self, forward_batch, layer_id, ratio, head_dim, positions, _diag)
        return

    norm_eps = getattr(compressor.norm, "variance_epsilon", getattr(compressor.norm, "eps", 1e-6))
    comp_bf16, bpos = _compress_prefill_core(
        kv_all, score_all, ape, compressor.norm.weight, norm_eps,
        getattr(compressor, "freqs_cis", None), positions, ext,
        ratio, head_dim, rope_dim, overlap,
        do_fp8=(_o.environ.get("INTEL_CPU_DSV4_KV_BF16_EXACT") != "1"),
        out_dtype=x.dtype,
    )
    if comp_bf16 is None:
        if _diag:
            _compressor_diag_write(f"[COMPDIAG] L{layer_id} r{ratio} no full window -> no write")
        return

    # COMPRESSOR-INPUT CAPTURE (reviewer): dump the producer operands at the FIRST compressed layer so
    # an independent reference computed from the RAW CHECKPOINT (wkv/wgate/ape/norm) on the SAME x can
    # be compared to the production comp_kv -> isolates which operand (fused-split / ape / norm / freqs)
    # is wrong. One-shot.
    if (overlap and _o.environ.get("INTEL_CPU_DSV4_COMP_INPUT_CAPTURE")
            and not _COMP_INPUT_CAP.get("done")):
        _COMP_INPUT_CAP["done"] = True
        try:
            fcis = getattr(compressor, "freqs_cis", None)
            snap = {
                "layer_id": int(layer_id), "ratio": int(ratio), "head_dim": int(head_dim),
                "rope_dim": int(rope_dim), "overlap": bool(overlap),
                "x": x.detach().float().cpu(),                    # compressor INPUT
                "kv_score": compressor.compute_kv_score(x, forward_batch).detach().float().cpu(),  # fused proj
                "ape_prod": ape.detach().float().cpu(),           # post undo-hotfix (what production used)
                "ape_param": compressor.ape.detach().float().cpu(),  # raw stored param (hotfixed)
                "ape_converted": bool(getattr(compressor, "ape_converted", False)),
                "norm_weight": compressor.norm.weight.detach().float().cpu(),
                "norm_eps": float(norm_eps),
                "wkv_gate_weight": compressor.wkv_gate.weight.detach().float().cpu(),  # live fused weight
                "positions": positions.detach().cpu(),
                "ext": [int(v) for v in ext],
                "comp_kv": comp_bf16.detach().float().cpu(),      # production OUTPUT
                "bpos": bpos.detach().cpu(),
                "freqs_at_bpos": (torch.view_as_real(fcis[bpos.long()]).detach().float().cpu()
                                  if fcis is not None else None),
            }
            torch.save(snap, _o.environ["INTEL_CPU_DSV4_COMP_INPUT_CAPTURE"])
            logger.warning("[COMPIN] L%d compressor-input snapshot saved (x=%s kv_score=%s)",
                           layer_id, tuple(x.shape), tuple(snap["kv_score"].shape))
        except Exception as _e:  # noqa: BLE001
            logger.warning("[COMPIN] capture failed: %r", _e)


    # the compressed KV + its block-start absolute positions + the query positions to a per-edp
    # HANDOFF that _batched_mla_attend consumes directly with a causal COVERAGE mask (block j visible
    # to query at abs pos p iff (j+1)*ratio <= p+1). Selection is effectively dense here (S<topk), so
    # the coverage mask equals the correct sparse selection.
    extra_k_cache = None
    try:
        pool = getattr(self, "token_to_kv_pool", None)
        if pool is not None and hasattr(pool, "get_extra_key_buffer"):
            extra_k_cache = pool.get_extra_key_buffer(layer_id)
    except Exception as e:  # noqa: BLE001
        if _diag:
            _compressor_diag_write(f"[COMPDIAG] L{layer_id} r{ratio} extra_key_buffer lookup failed: {e!r}")

    S = comp_bf16.shape[0]
    if extra_k_cache is None:
        _COMPRESSED_STASH[(layer_id, ratio)] = comp_bf16
        if _diag:
            _compressor_diag_write(f"[COMPDIAG] L{layer_id} r{ratio} NO extra_k_cache -> stashed only S={S}")
        return

    # Request-aware structure: which request each BLOCK and each QUERY (token) belongs to. positions
    # are per-request, so same-request masking (below) prevents a cross-request compressed-KV LEAK.
    ext_t = [int(v) for v in ext]
    nreq = len(ext_t)
    nwin = [L // ratio for L in ext_t]
    query_reqid = torch.repeat_interleave(torch.arange(nreq), torch.tensor(ext_t))        # [N]
    block_reqid = torch.repeat_interleave(torch.arange(nreq), torch.tensor(nwin))         # [S]

    # PERSIST per request for decode (keyed by req_pool_index, stable prefill->decode).
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is not None and block_reqid.numel() == S:
        rpi_l = rpi.detach().cpu().tolist()
        for r in range(nreq):
            bsel = block_reqid == r
            if bool(bsel.any()) and r < len(rpi_l):
                _COMPRESSED_PERSIST[(layer_id, int(rpi_l[r]))] = (
                    comp_bf16[bsel].clone(), bpos[bsel].clone(),
                )

    edp = extra_k_cache.data_ptr()
    _COMPRESSED_HANDOFF[edp] = {
        "comp_kv": comp_bf16, "block_pos": bpos.clone(), "block_reqid": block_reqid,
        "query_pos": positions.detach().clone(), "query_reqid": query_reqid, "ratio": ratio,
    }
    if _diag:
        _compressor_diag_write(
            f"[COMPDIAG] L{layer_id} r{ratio} coff{coff} HANDOFF S={S} nreq={nreq} edp={edp} "
            f"bpos[:4]={bpos[:4].tolist()} npos={positions.shape[0]}"
        )


def _decode_republish_compressed(self, forward_batch, layer_id, ratio, head_dim, positions, _diag):
    # Rebuild a request-aware handoff for the decode batch from each request's PERSISTED prefill
    # compressed KV. One query per request; all persisted blocks are in the past (visible).
    import torch

    pool = getattr(self, "token_to_kv_pool", None)
    extra_k_cache = pool.get_extra_key_buffer(layer_id) if (pool is not None and hasattr(pool, "get_extra_key_buffer")) else None
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if extra_k_cache is None or rpi is None:
        return
    rpi_l = rpi.detach().cpu().tolist()
    parts, bp, br = [], [], []
    for r, pool_idx in enumerate(rpi_l):
        pv = _COMPRESSED_PERSIST.get((layer_id, int(pool_idx)))
        if pv is not None and pv[0].shape[0] > 0:
            ckv, bpk = pv
            parts.append(ckv)
            bp.append(bpk)
            br.append(torch.full((ckv.shape[0],), r, dtype=torch.long))
    if not parts:
        return
    comp_all = torch.cat(parts, 0)
    bpos_all = torch.cat(bp, 0)
    breq_all = torch.cat(br, 0)
    query_reqid = torch.arange(len(rpi_l))               # one decode query per request
    edp = extra_k_cache.data_ptr()
    _COMPRESSED_HANDOFF[edp] = {
        "comp_kv": comp_all, "block_pos": bpos_all, "block_reqid": breq_all,
        "query_pos": positions.detach().clone(), "query_reqid": query_reqid, "ratio": ratio,
    }
    if _diag:
        _compressor_diag_write(
            f"[COMPDIAG] L{layer_id} r{ratio} DECODE republish S={comp_all.shape[0]} nreq={len(rpi_l)}"
        )


def _compressor_diag_write(msg: str) -> None:
    import os as _o

    try:
        logger.warning(msg)
        with open(_o.environ.get("ATTN_DBG_FILE", "/scratch/bkaul/dsv4_compressor_diag.txt"), "a") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001
        pass



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
    # FUSED C/C++ path (graph-loop analysis): indexer_logits.cpp fuses the einsum→relu→per-head-
    # weight→sum into ONE kernel (scores never materialize to DRAM); indexer_topk.cpp fuses the
    # selection. Gated by INTEL_CPU_DSV4_DSA_CKERNEL; the eager einsum/topk stays as the A/B oracle
    # (DSA_CKERNEL_CHECK=1 asserts C==torch per call). The authored kernel's contract is per-query
    # kv[N,S,D]; the serving's compressed KV `ck[S,D]` is SHARED across the N queries, so broadcast
    # it (step-1a: correctness-first dispatch+parity; a shared-KV kernel variant is the step-1b perf
    # follow-up that removes the broadcast).
    _use_ck_idx = False
    try:
        from . import _dsa_ckernels as _ck
        from . import dsa_indexer_cpu as _dix
        _use_ck_idx = _ck.enabled("indexer")
    except Exception:  # noqa: BLE001
        _use_ck_idx = False
    if _use_ck_idx:
        try:
            kv_bc = ck.unsqueeze(0).expand(q.shape[0], ck.shape[0], ck.shape[1]).contiguous()  # [N,S,chd]
            logits = _dix.indexer_logits(q, kv_bc, w)  # fused relu*weight*sum (C); marks c_kernel
        except Exception as _e:  # noqa: BLE001 — degenerate shapes (tiny S) -> eager oracle fallback
            if _ck.strict():
                raise
            logger.warning("[DSA] indexer C kernel fell back to eager: %r", _e)
            _use_ck_idx = False
            _ck.mark("indexer", "eager")
            scores = torch.einsum("nhd,sd->nhs", q, ck)
            logits = (torch.relu(scores) * w.unsqueeze(-1)).sum(dim=1)
    else:
        if "_ck" in dir():
            _ck.mark("indexer", "eager")
        scores = torch.einsum("nhd,sd->nhs", q, ck)  # [N, nh, S]
        logits = (torch.relu(scores) * w.unsqueeze(-1)).sum(dim=1)  # [N, S]
    causal = key_start.unsqueeze(0) <= positions.reshape(-1, 1)  # query sees only past windows
    logits = logits.masked_fill(~causal, float("-inf"))
    ktop = min(int(c4_indexer.index_topk), logits.shape[1])
    if _use_ck_idx:
        topk = _dix.indexer_topk(logits, ktop)  # fused top-k (C); marks c_kernel
    else:
        if "_ck" in dir():
            _ck.mark("topk", "eager")
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
