"""Hardware-agnostic PER-LAYER hidden-state capture for DeepSeek-V4 (CPU + GPU) — the DEBUGGER that
localizes a CPU/GPU (or CPU-C/CPU-eager) correctness divergence to the first divergent LAYER/op.

The output-logit capture (capture_logits.py) only localizes to a token POSITION; this localizes to a
LAYER. Env-gated + ONE-SHOT: when INTEL_CPU_DSV4_LAYER_CAPTURE=<outfile> is set, the FIRST model
forward records every decoder layer's last-token hidden vector (the next-token predictor position),
keyed by layer_id, plus the final-logits top-k, and dumps them. Compare the per-layer vectors across
two legs (compare_layers.py) -> the first layer whose cosine drops is where the divergence enters.

Installed via SGLANG_EXTERNAL_MODEL_PACKAGE so the WORKER subprocess applies the patch (the model runs
there). Safe on any device: it only wraps the existing forwards and moves captured tensors to cpu/float.
No EntryClass -> registers no model; native deepseek_v4 stays registered. On CPU it is also imported
(gated) by the intel_cpu_models plugin so the full CPU stack + capture coexist.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_LAYER_STATES: "dict[int, list]" = {}
_CAPTURE_ACTIVE = False
_CAPTURE_DONE = False
_INSTALLED = False

# MoE discrete expert-index capture (routing correctness): per-layer selected global expert ids.
# These are integers and TP-invariant (router runs replicated, topk_ids are global), so they
# sidestep the MHC [N,4,hidden] layout problem that makes continuous tensor captures non-comparable.
_MOE_TOPK: "dict[int, list]" = {}
_MOE_TOPK_INSTALLED = False
_MOE_TOPK_DONE = False

# Request-bound capture: only fire on the forward whose input_ids match the REAL frozen prompt
# (NOT the engine warmup/dummy batch, which otherwise pollutes a one-shot "first forward" capture).
_FROZEN_IDS = None
_CAPTURE_SEQ_IDX = None
_CAPTURE_INPUT_IDS = None


def _load_frozen_ids():
    global _FROZEN_IDS
    if _FROZEN_IDS is not None:
        return _FROZEN_IDS
    _FROZEN_IDS = []
    path = os.environ.get(
        "INTEL_CPU_DSV4_CAPTURE_FROZEN",
        "/data/nfs_home/bkaul/dsv4-tree-pilot/plugin/validate/results/p1_frozen_tokenids.json",
    )
    try:
        d = json.load(open(path))
        for it in d.get("items", []):
            _FROZEN_IDS.append(tuple(int(x) for x in it["token_ids"]))
        logger.warning("[LAYER_CAPTURE] loaded %d frozen seqs for request-bound gating", len(_FROZEN_IDS))
    except Exception as e:  # noqa: BLE001
        logger.warning("[LAYER_CAPTURE] frozen load failed (%r); capture will not bind to a request", e)
    return _FROZEN_IDS


def _target_seq_index(a, k):
    # Return (seq_idx, ids_list) if this forward's input_ids match a frozen seq (exact, or same
    # length + non-constant as a fallback for benign tokenization); else (None, None). This REJECTS
    # the warmup/dummy forward so the one-shot capture binds to the real prompt on BOTH legs.
    try:
        import torch

        ii = a[0] if a else k.get("input_ids")
        if ii is None or not torch.is_tensor(ii):
            return None, None
        ids = tuple(int(x) for x in ii.reshape(-1).tolist())
        frozen = _load_frozen_ids()
        for idx, f in enumerate(frozen):
            if ids == f:
                return idx, list(ids)
        for idx, f in enumerate(frozen):
            if len(ids) == len(f) and len(set(ids)) > 1:
                return idx, list(ids)
        return None, None
    except Exception:  # noqa: BLE001
        return None, None


def _install_layer_capture() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if not os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE"):
        return  # inert unless explicitly enabled — zero overhead on normal runs
    try:
        import sglang.srt.models.deepseek_v4 as _m
    except Exception as e:  # noqa: BLE001
        logger.warning("[LAYER_CAPTURE] deepseek_v4 import failed: %r", e)
        return
    _INSTALLED = True

    LayerCls = _m.DeepseekV4DecoderLayer
    ModelCls = _m.DeepseekV4ForCausalLM
    _orig_layer_fwd = LayerCls.forward
    _orig_model_fwd = ModelCls.forward

    def _layer_fwd(self, *a, **k):
        out = _orig_layer_fwd(self, *a, **k)
        if _CAPTURE_ACTIVE:
            try:
                hs = out[0] if isinstance(out, (tuple, list)) else out
                # robust to [num_tokens, hidden] or [batch, seq, hidden]: flatten leading dims,
                # take the LAST token's hidden vector (the next-token predictor position).
                v = hs.detach().float().reshape(-1, hs.shape[-1])[-1].cpu().tolist()
                lid = int(getattr(self, "layer_id", len(_LAYER_STATES)))
                _LAYER_STATES[lid] = [round(float(x), 6) for x in v]
            except Exception as e:  # noqa: BLE001
                logger.warning("[LAYER_CAPTURE] layer %s capture failed: %r", getattr(self, "layer_id", "?"), e)
        return out

    def _model_fwd(self, *a, **k):
        global _CAPTURE_ACTIVE, _CAPTURE_DONE, _CAPTURE_SEQ_IDX, _CAPTURE_INPUT_IDS
        outfile = os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE")
        seq_idx, ids = _target_seq_index(a, k) if (bool(outfile) and not _CAPTURE_DONE) else (None, None)
        activate = seq_idx is not None
        if activate:
            _CAPTURE_ACTIVE = True
            _CAPTURE_SEQ_IDX = seq_idx
            _CAPTURE_INPUT_IDS = ids
            _LAYER_STATES.clear()
            logger.warning("[LAYER_CAPTURE] request-bound activate on frozen seq %d (%d tokens)", seq_idx, len(ids))
        logits = _orig_model_fwd(self, *a, **k)
        if activate:
            _CAPTURE_ACTIVE = False
            _CAPTURE_DONE = True
            if not _is_rank0():
                return logits  # only rank 0 writes (fixes TP file-write races)
            try:
                fl = logits
                if hasattr(fl, "detach"):
                    last = fl[-1].detach().float().cpu()
                    topv, topi = last.topk(min(5, last.numel()))
                    final_top = [[int(i), round(float(v), 6)] for v, i in zip(topv.tolist(), topi.tolist())]
                else:
                    final_top = None
                payload = {"leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "unknown"),
                           "n_layers": len(_LAYER_STATES),
                           "seq_idx": _CAPTURE_SEQ_IDX,
                           "input_ids": _CAPTURE_INPUT_IDS,
                           "layers": {str(li): v for li, v in sorted(_LAYER_STATES.items())},
                           "final_logits_top5": final_top}
                tmp = outfile + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp, outfile)
                logger.warning("[LAYER_CAPTURE] wrote %d layers -> %s", len(_LAYER_STATES), outfile)
            except Exception as e:  # noqa: BLE001
                logger.warning("[LAYER_CAPTURE] dump failed: %r", e)
        return logits

    LayerCls.forward = _layer_fwd
    ModelCls.forward = _model_fwd
    logger.warning("[LAYER_CAPTURE] installed (one-shot per-layer last-token capture).")


_ATTN_IO_DONE = False


def _install_attn_io_capture() -> None:
    """Oracle-anchored ATTENTION SUB-OP I/O capture (hardware-agnostic: GPU oracle AND CPU plugin).

    Gated by INTEL_CPU_DSV4_ATTNIO_CAPTURE=<outfile.pt>. Taps the NATIVE MQALayer at layer 0 and saves
    FULL tensors at the exact sub-op boundaries the two implementations share:
      - 'attn_in'   : the MQALayer input hidden (post input_layernorm) — the attention INPUT
      - 'q_final'   : _compute_q_b output = Q post norm+rope (into the attention core)
      - 'kv'        : _compute_kv_to_cache / wkv output = the compressed KV
      - 'woa_in'    : the wo_a input = the attention-CORE output (pre o_proj) — THE key quantity
      - 'attn_out'  : the MQALayer output
    Comparing these across GPU-oracle vs CPU localizes the drift: if q_final/kv match but woa_in diverges,
    the bug is the attention CORE compute (QK/softmax/PV/sink). Cross-feeding GPU's q_final+kv through the
    CPU core and diffing vs GPU's woa_in pinpoints compute-vs-upstream. One-shot, layer-0, first forward.
    """
    out = os.environ.get("INTEL_CPU_DSV4_ATTNIO_CAPTURE")
    if not out:
        return
    try:
        import sglang.srt.models.deepseek_v4 as _m
        import torch
    except Exception as e:  # noqa: BLE001
        logger.warning("[ATTNIO] import failed: %r", e)
        return
    Attn = getattr(_m, "MQALayer", None)
    if Attn is None or getattr(Attn, "_attnio_hooked", False):
        return
    Attn._attnio_hooked = True
    _stash = {}

    def _save(name, t):
        try:
            if torch.is_tensor(t):
                _stash[name] = t.detach().float().cpu()
        except Exception:  # noqa: BLE001
            pass

    def _is_l0(self):
        return int(getattr(self, "layer_id", getattr(self, "layer_idx", -1))) == 0

    # woa_in = the attention-CORE output pre-o_proj, captured by tapping the NATIVE module-level
    # `_apply_wo_a_bf16_matmul(o, wo_a, is_decode)` (o is [T,G,D]) which BOTH CPU and GPU call. First
    # call while capture is active = layer 0's attention. Gated + one-shot per capture.
    try:
        import sglang.srt.models.deepseek_v4 as _dv4m
        if not getattr(_dv4m, "_dsv4_woa_cap_patched", False) and hasattr(_dv4m, "_apply_wo_a_bf16_matmul"):
            _owoa_fn = _dv4m._apply_wo_a_bf16_matmul

            def _woa_cap(o, wo_a, is_decode, *a, **k):
                if _CAPTURE_ACTIVE and "woa_in" not in _stash:
                    _save("woa_in", o)
                return _owoa_fn(o, wo_a, is_decode, *a, **k)

            _dv4m._apply_wo_a_bf16_matmul = _woa_cap
            _dv4m._dsv4_woa_cap_patched = True
    except Exception:  # noqa: BLE001
        pass

    # Capture the REAL attention operands, gated on the request-bound _CAPTURE_ACTIVE flag so they
    # bind to the frozen q640 forward (NOT an engine warmup/partial forward). wo_a / wkv are NOT
    # .forward modules here (low-rank projections applied via functions), so hook the actual methods:
    # _compute_q_b -> Q (post norm+rope), _compute_kv_bf16 -> KV. These are TP head-sharded, so every
    # rank saves its own shard to a per-rank file and the heads are concatenated offline.
    _oqb = getattr(Attn, "_compute_q_b", None)
    if _oqb is not None:
        def _qbw(self, q, positions, q_out=None, *a, **k):
            r = _oqb(self, q, positions, q_out, *a, **k)
            if _CAPTURE_ACTIVE and _is_l0(self) and "q_final" not in _stash:
                _save("q_final", r)        # [T, n_local_heads, head_dim] (TP shard)
                _save("q_lora", q)         # [T, q_lora_rank] (replicated)
            return r
        Attn._compute_q_b = _qbw

    _okvb = getattr(Attn, "_compute_kv_bf16", None)
    if _okvb is not None:
        def _kvbw(self, x, positions, *a, **k):
            r = _okvb(self, x, positions, *a, **k)
            if _CAPTURE_ACTIVE and _is_l0(self) and "kv" not in _stash:
                _save("kv", r[0] if isinstance(r, tuple) else r)
            return r
        Attn._compute_kv_bf16 = _kvbw

    _ofwd = Attn.forward

    def _fwd(self, *a, **k):
        global _ATTN_IO_DONE
        first_l0 = _is_l0(self) and not _ATTN_IO_DONE and _CAPTURE_ACTIVE
        if first_l0:
            # attn input hidden = MQALayer.forward(self, positions, hidden_states, forward_batch, ...)
            # -> hidden_states is positional arg index 1 (exact).
            xin = k.get("hidden_states")
            if xin is None:
                xin = k.get("x")
            if xin is None and len(a) >= 2 and torch.is_tensor(a[1]):
                xin = a[1]
            if xin is None:
                for z in a:
                    if torch.is_tensor(z) and z.is_floating_point() and z.dim() >= 2:
                        xin = z
                        break
            _save("attn_in", xin)
            # Patch the layer-0 instance's REAL down-projection modules (the executed path uses the
            # fused wqkv_a, not wkv.forward / _compute_kv_bf16) so the KV latent is actually captured.
            # qkv_a = fused q+kv down-proj; kv latent = qkv_a[..., q_lora_rank:]. Also catch wkv/wq_a
            # if the non-fused path runs. One-shot per module.
            for mod_name, stash_name in (("wqkv_a", "qkv_a"), ("wkv", "kv_a_only"), ("wq_a", "q_a_only")):
                mod = getattr(self, mod_name, None)
                if mod is not None and hasattr(mod, "forward") and not getattr(mod, "_dsv4_io_wrapped", False):
                    _omodf = mod.forward
                    def _mk(of, sn, m):
                        def _w(x, *aa, **kk):
                            # Record the ACTUAL input the module receives (x_linear), its dtype, and the
                            # module's weight dtype/quant dispatch — the input may be fp8-quantized on one
                            # leg (x_quant) and bf16 on the other, which is the precision/dispatch the
                            # replay must reflect.
                            if sn + "_in" not in _stash and torch.is_tensor(x) and x.is_floating_point():
                                _save(sn + "_in", x)
                            if "_dispatch" not in _stash:
                                _stash["_dispatch"] = {}
                            if sn not in _stash["_dispatch"]:
                                try:
                                    w = getattr(m, "weight", None)
                                    qm = getattr(m, "quant_method", None)
                                    _stash["_dispatch"][sn] = {
                                        "module": type(m).__name__,
                                        "input_dtype": str(getattr(x, "dtype", None)),
                                        "input_is_tensor": bool(torch.is_tensor(x)),
                                        "weight_dtype": str(getattr(w, "dtype", None)),
                                        "weight_shape": tuple(w.shape) if torch.is_tensor(w) else None,
                                        "quant_method": type(qm).__name__,
                                        "weight_block_size": getattr(qm, "weight_block_size", None),
                                        "w8a8_backend": getattr(getattr(qm, "w8a8_block_fp8_linear", None), "__name__",
                                                                str(getattr(qm, "w8a8_block_fp8_linear", None))),
                                        "block_quant": bool(getattr(qm, "block_quant", False)),
                                        "scale_attrs": [a for a in ("weight_scale", "weight_scale_inv", "input_scale")
                                                        if getattr(m, a, None) is not None],
                                    }
                                    # Capture the fp8 weight + any scale, to settle weights-vs-matmul
                                    # offline (same input, compare weights, replay dequant@input).
                                    if torch.is_tensor(w):
                                        _stash[sn + "_W"] = w.detach().to(torch.float32).cpu()
                                    for a in ("weight_scale", "weight_scale_inv", "input_scale"):
                                        sc = getattr(m, a, None)
                                        if torch.is_tensor(sc):
                                            _stash[f"{sn}_{a}"] = sc.detach().to(torch.float32).cpu()
                                    # STEP 1: capture GPU's ACTUAL quantized activation + scale via the
                                    # authoritative per-token-group fp8 quant the W8A8 kernel uses
                                    # (group size = weight_block_size[1], e.g. 128). Faithful, not a
                                    # reconstruction. On CPU (W8A16) this import/kernel is absent -> skip.
                                    try:
                                        wbs = getattr(qm, "weight_block_size", None)
                                        blk_k = int(wbs[1]) if wbs else 128
                                        from sglang.kernels.ops.quantization.fp8_kernel import (
                                            sglang_per_token_group_quant_fp8 as _ptg,
                                        )
                                        if torch.is_tensor(x) and x.is_floating_point():
                                            qa, qs = _ptg(x, blk_k)
                                            _stash[sn + "_act_fp8"] = qa.detach().to(torch.float32).cpu()
                                            _stash[sn + "_act_scale"] = qs.detach().to(torch.float32).cpu()
                                            _stash["_dispatch"][sn]["act_quant_group"] = blk_k
                                    except Exception as _qe:  # noqa: BLE001
                                        _stash["_dispatch"][sn]["act_quant"] = f"unavailable: {_qe!r}"
                                except Exception:  # noqa: BLE001
                                    pass
                            rr = of(x, *aa, **kk)
                            if sn not in _stash:
                                _save(sn, rr[0] if isinstance(rr, tuple) else rr)
                            return rr
                        return _w
                    mod.forward = _mk(_omodf, stash_name, mod)
                    mod._dsv4_io_wrapped = True
            _stash["_q_lora_rank"] = torch.tensor(int(getattr(self, "q_lora_rank", -1)))
            # REAL-FORWARD SHARED-INPUT REPLAY: if a shared attn_in is provided, run THIS leg's actual
            # wqkv_a fp8 kernel on it (identical input on both legs) and save the output. Comparing the
            # two legs' qkv_a_replay isolates the operator (each leg's real fp8 matmul) with ZERO input
            # difference and without touching the (differently-packed) raw weights.
            _rin = os.environ.get("INTEL_CPU_DSV4_REPLAY_INPUT")
            if _rin and os.path.exists(_rin):
                try:
                    shared = torch.load(_rin, map_location="cpu")
                    shared = shared["attn_in"] if isinstance(shared, dict) else shared
                    wq = getattr(self, "wqkv_a", None)
                    if wq is not None and torch.is_tensor(shared):
                        xin_dev = shared.to(device=next(wq.parameters()).device,
                                            dtype=next(wq.parameters()).dtype if False else torch.bfloat16)
                        rr = wq(xin_dev)
                        _save("qkv_a_replay", rr[0] if isinstance(rr, tuple) else rr)
                        _stash["_replay_input_sha"] = torch.tensor(float(shared.float().sum()))
                except Exception as _e:  # noqa: BLE001
                    logger.warning("[ATTNIO] shared-input replay failed: %r", _e)
        r = _ofwd(self, *a, **k)
        if first_l0:
            _ATTN_IO_DONE = True
            o = r[0] if isinstance(r, (tuple, list)) else r
            _save("attn_out", o)
            # Per-rank save (ALL ranks): sharded operands (q_final/kv) keep each rank's heads so they
            # can be concatenated offline; replicated ones (attn_in/attn_out/layer_in) are identical
            # across ranks. Rank in the filename -> no file-write race.
            try:
                from sglang.srt.distributed import get_tensor_model_parallel_rank as _r
                rank = int(_r())
            except Exception:  # noqa: BLE001
                rank = 0
            try:
                _stash["_meta"] = {"layer_id": 0, "rank": rank,
                                   "leg": os.environ.get("INTEL_CPU_DSV4_ATTNIO_LEG", "unknown")}
                dst = out[:-3] + f"_rank{rank}.pt" if out.endswith(".pt") else out + f".rank{rank}"
                torch.save(_stash, dst)
                logger.warning("[ATTNIO] rank %d saved %s -> %s", rank, list(_stash.keys()), dst)
            except Exception as e:  # noqa: BLE001
                logger.warning("[ATTNIO] save failed: %r", e)
        return r

    Attn.forward = _fwd
    logger.warning("[ATTNIO] installed (layer-0 attention sub-op I/O capture).")

    # DETERMINISTIC ANCHOR: the DecoderLayer's layer-0 INPUT hidden_states = the embedding (layer 0,
    # prev_residual=None -> no MHC combine yet). Token-identical frozen prompt -> this MUST match GPU.
    # If layer_in matches but attn_in (post input_layernorm + MHC-pre) diverges -> the bug is in
    # input_layernorm / MHC-pre, UPSTREAM of the attention core (which fp32-precision already ruled out).
    Layer = getattr(_m, "DeepseekV4DecoderLayer", None)
    if Layer is not None and not getattr(Layer, "_attnio_layer_hooked", False):
        Layer._attnio_layer_hooked = True
        _olfwd = Layer.forward

        def _lfwd(self, *a, **k):
            if int(getattr(self, "layer_id", -1)) == 0 and _CAPTURE_ACTIVE and "layer_in" not in _stash:
                # signature: forward(self, positions, hidden_states, input_ids, forward_batch, ...)
                # -> hidden_states is positional arg index 1 (exact, not a dim heuristic).
                hs = k.get("hidden_states")
                if hs is None and len(a) >= 2 and torch.is_tensor(a[1]):
                    hs = a[1]
                _save("layer_in", hs)
            return _olfwd(self, *a, **k)

        Layer.forward = _lfwd
        logger.warning("[ATTNIO] installed DecoderLayer-0 input (embedding anchor) tap.")


def _is_rank0() -> bool:
    """True on the single rank that should write capture files (fixes TP file-write races)."""
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank()) == 0
    except Exception:  # noqa: BLE001
        try:
            import torch.distributed as dist
            return (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:  # noqa: BLE001
            return True


# Per-layer MoE INPUT (post-attention residual stream fed to the router): full [N, hidden], ALL
# tokens, NOT flattened — the clean operator boundary for the layers-0-3 CPU-vs-GPU comparison.
_MOE_INPUT: "dict[int, object]" = {}
_MOE_INPUT_MAX_LAYER = int(os.environ.get("INTEL_CPU_DSV4_MOEIN_MAXLAYER", "6"))


def _install_moe_topk_capture() -> None:
    """Capture per-layer MoE selected expert ids (discrete, TP-invariant) for CPU-vs-GPU routing
    comparison. One-shot, env-gated on INTEL_CPU_DSV4_LAYER_CAPTURE; dumps to that file + '.moetopk'.
    """
    global _MOE_TOPK_INSTALLED
    if _MOE_TOPK_INSTALLED:
        return
    outfile = os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE")
    if not outfile:
        return
    try:
        import sglang.srt.models.deepseek_v2 as _dv2
        from sglang.srt.layers.moe.topk import TopK
    except Exception as e:  # noqa: BLE001
        logger.warning("[MOE_TOPK] import failed: %r", e)
        return
    try:
        from sglang.srt.layers.moe.hash_topk import HashTopK
    except Exception:  # noqa: BLE001
        HashTopK = None
    _MOE_TOPK_INSTALLED = True

    # Stash each topk module's last topk_ids on the instance.
    for Cls in [c for c in (TopK, HashTopK) if c is not None]:
        if getattr(Cls, "_dsv4_topk_patched", False):
            continue
        _orig = Cls.forward

        def _mk(orig):
            def _f(self, *a, **k):
                out = orig(self, *a, **k)
                try:
                    tids = getattr(out, "topk_ids", None)
                    if tids is not None:
                        self._dsv4_last_topk_ids = tids.detach().to("cpu")
                    # Capture the REAL (hidden, router_logits) this TopK received + the exact call
                    # args, so the router replay can match the real call signature (defect test V2).
                    _h = k.get("hidden_states", a[0] if len(a) > 0 else None)
                    _rl = k.get("router_logits", a[1] if len(a) > 1 else None)
                    if _h is not None and hasattr(_h, "detach"):
                        self._dsv4_last_topk_hidden = _h.detach().to("cpu")
                    if _rl is not None and hasattr(_rl, "detach"):
                        self._dsv4_last_router_logits = _rl.detach().float().to("cpu")
                    self._dsv4_last_topk_extrakeys = [kk for kk in k if kk not in ("hidden_states", "router_logits")]
                except Exception:  # noqa: BLE001
                    pass
                return out
            return _f

        Cls.forward = _mk(_orig)
        Cls._dsv4_topk_patched = True

    MoE = _dv2.DeepseekV2MoE
    if not getattr(MoE, "_dsv4_topkcap_patched", False):
        _omf = MoE.forward

        def _moe_fwd(self, *a, **k):
            import torch
            lid = int(getattr(self, "layer_id", -1))
            # ROUTER INPUT-SUBSTITUTION (defect test, robust): at the target gate layer, REPLACE the
            # hidden_states fed to the REAL MoE forward with an injected tensor, so the REAL gate+topk
            # (no re-call artifact) route on the injected input; the moetopk hook below captures the
            # result. Also save this layer's own bf16 input as the inject-source for the other leg.
            _inj_layer = os.environ.get("INTEL_CPU_DSV4_ROUTER_INJECT_LAYER")
            if (_inj_layer and _CAPTURE_ACTIVE and lid == int(_inj_layer)
                    and not getattr(self, "_dsv4_rinj_done", False)):
                self._dsv4_rinj_done = True
                try:
                    hin = k.get("hidden_states", a[0] if a else None)
                    own_bf16 = hin.detach().reshape(-1, hin.shape[-1]).to(torch.bfloat16).contiguous()
                    _rank0 = True
                    try:
                        from sglang.srt.distributed import get_tensor_model_parallel_rank as _r
                        _rank0 = int(_r()) == 0
                    except Exception:  # noqa: BLE001
                        pass
                    # save own input (the other leg loads this as the inject source)
                    _own_out = os.environ.get("INTEL_CPU_DSV4_ROUTER_INJECT_OWN_OUT")
                    if _own_out and _rank0:
                        torch.save({"layer": lid, "own_hidden_bf16": own_bf16.cpu(),
                                    "leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "?")},
                                   _own_out + ".tmp"); os.replace(_own_out + ".tmp", _own_out)
                    # substitute injected input into the REAL forward (replaces hidden for gate+topk+experts;
                    # we only read this layer's topk, so downstream corruption is irrelevant)
                    _inj_f = os.environ.get("INTEL_CPU_DSV4_ROUTER_INJECT_FILE")
                    if _inj_f and os.path.exists(_inj_f):
                        src = torch.load(_inj_f, map_location="cpu")
                        inj = (src["own_hidden_bf16"] if isinstance(src, dict) else src)
                        inj = inj.to(device=hin.device, dtype=hin.dtype).reshape(hin.shape).contiguous()
                        if "hidden_states" in k:
                            k["hidden_states"] = inj
                        else:
                            a = (inj,) + tuple(a[1:])
                        with open(os.environ.get("INTEL_CPU_DSV4_ROUTER_REPLAY_DBG",
                                                 "/scratch/bkaul/router_replay_dbg.txt"), "a") as fh:
                            fh.write(f"[RINJ] layer {lid}: SUBSTITUTED hidden with inject "
                                     f"(own_sum={float(own_bf16.float().sum()):.3f} "
                                     f"inj_sum={float(inj.float().sum()):.3f})\n")
                except Exception as _e:  # noqa: BLE001
                    with open("/scratch/bkaul/router_replay_dbg.txt", "a") as fh:
                        fh.write(f"[RINJ] layer {lid} FAILED: {_e!r}\n")
            if _CAPTURE_ACTIVE and not _MOE_TOPK_DONE:
                try:
                    if 0 <= lid <= _MOE_INPUT_MAX_LAYER and lid not in _MOE_INPUT:
                        hin = k.get("hidden_states")
                        if hin is None and a:
                            hin = a[0]
                        if hin is not None and hasattr(hin, "detach"):
                            _MOE_INPUT[lid] = hin.detach().float().reshape(-1, hin.shape[-1]).cpu()
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MOE_IN] layer %s input cap failed: %r", lid, e)
            out = _omf(self, *a, **k)
            if _CAPTURE_ACTIVE and not _MOE_TOPK_DONE:
                try:
                    tk = getattr(self, "topk", None)
                    tids = getattr(tk, "_dsv4_last_topk_ids", None)
                    if tids is not None:
                        _MOE_TOPK[lid] = {
                            "is_hash": bool(getattr(self, "is_hash", False)),
                            "topk_ids": tids.int().tolist(),
                        }
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MOE_TOPK] layer %s cap failed: %r", lid, e)
            return out

        MoE.forward = _moe_fwd
        MoE._dsv4_topkcap_patched = True

    # Dump _MOE_TOPK one-shot by wrapping the model forward (composes with the layer-capture wrap).
    ModelCls = _dv2.__dict__.get("DeepseekV2ForCausalLM")
    try:
        import sglang.srt.models.deepseek_v4 as _dv4
        ModelCls = _dv4.DeepseekV4ForCausalLM
    except Exception:  # noqa: BLE001
        pass
    if ModelCls is not None and not getattr(ModelCls, "_dsv4_moetopk_dump_patched", False):
        _omodf = ModelCls.forward

        def _model_fwd_dump(self, *a, **k):
            global _MOE_TOPK_DONE
            logits = _omodf(self, *a, **k)
            if not _MOE_TOPK_DONE and _MOE_TOPK:
                _MOE_TOPK_DONE = True
                if not _is_rank0():
                    return logits  # only rank 0 writes (fixes TP file-write races)
                try:
                    payload = {
                        "leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "unknown"),
                        "n_moe_layers": len(_MOE_TOPK),
                        "seq_idx": _CAPTURE_SEQ_IDX,
                        "input_ids": _CAPTURE_INPUT_IDS,
                        "layers": {str(li): v for li, v in sorted(_MOE_TOPK.items())},
                    }
                    dst = outfile + ".moetopk"
                    with open(dst + ".tmp", "w") as f:
                        json.dump(payload, f)
                    os.replace(dst + ".tmp", dst)
                    logger.warning("[MOE_TOPK] wrote %d MoE layers -> %s", len(_MOE_TOPK), dst)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MOE_TOPK] dump failed: %r", e)
                # MoE INPUT tensors (all tokens, full width) -> compact torch dump for layers 0-3 compare.
                try:
                    if _MOE_INPUT:
                        import torch
                        dstp = outfile + ".moein.pt"
                        torch.save({"leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "unknown"),
                                    "seq_idx": _CAPTURE_SEQ_IDX,
                                    "input_ids": _CAPTURE_INPUT_IDS,
                                    "moe_input": {int(li): t.to(torch.float16) for li, t in _MOE_INPUT.items()}},
                                   dstp + ".tmp")
                        os.replace(dstp + ".tmp", dstp)
                        logger.warning("[MOE_IN] wrote %d MoE-input layers -> %s", len(_MOE_INPUT), dstp)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MOE_IN] dump failed: %r", e)
            return logits

        ModelCls.forward = _model_fwd_dump
        ModelCls._dsv4_moetopk_dump_patched = True
    logger.warning("[MOE_TOPK] installed (per-layer expert-id capture).")


_ATTNOUT: "dict[int, list]" = {}
_QFINAL: "dict[int, list]" = {}
_ATTNOUT_INSTALLED = False
_ATTNOUT_DONE = False


def _install_attnout_capture() -> None:
    """Request-bound per-layer ATTENTION-MODULE output capture (last-token vector), to separate the
    attention path from the MoE/MHC path. Reuses the request-bound _CAPTURE_ACTIVE signal; dumps to
    INTEL_CPU_DSV4_LAYER_CAPTURE + '.attnout' in the compare_layers format. If attn_out diverges at
    layer 0 -> attention core; if attn_out matches but the layer output (MHC) diverges -> MHC/combine.
    """
    global _ATTNOUT_INSTALLED
    if _ATTNOUT_INSTALLED:
        return
    outfile = os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE")
    if not outfile:
        return
    try:
        import sglang.srt.models.deepseek_v4 as _m
    except Exception as e:  # noqa: BLE001
        logger.warning("[ATTNOUT] import failed: %r", e)
        return
    Attn = getattr(_m, "MQALayer", None)
    if Attn is None or getattr(Attn, "_attnout_hooked", False):
        return
    _ATTNOUT_INSTALLED = True
    Attn._attnout_hooked = True
    _ofwd = Attn.forward

    def _fwd(self, *a, **k):
        r = _ofwd(self, *a, **k)
        if _CAPTURE_ACTIVE and not _ATTNOUT_DONE:
            try:
                o = r[0] if isinstance(r, (tuple, list)) else r
                if hasattr(o, "detach"):
                    v = o.detach().float().reshape(-1, o.shape[-1])[-1].cpu().tolist()
                    lid = int(getattr(self, "layer_id", getattr(self, "layer_idx", -1)))
                    _ATTNOUT[lid] = [round(float(x), 6) for x in v]
            except Exception as e:  # noqa: BLE001
                logger.warning("[ATTNOUT] layer cap failed: %r", e)
        return r

    Attn.forward = _fwd

    # q post norm+rope tap: splits compress-rope/q from the compressed-KV/indexer/sparse-attn path.
    _oqb = getattr(Attn, "_compute_q_b", None)
    if _oqb is not None and not getattr(Attn, "_qfinal_hooked", False):
        Attn._qfinal_hooked = True

        def _qbw(self, q, positions, q_out=None, *a, **k):
            r = _oqb(self, q, positions, q_out, *a, **k)
            if _CAPTURE_ACTIVE and not _ATTNOUT_DONE:
                try:
                    t = r if (r is not None and hasattr(r, "detach")) else q_out
                    if t is not None and hasattr(t, "detach"):
                        v = t.detach().float().reshape(-1, t.shape[-1])[-1].cpu().tolist()
                        lid = int(getattr(self, "layer_id", getattr(self, "layer_idx", -1)))
                        _QFINAL[lid] = [round(float(x), 6) for x in v]
                except Exception as e:  # noqa: BLE001
                    logger.warning("[QFINAL] layer cap failed: %r", e)
            return r

        Attn._compute_q_b = _qbw

    ModelCls = getattr(_m, "DeepseekV4ForCausalLM", None)
    if ModelCls is not None and not getattr(ModelCls, "_attnout_dump_patched", False):
        ModelCls._attnout_dump_patched = True
        _omodf = ModelCls.forward

        def _dump(self, *a, **k):
            global _ATTNOUT_DONE
            logits = _omodf(self, *a, **k)
            if not _ATTNOUT_DONE and _ATTNOUT:
                _ATTNOUT_DONE = True
                if not _is_rank0():
                    return logits  # only rank 0 writes (fixes TP file-write races)
                try:
                    payload = {
                        "leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "unknown"),
                        "n_layers": len(_ATTNOUT),
                        "seq_idx": _CAPTURE_SEQ_IDX,
                        "input_ids": _CAPTURE_INPUT_IDS,
                        "layers": {str(li): v for li, v in sorted(_ATTNOUT.items())},
                        "final_logits_top5": None,
                    }
                    dst = outfile + ".attnout"
                    with open(dst + ".tmp", "w") as f:
                        json.dump(payload, f)
                    os.replace(dst + ".tmp", dst)
                    logger.warning("[ATTNOUT] wrote %d layers -> %s", len(_ATTNOUT), dst)
                    if _QFINAL:
                        qpayload = {
                            "leg": os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE_LEG", "unknown"),
                            "n_layers": len(_QFINAL),
                            "seq_idx": _CAPTURE_SEQ_IDX,
                            "input_ids": _CAPTURE_INPUT_IDS,
                            "layers": {str(li): v for li, v in sorted(_QFINAL.items())},
                            "final_logits_top5": None,
                        }
                        qdst = outfile + ".qfinal"
                        with open(qdst + ".tmp", "w") as f:
                            json.dump(qpayload, f)
                        os.replace(qdst + ".tmp", qdst)
                        logger.warning("[QFINAL] wrote %d layers -> %s", len(_QFINAL), qdst)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[ATTNOUT] dump failed: %r", e)
            return logits

        ModelCls.forward = _dump
    logger.warning("[ATTNOUT] installed (request-bound per-layer attn-output tap).")


_install_layer_capture()
_install_attn_io_capture()
_install_moe_topk_capture()
_install_attnout_capture()

try:
    from .rescue import install_rescue_hooks
    install_rescue_hooks()
except Exception as _e:  # noqa: BLE001
    logger.warning("[RESCUE] install skipped: %r", _e)
