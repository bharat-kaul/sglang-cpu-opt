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
        global _CAPTURE_ACTIVE, _CAPTURE_DONE
        outfile = os.environ.get("INTEL_CPU_DSV4_LAYER_CAPTURE")
        activate = bool(outfile) and not _CAPTURE_DONE
        if activate:
            _CAPTURE_ACTIVE = True
            _LAYER_STATES.clear()
        logits = _orig_model_fwd(self, *a, **k)
        if activate:
            _CAPTURE_ACTIVE = False
            _CAPTURE_DONE = True
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

    # wo_a input = attention-core output (pre o_proj); wkv output = compressed KV. Patch module
    # forwards per-instance on first layer-0 call (capture their I/O without re-invoking).
    _submod_patched = {"done": False}

    def _patch_submods(self):
        if _submod_patched["done"]:
            return
        _submod_patched["done"] = True
        woa = getattr(self, "wo_a", None)
        if woa is not None:
            _ow = woa.forward
            def _woaw(x, *a, **k):
                if "woa_in" not in _stash:
                    _save("woa_in", x)
                return _ow(x, *a, **k)
            woa.forward = _woaw
        wkv = getattr(self, "wkv", None)
        if wkv is not None:
            _okvf = wkv.forward
            def _wkvw(x, *a, **k):
                r = _okvf(x, *a, **k)
                if "kv" not in _stash:
                    _save("kv", r[0] if isinstance(r, tuple) else r)
                return r
            wkv.forward = _wkvw

    _oqb = getattr(Attn, "_compute_q_b", None)
    if _oqb is not None:
        def _qbw(self, q, positions, q_out=None, *a, **k):
            r = _oqb(self, q, positions, q_out, *a, **k)
            if _is_l0(self) and "q_final" not in _stash:
                _save("q_final", r)
                _save("q_lora", q)
            return r
        Attn._compute_q_b = _qbw

    _okv = getattr(Attn, "_compute_kv_to_cache", None)
    _owkv_attr = None
    _ofwd = Attn.forward

    def _fwd(self, *a, **k):
        global _ATTN_IO_DONE
        first_l0 = _is_l0(self) and not _ATTN_IO_DONE
        if first_l0:
            _patch_submods(self)
            # attn input hidden = positional arg (x) — find the first 2D+ float tensor
            xin = k.get("x")
            if xin is None:
                xin = k.get("hidden_states")
            if xin is None:
                for z in a:
                    if torch.is_tensor(z) and z.is_floating_point() and z.dim() >= 2:
                        xin = z
                        break
            _save("attn_in", xin)
        r = _ofwd(self, *a, **k)
        if first_l0:
            _ATTN_IO_DONE = True
            o = r[0] if isinstance(r, (tuple, list)) else r
            _save("attn_out", o)
            try:
                _stash["_meta"] = {"layer_id": 0, "leg": os.environ.get("INTEL_CPU_DSV4_ATTNIO_LEG", "unknown")}
                torch.save(_stash, out)
                logger.warning("[ATTNIO] saved %s -> %s", list(_stash.keys()), out)
            except Exception as e:  # noqa: BLE001
                logger.warning("[ATTNIO] save failed: %r", e)
        return r

    Attn.forward = _fwd
    logger.warning("[ATTNIO] installed (layer-0 attention sub-op I/O capture).")


_install_layer_capture()
_install_attn_io_capture()
