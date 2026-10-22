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


_install_layer_capture()
