"""GPU-safe per-layer capture hook for the reference oracle (loaded via SGLANG_EXTERNAL_MODEL_PACKAGE).

Installs ONLY the hidden-state/logits hook on DeepseekV4DecoderLayer + LogitsProcessor — no CPU
kernel patches — so it runs unchanged inside the native-GPU sglang container. Writes per-layer
input/output stats and the final logits top-5 to HID_DBG_FILE, matching the CPU plugin's format
so the two traces diff apples-to-apples to localize the first divergent layer.
"""
import os

_HID = {"n": 0}


def _install():
    try:
        import torch
        import sglang.srt.models.deepseek_v4 as _dv4
    except Exception:  # noqa: BLE001
        return
    Layer = getattr(_dv4, "DeepseekV4DecoderLayer", None)
    if Layer is None or getattr(Layer, "_gpu_ref_hooked", False):
        return
    f = os.environ.get("HID_DBG_FILE", "/scratch/bkaul/dsv4_gpu_hid.txt")
    _orig = Layer.forward

    def _stat(t):
        try:
            tf = t.float()
            fp = tf.reshape(-1)[:4].tolist()
            return f"mean={tf.abs().mean().item():.3e} max={tf.abs().max().item():.3e} nan={int(torch.isnan(tf).any())} fp={[round(x,4) for x in fp]}"
        except Exception:  # noqa: BLE001
            return "NA"

    def _find_hid(a, k):
        for x in list(a) + list(k.values()):
            if torch.is_tensor(x) and x.is_floating_point() and x.dim() >= 2 and x.shape[-1] in (4096, 7168):
                return x
        return None

    def _fwd(self, *a, **k):
        lid = getattr(self, "layer_id", getattr(self, "layer_idx", -1))
        cap = _HID["n"] < 48
        if cap:
            inp = _find_hid(a, k)
            try:
                with open(f, "a") as fh:
                    fh.write(f"L{lid} IN  {_stat(inp) if inp is not None else 'NA'}\n")
            except Exception:  # noqa: BLE001
                pass
        out = _orig(self, *a, **k)
        if cap:
            try:
                with open(f, "a") as fh:
                    if isinstance(out, (tuple, list)):
                        for j, o in enumerate(out[:2]):
                            if torch.is_tensor(o):
                                fh.write(f"L{lid} OUT{j} {_stat(o)}\n")
                    else:
                        fh.write(f"L{lid} OUT0 {_stat(out)}\n")
            except Exception:  # noqa: BLE001
                pass
            _HID["n"] += 1
        return out

    Layer.forward = _fwd
    Layer._gpu_ref_hooked = True

    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor

        if not getattr(LogitsProcessor, "_gpu_ref_hooked", False):
            _olp = LogitsProcessor.forward

            def _lp(self, *a, **k):
                r = _olp(self, *a, **k)
                try:
                    lg = getattr(r, "next_token_logits", None)
                    if lg is not None and _HID.get("lp", 0) < 16:
                        _HID["lp"] = _HID.get("lp", 0) + 1
                        v, i = torch.topk(lg[-1].float(), 5)
                        with open(f, "a") as fh:
                            fh.write(f"LOGITS top5 ids={i.tolist()} vals={[round(x, 3) for x in v.tolist()]} nan={int(torch.isnan(lg).any())}\n")
                except Exception:  # noqa: BLE001
                    pass
                return r

            LogitsProcessor.forward = _lp
            LogitsProcessor._gpu_ref_hooked = True
    except Exception:  # noqa: BLE001
        pass


if os.environ.get("GPU_REF_HID_DEBUG") == "1":
    _install()
