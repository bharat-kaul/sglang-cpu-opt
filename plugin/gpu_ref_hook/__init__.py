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

    # Also capture the MHC boundary intermediates: hc_pre output y + hc_post output (residual).
    for meth, tag in (("hc_pre", "HCPRE"), ("hc_post", "HCPOST")):
        orig = getattr(Layer, meth, None)
        if orig is None or getattr(Layer, "_" + meth + "_hooked", False):
            continue

        def _mk(orig, tag):
            key = tag.lower()

            def _w(self, *a, **k):
                r = orig(self, *a, **k)
                # Only capture the small real prompt-0 prefill (skip 256-tok warmup/padding passes).
                _sz = None
                if tag == "HCPRE" and len(a) > 0 and torch.is_tensor(a[0]):
                    _sz = a[0].shape[0]
                elif tag == "HCPOST" and len(a) > 1 and torch.is_tensor(a[1]):
                    _sz = a[1].shape[0]
                if _sz is not None and _sz >= 32:
                    return r
                if tag == "HCPRE" and _HID.get("wdump", 0) < 2:
                    # Skip CUDA-graph warmup (dummy/zero or huge activations).
                    xin = a[0] if len(a) > 0 and torch.is_tensor(a[0]) else None
                    _xm = xin.float().abs().mean().item() if xin is not None else 0.0
                    if xin is not None and 0.003 < _xm < 5.0:
                        _HID["wdump"] = _HID.get("wdump", 0) + 1
                        try:
                            if _HID["wdump"] == 1:
                                torch.save(xin.detach().float().cpu(), f + ".x0.pt")
                            with open(f, "a") as fh:
                                fh.write(f"XIN shape={tuple(xin.shape)} {_stat(xin)}\n")
                                for wn in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_scale", "hc_attn_base"):
                                    w = getattr(self, wn, None)
                                    if torch.is_tensor(w):
                                        fh.write(f"WDUMP {wn} shape={tuple(w.shape)} dtype={w.dtype} {_stat(w)}\n")
                        except Exception:  # noqa: BLE001
                            pass
                # Skip CUDA-graph warmup passes (huge dummy activations) for HCPOST.
                if tag == "HCPOST" and len(a) > 1 and torch.is_tensor(a[1]):
                    try:
                        if a[1].float().abs().mean().item() > 5.0:
                            return r
                    except Exception:  # noqa: BLE001
                        pass
                if _HID.get(key, 0) < 4:
                    _HID[key] = _HID.get(key, 0) + 1
                    try:
                        y = r[0] if isinstance(r, (tuple, list)) else r
                        with open(f, "a") as fh:
                            fh.write(f"{tag}#{_HID[key]} out {_stat(y)}\n")
                            if tag == "HCPOST":
                                # hc_post(self, x_branch, residual, post, comb)
                                names = ("x", "resid", "post", "comb")
                                for idx, nm in enumerate(names):
                                    if idx < len(a) and torch.is_tensor(a[idx]):
                                        fh.write(f"{tag}#{_HID[key]} in.{nm} {_stat(a[idx])}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            return _w

        setattr(Layer, meth, _mk(orig, tag))
        setattr(Layer, "_" + meth + "_hooked", True)

    try:
        import sglang.kernels.ops.layernorm.mhc as _mhc

        if not getattr(_mhc, "_sink_dbg_gpu", False):
            _osink = _mhc.hc_split_sinkhorn

            def _sink(mixes, *a, **k):
                r = _osink(mixes, *a, **k)
                if _HID.get("sink", 0) < 3:
                    _HID["sink"] = _HID.get("sink", 0) + 1
                    try:
                        last = mixes.shape[-1]
                        hc = last // 6
                        flat = mixes.reshape(-1, last)
                        post_slice = flat[:, hc : 2 * hc]
                        pre_o, post_o, comb_o = r
                        scale = a[0] if len(a) > 0 else k.get("hc_scale")
                        base = a[1] if len(a) > 1 else k.get("hc_base")
                        with open(f, "a") as fh:
                            fh.write(f"SINK#{_HID['sink']} mixes {_stat(mixes)}\n")
                            fh.write(f"SINK#{_HID['sink']} mix_postslice {_stat(post_slice)}\n")
                            fh.write(f"SINK#{_HID['sink']} post_out {_stat(post_o)}\n")
                            if torch.is_tensor(scale):
                                fh.write(f"SINK#{_HID['sink']} scale {[round(x,4) for x in scale.float().reshape(-1).tolist()]}\n")
                            if torch.is_tensor(base):
                                fh.write(f"SINK#{_HID['sink']} base8 {[round(x,4) for x in base.float().reshape(-1).tolist()[:8]]}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            _mhc.hc_split_sinkhorn = _sink
            _mhc._sink_dbg_gpu = True
            try:
                _dv4._get_mhc_ops.cache_clear()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

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
