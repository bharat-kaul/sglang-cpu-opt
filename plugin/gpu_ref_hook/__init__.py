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

    def _rank():
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001
            return -1

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
        inp = _find_hid(a, k)
        # Only capture the small real prompt-0 prefill (skip 256-tok warmup/padding passes).
        if inp is not None and inp.shape[0] >= 32:
            return _orig(self, *a, **k)
        cap = _HID["n"] < 48
        if cap:
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

    # Capture self_attn (MQALayer) input x + output to localize attention-internal divergence.
    Attn = getattr(_dv4, "MQALayer", None)
    if Attn is not None and not getattr(Attn, "_gpu_ref_attn_hooked", False):
        _oatt = Attn.forward

        def _attw(self, *a, **k):
            xin = k.get("x")
            if xin is None and len(a) > 0 and torch.is_tensor(a[0]):
                xin = a[0]
            # One-time tap of wo_a input = the pre-o_proj MLA attention-core output.
            _woa = getattr(self, "wo_a", None)
            if _woa is not None and not getattr(_woa, "_gpu_ref_woa_tapped", False):
                _owoa = _woa.forward

                def _woaw(x, *aa, **kk):
                    if torch.is_tensor(x) and x.shape[0] < 32 and _HID.get("woa", 0) < 6:
                        _HID["woa"] = _HID.get("woa", 0) + 1
                        try:
                            with open(f, "a") as fh:
                                fh.write(f"WOA#{_HID['woa']} rank{_rank()} preO {_stat(x)}\n")
                        except Exception:  # noqa: BLE001
                            pass
                    return _owoa(x, *aa, **kk)

                _woa.forward = _woaw
                _woa._gpu_ref_woa_tapped = True
            out = _oatt(self, *a, **k)
            if xin is not None and torch.is_tensor(xin) and xin.shape[0] < 32 and _HID.get("attn", 0) < 4:
                _HID["attn"] = _HID.get("attn", 0) + 1
                try:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    with open(f, "a") as fh:
                        fh.write(f"ATTN#{_HID['attn']} in {_stat(xin)}\n")
                        if torch.is_tensor(o):
                            fh.write(f"ATTN#{_HID['attn']} out {_stat(o)}\n")
                except Exception:  # noqa: BLE001
                    pass
            return out

        Attn.forward = _attw
        Attn._gpu_ref_attn_hooked = True

        # Split attention internals: final q (post norm+rope) and the pre-norm compressed kv (wkv).
        if hasattr(Attn, "_compute_q_b") and hasattr(Attn, "_compute_kv_to_cache"):
            _oqb = Attn._compute_q_b
            _okv = Attn._compute_kv_to_cache

            def _qbw(self, q, positions, q_out=None):
                r = _oqb(self, q, positions, q_out)
                if torch.is_tensor(r) and r.shape[0] < 32 and _HID.get("qb", 0) < 6:
                    _HID["qb"] = _HID.get("qb", 0) + 1
                    try:
                        with open(f, "a") as fh:
                            fh.write(f"QB#{_HID['qb']} rank{_rank()} qlora {_stat(q)}\n")
                            fh.write(f"QB#{_HID['qb']} rank{_rank()} q_final {_stat(r)}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return r

            def _kvw(self, x, positions, forward_batch, attn_backend, qkv_a=None):
                if torch.is_tensor(x) and x.shape[0] < 32 and _HID.get("kv", 0) < 3:
                    _HID["kv"] = _HID.get("kv", 0) + 1
                    try:
                        kv = qkv_a[..., self.q_lora_rank :] if qkv_a is not None else self.wkv(x)[0]
                        with open(f, "a") as fh:
                            fh.write(f"KV#{_HID['kv']} wkv_out {_stat(kv)}\n")
                    except Exception:  # noqa: BLE001
                        pass
                return _okv(self, x, positions, forward_batch, attn_backend, qkv_a)

            Attn._compute_q_b = _qbw
            Attn._compute_kv_to_cache = _kvw

    # Tap fused_q_norm_rope: input (wq_b out) + output, nope[0:4] AND rope[-4:] dims, rank-tagged.
    _ofqr = getattr(_dv4, "fused_q_norm_rope", None)
    if _ofqr is not None and not getattr(_dv4, "_fqr_gpu_tapped", False):

        def _fqr(qi, qo, *aa, **kk):
            r = _ofqr(qi, qo, *aa, **kk)
            try:
                if torch.is_tensor(qi) and qi.shape[0] < 32 and _HID.get("fqr", 0) < 6:
                    _HID["fqr"] = _HID.get("fqr", 0) + 1
                    qif = qi.reshape(qi.shape[0], -1)
                    qof = qo.reshape(qo.shape[0], -1)
                    with open(f, "a") as fh:
                        fh.write(
                            f"FQR#{_HID['fqr']} rank{_rank()} "
                            f"in_nope={[round(v,4) for v in qif[0,:4].float().tolist()]} "
                            f"in_rope={[round(v,4) for v in qif[0,448:452].float().tolist()]} "
                            f"out_nope={[round(v,4) for v in qof[0,:4].float().tolist()]} "
                            f"out_rope={[round(v,4) for v in qof[0,448:452].float().tolist()]}\n"
                        )
            except Exception:  # noqa: BLE001
                pass
            return r

        _dv4.fused_q_norm_rope = _fqr
        _dv4._fqr_gpu_tapped = True

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
