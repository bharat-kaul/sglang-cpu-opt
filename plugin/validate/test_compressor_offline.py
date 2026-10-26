#!/usr/bin/env python3
"""OFFLINE unit test for the C4/C128 compressor port (no model load).

Validates the ACTUAL PRODUCTION producer (_compress_prefill_core from the plugin, the same function
the live compressor calls) against a faithful transcription of the authors' reference
`Compressor.forward` (prefill, start_pos=0). Per reviewer: the test MUST invoke the production path,
not a second transcription. Identical synthetic operands (fused kv_score split + RAW ape + norm
weight/eps + freqs_cis + positions) feed BOTH the production core and the reference, incl. fp8.

Run: python plugin/validate/test_compressor_offline.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plugin"))

torch.manual_seed(0)
HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6


def ref_apply_rotary_emb(x, freqs_cis):
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.view(xc.size(0), xc.size(-1)) if xc.ndim == 2 else freqs_cis
    return torch.view_as_real(xc * fc).flatten(-2).to(y.dtype)


def ref_rmsnorm(x, weight, eps=EPS):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dt)


def ref_overlap_transform(tensor, ratio, d, value=0.0):
    b, s, _, _ = tensor.size()
    new = tensor.new_full((b, s, 2 * ratio, d), value)
    new[:, :, ratio:] = tensor[:, :, :, d:]
    new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new


def reference_compress_prefill(kv, score, ape, norm_w, freqs_cis, ratio, d, rd, overlap, do_fp8, fp8fn):
    bsz, seqlen, _ = kv.size()
    kv = kv.float()
    score = score.float()
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if remainder > 0:
        kv = kv[:, :cutoff]
        score = score[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape
    if overlap:
        kv = ref_overlap_transform(kv, ratio, d, 0.0)
        score = ref_overlap_transform(score, ratio, d, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    kv = ref_rmsnorm(kv.to(torch.bfloat16), norm_w)
    fc = freqs_cis[:cutoff:ratio]
    roped = ref_apply_rotary_emb(kv[..., -rd:], fc)
    kv = torch.cat([kv[..., :-rd], roped], dim=-1).squeeze(0)
    out = kv.to(torch.bfloat16)
    if do_fp8:
        out = fp8fn(out)
    return out


def make_freqs(seqlen, rd, base=160000.0):
    freqs = 1.0 / (base ** (torch.arange(0, rd, 2).float() / rd))
    t = torch.arange(seqlen).float()
    f = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(f), f)


def run():
    from intel_cpu_models._dsv4_cpu_infra import (
        _compress_prefill_core,
        _fp8_roundtrip_nope_bf16,
    )

    d, rd = HEAD_DIM, ROPE_DIM
    norm_w = torch.rand(d) * 0.5 + 0.75
    print(f"{'ratio':>5} {'ovlp':>5} {'len':>4} {'fp8':>5} {'nwin':>4} {'cos':>12} {'maxabs':>10}  verdict")
    fails = 0
    for ratio, overlap in [(4, True), (128, False)]:
        coff = 2 if overlap else 1
        ape = torch.randn(ratio, coff * d) * 0.1
        lens = [8, 9, 12, 16] if ratio == 4 else [128, 256]
        for L in lens:
            for do_fp8 in (False, True):
                kv_proj = torch.randn(L, coff * d) * 0.5
                score_proj = torch.randn(L, coff * d) * 0.5
                fc = make_freqs(L, rd)
                positions = torch.arange(L)
                ref = reference_compress_prefill(
                    kv_proj.reshape(1, L, coff * d), score_proj.reshape(1, L, coff * d),
                    ape, norm_w, fc, ratio, d, rd, overlap, do_fp8, _fp8_roundtrip_nope_bf16)
                got, _ = _compress_prefill_core(
                    kv_proj.float(), score_proj.float(), ape.float(), norm_w, EPS, fc, positions,
                    [L], ratio, d, rd, overlap, do_fp8, torch.bfloat16)
                cos = torch.nn.functional.cosine_similarity(
                    got.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
                maxabs = (got.float() - ref.float()).abs().max().item()
                ok = cos > 0.99999 and maxabs < 1e-2
                fails += 0 if ok else 1
                print(f"{ratio:>5} {str(overlap):>5} {L:>4} {str(do_fp8):>5} {ref.shape[0]:>4} "
                      f"{cos:>12.8f} {maxabs:>10.3e}  {'PASS' if ok else 'FAIL'}")
    print("\n" + ("ALL PASS — production _compress_prefill_core == reference" if fails == 0 else f"{fails} FAILURES"))
    return fails


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
