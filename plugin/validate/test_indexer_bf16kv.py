#!/usr/bin/env python3
"""Regression gate for the LANDED bf16-KV indexer lever (the one conformance-safe DSA win).

The shipped `indexer_logits` rounds the KV operand to bf16 internally to feed the AMX brgemm, so supplying a
**bf16 KV cache** (what a bf16-compute engine already holds) is NUMERICALLY IDENTICAL to supplying fp32 and
letting the kernel convert — while HALVING the dominant length-S KV read (and removing the per-tile convert).
This test pins that identity BIT-FOR-BIT (maxabs == 0 across the M-sweep), so the win is conformance-safe with
NO Phase-3 accuracy gate: the only remaining work is the Phase-2 serving-path cache dtype. Verified EMR
2026-10-14 (maxabs 0.0 at M=1/8/32/64); keep it green.

Usage: python test_indexer_bf16kv.py   (or pytest)  — EMR/AMX node.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KD = os.path.join(HERE, "..", "kernels", "dsa_pilot")
CF = ["-O3", "-fopenmp", "-march=native", "-mprefer-vector-width=512", "-DCPU_CAPABILITY_AVX512"]


def _load():
    from torch.utils.cpp_extension import load
    return load(name="idx_bf16kv_gate", sources=[os.path.join(KD, "indexer_logits.cpp")],
                extra_cflags=CF, verbose=False)


def test_bf16kv_bit_identical():
    m = _load()
    torch.manual_seed(0)
    for M in (1, 8, 32, 64):
        q = torch.randn(M, 64, 128)
        kv = torch.randn(M, 1024, 128)
        w = torch.randn(M, 64)
        fp32_path = m.indexer_logits(q, kv, w)                       # fp32 KV -> per-tile convert
        bf16_path = m.indexer_logits(q, kv.to(torch.bfloat16), w)    # bf16 KV cache -> kv_bf16 branch, no convert
        assert torch.equal(fp32_path, bf16_path), (
            f"bf16-KV not bit-identical at M={M}: maxabs={(fp32_path - bf16_path).abs().max().item():.3e}")


if __name__ == "__main__":
    test_bf16kv_bit_identical()
    print("PASS: bf16-KV is bit-identical to the fp32 indexer path (M=1/8/32/64) — conformance-safe, Phase-2 ready")
    sys.exit(0)
