"""Read-only taps at the native FP8 GEMM boundary during offline replay."""

from contextlib import contextmanager
import hashlib

import torch


@contextmanager
def capture_gemm(stage, projection, directory, layer, rank, job, weight_cache):
    from sglang.srt.layers.quantization import fp8_utils

    originals = {
        name: getattr(fp8_utils, name) for name in
        ("w8a8_block_fp8_matmul_deepgemm", "w8a8_block_fp8_matmul_triton")
    }
    key = f"{projection}_gemm"

    def wrapper(name, original):
        def call(*args, **kwargs):
            if key in stage or len(args) < 5:
                raise RuntimeError("unexpected native shared GEMM call structure")
            quantized, weight, scale, weight_scale, block = args[:5]
            if quantized.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("native GEMM did not consume FP8 operands")
            if list(block) != [128, 128] or scale.dtype != torch.float32:
                raise RuntimeError("unsupported native activation scale format")
            output = original(*args, **kwargs)
            if (not torch.isfinite(output).all() or not torch.isfinite(scale).all() or
                    not torch.isfinite(quantized.float()).all() or not (scale > 0).all()):
                raise RuntimeError("invalid native GEMM output/activation/scale")
            filename = f"weights.L{layer}.{projection}.pt"
            identity = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
                        weight_scale.data_ptr(), tuple(weight_scale.shape), tuple(weight_scale.stride()))
            if filename not in weight_cache:
                saved = directory / filename
                torch.save({
                    "weight": weight.detach().cpu().clone(),
                    "scale": weight_scale.detach().cpu().clone(), "block": list(block),
                    "layer": layer, "projection": projection, "rank": rank, "job": job,
                }, saved)
                weight_cache[filename] = (identity, hashlib.sha256(saved.read_bytes()).hexdigest())
            if weight_cache[filename][0] != identity:
                raise RuntimeError("native projection weights changed during replay")
            stage[key] = {
                "backend": name, "quantized_input": quantized.detach().cpu().clone(),
                "input_scale": scale.detach().cpu().clone(),
                "output": output.detach().cpu().clone(),
                "weight_file": filename, "weight_sha256": weight_cache[filename][1],
                "block": list(block),
            }
            return output
        return call

    for name, original in originals.items():
        setattr(fp8_utils, name, wrapper(name, original))
    try:
        yield
        if key not in stage:
            raise RuntimeError("native FP8 GEMM boundary was not observed")
    finally:
        for name, original in originals.items():
            setattr(fp8_utils, name, original)
