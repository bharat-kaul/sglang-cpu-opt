"""Neutralize pin_memory on CPU hosts.

Newer sglang calls ``torch.*(pin_memory=True)`` and ``Tensor.pin_memory()`` in several
unconditional paths (DSA attention backend, KV/kpool plan, scheduler), which raises
``pin_memory=True requires a CUDA or other accelerator backend`` on a CPU-only node.
pinned memory is a host<->device transfer optimization with no benefit (and no allocator)
without CUDA, so we strip it. Gated on CUDA being unavailable; installed before any
sglang allocation so the subprocess workers inherit it via the external model package.
"""
import logging

logger = logging.getLogger(__name__)


def install_cpu_pin_memory_shim() -> None:
    import torch

    if torch.cuda.is_available() or getattr(torch, "_cpu_pin_shim", False):
        return
    torch.Tensor.pin_memory = lambda self, *a, **k: self  # no-op on CPU

    def _strip(orig):
        def _f(*a, **k):
            k.pop("pin_memory", None)
            return orig(*a, **k)
        return _f

    for _name in ("tensor", "empty", "zeros", "ones", "full", "empty_strided", "as_tensor"):
        _orig = getattr(torch, _name, None)
        if _orig is not None:
            setattr(torch, _name, _strip(_orig))
    torch._cpu_pin_shim = True
    logger.info("intel_cpu_models: pin_memory neutralized on CPU (no CUDA backend).")
