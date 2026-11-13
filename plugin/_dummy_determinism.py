"""Deterministic, device- and torch-version-independent dummy weight init.

sglang's ``initialize_dummy_weights`` seeds a torch ``Generator`` on the PARAM's
DEVICE with a fixed seed. But the CPU and CUDA RNG streams differ, and the stream
can shift across torch versions — our CPU engine is torch 2.12+cpu while the GPU
reference container is torch 2.13+cu130. For a CPU-vs-GPU per-layer PARITY diff the
two runs MUST see BIT-IDENTICAL dummy weights, otherwise a weight mismatch is
indistinguishable from a kernel bug.

This patches ``initialize_dummy_weights`` to fill each floating param from numpy
(MT19937 — stable across platforms/versions) with a per-parameter seed derived from
the param NAME, so both engines produce identical weights regardless of device,
torch version, or tp-rank-0 layout (both sides run tp=1, full params). Matches
sglang's convention of filling ``weight_scale_inv`` with 1.0. Gated by
DETERMINISTIC_DUMMY=1; no-op otherwise.
TP>1 is rejected: use make_dummy_checkpoint.py and the normal checkpoint loaders.
"""
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def install_deterministic_dummy() -> None:
    if os.environ.get("DETERMINISTIC_DUMMY") != "1":
        return
    import numpy as np
    import torch
    import sglang.srt.model_loader.weight_utils as _wu
    import sglang.srt.model_loader.loader as _ld
    from sglang.srt.distributed import get_tensor_model_parallel_world_size
    if getattr(_wu.initialize_dummy_weights, "_deterministic", False):
        return

    def _seed_for(name: str) -> int:
        return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)

    def _init(model, low: float = -1e-3, high: float = 1e-3, seed: int = 1234) -> None:
        if get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError(
                "DETERMINISTIC_DUMMY is TP=1 only; generate a shared oracle checkpoint "
                "with make_dummy_checkpoint.py and use load_format=safetensors on both sides"
            )
        for name, param in model.state_dict().items():
            if not torch.is_floating_point(param):
                continue
            if name.endswith("weight_scale_inv"):
                param.fill_(1.0)
                continue
            rng = np.random.RandomState(_seed_for(name))
            vals = rng.uniform(low, high, size=tuple(param.shape)).astype("float32")
            t = torch.from_numpy(vals).to(param.dtype)
            param.data.copy_(t.to(param.device))

    _init._deterministic = True
    _wu.initialize_dummy_weights = _init
    # DummyModelLoader imported the symbol by value (loader.py `from ... import
    # initialize_dummy_weights`), so patch the binding it actually calls too.
    _ld.initialize_dummy_weights = _init
    logger.info("deterministic dummy weights installed (numpy name-seeded, CPU/GPU bit-identical).")
