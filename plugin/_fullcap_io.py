"""Single-writer capture output for tensor-parallel oracle runs."""

import os


def save_fullcap(tensors, path) -> None:
    import torch

    selected = os.environ.get("FULLCAP_RANK")
    if selected is not None:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = int(selected)
        if rank < 0:
            raise ValueError("FULLCAP_RANK must be nonnegative")
        if get_tensor_model_parallel_rank() != rank:
            return
    torch.save(tensors, path)
