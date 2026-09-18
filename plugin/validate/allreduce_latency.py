"""All-reduce latency microbench (CPU, gloo) to characterize the TP-comm ceiling.

Our decode TP all-reduce is a single [1, 4096] bf16 tensor = 8 KB, ~2/layer. This measures
what a standard collective costs for that size across N ranks on one node, to compare against
SGLang's custom sgl_kernel shm all-reduce (measured ~6 s/layer in-model). Run:
  torchrun --standalone --nproc_per_node=4 allreduce_latency.py
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"gloo all_reduce, {world} ranks, threads/rank={torch.get_num_threads()}")
    for nbytes, label in [(8 * 1024, "  8KB"), (64 * 1024, " 64KB"), (1024 * 1024, "  1MB")]:
        n = nbytes // 4  # float32
        x = torch.ones(n, dtype=torch.float32)
        for _ in range(50):
            dist.all_reduce(x)
        dist.barrier()
        iters = 3000
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        dist.barrier()
        dt = (time.perf_counter() - t0) / iters
        if rank == 0:
            print(f"  {label}: {dt * 1e6:9.1f} us/op   ({nbytes / dt / 1e9:6.2f} GB/s effective)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
