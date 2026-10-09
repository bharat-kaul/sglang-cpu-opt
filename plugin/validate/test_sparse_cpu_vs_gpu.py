"""Part B of the sparse GPU-oracle: diff our COMPILED CPU sparse_attend paths (scalar / fp32bmm / amx /
bestof) against the REAL TileLang kernel output saved by test_sparse_gpu_oracle.py. Runs on an AMX node.

Decision bar: the real kernel's own bf16 noise floor is ~1.95e-3 max-err (bf16-replica vs GPU). A CPU path
whose max-err vs the GPU output is <= that floor is INDISTINGUISHABLE from the real kernel's precision
(i.e. conformant). This tells us whether the bf16-AMX path is a legitimate (conformant) class."""
import os

import torch
from torch.utils.cpp_extension import load

IO = "/scratch/bkaul/sparse_oracle_io.pt"
BF16_FLOOR = 1.953e-3  # bf16-replica vs GPU worst max-err (job 384502) = the kernel's own precision noise

K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sparse_attend.cpp")
mod = load(name="sparse_cpu_vs_gpu", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)


def metrics(ref, got):
    r, g = ref.flatten().float(), got.flatten().float()
    cos = torch.nn.functional.cosine_similarity(r, g, dim=0).item()
    mae = (r - g).abs().max().item()
    return cos, mae


def main():
    recs = torch.load(IO)
    print(f"bf16 noise floor (kernel's own, vs its replica) = {BF16_FLOOR:.3e}")
    print(f"{'N':>4} {'path':>10} {'cos_vs_gpu':>11} {'mae_vs_gpu':>11} {'<=floor?':>9}")
    for r in recs:
        N, scale = r["N"], r["scale"]
        q, kv, sink, gpu = r["q"].float(), r["kv"].float(), r["sink"].float(), r["gpu_out"].float()
        paths = {
            "scalar": mod.sparse_attend(q, kv, sink, scale),
            "fp32bmm": mod.sparse_attend_fp32bmm(q, kv, sink, scale),
            "amx": mod.sparse_attend_amx(q, kv, sink, scale),
            "bestof": mod.sparse_attend_bestof(q, kv, sink, scale),
        }
        for name, out in paths.items():
            cos, mae = metrics(gpu, out)
            print(f"{N:>4} {name:>10} {cos:>11.6f} {mae:>11.3e} {'YES' if mae <= BF16_FLOOR else 'no':>9}")
    print(">>> done")


if __name__ == "__main__":
    main()
