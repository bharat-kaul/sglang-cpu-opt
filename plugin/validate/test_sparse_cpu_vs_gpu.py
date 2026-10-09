"""Part B of the sparse GPU-oracle: diff our COMPILED CPU sparse_attend paths (scalar / fp32bmm / amx /
bestof) against the REAL TileLang kernel output saved by test_sparse_gpu_oracle.py. Runs on an AMX node.

P1-F4 corrections: (a) the GPU kernel received BF16-ROUNDED operands; to measure CPU-vs-GPU APPROXIMATION
honestly we feed the CPU kernels the SAME bf16-rounded q/kv (not the original fp32), and report the fp32-
operand result only as a secondary column. (b) These are APPROXIMATION-error observations vs a NON-source-
faithful (global, not 64-block) replica's GPU output — NOT an intrinsic hardware-noise floor. The reference
threshold below is a PROPOSAL pending ratification + an independently justified downstream budget."""
import os

import torch
from torch.utils.cpp_extension import load

IO = "/scratch/bkaul/sparse_oracle_io.pt"
# PROPOSED reference (NOT ratified): bf16-replica-vs-GPU worst max-err (job 384502). It is approximation
# error vs a globally-normalized replica, not characterized hardware noise (repeated GPU runs not used).
PROPOSED_REF = 1.953e-3

K = os.path.join(os.path.dirname(__file__), "..", "kernels", "dsa_pilot", "sparse_attend.cpp")
mod = load(name="sparse_cpu_vs_gpu", sources=[K], extra_cflags=["-O3", "-fopenmp", "-march=native"], verbose=False)


def metrics(ref, got):
    r, g = ref.flatten().float(), got.flatten().float()
    cos = torch.nn.functional.cosine_similarity(r, g, dim=0).item()
    mae = (r - g).abs().max().item()
    return cos, mae


def main():
    # R3-F5: restricted deserialization (no arbitrary pickle execution) + ENFORCE provenance, not just print.
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass
    blob = torch.load(IO, weights_only=True)
    if not isinstance(blob, dict) or "records" not in blob or "provenance" not in blob:
        raise SystemExit("FATAL: io archive missing provenance/records (expected the stamped dict format)")
    prov = blob["provenance"]
    if not prov.get("kernel_py_sha256_validated"):
        raise SystemExit(f"FATAL: saved io provenance has UNVALIDATED kernel sha ({prov.get('kernel_py_sha256')!r})")
    print(f"[provenance] {prov}")
    recs = blob["records"]
    print(f"PROPOSED reference (approx-err vs the SOURCE-FAITHFUL 64-block replica's GPU out; NOT a ratified "
          f"hardware-noise floor) = {PROPOSED_REF:.3e}")
    print(f"{'N':>4} {'path':>10} {'operands':>10} {'cos_vs_gpu':>11} {'mae_vs_gpu':>11}")
    for r in recs:
        N, scale = r["N"], r["scale"]
        sink, gpu = r["sink"].float(), r["gpu_out"].float()
        # Two operand conventions: MATCHED = bf16-rounded (what the GPU kernel actually received) is the
        # honest approximation measurement; FP32 = original operands (secondary, diverges more at large N).
        for conv in ("bf16-matched", "fp32"):
            q = r["q"].bfloat16().float() if conv == "bf16-matched" else r["q"].float()
            kv = r["kv"].bfloat16().float() if conv == "bf16-matched" else r["kv"].float()
            paths = {
                "scalar": mod.sparse_attend(q, kv, sink, scale),
                "fp32bmm": mod.sparse_attend_fp32bmm(q, kv, sink, scale),
                "amx": mod.sparse_attend_amx(q, kv, sink, scale),
                "bestof": mod.sparse_attend_bestof(q, kv, sink, scale),
            }
            for name, out in paths.items():
                ob = out.bfloat16().float() if conv == "bf16-matched" else out   # match the bf16 OUTPUT boundary
                cos, mae = metrics(gpu, ob)
                print(f"{N:>4} {name:>10} {conv:>10} {cos:>11.6f} {mae:>11.3e}")
    print(">>> done")


if __name__ == "__main__":
    main()
