"""Load native GPU model once and replay the frozen CPU shared operands."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "gpu_manifest.json").read_text())
    import sglang as sgl

    engine = sgl.Engine(
        model_path="/scratch/bkaul/models/DeepSeek-V4-Flash",
        device="cuda", tp_size=2, dtype="bfloat16", trust_remote_code=True,
        mem_fraction_static=0.85, disable_cuda_graph=True, disable_overlap_schedule=True,
        disable_radix_cache=True, context_length=4096, max_total_tokens=4096,
        swa_full_tokens_ratio=0.5,
        chunked_prefill_size=16384, watchdog_timeout=100000,
    )
    try:
        engine.generate(input_ids=[manifest["trigger_ids"]],
                        sampling_params={"temperature": 0.0, "max_new_tokens": 1})
        destination = args.root / f"gpu_{os.environ['SLURM_JOB_ID']}"
        (destination / "runtime.json").write_text(json.dumps(engine.get_server_info(), indent=2))
        for rank in range(2):
            result = json.loads((destination / f"rank{rank}/complete.json").read_text())
            if len(result["records"]) != len(manifest["captures"]):
                raise RuntimeError("incomplete native GPU replay")
    finally:
        engine.shutdown()
    print("SHARED_REGRESSION_GPU_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
