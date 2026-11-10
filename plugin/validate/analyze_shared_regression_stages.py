"""Localize saved shared-output differences without loading the full model.

The two standalone CPU GEMMs retain packed FP8 weights. Their reconstruction
is a diagnostic, not a substitute for the captured fused CPU kernel: its
disagreement with that kernel is explicitly measured.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from compare_shared_regression_gpu import metrics


def activation(projection, limit):
    gate, up = projection.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.clamp(max=limit)) *
            up.clamp(-limit, limit)).to(projection.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("gpu_job")
    parser.add_argument("--cpu-job", choices=("398194", "398192", "398193", "398195"))
    args = parser.parse_args()
    import sgl_kernel.common_ops  # noqa: F401

    torch.set_num_threads(8)
    gpu_root = args.root / f"gpu_{args.gpu_job}"
    analysis_job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    output = gpu_root / f"stage_analysis.cpu{args.cpu_job or 'all'}.job{analysis_job}.json"
    partial = output.with_suffix(".jsonl")
    partial.touch(exist_ok=False)
    manifest_path = args.root / "gpu_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for rank in range(2):
        complete = json.loads((gpu_root / f"rank{rank}/complete.json").read_text())
        if (complete["job"] != args.gpu_job or complete["rank"] != rank or
                complete["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
            raise RuntimeError("GPU completion identity mismatch")
    records = []
    coordinates = set()
    for item in manifest["captures"]:
        path = Path(item["path"])
        if args.cpu_job and not path.parent.parent.name.endswith(f"_{args.cpu_job}"):
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("CPU capture changed")
        source = torch.load(path, weights_only=False)
        if args.cpu_job and source["job"] != args.cpu_job:
            raise RuntimeError("CPU source job does not match selected path")
        if source["decision"] == 0:
            continue
        coordinate = (source["q"], source["job"], source["decision"], source["layer"])
        if coordinate in coordinates:
            raise RuntimeError("duplicate stage coordinate")
        coordinates.add(coordinate)
        row = source["target_row"]
        name = f"q{source['q']}.cpu{source['job']}.d{source['decision']}.L{source['layer']}.pt"
        ranks = [torch.load(gpu_root / f"rank{r}" / name, weights_only=False) for r in range(2)]
        for r, gpu in enumerate(ranks):
            if (gpu["source_sha256"] != item["sha256"] or gpu["job"] != args.gpu_job or
                    gpu["rank"] != r or gpu["shared_tp1"] or gpu["down_reduce_results"]):
                raise RuntimeError("unsupported GPU stage identity/TP configuration")
            if not torch.equal(gpu["input"], source["hidden"]):
                raise RuntimeError("GPU input mismatch")
        weights = torch.load(path.parent / f"weights.L{source['layer']}.pt", weights_only=False)
        cpu = torch.load(path.with_suffix(".replay.pt"), weights_only=False)
        active = "on" if source["shared_clamp"] == "1" else "off"
        if not torch.equal(cpu[f"{active}_combined"], source["combined"]):
            raise RuntimeError("CPU native replay mismatch")

        def linear(x, index):
            return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                x.contiguous(), weights[f"w{index}"], weights[f"s{index}"],
                weights["block"], None, torch.bfloat16, True,
            )

        cpu_gate = linear(source["hidden"], 1)
        stages = {}
        for variant, limit, cpu_arm in (("on", 10.0, "on"), ("inactive", 1e30, "off")):
            parts = [gpu["variants"][variant] for gpu in ranks]
            if not torch.equal(parts[0]["output"], parts[1]["output"]):
                raise RuntimeError("GPU reduced outputs differ")
            split = [part["gate_output"].chunk(2, dim=-1) for part in parts]
            gpu_gate = torch.cat([split[0][0], split[1][0], split[0][1], split[1][1]], dim=-1)
            gpu_activation = torch.cat([part["down_input"] for part in parts], dim=-1)
            cpu_activation = activation(cpu_gate, limit)
            reconstructed = linear(cpu_activation, 2)
            matched_down = linear(gpu_activation, 2)
            formula = activation(gpu_gate, limit)
            stages[variant] = {
                "gate_projection": metrics(cpu_gate, gpu_gate, row),
                "activation": metrics(cpu_activation, gpu_activation, row),
                "gpu_activation_formula": metrics(formula, gpu_activation, row),
                "gpu_activation_formula_equal": torch.equal(formula, gpu_activation),
                "reconstruction_vs_native_cpu": metrics(reconstructed, cpu[f"{cpu_arm}_direct"], row),
                "reconstruction_equals_native_cpu": torch.equal(reconstructed, cpu[f"{cpu_arm}_direct"]),
                "matched_down_input": metrics(matched_down, parts[0]["output"], row),
                "native_output": metrics(cpu[f"{cpu_arm}_direct"], parts[0]["output"], row),
            }
        records.append({"q": source["q"], "cpu_job": source["job"], "layer": source["layer"],
                        "decision": source["decision"], "target_row": row, "stages": stages})
        with partial.open("a") as stream:
            stream.write(json.dumps({
                "cpu_analysis_job": analysis_job, "gpu_job": args.gpu_job,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "record": records[-1],
            }, allow_nan=False) + "\n")
        print("SHARED_STAGE", coordinate, flush=True)
    expected = {(q, job, d, layer) for q, jobs, d in
                ((1175, ("398194", "398192"), 22), (1239, ("398193", "398195"), 33))
                for job in jobs for layer in range(43)}
    if args.cpu_job:
        expected = {coordinate for coordinate in expected if coordinate[1] == args.cpu_job}
    if coordinates != expected:
        raise RuntimeError("incomplete first-divergence stage coverage")
    result = {
        "gpu_job": args.gpu_job, "cpu_analysis_job": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scope": "Standalone native CPU GEMMs vs saved GPU stages; no task-accuracy intervention",
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("SHARED_STAGE_COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
