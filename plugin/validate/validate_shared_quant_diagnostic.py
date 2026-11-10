"""Validate the disabled native path and measure an enabled A8 diagnostic."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch

from compare_shared_regression_gpu import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("gpu_job")
    parser.add_argument("--cpu-job", required=True,
                        choices=("398194", "398192", "398193", "398195"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    loader = Path(__file__).parents[1] / "intel_cpu_models/_dsv4_shared_clamp.py"
    so = runpy.run_path(str(loader))["load"](precision_diagnostic=True)
    manifest_path = args.root / "gpu_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    directory = args.root / f"gpu_{args.gpu_job}"
    output = directory / f"quant_diagnostic.cpu{args.cpu_job}.job{os.environ['SLURM_JOB_ID']}.json"
    partial = output.with_suffix(".jsonl")
    partial.touch(exist_ok=False)
    records = []
    for item in manifest["captures"]:
        path = Path(item["path"])
        if not path.parent.parent.name.endswith(f"_{args.cpu_job}"):
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("CPU capture changed")
        source = torch.load(path, weights_only=False)
        if source["job"] != args.cpu_job:
            raise RuntimeError("CPU job mismatch")
        weights = torch.load(path.parent / f"weights.L{source['layer']}.pt", weights_only=False)
        baseline = torch.load(path.with_suffix(".replay.pt"), weights_only=False)
        native_arm = "on" if source["shared_clamp"] == "1" else "off"
        if not torch.equal(source["combined"], baseline[f"{native_arm}_combined"]):
            raise RuntimeError("historical native replay mismatch")

        def shared(combined):
            return so.shared_experts(
                source["hidden"].clone(), weights["w1"], weights["w2"],
                source["routed"] if combined else None, source["factor"] if combined else None,
                False, False, True, weights["s1"], weights["s2"], weights["block"], True)

        so.set_shared_quantize(False)
        for arm, limit in (("off", -1.0), ("on", 10.0)):
            so.set_shared_clamp(limit)
            for boundary in ("direct", "combined"):
                if not torch.equal(shared(boundary == "combined"), baseline[f"{arm}_{boundary}"]):
                    raise RuntimeError(f"disabled diagnostic changed native {arm}/{boundary}: {path}")
        so.set_shared_quantize(True)
        candidate = shared(False)
        name = f"q{source['q']}.cpu{args.cpu_job}.d{source['decision']}.L{source['layer']}.pt"
        gpu = torch.load(directory / "rank0" / name, weights_only=False)
        if gpu["source_sha256"] != item["sha256"] or gpu["job"] != args.gpu_job:
            raise RuntimeError("GPU source mismatch")
        ref = gpu["variants"]["on"]["output"]
        record = {
            "q": source["q"], "cpu_job": args.cpu_job, "layer": source["layer"],
            "decision": source["decision"], "disabled_equal": True,
            "baseline": metrics(baseline["on_direct"], ref, source["target_row"]),
            "candidate": metrics(candidate, ref, source["target_row"]),
        }
        if source["decision"] > 0:
            probes = []
            for rank in range(2):
                data = torch.load(directory / f"rank{rank}" / name, weights_only=False)
                for projection in ("gate", "down"):
                    inp = data["input"] if projection == "gate" else data["variants"]["on"]["down_input"]
                    restored, quantized, scale = so.quantize_activation(inp.contiguous())
                    native = data["variants"]["on"][f"{projection}_gemm"]
                    if not torch.equal(scale, native["input_scale"]):
                        raise RuntimeError("CPU diagnostic scale differs from native GPU scale")
                    native_restored = (native["quantized_input"].float() *
                                       scale.repeat_interleave(128, 1)).to(torch.bfloat16)
                    probes.append({
                        "rank": rank, "projection": projection, "elements": quantized.numel(),
                        "fp8_code_differences": int((quantized.view(torch.uint8) !=
                                                   native["quantized_input"].view(torch.uint8)).sum()),
                        "restored": metrics(restored, native_restored, source["target_row"]),
                    })
            record["quantizer_probes"] = probes
        records.append(record)
        with partial.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("QUANT_DIAGNOSTIC", source["q"], source["decision"], source["layer"], flush=True)
    decisions = {0, 22 if args.cpu_job in ("398194", "398192") else 33}
    if len(records) != 86 or {(r["decision"], r["layer"]) for r in records} != {
            (d, layer) for d in decisions for layer in range(43)}:
        raise RuntimeError("incomplete diagnostic validation coverage")
    output.write_text(json.dumps({
        "job": os.environ["SLURM_JOB_ID"], "gpu_job": args.gpu_job,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scope": "CPU A8 roundtrip diagnostic, not an exact GPU kernel; native disabled path checked",
        "records": records,
    }, indent=2, allow_nan=False))
    print("QUANT_DIAGNOSTIC_COMPLETE", args.cpu_job, flush=True)


if __name__ == "__main__":
    main()
