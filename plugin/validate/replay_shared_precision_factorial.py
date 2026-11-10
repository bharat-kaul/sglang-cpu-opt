"""Stagewise precision/TP controls using actual native GPU FP8 operands."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from compare_shared_regression_gpu import metrics


def same_fp8(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def restored_activation(data):
    q, scale = data["quantized_input"], data["input_scale"]
    if (q.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or
            scale.shape != (q.shape[0], q.shape[1] // 128)):
        raise RuntimeError("unsupported native activation scale layout")
    wide = q.float() * scale.repeat_interleave(128, dim=1)
    if not torch.isfinite(wide).all() or not (scale > 0).all():
        raise RuntimeError("invalid native quantized activation")
    return wide.to(torch.bfloat16), wide


def join_gate(parts):
    split = [part.chunk(2, dim=0) for part in parts]
    return torch.cat((split[0][0], split[1][0], split[0][1], split[1][1]), dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("gpu_job")
    parser.add_argument("--cpu-job", required=True,
                        choices=("398194", "398192", "398193", "398195"))
    args = parser.parse_args()
    import sgl_kernel.common_ops  # noqa: F401
    torch.set_num_threads(8)
    root = args.root / f"gpu_{args.gpu_job}"
    manifest_path = args.root / "gpu_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    analysis_job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    stem = root / f"precision.cpu{args.cpu_job}.job{analysis_job}"
    partial = Path(f"{stem}.jsonl")
    partial.touch(exist_ok=False)
    for rank in range(2):
        complete = json.loads((root / f"rank{rank}/complete.json").read_text())
        if (complete["job"] != args.gpu_job or complete["rank"] != rank or
                complete["manifest_sha256"] != manifest_hash or not complete["quant_capture"]):
            raise RuntimeError("GPU precision capture identity mismatch")
    records = []
    weight_cache = {}
    pack = torch.ops.sgl_kernel.convert_weight_packed

    def linear(x, weight, scale):
        return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            x.contiguous(), weight, scale.contiguous(), [128, 128], None, torch.bfloat16, True)

    def weights(data, rank, projection, layer):
        path = root / f"rank{rank}" / data["weight_file"]
        if str(path) not in weight_cache:
            if hashlib.sha256(path.read_bytes()).hexdigest() != data["weight_sha256"]:
                raise RuntimeError("GPU weight artifact changed")
            w = torch.load(path, weights_only=False)
            if (w["job"] != args.gpu_job or w["rank"] != rank or
                    w["projection"] != projection or w["layer"] != layer):
                raise RuntimeError("GPU weight identity mismatch")
            matrix, scale = w["weight"], w["scale"]
            if matrix.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
                raise RuntimeError("unsupported native GPU weight scale dtype")
            if scale.shape == (matrix.shape[0], matrix.shape[1] // 128):
                compact = scale[::128]
                if not torch.equal(compact.repeat_interleave(128, 0), scale):
                    raise RuntimeError("GPU weight scale is not blockwise constant")
                scale = compact.contiguous()
            if scale.shape != (matrix.shape[0] // 128, matrix.shape[1] // 128):
                raise RuntimeError("unsupported native GPU weight scale shape")
            weight_cache[str(path)] = (matrix, scale, pack(matrix))
        return weight_cache[str(path)]

    for item in manifest["captures"]:
        path = Path(item["path"])
        if not path.parent.parent.name.endswith(f"_{args.cpu_job}") or path.name.startswith("d0."):
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("CPU operand artifact changed")
        source = torch.load(path, weights_only=False)
        if source["job"] != args.cpu_job:
            raise RuntimeError("CPU job mismatch")
        layer, row = source["layer"], source["target_row"]
        filename = f"q{source['q']}.cpu{args.cpu_job}.d{source['decision']}.L{layer}.pt"
        ranks = [torch.load(root / f"rank{r}" / filename, weights_only=False) for r in range(2)]
        for rank, gpu in enumerate(ranks):
            if (gpu["source_sha256"] != item["sha256"] or gpu["job"] != args.gpu_job or
                    gpu["rank"] != rank or not torch.equal(gpu["input"], source["hidden"])):
                raise RuntimeError("GPU operand/run identity mismatch")
            if gpu["shared_tp1"] or gpu["down_reduce_results"]:
                raise RuntimeError("unexpected GPU TP configuration")
            previous = torch.load(args.root / "gpu_398227" / f"rank{rank}" / filename,
                                  weights_only=False)
            if not torch.equal(gpu["variants"]["on"]["output"], previous["variants"]["on"]["output"]):
                raise RuntimeError("instrumented GPU replay differs from previous native output")
        cpu_weights = torch.load(path.parent / f"weights.L{layer}.pt", weights_only=False)
        stage_result = {}
        for projection, index in (("gate", 1), ("down", 2)):
            stages = [g["variants"]["on"] for g in ranks]
            native = [s[f"{projection}_gemm"] for s in stages]
            ws = [weights(n, r, projection, layer) for r, n in enumerate(native)]
            combined_weight = join_gate([w[0] for w in ws]) if projection == "gate" else torch.cat(
                [w[0] for w in ws], dim=1)
            combined_scale = join_gate([w[1] for w in ws]) if projection == "gate" else torch.cat(
                [w[1] for w in ws], dim=1)
            if not same_fp8(pack(combined_weight), cpu_weights[f"w{index}"]) or not torch.equal(
                    combined_scale, cpu_weights[f"s{index}"]):
                raise RuntimeError(f"effective CPU/GPU weights differ: L{layer} {projection}")
            restored = [restored_activation(n) for n in native]
            if projection == "gate":
                if not torch.equal(restored[0][1], restored[1][1]):
                    raise RuntimeError("GPU ranks quantized identical gate input differently")
                x = source["hidden"]
                qx, wide = restored[0]
                reference = join_gate([s["gate_output"].T for s in stages]).T.contiguous()

                def split(x):
                    return join_gate([linear(x, w[2], w[1]).T for w in ws]).T.contiguous()
            else:
                x = torch.cat([s["down_input"] for s in stages], dim=1)
                qx = torch.cat([r[0] for r in restored], dim=1)
                wide = torch.cat([r[1] for r in restored], dim=1)
                reference = stages[0]["output"]
                if not torch.equal(reference, stages[1]["output"]):
                    raise RuntimeError("GPU TP-reduced output differs across ranks")

                def split(x):
                    parts = x.chunk(2, dim=1)
                    return linear(parts[0], ws[0][2], ws[0][1]) + linear(parts[1], ws[1][2], ws[1][1])
            for rank, n in enumerate(native):
                boundary = stages[rank]["gate_output" if projection == "gate" else "down_local_output"]
                if not torch.equal(n["output"], boundary):
                    raise RuntimeError("native GEMM output differs from projection boundary")
            variants = {
                "original": linear(x, cpu_weights[f"w{index}"], cpu_weights[f"s{index}"]),
                "quant_only": linear(qx, cpu_weights[f"w{index}"], cpu_weights[f"s{index}"]),
                "tp_only": split(x),
                "both": split(qx),
            }
            stage_result[projection] = {
                "weights_equal": True,
                "quantized_rescale_bf16_rounding": metrics(qx.float(), wide, row),
                "variants": {name: metrics(value, reference, row) for name, value in variants.items()},
            }
        record = {"q": source["q"], "cpu_job": args.cpu_job, "decision": source["decision"],
                  "layer": layer, "target_row": row, "stages": stage_result}
        records.append(record)
        with partial.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("PRECISION_STAGE", source["q"], layer, json.dumps(stage_result), flush=True)
    if len(records) != 43 or {r["layer"] for r in records} != set(range(43)):
        raise RuntimeError("incomplete precision factorial coverage")
    Path(f"{stem}.json").write_text(json.dumps({
        "gpu_job": args.gpu_job, "cpu_analysis_job": analysis_job,
        "manifest_sha256": manifest_hash,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Stagewise controls; native GPU quantized/rescaled activations stored as BF16 for CPU GEMM",
        "records": records,
    }, indent=2, allow_nan=False))
    print("PRECISION_FACTORIAL_COMPLETE", args.cpu_job, flush=True)


if __name__ == "__main__":
    main()
