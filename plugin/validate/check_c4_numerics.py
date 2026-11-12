#!/usr/bin/env python3
"""Matched-capture CPU/GPU sparse diagnostics, not a task-accuracy acceptance gate."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

import bench_long_prefill_sparse as bench
from dsv4_sparse_trial import shared_chunked
from replay_serving_sparse import same_bytes, validate_sample
from screen_serving_opportunities import write_json


MODEL = Path("/scratch/bkaul/models/DeepSeek-V4-Flash")
GPU_IMAGE = "7fafb457002ab4d5a2aa6796d7267f086401065d022347b2f49ae366b6c2687a"
LAYERS = (0, 2, 3)


def load_capture(layer):
    path = bench.CAPTURE_ROOT / f"1971307-5-L{layer}.pt"
    if bench.sha(path) != bench.CAPTURE_HASHES[layer]:
        raise RuntimeError("capture hash mismatch")
    sample = torch.load(path, weights_only=True, map_location="cpu")
    expected = {
        "job": "401206", "pid": 1971307, "sequence": 5, "case": "b1-s3072-meaningful",
        "phase": "prefill", "tokens": 512, "seq_lens": [3072],
        "threads_before": 32, "affinity": [0],
    }
    validate_sample(sample, expected, layer)
    if any(not same_bytes(sample[k], sample[k].bfloat16().float()) for k in ("q", "kv")):
        raise RuntimeError("GPU BF16 inputs would change captured operands")
    return sample


def gathered_fp64(q, kv, mask, sink, scale):
    """Finite-input mathematical reference; not the GPU blockwise precision policy."""
    outputs = []
    for row in range(q.shape[0]):
        keys = kv[mask[row]].double()
        if not len(keys):
            outputs.append(torch.zeros_like(q[row], dtype=torch.float64))
            continue
        scores = q[row].double() @ keys.t() * scale
        maximum = torch.maximum(scores.max(-1).values, sink.double())
        weights = (scores - maximum[:, None]).exp()
        denominator = weights.sum(-1) + (sink.double() - maximum).exp()
        outputs.append((weights @ keys) / denominator[:, None])
    return torch.stack(outputs)


def differences(candidate, reference):
    if (candidate.shape != reference.shape or candidate.numel() == 0
            or not candidate.is_floating_point() or not reference.is_floating_point()
            or candidate.device.type != "cpu" or reference.device.type != "cpu"
            or not torch.isfinite(candidate).all() or not torch.isfinite(reference).all()):
        raise RuntimeError("invalid numerical evidence")
    # Explicit widening allows comparison at native and common BF16 output boundaries.
    metrics = bench.metrics(candidate.double(), reference.double())
    metrics["byte_equal"] = same_bytes(candidate, reference)
    metrics["candidate_dtype"], metrics["reference_dtype"] = str(candidate.dtype), str(reference.dtype)
    return metrics


def mask_indices(mask):
    lengths = mask.sum(1).to(torch.int32)
    if int(lengths.min()) < 1:
        raise RuntimeError("GPU replay domain requires at least one visible key per query")
    width = ((int(lengths.max()) + 63) // 64) * 64
    indices = torch.full((mask.shape[0], width), -1, dtype=torch.int32)
    for row in range(mask.shape[0]):
        indices[row, :int(lengths[row])] = mask[row].nonzero().flatten().to(torch.int32)
    rebuilt = torch.zeros_like(mask)
    for row in range(mask.shape[0]):
        rebuilt[row, indices[row, :int(lengths[row])].long()] = True
    if not torch.equal(rebuilt, mask):
        raise RuntimeError("mask to sparse-index conversion changed selection")
    return indices, lengths


def cpu_outputs(sample, native):
    args = tuple(sample[k] for k in ("q", "kv", "mask", "sink", "scale"))
    baseline = native.sparse_attend_masked(*args)
    if not same_bytes(baseline, sample["output"]):
        raise RuntimeError("native replay no longer reproduces captured output")
    candidate = shared_chunked(*args)
    if not same_bytes(candidate, bench.shared_chunked(*args)):
        raise RuntimeError("trial and measured candidate differ")
    reference = gathered_fp64(*args)
    return {"online": baseline, "candidate": candidate, "fp64": reference}


def gpu_outputs(sample):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from inference.kernel import sparse_attn

    q, kv = (sample[k].bfloat16().cuda() for k in ("q", "kv"))
    sink = sample["sink"].cuda()
    indices, lengths = mask_indices(sample["mask"])
    indices, lengths = indices.cuda(), lengths.cuda()
    outputs = {}
    for heads in (16, 64):
        for name in ("flashmla", "tilelang"):
            first = None
            for rep in range(2):
                pieces = []
                for start in range(0, q.shape[1], heads):
                    query, bias = q[:, start:start + heads].contiguous(), sink[start:start + heads].contiguous()
                    if name == "flashmla":
                        result, _, _ = flash_mla_sparse_fwd(
                            q=query, kv=kv.unsqueeze(1), indices=indices.unsqueeze(1),
                            sm_scale=sample["scale"], d_v=q.shape[-1],
                            attn_sink=bias, topk_length=lengths)
                    else:
                        result = sparse_attn(query.unsqueeze(0), kv.unsqueeze(0),
                                             bias, indices.unsqueeze(0), sample["scale"]).squeeze(0)
                    if result.shape != query.shape or result.dtype != torch.bfloat16 or not torch.isfinite(result).all():
                        raise RuntimeError("invalid real GPU kernel output")
                    pieces.append(result)
                result = torch.cat(pieces, 1).cpu()
                if rep == 0:
                    first = result
                elif not same_bytes(first, result):
                    raise RuntimeError("GPU kernel output not repeatable")
            outputs[f"{name}_heads{heads}"] = first
    return outputs


def run(device, output):
    if output.exists():
        raise RuntimeError("refusing to overwrite numerical evidence")
    samples = [load_capture(layer) for layer in LAYERS]
    sources = [Path(__file__), Path(bench.__file__), Path(__file__).parents[1] / "dsv4_sparse_trial.py",
               MODEL / "inference/kernel.py"]
    native = None
    if device == "cpu":
        if torch.get_num_threads() != 32 or sorted(os.sched_getaffinity(0)) != list(range(32)):
            raise RuntimeError("CPU replay requires fixed32 CPU0-31")
        if bench.sha(bench.NATIVE) != bench.NATIVE_HASH:
            raise RuntimeError("native binary hash mismatch")
        expected_cpp = bench.FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp"
        if str(expected_cpp) not in bench.NATIVE.with_name("build.ninja").read_text():
            raise RuntimeError("native build source mismatch")
        spec = importlib.util.spec_from_file_location("dsv4ck_sa", bench.NATIVE)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        sources += [expected_cpp, bench.NATIVE, bench.NATIVE.with_name("build.ninja"),
                    Path(torch.__file__).parent / "lib/libtorch_cpu.so"]
    else:
        if os.environ["GPU_IMAGE_ID"] != GPU_IMAGE or not torch.cuda.is_available():
            raise RuntimeError("wrong GPU oracle image or missing CUDA device")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("expected exactly one SM90 GPU")
        sys.path.insert(0, str(MODEL))
        import sgl_kernel.flash_mla as flash
        root = Path(importlib.util.find_spec("sglang").origin).parent
        sources += [Path(flash.__file__), root / "srt/layers/attention/deepseek_v4_backend.py",
                    root / "srt/layers/attention/dsv4/sparse_prefill_utils.py",
                    root / "srt/layers/attention/dsv4/metadata.py", root / "srt/environ.py"]
    hashes = {str(p): bench.sha(p) for p in sources}
    report = {
        "job": os.environ["SLURM_JOB_ID"], "device": device, "node": os.uname().nodename,
        "pid": os.getpid(), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)), "sources": hashes,
        "capture_sha256": {str(k): bench.CAPTURE_HASHES[k] for k in LAYERS},
        "gpu_image": GPU_IMAGE if device == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "scope": "Exact job401206 final-prefill operands; no weights loaded. FP64 is a "
                 "mathematical diagnostic, not GPU precision emulation. GPU heads16 models "
                 "TP4-local head grouping without distributed execution; heads64 is a control. "
                 "CPU outputs are FP32, GPU outputs BF16. No task accuracy or promotion.",
        "cases": [],
    }
    output.mkdir(parents=True)
    for sample in samples:
        values = cpu_outputs(sample, native) if device == "cpu" else gpu_outputs(sample)
        row = {"layer": sample["layer"], "q_shape": list(sample["q"].shape),
               "kv_shape": list(sample["kv"].shape), "capture_identity": sample["identity"]}
        for name, value in values.items():
            differences(value, sample["output"])
        if device == "cpu":
            row["vs_fp64"] = {name: differences(values[name], values["fp64"])
                              for name in ("online", "candidate")}
            row["candidate_vs_online"] = differences(values["candidate"], values["online"])
        else:
            row["online_vs_gpu"] = {name: differences(sample["output"], value)
                                    for name, value in values.items()}
        path = output / f"L{sample['layer']}.pt"
        torch.save({"job": report["job"], "device": device, "layer": sample["layer"],
                    "capture_sha256": bench.CAPTURE_HASHES[sample["layer"]], "outputs": values}, path)
        row.update(output_file=str(path), output_sha256=bench.sha(path))
        report["cases"].append(row)
        write_json(output / "partial.json", report)
        print(f"SPARSE_NUMERICAL_CASE device={device} layer={sample['layer']}", flush=True)
    if any(bench.sha(p) != h for p, h in hashes.items()):
        raise RuntimeError("source changed during numerical evaluation")
    write_json(output / "complete.json", report)
    print(f"SPARSE_NUMERICAL_COMPLETE device={device}", flush=True)


def compare(cpu_root, gpu_root, output, cpu_job, gpu_job):
    if output.exists():
        raise RuntimeError("refusing to overwrite numerical comparison")
    reports = []
    values = []
    for root, device, job in ((cpu_root, "cpu", cpu_job), (gpu_root, "cuda", gpu_job)):
        report = json.loads((root / "complete.json").read_text())
        if report["device"] != device or report["job"] != job:
            raise RuntimeError("comparison job/device mismatch")
        for expected_path in (Path(__file__), Path(bench.__file__),
                              Path(__file__).parents[1] / "dsv4_sparse_trial.py"):
            matches = [h for p, h in report["sources"].items() if Path(p).name == expected_path.name]
            if matches != [bench.sha(expected_path)]:
                raise RuntimeError("comparison evaluated source mismatch")
        if report["capture_sha256"] != {str(k): bench.CAPTURE_HASHES[k] for k in LAYERS}:
            raise RuntimeError("comparison capture mismatch")
        if [row["layer"] for row in report["cases"]] != list(LAYERS):
            raise RuntimeError("comparison layer coverage mismatch")
        if device == "cuda" and report["gpu_image"] != GPU_IMAGE:
            raise RuntimeError("comparison GPU image mismatch")
        layers = {}
        for row in report["cases"]:
            path = root / f"L{row['layer']}.pt"
            if Path(row["output_file"]).resolve() != path.resolve() or bench.sha(path) != row["output_sha256"]:
                raise RuntimeError("comparison output path/hash mismatch")
            data = torch.load(path, weights_only=True, map_location="cpu")
            if any(data[k] != expected for k, expected in {
                    "job": job, "device": device, "layer": row["layer"],
                    "capture_sha256": bench.CAPTURE_HASHES[row["layer"]]}.items()):
                raise RuntimeError("comparison tensor identity mismatch")
            expected = ({"online", "candidate", "fp64"} if device == "cpu" else
                        {"flashmla_heads16", "flashmla_heads64", "tilelang_heads16", "tilelang_heads64"})
            if set(data["outputs"]) != expected:
                raise RuntimeError("comparison output coverage mismatch")
            for name, tensor in data["outputs"].items():
                dtype = (torch.float64 if name == "fp64" else torch.float32) if device == "cpu" else torch.bfloat16
                if tensor.shape != (512, 64, 512) or tensor.dtype != dtype:
                    raise RuntimeError("comparison tensor shape/dtype mismatch")
                differences(tensor, tensor)
            sample = load_capture(row["layer"])
            if (row["capture_identity"] != sample["identity"] or row["q_shape"] != list(sample["q"].shape)
                    or row["kv_shape"] != list(sample["kv"].shape)):
                raise RuntimeError("comparison capture coordinates mismatch")
            if device == "cpu" and not same_bytes(data["outputs"]["online"], sample["output"]):
                raise RuntimeError("comparison baseline differs from approved capture")
            layers[row["layer"]] = data["outputs"]
        reports.append(report)
        values.append(layers)
    results = []
    for layer in LAYERS:
        cpu, gpu = values[0][layer], values[1][layer]
        row = {"layer": layer, "comparisons": {}}
        for name, reference in gpu.items():
            row["comparisons"][name] = {
                "online_fp32": differences(cpu["online"], reference),
                "candidate_fp32": differences(cpu["candidate"], reference),
                "online_bf16": differences(cpu["online"].bfloat16(), reference),
                "candidate_bf16": differences(cpu["candidate"].bfloat16(), reference),
                "gpu_vs_fp64": differences(reference, cpu["fp64"]),
            }
        row["cpu_vs_fp64"] = {name: differences(cpu[name], cpu["fp64"]) for name in ("online", "candidate")}
        results.append(row)
    write_json(output, {
        "cpu_job": cpu_job, "gpu_job": gpu_job, "cases": results,
        "input_report_sha256": {str(root / "complete.json"): bench.sha(root / "complete.json")
                                for root in (cpu_root, gpu_root)},
        "scope": reports[0]["scope"],
        "disposition": "Operator comparison only. GPU proximity is not task accuracy; no promotion.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-results", type=Path)
    parser.add_argument("--gpu-results", type=Path)
    parser.add_argument("--cpu-job")
    parser.add_argument("--gpu-job")
    args = parser.parse_args()
    if args.device == "compare":
        if not all((args.cpu_results, args.gpu_results, args.cpu_job, args.gpu_job)):
            parser.error("comparison requires both result directories and expected job IDs")
        compare(args.cpu_results.resolve(), args.gpu_results.resolve(), args.output.resolve(),
                args.cpu_job, args.gpu_job)
    else:
        run(args.device, args.output.resolve())
