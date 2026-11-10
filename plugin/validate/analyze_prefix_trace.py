"""Localize native boundary disagreements, without treating them as local defects."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from compare_shared_regression_gpu import metrics
from task_gsm8k import extract_final_answer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(cpu, gpu):
    result = metrics(cpu.float().unsqueeze(0), gpu.float().unsqueeze(0), 0)
    result.update(cpu_dtype=str(cpu.dtype), gpu_dtype=str(gpu.dtype),
                  values_equal=torch.equal(cpu, gpu), shape=list(cpu.shape))
    return result


def load_capture(path, *, q, decision, phase, job, plan_hash, sequence):
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"q": q, "decision": decision, "phase": phase,
                "job": str(job), "plan_sha256": plan_hash, "sequence_ids": sequence}
    for key, item in expected.items():
        if value[key] != item:
            raise RuntimeError(f"{path}: inconsistent {key}")
    row, logit_row = value["target_row"], value["logit_row"]
    if (value["positions"][row] != len(sequence) - 1
            or value["input_ids"][row] != sequence[-1]
            or value["slots"][logit_row] != value["slot"]):
        raise RuntimeError(f"{path}: inconsistent request coordinates")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cpu_job")
    parser.add_argument("gpu_job")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan_path = args.root / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = digest(plan_path)
    source_hash = digest(args.root / "source_hashes.json")
    for relative, expected in json.loads((args.root / "source_hashes.json").read_text()).items():
        if digest(args.root / "source" / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    cpu_jobs = args.cpu_job.split(",")
    if len(cpu_jobs) == 1:
        cpu_jobs *= len(plan["batches"])
    if len(cpu_jobs) != len(plan["batches"]):
        raise RuntimeError("expected one CPU job or one per original batch")
    cpu_directories = [args.root / f"cpu_{job}" for job in cpu_jobs]
    gpu_dir = args.root / f"gpu_{args.gpu_job}"
    runs = list({(directory, job, "cpu") for directory, job in zip(cpu_directories, cpu_jobs)})
    runs.append((gpu_dir, args.gpu_job, "cuda"))
    for directory, job, device in runs:
        identity = json.loads((directory / "identity.json").read_text())
        complete = json.loads((directory / "complete.json").read_text())
        for record in (identity, complete):
            if (record["job"] != job or record["device"] != device
                    or record["plan_sha256"] != plan_hash):
                raise RuntimeError(f"invalid run identity: {directory}")
        if identity["source_manifest_sha256"] != source_hash:
            raise RuntimeError("run used a different frozen source")
    originals = {item["q"]: item for batch in plan["batches"] for item in batch}
    cpu_sources = {
        item["q"]: (directory, job)
        for batch, directory, job in zip(plan["batches"], cpu_directories, cpu_jobs)
        for item in batch
    }
    gpu_natural = {}
    for index, batch in enumerate(plan["batches"]):
        cpu_dir = cpu_directories[index]
        expected_qs = [item["q"] for item in batch]
        cpu = json.loads((cpu_dir / f"natural.batch{index}.json").read_text())
        gpu = json.loads((gpu_dir / f"natural.batch{index}.json").read_text())
        forced = json.loads((gpu_dir / f"forced.batch{index}.json").read_text())
        for rows in (cpu, gpu, forced):
            if [r["q"] for r in rows] != expected_qs:
                raise RuntimeError("batch order or coverage changed")
            for record in rows:
                if extract_final_answer(record["text"]) != record["pred"]:
                    raise RuntimeError("saved answer does not match independent extraction")
        for item, actual, replay in zip(batch, cpu, forced):
            if (actual["text"] != item["text"]
                    or actual["output_token_ids"] != item["output_token_ids"]):
                raise RuntimeError("CPU original output reproduction failed")
            expected_ids = item["output_token_ids"][:plan["last_decision"] + 1]
            if replay["output_token_ids"] != expected_ids:
                raise RuntimeError("GPU forcing did not consume the requested trajectory")
        gpu_natural.update({r["q"]: r for r in gpu})
    summaries = []
    detail_path = args.root / "boundary_metrics.jsonl"
    temporary = detail_path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for q in plan["targets"]:
            cpu_dir, cpu_job = cpu_sources[q]
            item, natural = originals[q], gpu_natural[q]
            cpu_ids, gpu_ids = item["output_token_ids"], natural["output_token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(cpu_ids, gpu_ids)) if a != b), None)
            if first is None and len(cpu_ids) != len(gpu_ids):
                first = min(len(cpu_ids), len(gpu_ids))
            summary = {
                "q": q, "gold": item["gold"], "cpu_answer": extract_final_answer(item["text"]),
                "gpu_natural_answer": extract_final_answer(natural["text"]),
                "natural_first_token_difference": first, "selected_decisions": [],
            }
            for decision in range(plan["last_decision"] + 1):
                sequence = item["prompt_ids"] + cpu_ids[:decision]
                common = dict(q=q, decision=decision, plan_hash=plan_hash, sequence=sequence)
                cpu = load_capture(
                    cpu_dir / f"rank0/natural.q{q}.d{decision}.pt",
                    phase="natural", job=cpu_job, **common,
                )
                gpu = [
                    load_capture(gpu_dir / f"rank{rank}/forced.q{q}.d{decision}.pt",
                                 phase="forced", job=args.gpu_job, **common)
                    for rank in (0, 1)
                ]
                logits = [gpu[rank]["logits"] for rank in (0, 1)]
                if logits[0].dtype != logits[1].dtype or not torch.equal(*logits):
                    raise RuntimeError("GPU rank logits disagree")
                repeated = None
                native = None
                if gpu_ids[:decision] == cpu_ids[:decision] and decision < len(gpu_ids):
                    native = load_capture(
                        gpu_dir / f"rank0/natural.q{q}.d{decision}.pt",
                        phase="natural", job=args.gpu_job, **common,
                    )
                    repeated = {
                        "logits": difference(native["logits"], logits[0]),
                        "batch_tokens_equal": native["input_ids"] == gpu[0]["input_ids"],
                        "batch_positions_equal": native["positions"] == gpu[0]["positions"],
                        "first_canonical_boundary_difference": None,
                    }
                layers = []
                for layer in range(43):
                    diffs = {}
                    for boundary in ("attn.in", "attn.out", "mlp.in", "mlp.out"):
                        key = f"L{layer}.{boundary}"
                        left, right = [g["boundaries"][key] for g in gpu]
                        if left.dtype != right.dtype or not torch.equal(left, right):
                            raise RuntimeError(f"GPU canonical rank outputs disagree: {key}")
                        diffs[boundary] = difference(cpu["boundaries"][key], left)
                        if native is not None and repeated["first_canonical_boundary_difference"] is None:
                            native_value = native["boundaries"][key]
                            if not torch.equal(native_value, left):
                                repeated["first_canonical_boundary_difference"] = {
                                    "boundary": key, "difference": difference(native_value, left),
                                }
                    cpu_routes, cpu_weights = cpu["boundaries"][f"L{layer}.topk.out"]
                    gpu_routes, gpu_weights = gpu[0]["boundaries"][f"L{layer}.topk.out"]
                    rank_routes, rank_weights = gpu[1]["boundaries"][f"L{layer}.topk.out"]
                    if not torch.equal(gpu_routes, rank_routes) or not torch.equal(gpu_weights, rank_weights):
                        raise RuntimeError("GPU rank routing differs")
                    cpu_set, gpu_set = set(cpu_routes.tolist()), set(gpu_routes.tolist())
                    layers.append({
                        "layer": layer, "differences": diffs,
                        "cpu_route_ids": cpu_routes.tolist(), "gpu_route_ids": gpu_routes.tolist(),
                        "route_set_changed": cpu_set != gpu_set,
                        "cpu_route_weights": cpu_weights.tolist(),
                        "gpu_route_weights": gpu_weights.tolist(),
                    })
                cpu_token, gpu_token = int(cpu["logits"].argmax()), int(logits[0].argmax())
                if cpu_token != cpu_ids[decision]:
                    raise RuntimeError("captured CPU raw argmax differs from generated token")
                heads = {
                    key: difference(cpu["boundaries"][key], gpu[0]["boundaries"][key])
                    for key in ("hc_head.in", "hc_head.out", "final_norm.out", "lm_head.in")
                }
                record = {
                    "q": q, "decision": decision, "cpu_token": cpu_token,
                    "gpu_conditional_token": gpu_token,
                    "cpu_old_minus_gpu_token": float(cpu["logits"][cpu_token].float()
                                                    - cpu["logits"][gpu_token].float()),
                    "gpu_cpu_minus_own_token": float(logits[0][cpu_token].float()
                                                    - logits[0][gpu_token].float()),
                    "layers": layers, "head": heads,
                    "gpu_natural_vs_forced_on_common_prefix": repeated,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                if decision in (0, first):
                    selected = {k: v for k, v in record.items() if k != "layers"}
                    selected["route_set_changed_layers"] = [
                        row["layer"] for row in layers if row["route_set_changed"]
                    ]
                    selected["layer_errors"] = [
                        {"layer": row["layer"], **{
                            name: metric["target_rel_l2"]
                            for name, metric in row["differences"].items()
                        }} for row in layers
                    ]
                    summary["selected_decisions"].append(selected)
            summaries.append(summary)
    temporary.replace(detail_path)
    output = {
        "cpu_job": args.cpu_job, "gpu_job": args.gpu_job,
        "plan_sha256": plan_hash, "source_manifest_sha256": source_hash,
        "analysis_sha256": digest(Path(__file__)),
        "scope": "Native subsystem disagreement on identical consumed prefixes; "
                 "different subsystem operands are not matched-operand kernel comparisons.",
        "cpu_reproduced_original_cases": len(originals),
        "matched_decisions": len(plan["targets"]) * (plan["last_decision"] + 1),
        "targets": summaries,
    }
    (args.root / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for target in summaries:
        print({k: v for k, v in target.items() if k != "selected_decisions"}, flush=True)


if __name__ == "__main__":
    main()
