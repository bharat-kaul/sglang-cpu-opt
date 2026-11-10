"""Original answer batches plus a separately labeled GPU forced-prefix trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

from task_gsm8k import extract_final_answer


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-index", type=int, choices=(0, 1))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(exist_ok=False)
    phase_file = args.out / "phase"
    phase_file.write_text("warmup")
    os.environ["DSV4_PREFIX_TRACE_PLAN"] = str(args.plan)
    os.environ["DSV4_PREFIX_TRACE_OUT"] = str(args.out)
    os.environ["DSV4_PREFIX_TRACE_PHASE"] = str(phase_file)
    if args.device == "cpu" and os.environ.get("INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC") != "0":
        raise RuntimeError("CPU trace requires the evaluated baseline, quant diagnostic OFF")
    write_json(args.out / "identity.json", {
        "job": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
        "pid": os.getpid(), "device": args.device,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            (args.plan.parent / "source_hashes.json").read_bytes()
        ).hexdigest(),
        "environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("INTEL_CPU_DSV4_", "OMP_", "KMP_", "GOMP_", "SGLANG_"))
        },
    })
    import sglang as sgl

    engine = sgl.Engine(
        model_path=plan["model"], device=args.device,
        tp_size=1 if args.device == "cpu" else 2, dtype="bfloat16",
        disable_overlap_schedule=True, disable_radix_cache=True,
        disable_cuda_graph=args.device == "cuda", trust_remote_code=True,
        mem_fraction_static=0.5 if args.device == "cpu" else 0.85,
        log_level="warning", watchdog_timeout=7000,
        context_length=4096, max_total_tokens=131072,
        swa_full_tokens_ratio=0.5, chunked_prefill_size=512,
    )
    try:
        write_json(args.out / "runtime.json", engine.get_server_info())
        phases = ("natural",) if args.device == "cpu" else ("natural", "forced")
        expected = set()
        for phase in phases:
            temporary = phase_file.with_suffix(".tmp")
            temporary.write_text(phase)
            temporary.replace(phase_file)
            for batch_index, batch in enumerate(plan["batches"]):
                if args.batch_index is not None and batch_index != args.batch_index:
                    continue
                limits = [
                    256 if phase == "natural" else min(
                        plan["last_decision"] + 1, len(item["output_token_ids"])
                    )
                    for item in batch
                ]
                responses = engine.generate(
                    [item["prompt"] for item in batch],
                    sampling_params=[
                        {"temperature": 0.0, "max_new_tokens": limit,
                         "stop": ["Question", "Assistant:"]}
                        for limit in limits
                    ],
                )
                write_json(args.out / f"{phase}.raw.batch{batch_index}.json", responses)
                if len(responses) != len(batch):
                    raise RuntimeError("missing batch responses")
                records = []
                for item, response, limit in zip(batch, responses, limits):
                    ids = response["output_ids"]
                    expected_ids = item["output_token_ids"][:limit]
                    records.append({
                        "q": item["q"], "gold": item["gold"], "text": response["text"],
                        "pred": extract_final_answer(response["text"]),
                        "output_token_ids": ids, "expected_ids_equal": ids == expected_ids,
                        "expected_text_equal": response["text"] == item["text"],
                        "meta_info": response["meta_info"],
                    })
                    if item["q"] in plan["targets"]:
                        expected.update(
                            f"{phase}.q{item['q']}.d{decision}.pt"
                            for decision in range(min(plan["last_decision"] + 1, len(ids)))
                        )
                write_json(args.out / f"{phase}.batch{batch_index}.json", records)
                if phase == "forced" and not all(r["expected_ids_equal"] for r in records):
                    raise RuntimeError("forced GPU output differs from requested CPU prefix")
                if args.device == "cpu" and not all(
                    r["expected_ids_equal"] and r["expected_text_equal"] for r in records
                ):
                    raise RuntimeError("CPU baseline did not reproduce the original batch")
                print(f"PREFIX_BATCH_COMPLETE phase={phase} batch={batch_index}", flush=True)
        for rank in range(1 if args.device == "cpu" else 2):
            actual = {p.name for p in (args.out / f"rank{rank}").glob("*.q*.pt")}
            if actual != expected:
                raise RuntimeError(f"rank {rank} capture coverage mismatch: {expected - actual}")
        write_json(args.out / "complete.json", {
            "device": args.device, "job": os.environ["SLURM_JOB_ID"],
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "captures_per_rank": len(expected), "phases": phases,
            "batch_index": args.batch_index,
        })
        print("PREFIX_TRACE_COMPLETE", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
