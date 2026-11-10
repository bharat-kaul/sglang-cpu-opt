"""Freeze the original two production batches and verify their consumed prompts."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from dsv4_prefix_trace import Histories
from task_gsm8k import one_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model = "/scratch/bkaul/models/DeepSeek-V4-Flash"
    data_path = Path("/scratch/bkaul/gsm8k_test.jsonl")
    data = [json.loads(line) for line in data_path.read_text().splitlines()]
    shots = "".join(one_example(row, True) + "\n\n" for row in data[:8])
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    root = Path("/scratch/bkaul/dsv4_shared_regression_20261106")
    batches, hashes = [], {}
    for case, job, start in (("q1175", 398192, 1148), ("q1239", 398195, 1230)):
        directory = root / case / f"shared_{job}"
        gens_path = directory / "gens.json"
        records = json.loads(gens_path.read_text())["gens"]
        if [r["q"] for r in records] != list(range(start, start + 32)):
            raise RuntimeError("original production batch identity mismatch")
        for item in records:
            item["prompt"] = shots + one_example(data[item["q"] + 8], False)
            item["prompt_ids"] = tokenizer.encode(item["prompt"])
            if len(item["prompt_ids"]) != item["prompt_tokens"]:
                raise RuntimeError("prompt tokenization differs from evaluated source")
        traces = list(directory.glob("forward.*.jsonl"))
        if len(traces) != 1:
            raise RuntimeError("ambiguous original execution trace")
        history, observed = Histories(), set()
        for line in traces[0].read_text().splitlines():
            record = json.loads(line)
            if record["mode"] == "IDLE":
                continue
            slots = record["req_pool_indices"]
            history.update(
                record["input_ids"], record["positions"], slots,
                record["extend_prefix_lens"], record["extend_seq_lens"],
            )
            for slot in slots:
                for item in records:
                    if history.sequences[slot] == item["prompt_ids"]:
                        observed.add(item["q"])
        if observed != {item["q"] for item in records}:
            raise RuntimeError("not all exact prompt IDs occurred in the original runtime")
        for path in (gens_path, traces[0], directory / "runtime.json"):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        batches.append(records)
    hashes[str(data_path)] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    plan = {
        "model": model, "targets": [1175, 1239, 1168], "last_decision": 40,
        "batches": batches, "inputs_sha256": hashes,
        "scope": "CPU natural reproduction; GPU natural control and CPU-prefix forcing. "
                 "TP2 GPU eager traces are not the historical TP4 accuracy result.",
    }
    with args.out.open("x") as stream:
        json.dump(plan, stream, indent=2)
    print("PREFIX_PLAN_COMPLETE exact_original_prompts=64")


if __name__ == "__main__":
    main()
