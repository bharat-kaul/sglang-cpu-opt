"""Recheck the fixed 2026-11-09 campaign from its retained external evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def main():
    if not __debug__:
        raise RuntimeError("Run without -O: this campaign audit uses assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Retained campaign evidence directory")
    parser.add_argument("--data", type=Path, required=True, help="Original GSM8K JSONL")
    parser.add_argument("--out", type=Path, required=True, help="New summary output path")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.out}")
    plan = json.loads((ROOT / "plan.json").read_text())
    data_path = args.data
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == plan["data_sha256"]
    assert hashlib.sha256((ROOT / "source_hashes.json").read_bytes()).hexdigest() == (
        "ab7753eae9f726c55708becf29d9b2815ff29457fa4098f2249fc0f39bdd1c29"
    )
    for relative, expected in json.loads((ROOT / "source_hashes.json").read_text()).items():
        assert hashlib.sha256((ROOT / "source" / relative).read_bytes()).hexdigest() == expected, relative
    sys.path.insert(0, str(ROOT / "source/plugin/validate"))
    from task_gsm8k import INVALID, answer_value, extract_final_answer

    data = [json.loads(line) for line in data_path.read_text().splitlines()][8:]
    results = {}
    runtime_reference = None
    for arm in ("cpu", "gpu"):
        records = {}
        for shard in range(16):
            directory = ROOT / arm / f"shard_{shard}"
            assert (directory / "complete").exists()
            score = json.loads((directory / "score.json").read_text())
            generation = json.loads((directory / "gens.json").read_text())
            runtime = json.loads((directory / "runtime.json").read_text())
            identity = generation["run_identity"]
            assert score["run_identity"] == identity
            for key, value in {
                "num_shards": 16, "shard_index": shard, "chunk_size": 32,
                "chunked_prefill_size": 512, "num_shots": 8, "max_new_tokens": 256,
                "tp": 1 if arm == "cpu" else 4,
                "device": "cpu" if arm == "cpu" else "cuda",
            }.items():
                assert identity[key] == value, (arm, shard, key)
            assert identity["sampling_params"] == {
                "temperature": 0.0, "max_new_tokens": 256, "stop": ["Question", "Assistant:"]
            }
            if arm == "cpu":
                for flag in ("CHUNK_ACCUM", "MXFP4_CLAMP", "SHARED_CLAMP"):
                    assert identity["env_toggles"]["INTEL_CPU_DSV4_" + flag] == "1"
                assert identity["env_toggles"]["INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC"] == "0"
                assert "INTEL_CPU_DSV4_EXECUTION_TRACE" not in identity["env_toggles"]
            settings = {key: runtime[key] for key in (
                "disable_radix_cache", "chunked_prefill_size", "context_length",
                "max_total_tokens", "swa_full_tokens_ratio", "disable_overlap_schedule",
                "max_running_requests",
            )}
            assert settings["disable_radix_cache"] is True
            if runtime_reference is None:
                runtime_reference = settings
            assert settings == runtime_reference, (arm, shard, settings, runtime_reference)
            rows = generation["gens"]
            assert [row["q"] for row in rows] == list(range(shard * 82, min((shard + 1) * 82, 1311)))
            correct = invalid = 0
            for row in rows:
                q = row["q"]
                assert q not in records
                gold = answer_value(data[q]["answer"])
                pred = extract_final_answer(row["text"])
                assert (gold, pred) == (row["gold"], row["pred"])
                assert isinstance(row["output_token_ids"], list) and row["output_token_ids"]
                correct += pred == gold
                invalid += pred == INVALID
                records[q] = row
            assert score["correct"] == correct and score["invalid"] == invalid
            assert score["done"] == score["total"] == len(rows) and score["partial"] is False
        assert set(records) == set(range(1311))
        results[arm] = records
    cpu, gpu = results["cpu"], results["gpu"]
    summary = {
        "scope": "One full matched custom-protocol GSM8K evaluation; not published-score reproduction or repeatability certification.",
        "cpu_jobs": json.loads((ROOT / "cpu_shard_jobs.json").read_text()),
        "gpu_job": 401058,
        "source_manifest_sha256": hashlib.sha256((ROOT / "source_hashes.json").read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matched_runtime": runtime_reference,
        "questions": 1311,
        "results": {},
        "cpu_right_gpu_wrong": [],
        "gpu_right_cpu_wrong": [],
    }
    for arm, rows in results.items():
        correct = sum(row["pred"] == row["gold"] for row in rows.values())
        summary["results"][arm] = {
            "correct": correct, "accuracy": correct / 1311,
            "invalid": sum(row["pred"] == INVALID for row in rows.values()),
        }
    for q in range(1311):
        cr, gr = cpu[q]["pred"] == cpu[q]["gold"], gpu[q]["pred"] == gpu[q]["gold"]
        if cr and not gr:
            summary["cpu_right_gpu_wrong"].append(q)
        if gr and not cr:
            summary["gpu_right_cpu_wrong"].append(q)
    summary["answer_agreement"] = sum(cpu[q]["pred"] == gpu[q]["pred"] for q in cpu)
    summary["cpu_minus_gpu"] = summary["results"]["cpu"]["correct"] - summary["results"]["gpu"]["correct"]
    historical = {}
    for path in Path(plan["historical_cpu_root"]).glob("shard_*/shared_*/gens.json"):
        value = json.loads(path.read_text())
        historical.update({row["q"]: row for row in value["gens"]})
    assert set(historical) == set(cpu)
    summary["cpu_vs_historical"] = {
        "changed_answers": [q for q in cpu if cpu[q]["pred"] != historical[q]["pred"]],
        "changed_texts": [q for q in cpu if cpu[q]["text"] != historical[q]["text"]],
    }
    with args.out.open("x") as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
