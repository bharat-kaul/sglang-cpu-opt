#!/usr/bin/env python3
"""Six resident workers, sixteen logical shards, matched C4 task-accuracy arms."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from screen_serving_opportunities import BASE_COMMIT, EXPECTED_ENV, digest, prepare_inputs, write_json
from task_gsm8k import INVALID, answer_value, extract_final_answer, one_example, read_jsonl


MODEL = Path("/scratch/bkaul/models/DeepSeek-V4-Flash")
DATA = Path("/scratch/bkaul/gsm8k_test.jsonl")
DATA_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
WORKERS, SHARDS, QUESTIONS, BATCH = 6, 16, 1311, 32
ARMS = ("baseline", "candidate")
ENV = {**EXPECTED_ENV, "INTEL_CPU_DSV4_CK_STRICT": "1"}
SAMPLING = {"temperature": 0.0, "max_new_tokens": 256, "stop": ["Question", "Assistant:"]}
PERF_POINTS = ((1, 3072), (32, 128), (64, 128))
PERF_SAMPLING = {"temperature": 0.0, "max_new_tokens": 5, "ignore_eos": True}


def performance_cases(index):
    for batch, length in PERF_POINTS:
        for stage in ("warm", "measured"):
            for arm in (ARMS if index % 2 == 0 else tuple(reversed(ARMS))):
                yield f"perf-b{batch}-s{length}-{stage}-{arm}", batch, length, stage, arm


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def inputs():
    if digest(DATA) != DATA_SHA:
        raise RuntimeError("accuracy dataset changed")
    data = read_jsonl(DATA)
    if len(data) != QUESTIONS + 8:
        raise RuntimeError("unexpected question coverage")
    shots = "".join(one_example(row, True) + "\n\n" for row in data[:8])
    return [shots + one_example(row, False) for row in data[8:]], [
        answer_value(row["answer"]) for row in data[8:]]


def batches(worker):
    for shard in range(worker, SHARDS, WORKERS):
        ids = list(range(shard * 82, min((shard + 1) * 82, QUESTIONS)))
        for batch, start in enumerate(range(0, len(ids), BATCH)):
            yield shard, batch, ids[start:start + BATCH]


def case_name(shard, batch, arm):
    return f"s{shard:02d}-b{batch:02d}-{arm}"


def check_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    expected = {"workers": WORKERS, "shards": SHARDS, "questions": QUESTIONS, "batch": BATCH,
                "environment": ENV, "sampling": SAMPLING, "base_commit": BASE_COMMIT,
                "performance_points": [list(p) for p in PERF_POINTS], "performance_sampling": PERF_SAMPLING,
                "data_sha256": DATA_SHA, "driver_sha256": digest(__file__)}
    if any(plan[k] != value for k, value in expected.items()):
        raise RuntimeError("wave plan contract mismatch")
    if digest(root / "source-sha256.txt") != plan["source_manifest_sha256"]:
        raise RuntimeError("wave source manifest changed")
    if digest(root / "performance-inputs.json") != plan["performance_inputs_sha256"]:
        raise RuntimeError("performance inputs changed")
    if digest(MODEL / "config.json") != plan["model_config_sha256"]:
        raise RuntimeError("wave model config changed")
    for path, value in plan["installed_sources"].items():
        if digest(path) != value:
            raise RuntimeError("installed framework changed")
    return plan


def prepare(root):
    if (root / "plan.json").exists():
        raise RuntimeError("refusing to overwrite accuracy plan")
    prompts, _ = inputs()
    if (root / "source-base.txt").read_text().strip() != BASE_COMMIT:
        raise RuntimeError("wave must use pinned main baseline")
    installed = Path("/scratch/bkaul/venvs/sglang-cpu/lib/python3.12/site-packages/sglang")
    files = [installed / name for name in (
        "srt/models/deepseek_v4.py", "srt/layers/attention/deepseek_v4_backend.py",
        "srt/managers/scheduler.py", "srt/entrypoints/engine.py")]
    perf_inputs = prepare_inputs(str(MODEL), str(DATA), [1, 32, 64], [128, 3072])
    write_json(root / "performance-inputs.json", perf_inputs)
    plan = {
        "workers": WORKERS, "shards": SHARDS, "questions": QUESTIONS, "batch": BATCH,
        "environment": ENV, "sampling": SAMPLING, "base_commit": BASE_COMMIT,
        "data_sha256": DATA_SHA, "driver_sha256": digest(__file__),
        "performance_points": [list(p) for p in PERF_POINTS], "performance_sampling": PERF_SAMPLING,
        "performance_inputs_sha256": digest(root / "performance-inputs.json"),
        "source_manifest_sha256": digest(root / "source-sha256.txt"),
        "model_config_sha256": digest(MODEL / "config.json"),
        "prompt_sha256": [prompt_hash(p) for p in prompts],
        "installed_sources": {str(p): digest(p) for p in files},
        "decision": "No observed aggregate loss versus fresh matched fixed32 baseline, "
                    "zero invalid answers, complete validated coverage. Not statistical "
                    "noninferiority or automatic production promotion.",
        "lifecycle": "Two concurrent loads, six resident workers. Scheduler + plugin history "
                     "reset per batch/arm. Alternating arm order. No model reload between logical shards.",
    }
    write_json(root / "plan.json", plan)
    (root / "workers").mkdir()
    (root / "load_slots").mkdir()


def validate_events(events, case, arm, job, *, affinity=(0,)):
    if not events or len({e["pid"] for e in events}) != 1:
        raise RuntimeError("missing or multiple inference workers")
    ordered = sorted(events, key=lambda e: e["sequence"])
    seqs = [e["sequence"] for e in ordered]
    if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
        raise RuntimeError("noncontiguous or duplicate forward sequence")
    prefill_tokens = 0
    for i, e in enumerate(ordered):
        if (e["case"] != case or e["arm"] != arm or e["job"] != job or e.get("error")
                or e["threads"] != 32 or e["affinity"] != list(affinity) or e["reset"] is not (i == 0)
                or type(e["tokens"]) is not int or e["tokens"] <= 0
                or type(e["forward_ns"]) is not int or e["forward_ns"] <= 0):
            raise RuntimeError("invalid inference event")
        if e["phase"] not in ("prefill", "decode") or (i == 0 and e["phase"] != "prefill"):
            raise RuntimeError("invalid forward phase")
        if e["phase"] == "prefill":
            prefill_tokens += e["tokens"]
        calls = e["sparse"]
        if len(calls) != 43 or {c["layer"] for c in calls} != set(range(43)):
            raise RuntimeError("incomplete sparse coverage")
        for c in calls:
            ratio = 0 if c["layer"] < 2 else 4 if c["layer"] % 2 == 0 else 128
            selected = arm == "candidate" and e["phase"] == "prefill" and ratio == 4
            if c["ratio"] != ratio or c["candidate"] is not selected:
                raise RuntimeError("illegal candidate dispatch")
    return prefill_tokens


def score_responses(responses, ids, prompts, golds):
    if not isinstance(responses, list) or len(responses) != len(ids):
        raise RuntimeError("incorrect response count")
    records = []
    for qid, response in zip(ids, responses):
        text, tokens, meta = response["text"], response["output_ids"], response["meta_info"]
        if (not isinstance(text, str) or not isinstance(tokens, list) or not 1 <= len(tokens) <= 256
                or any(type(t) is not int or not 0 <= t < 129280 for t in tokens)
                or type(meta["prompt_tokens"]) is not int or meta["prompt_tokens"] <= 0):
            raise RuntimeError("invalid generated response")
        pred = extract_final_answer(text)
        records.append({
            "q": qid, "gold": golds[qid], "text": text, "pred": pred,
            "correct": pred == golds[qid], "invalid": pred == INVALID,
            "output_token_ids": tokens, "prompt_sha256": prompt_hash(prompts[qid]),
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta.get("completion_tokens"), "finish_reason": meta.get("finish_reason"),
        })
    return records


def case_events(out, case):
    return [event for path in (out / "events").glob("events.*.jsonl")
            for line in path.read_text().splitlines()
            if (event := json.loads(line))["case"] == case]


def performance_sample(packet, events, identity, spec, *, affinity=(0,)):
    case, batch, length, stage, arm = spec
    if (packet["identity"] != identity or packet["case"] != case or packet["batch"] != batch
            or packet["length"] != length or packet["stage"] != stage or packet["arm"] != arm):
        raise RuntimeError("performance coordinates/identity mismatch")
    if validate_events(events, case, arm, identity["job"], affinity=affinity) != batch * length:
        raise RuntimeError("performance prefill work mismatch")
    decode = [e for e in events if e["phase"] == "decode"]
    if sum(e["tokens"] for e in decode) != batch * 4:
        raise RuntimeError("performance decode work mismatch")
    for e in events:
        if (not isinstance(e["seq_lens"], list) or not e["seq_lens"]
                or any(type(n) is not int or n <= 0 or n > length + 4 for n in e["seq_lens"])
                or (e["phase"] == "decode" and len(e["seq_lens"]) != e["tokens"])):
            raise RuntimeError("invalid performance sequence coordinates")
    elapsed = packet["elapsed_s"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("invalid performance client duration")
    responses = packet["responses"]
    if not isinstance(responses, list) or len(responses) != batch:
        raise RuntimeError("performance response count mismatch")
    for response in responses:
        ids, meta = response["output_ids"], response["meta_info"]
        if (not isinstance(ids, list) or len(ids) != 5
                or any(type(t) is not int or not 0 <= t < 129280 for t in ids)
                or type(meta["prompt_tokens"]) is not int or meta["prompt_tokens"] != length
                or type(meta["completion_tokens"]) is not int or meta["completion_tokens"] != 5):
            raise RuntimeError("performance generated work mismatch")
    prefill_s = sum(e["forward_ns"] for e in events if e["phase"] == "prefill") / 1e9
    decode_s = sum(e["forward_ns"] for e in decode) / 1e9
    if elapsed < prefill_s + decode_s:
        raise RuntimeError("forward time exceeds end-to-end request time")
    first = min(events, key=lambda e: e["sequence"])
    return {
        "identity": identity, "inference_pid": first["pid"],
        "prefill_forward_s": prefill_s, "decode_forward_s": decode_s, "client_s": elapsed,
        "prefill_forward_tokens_s": batch * length / prefill_s,
        "decode_forward_tokens_s": batch * 4 / decode_s,
        "schedule": [(e["phase"], e["tokens"], e["seq_lens"])
                     for e in sorted(events, key=lambda e: e["sequence"])],
        "anchors": [first[k] for k in ("first_input_hash", "first_positions_hash", "first_cache_hash")],
    }


def run_performance(engine, root, out, identity, index):
    prompts = json.loads((root / "performance-inputs.json").read_text())
    paired = {}
    for spec in performance_cases(index):
        case, batch, length, stage, arm = spec
        if engine.flush_cache().success is not True:
            raise RuntimeError("performance scheduler reset failed")
        write_json(out / "control.json", {"case": case, "arm": arm})
        started = time.monotonic()
        responses = engine.generate(input_ids=prompts[str(length)][:batch], sampling_params=PERF_SAMPLING)
        elapsed = time.monotonic() - started
        events = case_events(out, case)
        packet = {"identity": identity, "case": case, "batch": batch, "length": length,
                  "stage": stage, "arm": arm, "elapsed_s": elapsed, "responses": responses}
        sample = performance_sample(packet, events, identity, spec)
        paired[(batch, length, stage, arm)] = sample
        other = paired.get((batch, length, stage, "candidate" if arm == "baseline" else "baseline"))
        if other is not None and any(sample[k] != other[k] for k in ("anchors", "schedule")):
            raise RuntimeError("performance paired inputs/schedules differ")
        ep = out / "performance" / f"{case}-events.json"
        write_json(ep, events)
        packet["events_sha256"] = digest(ep)
        write_json(out / "performance" / f"{case}.json", packet)
        print(f"C4_PERFORMANCE_CASE {case} client_s={elapsed:.3f}", flush=True)


def summarize_performance(root, identities, final):
    samples = {f"b{b}-s{s}": [] for b, s in PERF_POINTS}
    for index, identity in identities.items():
        pairs = {}
        for spec in performance_cases(index):
            case, batch, length, stage, arm = spec
            path = root / "workers" / str(index) / "performance" / f"{case}.json"
            if not path.exists():
                if final:
                    raise RuntimeError("missing performance case")
                continue
            packet = json.loads(path.read_text())
            ep = path.with_name(f"{case}-events.json")
            if digest(ep) != packet["events_sha256"]:
                raise RuntimeError("performance event hash mismatch")
            pairs[(batch, length, stage, arm)] = performance_sample(
                packet, json.loads(ep.read_text()), identity, spec)
        for batch, length in PERF_POINTS:
            for stage in ("warm", "measured"):
                a, b = (pairs.get((batch, length, stage, arm)) for arm in ARMS)
                if a is None or b is None:
                    continue
                if any(a[k] != b[k] for k in ("anchors", "schedule", "inference_pid")):
                    raise RuntimeError("performance paired inputs/schedules/processes differ")
                if stage == "measured":
                    samples[f"b{batch}-s{length}"].append({"baseline": a, "candidate": b})
    result = {}
    for point, rows in samples.items():
        if final and len(rows) != WORKERS:
            raise RuntimeError("incomplete performance replicas")
        reductions = {}
        for metric in ("prefill_forward_s", "decode_forward_s", "client_s"):
            values = [100 * (1 - r["candidate"][metric] / r["baseline"][metric]) for r in rows]
            reductions[metric] = ({"median": statistics.median(values), "min": min(values), "max": max(values)}
                                  if values else None)
        result[point] = {"process_pairs": len(rows), "samples": rows, "time_reduction_percent": reductions}
    return {
        "points": result,
        "scope": "Six independent resident-process paired samples per point, alternating order across workers. "
                 "Separate warmup of both arms excluded. Median and min/max of paired time reductions. "
                 "Phase timings are instrumented model-forward sums, NOT serving TTFT or inter-token latency. "
                 "Client duration is end-to-end fixed-length request wall time. Four decode steps cover C4, "
                 "not a complete C128 cycle; no long steady-state decode claim.",
    }


def worker(root, index):
    if not 0 <= index < WORKERS:
        raise RuntimeError("invalid resident worker")
    plan = check_plan(root)
    for key, value in ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"incorrect wave environment: {key}")
    prompts, golds = inputs()
    if [prompt_hash(p) for p in prompts] != plan["prompt_sha256"]:
        raise RuntimeError("prompt construction differs from plan")
    out = root / "workers" / str(index)
    out.mkdir(exist_ok=True)
    if (out / "identity.json").exists():
        raise RuntimeError("worker already started; no implicit resume")
    cases = {case_name(s, b, arm): arm for s, b, _ in batches(index) for arm in ARMS}
    cases.update({case: arm for case, _, _, _, arm in performance_cases(index)})
    write_json(out / "control-plan.json", cases)
    write_json(out / "control.json", {"case": "initialization", "arm": "baseline"})
    (out / "cases").mkdir()
    (out / "events").mkdir()
    (out / "performance").mkdir()
    os.environ["DSV4_C4_ACCURACY_CONTROL"] = str(out / "control.json")
    os.environ["DSV4_C4_ACCURACY_PLAN"] = str(out / "control-plan.json")
    os.environ["DSV4_C4_ACCURACY_OUT"] = str(out / "events")
    import intel_cpu_models  # noqa: F401
    import sglang as sgl
    import sgl_kernel
    import torch

    identity = {
        "worker": index, "job": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename,
        "array_job": os.environ["SLURM_ARRAY_JOB_ID"],
        "pid": os.getpid(), "plan_sha256": digest(root / "plan.json"),
        "common_ops_path": str(Path(sgl_kernel.common_ops.__file__).resolve()),
        "common_ops_sha256": digest(sgl_kernel.common_ops.__file__), "torch": torch.__version__,
        "sglang": sgl.__version__, "environment": ENV,
    }
    write_json(out / "identity.json", identity)
    engine = sgl.Engine(
        model_path=str(MODEL), device="cpu", tp_size=1, dtype="bfloat16",
        disable_overlap_schedule=True, trust_remote_code=True, mem_fraction_static=0.5,
        log_level="warning", watchdog_timeout=7000, chunked_prefill_size=512,
        disable_radix_cache=True, context_length=4096, max_total_tokens=131072,
        swa_full_tokens_ratio=0.5, max_running_requests=256, enable_mixed_chunk=False, random_seed=0,
    )
    try:
        runtime = engine.get_server_info()
        if runtime["enable_mixed_chunk"] is not False:
            raise RuntimeError("mixed decode/prefill would invalidate candidate gating")
        write_json(out / "runtime.json", runtime)
        print(f"C4_ACCURACY_READY worker={index}", flush=True)
        run_performance(engine, root, out, identity, index)
        for shard, batch, ids in batches(index):
            order = ARMS if (shard + batch) % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                case = case_name(shard, batch, arm)
                if engine.flush_cache().success is not True:
                    raise RuntimeError("scheduler cache reset failed")
                write_json(out / "control.json", {"case": case, "arm": arm})
                print(f"C4_ACCURACY_BEGIN {case} count={len(ids)}", flush=True)
                started = time.monotonic()
                responses = engine.generate([prompts[q] for q in ids], SAMPLING)
                elapsed = time.monotonic() - started
                records = score_responses(responses, ids, prompts, golds)
                events = case_events(out, case)
                tokens = validate_events(events, case, arm, identity["job"])
                if tokens != sum(r["prompt_tokens"] for r in records):
                    raise RuntimeError("prefill token work does not match responses")
                event_path = out / "cases" / f"{case}-events.json"
                write_json(event_path, events)
                packet = {"identity": identity, "case": case, "arm": arm, "shard": shard, "batch": batch,
                          "records": records, "elapsed_s": elapsed, "events_sha256": digest(event_path)}
                write_json(out / "cases" / f"{case}.json", packet)
                print(f"C4_ACCURACY_BATCH {case} correct={sum(r['correct'] for r in records)}/{len(ids)} "
                      f"invalid={sum(r['invalid'] for r in records)} elapsed_s={elapsed:.1f}", flush=True)
        check_plan(root)
        if digest(identity["common_ops_path"]) != identity["common_ops_sha256"]:
            raise RuntimeError("native dependency changed during accuracy run")
        write_json(out / "complete.json", identity)
        print(f"C4_ACCURACY_WORKER_COMPLETE worker={index}", flush=True)
    finally:
        engine.shutdown()


def summarize(root, final=False):
    plan = check_plan(root)
    prompts, golds = inputs()
    if [prompt_hash(p) for p in prompts] != plan["prompt_sha256"]:
        raise RuntimeError("summary prompt binding failed")
    scored = {arm: {} for arm in ARMS}
    completed, failures, jobs, environments = [], [], set(), set()
    anchors, identities = {}, {}
    for index in range(WORKERS):
        out = root / "workers" / str(index)
        if (out / "exit_code").exists() and (out / "exit_code").read_text().strip() != "0":
            failures.append(index)
        if not (out / "identity.json").exists():
            if final:
                raise RuntimeError("missing worker identity")
            continue
        identity = json.loads((out / "identity.json").read_text())
        if (identity["worker"] != index or identity["plan_sha256"] != digest(root / "plan.json")
                or identity["environment"] != ENV or not identity["job"] or identity["job"] in jobs
                or identity["array_job"] != (root / "array-job.txt").read_text().strip()):
            raise RuntimeError("worker identity mismatch")
        jobs.add(identity["job"])
        identities[index] = identity
        environments.add((identity["torch"], identity["sglang"], identity["common_ops_sha256"]))
        if digest(identity["common_ops_path"]) != identity["common_ops_sha256"]:
            raise RuntimeError("native dependency no longer matches")
        expected_cases = set()
        for shard, batch, ids in batches(index):
            for arm in ARMS:
                case = case_name(shard, batch, arm)
                expected_cases.add(case)
                path = out / "cases" / f"{case}.json"
                if not path.exists():
                    if final:
                        raise RuntimeError(f"missing completed batch {case}")
                    continue
                packet = json.loads(path.read_text())
                if (packet["identity"] != identity or packet["case"] != case or packet["arm"] != arm
                        or packet["shard"] != shard or packet["batch"] != batch
                        or [r["q"] for r in packet["records"]] != ids):
                    raise RuntimeError("batch coordinates/identity mismatch")
                ep = path.with_name(f"{case}-events.json")
                if digest(ep) != packet["events_sha256"]:
                    raise RuntimeError("batch event hash mismatch")
                events = json.loads(ep.read_text())
                tokens = validate_events(events, case, arm, identity["job"])
                if tokens != sum(r["prompt_tokens"] for r in packet["records"]):
                    raise RuntimeError("batch token work mismatch")
                first = min(events, key=lambda e: e["sequence"])
                anchors[(shard, batch, arm)] = tuple(first[k] for k in (
                    "first_input_hash", "first_positions_hash", "first_cache_hash"))
                for record in packet["records"]:
                    qid = record["q"]
                    pred = extract_final_answer(record["text"])
                    if (record["gold"] != golds[qid] or record["pred"] != pred
                            or record["correct"] is not (pred == golds[qid])
                            or record["invalid"] is not (pred == INVALID)
                            or record["prompt_sha256"] != prompt_hash(prompts[qid])
                            or qid in scored[arm]):
                        raise RuntimeError("independent rescoring/coverage mismatch")
                    token_ids = record["output_token_ids"]
                    if (not isinstance(token_ids, list) or not 1 <= len(token_ids) <= 256
                            or any(type(t) is not int or not 0 <= t < 129280 for t in token_ids)):
                        raise RuntimeError("invalid token evidence")
                    scored[arm][qid] = record
        present = {p.stem for p in (out / "cases").glob("*.json") if not p.stem.endswith("-events")}
        if not present <= expected_cases:
            raise RuntimeError("unexpected case artifact")
        if (out / "complete.json").exists():
            if json.loads((out / "complete.json").read_text()) != identity or present != expected_cases:
                raise RuntimeError("invalid worker completion")
            if (out / "exit_code").exists() and (out / "exit_code").read_text().strip() == "0":
                completed.append(index)
        if final and index not in completed:
            raise RuntimeError("worker not successfully terminated")
    if len(environments) > 1:
        raise RuntimeError("worker runtime/build identities differ")
    for shard in range(SHARDS):
        for batch in range(3):
            a, b = (anchors.get((shard, batch, arm)) for arm in ARMS)
            if a is not None and b is not None and a != b:
                raise RuntimeError("paired initial batch anchors differ")
    paired = sorted(set(scored["baseline"]) & set(scored["candidate"]))
    counts = {arm: {"done": len(rows), "correct": sum(r["correct"] for r in rows.values()),
                    "invalid": sum(r["invalid"] for r in rows.values())} for arm, rows in scored.items()}
    regressions = [q for q in paired if scored["baseline"][q]["correct"] and not scored["candidate"][q]["correct"]]
    recoveries = [q for q in paired if not scored["baseline"][q]["correct"] and scored["candidate"][q]["correct"]]
    complete = len(completed) == WORKERS and all(set(rows) == set(range(QUESTIONS)) for rows in scored.values())
    performance = summarize_performance(root, identities, final=final or complete)
    result = {
        "plan_sha256": digest(root / "plan.json"), "complete": complete, "completed_workers": completed,
        "failed_workers": failures, "counts": counts, "paired": len(paired),
        "regressions": regressions, "recoveries": recoveries, "net_correct_delta": len(recoveries) - len(regressions),
        "answer_changes": [q for q in paired if scored["baseline"][q]["pred"] != scored["candidate"][q]["pred"]],
        "decision": "PENDING" if not complete else (
            "NO_OBSERVED_LOSS" if len(recoveries) >= len(regressions)
            and all(c["invalid"] == 0 for c in counts.values()) else "DO_NOT_PROMOTE"),
        "performance": performance,
        "scope": "Matched fixed32 CPU task comparison, not a GPU rerun, statistical noninferiority "
                 "or automatic default promotion. Controlled performance probes reported separately.",
    }
    if final and not complete:
        raise RuntimeError("incomplete question coverage")
    return result


def watch(root, job):
    if not job.isdecimal():
        raise RuntimeError("invalid array job ID")
    empty = 0
    while True:
        try:
            result = summarize(root)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError):
            subprocess.run(["scancel", job], check=True)
            raise
        write_json(root / "progress.json", result)
        print(f"C4_WAVE_PROGRESS paired={result['paired']} counts={result['counts']} "
              f"net={result['net_correct_delta']} failed={result['failed_workers']}", flush=True)
        catastrophic = (result["paired"] >= 160 and
                        result["net_correct_delta"] / result["paired"] <= -0.05)
        invalid = any(c["done"] >= 160 and c["invalid"] / c["done"] > 0.05
                      for c in result["counts"].values())
        if result["failed_workers"] or catastrophic or invalid:
            subprocess.run(["scancel", job], check=True)
            raise RuntimeError("wave aborted: worker failure or severe partial regression")
        if result["complete"]:
            write_json(root / "summary.json", summarize(root, final=True))
            print("C4_WAVE_COMPLETE", flush=True)
            return
        queued = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T"],
                                check=True, text=True, capture_output=True).stdout.strip()
        empty = empty + 1 if not queued else 0
        if empty >= 2:
            raise RuntimeError("array absent without complete evidence; not a successful wave")
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "summary", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--job")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "worker":
        if args.worker is None:
            parser.error("--worker required")
        worker(root, args.worker)
    elif args.action == "watch":
        if not args.job:
            parser.error("--job required")
        watch(root, args.job)
    else:
        write_json(root / "summary.json", summarize(root, final=True))
