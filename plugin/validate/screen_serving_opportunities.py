#!/usr/bin/env python3
"""One-load serving attribution, with an explicit diagnostic C4 trial mode."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[2]
ACCURACY = Path("/scratch/bkaul/dsv4_latest_fullacc_20261107")
ACCURACY_MANIFEST = "ab7753eae9f726c55708becf29d9b2815ff29457fa4098f2249fc0f39bdd1c29"
BASE_COMMIT = "34fe50f1aca4dc9d1ff17111a38ff5f323d7b106"
MEANINGFUL_POINTS = [(1, 3072), (32, 128), (64, 128)]
EXPECTED_ENV = {
    "SGLANG_USE_CPU_ENGINE": "1", "SGLANG_EXTERNAL_MODEL_PACKAGE": "intel_cpu_models",
    "INTEL_CPU_DSV4_DSA_CPU": "1", "INTEL_CPU_DSV4_MXFP4_MOE": "1",
    "INTEL_CPU_DSV4_DSA_CKERNEL": "1", "INTEL_CPU_DSV4_CHUNK_ACCUM": "1",
    "INTEL_CPU_DSV4_MXFP4_CLAMP": "1", "INTEL_CPU_DSV4_SHARED_CLAMP": "1",
    "INTEL_CPU_DSV4_SHARED_QUANT_DIAGNOSTIC": "0", "INTEL_CPU_DSV4_COOP_MHC": "0",
    "INTEL_CPU_DSV4_DECODE_THREADS": "32",
    "INTEL_CPU_DSV4_DECODE_THREADS_STICKY": "0",
    "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "1000000",
    "OMP_NUM_THREADS": "32", "SGLANG_CPU_OMP_THREADS_BIND": "0-31",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_events(events, case, enabled, layers, *, job, batch, length, output_len,
                    sparse_scope="native_wrapper.sparse_attend_masked"):
    if not events:
        raise RuntimeError(f"no worker events for {case}")
    identities = set()
    phases = set()
    token_counts = {"prefill": 0, "decode": 0}
    for event in events:
        if event["job"] != job:
            raise RuntimeError("worker job mismatch")
        if (type(event["pid"]) is not int or event["pid"] <= 0
                or type(event["sequence"]) is not int or event["sequence"] < 0):
            raise RuntimeError("invalid worker identity")
        identity = (event["pid"], event["sequence"])
        if identity in identities:
            raise RuntimeError("duplicate forward identity")
        identities.add(identity)
        if event["case"] != case or event["enabled"] is not enabled:
            raise RuntimeError("worker control mismatch")
        if event.get("error") or event["failures"]:
            raise RuntimeError(f"worker failure: {event}")
        if type(event["forward_ns"]) is not int or event["forward_ns"] <= 0:
            raise RuntimeError("invalid forward duration")
        if event["forward_ns"] > 180_000_000_000:
            raise RuntimeError("forward exceeded screening budget")
        if event["threads_before"] != 32 or event["threads_after"] != 32:
            raise RuntimeError("unexpected worker thread count")
        phase = event["phase"]
        if phase not in token_counts:
            raise RuntimeError("invalid phase")
        seq_lens = event["seq_lens"]
        if (type(event["tokens"]) is not int or event["tokens"] <= 0
                or not isinstance(seq_lens, list) or not 1 <= len(seq_lens) <= batch
                or any(type(n) is not int or not 1 <= n <= length + output_len - 1
                       for n in seq_lens)):
            raise RuntimeError("invalid forward coordinates")
        if phase == "decode" and (event["tokens"] != len(seq_lens)
                                  or any(n <= length for n in seq_lens)):
            raise RuntimeError("decode coordinates disagree")
        if phase == "prefill" and any(n > length for n in seq_lens):
            raise RuntimeError("prefill coordinates disagree")
        token_counts[phase] += event["tokens"]
        phases.add(phase)
        if not enabled and event["rows"]:
            raise RuntimeError("off arm has active instrumentation")
        for row in event["rows"]:
            if (not all(type(row[k]) is int for k in ("calls", "inclusive_ns", "exclusive_ns"))
                    or row["calls"] <= 0 or row["inclusive_ns"] < row["exclusive_ns"]
                    or row["exclusive_ns"] < 0):
                raise RuntimeError("invalid nested timing")
        if enabled:
            seen = {r["layer"] for r in event["rows"] if r["op"] == "layer.total"}
            if seen != set(range(layers)):
                raise RuntimeError(f"incomplete layer coverage: {seen}")
            if any(r["metadata"].get("threads") != 32 for r in event["rows"]
                   if r["op"] == "layer.total"):
                raise RuntimeError("unexpected in-layer thread count")
            required = {"_batched_mla_attend", "compressor.projection", sparse_scope}
            if not required <= {r["op"] for r in event["rows"]}:
                raise RuntimeError("missing serving attention/compressor observations")
            if any(":sparse_attend:eager" in k and v for k, v in event["dispatch_marks"].items()):
                raise RuntimeError("sparse attention fell back to eager")
    if phases != {"prefill", "decode"}:
        raise RuntimeError(f"incomplete phase coverage: {phases}")
    if token_counts != {"prefill": batch * length, "decode": batch * (output_len - 1)}:
        raise RuntimeError(f"incomplete workload token coverage: {token_counts}")


def opportunity_summary(events):
    result = {}
    for phase in ("prefill", "decode"):
        phase_events = [e for e in events if e["phase"] == phase]
        wall = sum(e["forward_ns"] for e in phase_events)
        if wall <= 0:
            raise RuntimeError("missing phase wall time")
        names = {
            "sparse_native_core": lambda n: n == "native_wrapper.sparse_attend_masked",
            "sparse_whole_pipeline": lambda n: n == "_batched_mla_attend",
            "indexer_replication": lambda n: n == "indexer.shared_kv_expand",
            "compressor_projection_whole": lambda n: n == "compressor.projection",
            "compressor_casts": lambda n: n.startswith("compressor.aten.aten._to_copy."),
            "compressor_mm": lambda n: n.startswith("compressor.aten.aten.mm."),
        }
        buckets = {}
        for label, matches in names.items():
            selected = [r for e in phase_events for r in e["rows"] if matches(r["op"])]
            cost = sum(r["inclusive_ns"] for r in selected)
            calls = sum(r["calls"] for r in selected)
            share = cost / wall
            if not 0 <= share <= 1:
                raise RuntimeError(f"invalid share for {label}")
            buckets[label] = {
                "ns": cost, "calls": calls, "observed": bool(calls),
                "share_of_instrumented_forward": share if calls else None,
                "ideal_speedup_if_removed": None if not calls or share == 1 else 1 / (1 - share),
                "conditional_speedup_if_halved": 1 / (1 - share / 2) if calls else None,
            }
        result[phase] = {"forward_ns": wall, "forwards": len(phase_events), "buckets": buckets}
    return result


def prepare_inputs(model, data, batches, lengths):
    from transformers import AutoTokenizer
    from task_gsm8k import one_example, read_jsonl

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    examples = read_jsonl(data)
    prompts = {}
    for length in lengths:
        rows = []
        for i in range(max(batches)):
            text = "\n\n".join(one_example(examples[(i + j) % len(examples)], True) for j in range(32))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < length:
                raise RuntimeError("not enough natural-text prompt tokens")
            rows.append(tokens[:length])
        prompts[length] = rows
    return prompts


def run(args):
    if getattr(args, "meaningful", False) and getattr(args, "c4_trial", False):
        raise RuntimeError("meaningful capture and C4 trial modes are mutually exclusive")
    if any(os.environ.get(key) for key in ("DSV4_SPARSE_CAPTURE_PLAN", "DSV4_SPARSE_TRIAL_PLAN")):
        raise RuntimeError("probe plans must be owned by this driver")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key, value in EXPECTED_ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} must equal {value}")
    if digest(ACCURACY / "source_hashes.json") != ACCURACY_MANIFEST:
        raise RuntimeError("accuracy manifest changed")
    if (ROOT.parent / "source-base.txt").read_text().strip() != BASE_COMMIT:
        raise RuntimeError("screen must be frozen from the pinned main baseline")
    os.environ["DSV4_SERVING_PROBE_CONTROL"] = str(out / "control.json")
    os.environ["DSV4_SERVING_PROBE_OUT"] = str(out / "worker")
    control = out / "control.json"
    write_json(control, {"case": "initialization", "enabled": False})
    meaningful = getattr(args, "meaningful", False)
    trial = getattr(args, "c4_trial", False)
    if trial:
        os.environ["DSV4_SERVING_PROBE_LIGHT"] = "1"
        points = [(1, 3072)]
        args.output_len = 2
        plan = {"b1-s3072-trial-a": False, "b1-s3072-trial-candidate": True,
                "b1-s3072-trial-b": False}
        write_json(out / "trial-plan.json", plan)
        os.environ["DSV4_SPARSE_TRIAL_PLAN"] = str(out / "trial-plan.json")
    elif meaningful:
        os.environ["DSV4_SERVING_PROBE_LIGHT"] = "1"
        points = MEANINGFUL_POINTS
        args.output_len = 5
        plan = {f"b{b}-s{s}-meaningful": {"batch": b, "length": s} for b, s in points}
        write_json(out / "capture-plan.json", plan)
        os.environ["DSV4_SPARSE_CAPTURE_PLAN"] = str(out / "capture-plan.json")
    else:
        points = [(b, s) for b in args.batches for s in args.lengths]
    source_files = sorted(p for p in (ROOT / "plugin").rglob("*")
                          if p.suffix in {".py", ".cpp", ".h"})
    identity = {
        "job": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename, "pid": os.getpid(),
        "accuracy_manifest_sha256": ACCURACY_MANIFEST, "environment": EXPECTED_ENV,
        "base_commit": BASE_COMMIT,
        "runtime_difference": "Warm fixed-32 pool, sticky=0; accuracy job uses passive fixed-8 sticky=1.",
        "scope": ("One request per point; capture forwards excluded from attribution. No off/on "
                  "perturbation or generation-repeatability claim." if meaningful else
                  "One resident process; repeated requests, not independent process replicas. Provisional attribution."),
        "model_config_sha256": digest(Path(args.model) / "config.json"),
        "model_config_path": str(Path(args.model) / "config.json"),
        "data_sha256": digest(args.data),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "cases": [], "disposition": {
            "sparse": "Measure current core and preparation; do not assume BF16 contract equivalence.",
            "indexer": "Observe expansion; zero calls (e.g. short decode) is absence, not free speedup.",
            "projection": "Separate casts/mm with dispatch observer; retain observed perturbation.",
            "materiality": "Ideal removal below 1.10x rules out >10% total gain at that phase/point.",
            "perturbation": "If on/off phase time differs by >5%, attribution is diagnostic only.",
        },
    }
    if trial:
        identity["scope"] = ("Baseline/candidate/baseline B1S3072G2, one resident load. Explicit "
                             "scheduler and plugin state resets. Numerical diagnostics, no promotion.")
    write_json(out / "identity.json", identity)
    prompts = prepare_inputs(args.model, args.data, sorted({b for b, _ in points}),
                             sorted({s for _, s in points}))
    write_json(out / "inputs.json", prompts)
    __import__("intel_cpu_models")
    import sglang as sgl
    import sgl_kernel
    import torch

    installed = Path(sgl.__file__).resolve().parent
    inspected = [
        installed / "srt/models/deepseek_v4.py",
        installed / "srt/layers/attention/dsv4/compressor.py",
        installed / "kernels/ops/attention/dsv4/gemm.py",
        installed / "srt/layers/attention/deepseek_v4_backend.py",
        Path(sgl_kernel.common_ops.__file__).resolve(),
    ]
    donor = installed / "kernels/aot/csrc/cpu"
    inspected.extend(sorted(donor.rglob("*.h")))
    inspected.extend([donor / "moe.cpp", donor / "moe_fp8.cpp"])
    identity["installed_sources_and_library"] = {str(p): digest(p) for p in inspected}
    identity["torch"] = torch.__version__
    identity["prompt_sha256"] = digest(out / "inputs.json")
    write_json(out / "identity.json", identity)

    engine = sgl.Engine(
        model_path=args.model, device="cpu", tp_size=1, dtype="bfloat16",
        load_format="auto", disable_overlap_schedule=True, trust_remote_code=True,
        mem_fraction_static=0.5, log_level="warning", disable_radix_cache=True,
        context_length=4096, max_total_tokens=131072, swa_full_tokens_ratio=0.5,
        chunked_prefill_size=512, watchdog_timeout=300, random_seed=0,
    )
    try:
        write_json(out / "runtime.json", engine.get_server_info())
        print("SERVING_SCREEN_READY", flush=True)
        for batch, length in points:
                inputs = prompts[length][:batch]
                expected_tokens = None
                measurements = []
                # Warm both arms; exclude their compilation and allocator transients.
                arms = [(f"warm-{on}", on, -1) for on in (False, True)]
                for rep in range(args.reps):
                    arms.extend((f"r{rep}-{on}", on, rep) for on in
                                ((False, True) if rep % 2 == 0 else (True, False)))
                if meaningful:
                    arms = [("meaningful", True, 0)]
                if trial:
                    arms = [(label, True, 0) for label in
                            ("trial-a", "trial-candidate", "trial-b")]
                for label, enabled, rep in arms:
                    case = f"b{batch}-s{length}-{label}"
                    if trial:
                        flushed = engine.flush_cache()
                        if flushed.success is not True:
                            raise RuntimeError("scheduler cache reset failed")
                    write_json(control, {"case": case, "enabled": enabled})
                    start = time.perf_counter()
                    responses = engine.generate(
                        input_ids=inputs,
                        sampling_params={"temperature": 0.0, "max_new_tokens": args.output_len,
                                         "ignore_eos": True},
                    )
                    elapsed = time.perf_counter() - start
                    write_json(out / f"{case}-responses.json", responses)
                    if not isinstance(responses, list) or len(responses) != batch:
                        raise RuntimeError("unexpected generate response count")
                    token_ids = [r.get("output_ids") for r in responses]
                    if not all(isinstance(ids, list) and len(ids) == args.output_len
                               and all(type(i) is int for i in ids) for ids in token_ids):
                        raise RuntimeError("missing generated token IDs")
                    if expected_tokens is None:
                        expected_tokens = token_ids
                    elif token_ids != expected_tokens and not trial:
                        raise RuntimeError(f"off/on/repeat generation changed for {case}")
                    events = []
                    for path in (out / "worker").glob("events.*.jsonl"):
                        events.extend(e for line in path.read_text().splitlines()
                                      if (e := json.loads(line))["case"] == case)
                    validate_events(events, case, enabled, 43, job=identity["job"],
                                    batch=batch, length=length, output_len=args.output_len,
                                    sparse_scope="trial.sparse" if trial else "native_wrapper.sparse_attend_masked")
                    if meaningful:
                        captures = {(e["phase"], c["layer"])
                                    for e in events for c in e.get("captures", [])}
                        if captures != {(p, l) for p in ("prefill", "decode") for l in (0, 2, 3)}:
                            raise RuntimeError(f"incomplete sparse capture: {captures}")
                    if meaningful or trial:
                        if any(len({r["layer"] for r in e["rows"] if r["op"] == op}) != 43
                               for e in events for op in ("moe.total", "moe.routed", "moe.shared",
                                                         "mhc.pre", "mhc.post", "attention.total")):
                            raise RuntimeError("incomplete lightweight boundary coverage")
                    row = {"case": case, "enabled": enabled, "rep": rep,
                           "client_s": elapsed, "output_ids": token_ids,
                           "phases": ({"status": "Use replay.json boundaries; captured forwards "
                                      "must not enter timing shares."} if meaningful
                                      else opportunity_summary(events))}
                    measurements.append(row)
                    write_json(out / f"{case}.json", row)
                    print(f"SERVING_SCREEN_CASE {case} client_s={elapsed:.3f}", flush=True)
                point = {"batch": batch, "input_len": length, "measurements": measurements}
                point["perturbation"] = {}
                if meaningful or trial:
                    point["perturbation"] = {"status": "not measured; diagnostic attribution only"}
                    identity["cases"].append(point)
                    write_json(out / "partial.json", identity)
                    continue
                for phase in ("prefill", "decode"):
                    medians = {
                        str(on): statistics.median(
                            r["phases"][phase]["forward_ns"] for r in measurements
                            if r["rep"] >= 0 and r["enabled"] == on
                        ) for on in (False, True)
                    }
                    ratio = medians["True"] / medians["False"]
                    point["perturbation"][phase] = {
                        "on_over_off": ratio, "diagnostic_only": abs(ratio - 1) > 0.05,
                        "median_phase_ns": medians,
                    }
                identity["cases"].append(point)
                write_json(out / "partial.json", identity)
        if any(digest(ROOT / rel) != sha for rel, sha in identity["sources"].items()):
            raise RuntimeError("screen source changed during execution")
        if any(digest(path) != sha for path, sha in identity["installed_sources_and_library"].items()):
            raise RuntimeError("installed dependency changed during execution")
        write_json(out / "complete.json", identity)
        print(f"SERVING_SCREEN_COMPLETE job={os.environ['SLURM_JOB_ID']}", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/scratch/bkaul/models/DeepSeek-V4-Flash")
    parser.add_argument("--data", default="/scratch/bkaul/gsm8k_test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meaningful", action="store_true",
                        help="One request per selected point, lightweight timers and sparse capture")
    parser.add_argument("--c4-trial", action="store_true",
                        help="Bounded baseline/candidate/baseline C4-prefill trial")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 3072])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=8)
    arguments = parser.parse_args()
    if (arguments.reps < 3 or arguments.output_len < 2 or not arguments.batches
            or min(arguments.batches) < 1 or min(arguments.lengths) < 1
            or max(arguments.lengths) + arguments.output_len > 4096):
        parser.error("invalid screen dimensions/repetitions")
    run(arguments)
