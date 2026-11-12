#!/usr/bin/env python3
"""Replay captured FP32 serving calls, not full requests or a new numerical path."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from screen_serving_opportunities import BASE_COMMIT, MEANINGFUL_POINTS, ROOT, digest, write_json


def validate_sample(sample, event, layer):
    expected = {k: event[k] for k in ("job", "pid", "sequence", "case", "phase",
                                     "tokens", "seq_lens", "threads_before", "affinity")}
    if sample["identity"] != expected or sample["layer"] != layer:
        raise RuntimeError("capture identity does not match worker event")
    if type(sample["scale"]) is not float or not math.isfinite(sample["scale"]) or sample["scale"] <= 0:
        raise RuntimeError("invalid captured scale")
    for name in ("q", "kv", "mask", "sink", "output"):
        t = sample[name]
        dtype = torch.bool if name == "mask" else torch.float32
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.dtype != dtype:
            raise RuntimeError(f"invalid capture tensor: {name}")
        if not torch.isfinite(t).all():
            raise RuntimeError(f"nonfinite capture tensor: {name}")
    q, kv, mask, sink, out = (sample[k] for k in ("q", "kv", "mask", "sink", "output"))
    if (q.ndim != 3 or kv.ndim != 2 or mask.ndim != 2 or sink.ndim != 1
            or min(q.shape) <= 0 or kv.shape[0] <= 0
            or q.shape[0] != event["tokens"] or q.shape[1:] != (64, 512)
            or kv.shape[1] != q.shape[2] or mask.shape != (q.shape[0], kv.shape[0])
            or sink.shape != (q.shape[1],) or out.shape != q.shape):
        raise RuntimeError("capture shape/coordinate mismatch")


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def compact_pool(kv, mask):
    # Ascending union indices preserve each query's original visible-key reduction order.
    indices = mask.any(dim=0).nonzero().flatten()
    return kv.index_select(0, indices), mask.index_select(1, indices)


def replay_sample(sample, native):
    q, kv, mask, sink, reference = (sample[k] for k in ("q", "kv", "mask", "sink", "output"))
    scale = sample["scale"]
    compact_kv, compact_mask = compact_pool(kv, mask)
    empty_mask = torch.zeros_like(mask)

    def compact_with_preparation():
        k, m = compact_pool(kv, mask)
        return native(q, k, m, sink, scale)

    arms = {
        "original": lambda: native(q, kv, mask, sink, scale),
        "compact_prepared": lambda: native(q, compact_kv, compact_mask, sink, scale),
        "compact_including_preparation": compact_with_preparation,
        "all_masked_diagnostic": lambda: native(q, kv, empty_mask, sink, scale),
    }
    empty_reference = torch.zeros_like(reference)
    samples = {key: [] for key in arms}
    for name, fn in arms.items():
        output = fn()
        expected = empty_reference if name == "all_masked_diagnostic" else reference
        if not same_bytes(output, expected):
            raise RuntimeError(f"captured-output byte mismatch: {name}")
    for rep in range(3):
        for name in list(arms) if rep % 2 == 0 else list(reversed(arms)):
            start = time.perf_counter_ns()
            output = arms[name]()
            elapsed = time.perf_counter_ns() - start
            expected = empty_reference if name == "all_masked_diagnostic" else reference
            if not torch.isfinite(output).all() or not same_bytes(output, expected):
                raise RuntimeError(f"replay changed output: {name}")
            samples[name].append(elapsed)
    visible = mask.sum(dim=1)
    return {
        "identity": sample["identity"], "layer": sample["layer"],
        "q_shape": list(q.shape), "pool_rows": kv.shape[0],
        "union_visible_rows": compact_kv.shape[0], "visible_per_query": visible.tolist(),
        "head_mask_tests": q.shape[0] * q.shape[1] * kv.shape[0],
        "head_visible_visits": int(visible.sum()) * q.shape[1],
        "logical_visible_flops_excluding_exp": int(visible.sum()) * q.shape[1] * (5 * q.shape[2]),
        "samples_ns": samples,
        "median_ns": {key: statistics.median(values) for key, values in samples.items()},
        "range_ns": {key: [min(values), max(values)] for key, values in samples.items()},
        "output_bytes_match": True,
        "scope": "Three alternating replay trials in one process, reused buffers; not a DRAM roofline "
                 "or serving gain. All-masked control changes work/branch behavior, not an exact "
                 "subtraction of mask-scan cost. Prepared compaction excludes preparation.",
    }


def summarize_boundaries(events):
    result = {}
    for case in sorted({e["case"] for e in events}):
        result[case] = {}
        for phase in ("prefill", "decode"):
            group = [e for e in events if e["case"] == case and e["phase"] == phase]
            # First prefill may compile private kernels; captures perturb their whole forward.
            clean = [e for i, e in enumerate(group)
                     if not e.get("captures") and not (phase == "prefill" and i == 0)]
            wall = sum(e["forward_ns"] for e in clean)
            ops = {}
            for event in clean:
                if sum(r["exclusive_ns"] for r in event["rows"]) > event["forward_ns"]:
                    raise RuntimeError("exclusive scopes exceed forward time")
                for row in event["rows"]:
                    bucket = ops.setdefault(row["op"], {"inclusive_ns": 0, "exclusive_ns": 0, "calls": 0})
                    for key in bucket:
                        bucket[key] += row[key]
            result[case][phase] = {
                "sequences": [e["sequence"] for e in clean],
                "forward_ns": [e["forward_ns"] for e in clean],
                "ops": {name: {**v, "inclusive_share": v["inclusive_ns"] / wall,
                               "exclusive_share": v["exclusive_ns"] / wall}
                        for name, v in ops.items()},
                "status": "diagnostic; no matched off/on perturbation check" if clean else "no clean forwards",
            }
    return result


def run(results):
    if (results / "replay.json").exists():
        raise RuntimeError("refusing to overwrite replay result")
    identity = json.loads((results / "complete.json").read_text())
    if identity["job"] != os.environ["SLURM_JOB_ID"] or identity["base_commit"] != BASE_COMMIT:
        raise RuntimeError("wrong serving job/base")
    for relative, sha in identity["sources"].items():
        if digest(ROOT / relative) != sha:
            raise RuntimeError("capture source changed")
    for path, sha in identity["installed_sources_and_library"].items():
        if digest(path) != sha:
            raise RuntimeError("capture installed dependency changed")
    if digest(results / "inputs.json") != identity["prompt_sha256"]:
        raise RuntimeError("capture prompts changed")
    if [(p["batch"], p["input_len"]) for p in identity["cases"]] != MEANINGFUL_POINTS:
        raise RuntimeError("incomplete meaningful point coverage")
    events = [json.loads(line) for path in sorted((results / "worker").glob("events.*.jsonl"))
              for line in path.read_text().splitlines()]
    if len({(e["pid"], e["sequence"]) for e in events}) != len(events):
        raise RuntimeError("duplicate forward identity")
    if {e["case"] for e in events} != {f"b{b}-s{s}-meaningful" for b, s in MEANINGFUL_POINTS}:
        raise RuntimeError("unexpected worker cases")
    from screen_serving_opportunities import validate_events
    expected = set()
    for point in identity["cases"]:
        case = f'b{point["batch"]}-s{point["input_len"]}-meaningful'
        selected = [e for e in events if e["case"] == case]
        validate_events(selected, case, True, 43, job=identity["job"], batch=point["batch"],
                        length=point["input_len"], output_len=5)
        expected.update((case, phase, layer) for phase in ("prefill", "decode") for layer in (0, 2, 3))
    captures = [(e, c) for e in events for c in e.get("captures", [])]
    actual = [(e["case"], e["phase"], c["layer"]) for e, c in captures]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("missing/duplicate capture coverage")
    torch.set_num_threads(32)
    from intel_cpu_models import _dsa_ckernels as ck
    module = ck._mod("sa", "sparse_attend.cpp")
    native = module.sparse_attend_masked
    report = {
        "job": identity["job"], "base_commit": BASE_COMMIT, "node": os.uname().nodename,
        "pid": os.getpid(), "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)), "capture_identity_sha256": digest(results / "identity.json"),
        "replay_source_sha256": digest(__file__),
        "native_library": str(module.__file__), "native_sha256": digest(module.__file__),
        "boundaries": summarize_boundaries(events), "sparse": [],
    }
    for event, capture in captures:
        path = Path(capture["file"])
        if path.resolve().parent != (results / "worker/captures").resolve() or digest(path) != capture["sha256"]:
            raise RuntimeError("capture path/hash mismatch")
        sample = torch.load(path, weights_only=True, map_location="cpu")
        validate_sample(sample, event, capture["layer"])
        row = replay_sample(sample, native)
        row["capture_sha256"] = capture["sha256"]
        report["sparse"].append(row)
        write_json(results / "replay-partial.json", report)
        print(f'SPARSE_REPLAY {event["case"]} {event["phase"]} L{capture["layer"]} '
              f'pool={row["pool_rows"]} union={row["union_visible_rows"]} '
              f'median_ns={row["median_ns"]}', flush=True)
    write_json(results / "replay.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    run(parser.parse_args().results.resolve())
