"""Reconcile retained job401145 partials; this is not a general acceptance gate."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from screen_serving_opportunities import EXPECTED_ENV, opportunity_summary, validate_events


RUN = Path("/scratch/bkaul/dsv4-serving-screen/job-401145")
FROZEN = Path("/scratch/bkaul/dsv4-serving-screen/main-20261110T061822Z/source")
BASE = "34fe50f1aca4dc9d1ff17111a38ff5f323d7b106"
MANIFEST_SHA = "ff4a027a7c60c1d60edf6c9790fc8c3e514f1a6c8868e3888b2709b1b67bd5ae"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def phase_stats(events, phase):
    selected = [e for e in events if e["phase"] == phase]
    wall = sum(e["forward_ns"] for e in selected)
    inclusive, exclusive, calls = Counter(), Counter(), Counter()
    shapes = defaultdict(set)
    for event in selected:
        if sum(r["exclusive_ns"] for r in event["rows"]) > event["forward_ns"]:
            raise RuntimeError("exclusive scopes exceed forward duration")
        for row in event["rows"]:
            name = row["op"]
            inclusive[name] += row["inclusive_ns"]
            exclusive[name] += row["exclusive_ns"]
            calls[name] += row["calls"]
            shapes[name].add(json.dumps(row["metadata"], sort_keys=True))
    return {
        "wall_ns": wall,
        "forwards": len(selected),
        "forward_tokens": dict(Counter(e["tokens"] for e in selected)),
        "inclusive_share": {k: v / wall for k, v in inclusive.items()},
        "exclusive_share": {k: v / wall for k, v in exclusive.items()},
        "calls": dict(calls),
        "shapes": {k: [json.loads(v) for v in sorted(vs)] for k, vs in shapes.items()},
    }


def analyze():
    root = RUN / "results"
    identity = json.loads((root / "identity.json").read_text())
    if (identity["job"] != "401145" or identity["base_commit"] != BASE
            or identity["environment"] != EXPECTED_ENV):
        raise RuntimeError("unexpected run identity/runtime")
    manifest = FROZEN.parent / "source-sha256.txt"
    if digest(manifest) != MANIFEST_SHA:
        raise RuntimeError("unexpected frozen manifest")
    for line in manifest.read_text().splitlines():
        sha, relative = line.split("  ", 1)
        if digest(FROZEN / relative) != sha:
            raise RuntimeError(f"frozen source drift: {relative}")
    for relative, sha in identity["sources"].items():
        if digest(FROZEN / relative) != sha:
            raise RuntimeError(f"producer source mismatch: {relative}")
    for path, sha in identity["installed_sources_and_library"].items():
        if digest(Path(path)) != sha:
            raise RuntimeError(f"installed dependency drift: {path}")
    if digest(root / "inputs.json") != identity["prompt_sha256"]:
        raise RuntimeError("prompt drift")
    if (RUN / "exit_code").read_text().strip() != "1" or (root / "complete.json").exists():
        raise RuntimeError("unexpected terminal artifacts for the failed screen")
    log = Path("/scratch/bkaul/dsv4_serving_screen_401145.log")
    if "off/on/repeat generation changed for b32-s3072-warm-True" not in log.read_text():
        raise RuntimeError("expected failure reason absent")

    raw_files = sorted((root / "worker").glob("events.*.jsonl"))
    if len(raw_files) != 1:
        raise RuntimeError("expected one TP1 worker file")
    by_case = defaultdict(list)
    seen = set()
    for path in raw_files:
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                key = event["pid"], event["sequence"]
                if key in seen:
                    raise RuntimeError("duplicate global forward identity")
                seen.add(key)
                by_case[event["case"]].append(event)
    if {pid for pid, _ in seen} != {3839623}:
        raise RuntimeError("unexpected worker PID")

    expected_cases = set()
    points = []
    artifact_paths = [root / "identity.json", root / "inputs.json", root / "runtime.json",
                      RUN / "exit_code", log, manifest, *raw_files]
    for batch, length in ((1, 128), (1, 3072), (32, 128), (32, 3072)):
        stem = f"b{batch}-s{length}"
        labels = ["warm-False", "warm-True"]
        if (batch, length) != (32, 3072):
            labels += [f"r{rep}-{on}" for rep in range(3) for on in (False, True)]
        records, outputs = {}, {}
        for label in labels:
            case = f"{stem}-{label}"
            expected_cases.add(case)
            enabled = label.endswith("True")
            events = by_case[case]
            validate_events(events, case, enabled, 43, job="401145", batch=batch,
                            length=length, output_len=8)
            response_path = root / f"{case}-responses.json"
            artifact_paths.append(response_path)
            responses = json.loads(response_path.read_text())
            tokens = [r["output_ids"] for r in responses]
            if (len(tokens) != batch or any(len(row) != 8
                    or any(type(t) is not int or t < 0 for t in row) for row in tokens)):
                raise RuntimeError("invalid output token records")
            outputs[label] = tokens
            saved_path = root / f"{case}.json"
            if saved_path.exists():
                saved = json.loads(saved_path.read_text())
                artifact_paths.append(saved_path)
                if (saved["output_ids"] != tokens
                        or saved["phases"] != opportunity_summary(events)):
                    raise RuntimeError("saved summary disagrees with raw events")
            elif case != "b32-s3072-warm-True":
                raise RuntimeError(f"unexpected missing summary: {case}")
            records[label] = {phase: phase_stats(events, phase)
                              for phase in ("prefill", "decode")}
        reference = outputs["warm-False"]
        mismatches = {
            label: [{"row": i, "first_different_token": next(
                j for j, (a, b) in enumerate(zip(ref, row)) if a != b)}
                for i, (ref, row) in enumerate(zip(reference, tokens)) if row != ref]
            for label, tokens in outputs.items()
        }
        coordinate_match = all(
            [(e["phase"], e["tokens"], e["seq_lens"]) for e in by_case[f"{stem}-{label}"]]
            == [(e["phase"], e["tokens"], e["seq_lens"]) for e in by_case[f"{stem}-warm-False"]]
            for label in labels
        )
        parity = not any(mismatches.values())
        expected_parity = (batch, length) != (32, 3072)
        if parity != expected_parity:
            raise RuntimeError("unexpected parity disposition")
        point = {
            "batch": batch, "input_len": length, "output_parity": parity,
            "coordinate_match": coordinate_match, "mismatches": mismatches,
            "records": records, "phase_summary": {},
        }
        for phase in ("prefill", "decode"):
            chosen = [f"r{r}" for r in range(3)] if expected_parity else ["warm"]
            off = [records[f"{label}-False"][phase] for label in chosen]
            on = [records[f"{label}-True"][phase] for label in chosen]
            ratios = [b["wall_ns"] / a["wall_ns"] for a, b in zip(off, on)]
            ops = sorted({op for record in on for op in record["inclusive_share"]})
            split = {}
            for kind in ("inclusive_share", "exclusive_share"):
                split[kind] = {
                    op: distribution([record[kind].get(op, 0) for record in on])
                    for op in ops
                }
            point["phase_summary"][phase] = {
                "request_pairs": len(chosen),
                "off_ms_per_forward": distribution(
                    [e["wall_ns"] / e["forwards"] / 1e6 for e in off]),
                "on_ms_per_forward": distribution(
                    [e["wall_ns"] / e["forwards"] / 1e6 for e in on]),
                "paired_on_over_off": distribution(ratios),
                "diagnostic_only": not parity or not coordinate_match
                                   or any(abs(r - 1) > 0.05 for r in ratios),
                **split,
            }
        points.append(point)
    if set(by_case) != expected_cases:
        raise RuntimeError("unexpected or incomplete case set")
    return {
        "job": "401145", "base_commit": BASE, "worker_pid": 3839623,
        "node": identity["node"], "environment": identity["environment"],
        "scope": "Partial failed run; three complete points and one parity-failed warm pair. "
                 "One resident process, no independent-process performance acceptance.",
        "estimator": "Median/min/max of per-request phase shares or paired ratios; "
                     "per-forward times are each request's phase total divided by its forward count.",
        "posthoc_artifact_sha256": {str(p): digest(p) for p in artifact_paths},
        "posthoc_private_library_sha256": {
            str(p): digest(p) for p in sorted((RUN / "extensions").rglob("*.so"))},
        "points": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for point in result["points"]:
        for phase, values in point["phase_summary"].items():
            print(point["batch"], point["input_len"], phase,
                  "parity", point["output_parity"],
                  "coordinates", point["coordinate_match"],
                  "on/off", values["paired_on_over_off"],
                  "diagnostic_only", values["diagnostic_only"])
