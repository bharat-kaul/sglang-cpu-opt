#!/usr/bin/env python3
"""Reconcile the bounded C4 trial; keep numerical disposition separate from timing."""

import argparse
import json
import os
from pathlib import Path
import re

import torch

from bench_long_prefill_sparse import metrics
from screen_serving_opportunities import (
    BASE_COMMIT, EXPECTED_ENV, ROOT, digest, validate_events, write_json,
)


CASES = ("b1-s3072-trial-a", "b1-s3072-trial-candidate", "b1-s3072-trial-b")


def load_payload(event, root, config):
    row = event["trial"]
    path = Path(row["tensors"])
    if path.resolve().parent != (root / "worker/trial").resolve() or digest(path) != row["tensors_sha256"]:
        raise RuntimeError("trial tensor path/hash mismatch")
    data = torch.load(path, weights_only=True, map_location="cpu")
    if any(data[k] != event[k] for k in ("job", "pid", "sequence", "case", "phase")):
        raise RuntimeError("trial tensor identity mismatch")
    if set(data["hidden_last_row"]) != set(range(43)):
        raise RuntimeError("missing hidden layer rows")
    for tensor in data["hidden_last_row"].values():
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim not in (2, 3)
                or tensor.shape[0] != 1 or tensor.shape[-1] != config["hidden_size"]
                or not tensor.is_floating_point() or not torch.isfinite(tensor).all()
                or (tensor.ndim == 3 and tensor.shape[1] != config["hc_mult"])):
            raise RuntimeError("invalid hidden layer tensor")
    logits = data["logits"]
    required = event["phase"] == "decode" or event["seq_lens"] == [3072]
    if logits is None:
        if required:
            raise RuntimeError("missing final logits")
    elif (not isinstance(logits, torch.Tensor) or logits.shape != (1, config["vocab_size"])
          or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise RuntimeError("invalid logits")
    return data


def compare_tensors(a, b, logits=False):
    if a is None or b is None:
        if a is not None or b is not None:
            raise RuntimeError("logit presence differs")
        return {"byte_equal": True, "present": False}
    out = metrics(a, b)
    if logits:
        out["argmax_equal"] = bool(torch.equal(a.argmax(-1), b.argmax(-1)))
        out["argmax_ids"] = [int(x.argmax(-1).item()) for x in (a, b)]
        out["top1_minus_top2"] = [
            float(x.double().topk(2).values.diff().neg().item()) for x in (a, b)]
    return out


def analyze(root):
    if (root / "trial-analysis.json").exists():
        raise RuntimeError("refusing to overwrite trial analysis")
    identity = json.loads((root / "complete.json").read_text())
    if identity["job"] != os.environ["SLURM_JOB_ID"] or identity["base_commit"] != BASE_COMMIT:
        raise RuntimeError("wrong trial job/base")
    if identity["environment"] != EXPECTED_ENV or identity["node"] != os.uname().nodename:
        raise RuntimeError("wrong trial runtime/node")
    initial = json.loads((root / "identity.json").read_text())
    if {k: v for k, v in identity.items() if k != "cases"} != {
            k: v for k, v in initial.items() if k != "cases"}:
        raise RuntimeError("trial identities differ")
    expected_sources = {str(p.relative_to(ROOT)) for p in (ROOT / "plugin").rglob("*")
                        if p.suffix in {".py", ".cpp", ".h"}}
    if not expected_sources or set(identity["sources"]) != expected_sources:
        raise RuntimeError("source inventory mismatch")
    for relative, expected in identity["sources"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError("trial source changed")
    if not identity["installed_sources_and_library"]:
        raise RuntimeError("missing dependency inventory")
    for path, expected in identity["installed_sources_and_library"].items():
        if digest(path) != expected:
            raise RuntimeError("trial dependency changed")
    if digest(root / "inputs.json") != identity["prompt_sha256"]:
        raise RuntimeError("trial prompt changed")
    config_path = Path(identity["model_config_path"])
    if digest(config_path) != identity["model_config_sha256"]:
        raise RuntimeError("model config changed")
    config = json.loads(config_path.read_text())
    if config["num_hidden_layers"] != 43 or config["num_attention_heads"] != 64 or config["head_dim"] != 512:
        raise RuntimeError("unexpected trial model")
    events = [json.loads(line) for path in (root / "worker").glob("events.*.jsonl")
              for line in path.read_text().splitlines()]
    if len({(e["pid"], e["sequence"]) for e in events}) != len(events):
        raise RuntimeError("duplicate forward")
    if len({e["pid"] for e in events}) != 1 or {e["sequence"] for e in events} != set(range(21)):
        raise RuntimeError("trial must run sequentially in one worker")
    if {e["case"] for e in events} != set(CASES):
        raise RuntimeError("unexpected trial cases")
    groups = {case: sorted([e for e in events if e["case"] == case], key=lambda e: e["sequence"])
              for case in CASES}
    for arm, (case, group) in enumerate(groups.items()):
        validate_events(group, case, True, 43, job=identity["job"], batch=1,
                        length=3072, output_len=2, sparse_scope="trial.sparse")
        if len(group) != 7:
            raise RuntimeError("expected six prefill chunks and one decode forward")
        for i, e in enumerate(group):
            if (e["sequence"] != arm * 7 + i or e["affinity"] != [0]
                    or e["phase"] != ("prefill" if i < 6 else "decode")
                    or e["tokens"] != (512 if i < 6 else 1)
                    or e["seq_lens"] != ([512 * (i + 1)] if i < 6 else [3073])):
                raise RuntimeError("unexpected worker affinity/chunk schedule")
            trial = e["trial"]
            if (not isinstance(trial["req_pool_indices"], list) or len(trial["req_pool_indices"]) != 1
                    or type(trial["req_pool_indices"][0]) is not int or trial["req_pool_indices"][0] < 0):
                raise RuntimeError("invalid request pool coordinate")
            for key in ("positions_hash", "input_ids_hash", "out_cache_loc_hash", "tensors_sha256"):
                if not isinstance(trial[key], str) or re.fullmatch("[0-9a-f]{64}", trial[key]) is None:
                    raise RuntimeError("invalid trial tensor hash")
            if trial["state_reset"] is not (i == 0) or trial["candidate"] is not (case == CASES[1]):
                raise RuntimeError("incorrect trial arm/reset")
            calls = trial["calls"]
            if len(calls) != 43 or {c["layer"] for c in calls} != set(range(43)):
                raise RuntimeError("missing/duplicate sparse trial calls")
            for c in calls:
                for key in (("mask_hash", "q_hash", "kv_hash") if c["layer"] == 0 else ("mask_hash",)):
                    if not isinstance(c[key], str) or re.fullmatch("[0-9a-f]{64}", c[key]) is None:
                        raise RuntimeError("invalid sparse input hash")
                ratio = config["compress_ratios"][c["layer"]]
                selected = case == CASES[1] and e["phase"] == "prefill" and ratio == 4
                if c["ratio"] != ratio or c["candidate"] is not selected:
                    raise RuntimeError("candidate used outside C4 prefill or missing")
                if (c["q_shape"] != [e["tokens"], 64, 512]
                        or len(c["kv_shape"]) != 2 or c["kv_shape"][1] != 512
                        or type(c["kv_shape"][0]) is not int or c["kv_shape"][0] < 1):
                    raise RuntimeError("invalid sparse operand dimensions")
            timings = [r for r in e["rows"] if r["op"] == "trial.sparse"]
            if len(timings) != 43 or {r["layer"] for r in timings} != set(range(43)):
                raise RuntimeError("missing trial sparse timings")
            by_layer = {c["layer"]: c for c in calls}
            if any(r["calls"] != 1 or r["metadata"] != {
                    "ratio": by_layer[r["layer"]]["ratio"],
                    "candidate": by_layer[r["layer"]]["candidate"]} for r in timings):
                raise RuntimeError("sparse timing/dispatch mismatch")
    rows = []
    baseline_equal = True
    for a, candidate, b in zip(*(groups[case] for case in CASES)):
        for other in (candidate, b):
            for key in ("phase", "tokens", "seq_lens"):
                if a[key] != other[key]:
                    raise RuntimeError("forward schedules differ")
            for key in ("positions_hash", "req_pool_indices", "out_cache_loc_hash"):
                if a["trial"][key] != other["trial"][key]:
                    raise RuntimeError(f"allocator/position schedules differ: {key}")
            if a["phase"] == "prefill" and a["trial"]["input_ids_hash"] != other["trial"]["input_ids_hash"]:
                raise RuntimeError("prefill inputs differ")
        ac, cc, bc = ({c["layer"]: c for c in e["trial"]["calls"]} for e in (a, candidate, b))
        if ac != bc:
            baseline_equal = False
        if a["trial"]["input_ids_hash"] != b["trial"]["input_ids_hash"]:
            baseline_equal = False
        for layer in range(43):
            if any(ac[layer][key] != cc[layer][key] for key in ("q_shape", "kv_shape")):
                raise RuntimeError("candidate sparse shapes differ")
        # Changed decode input after divergent generation is recorded, not called matched.
        inputs_equal = a["trial"]["input_ids_hash"] == candidate["trial"]["input_ids_hash"]
        if inputs_equal and any(ac[0][key] != cc[0][key] for key in ("q_hash", "kv_hash", "mask_hash")):
            raise RuntimeError("layer0 anchors differ despite matched input")
        ap, cp, bp = (load_payload(e, root, config) for e in (a, candidate, b))
        control = [compare_tensors(ap["hidden_last_row"][layer], bp["hidden_last_row"][layer])
                   for layer in range(43)]
        control_logits = compare_tensors(ap["logits"], bp["logits"], logits=True)
        baseline_equal &= all(x["byte_equal"] for x in control) and control_logits["byte_equal"]
        row = {
            "phase": a["phase"], "seq_lens": a["seq_lens"],
            "sequences": [e["sequence"] for e in (a, candidate, b)],
            "candidate_input_ids_match": inputs_equal,
            "baseline_hidden_bytes_equal": all(x["byte_equal"] for x in control),
            "baseline_logits": control_logits,
            "candidate_hidden_last_row": {
                layer: compare_tensors(cp["hidden_last_row"][layer], ap["hidden_last_row"][layer])
                for layer in range(43)},
            "candidate_logits": compare_tensors(cp["logits"], ap["logits"], logits=True),
            "candidate_mask_changed_layers": [layer for layer in range(43)
                                              if ac[layer]["mask_hash"] != cc[layer]["mask_hash"]],
            "forward_ns": [e["forward_ns"] for e in (a, candidate, b)],
        }
        rows.append(row)
    responses = [json.loads((root / f"{case}-responses.json").read_text()) for case in CASES]
    if any(not isinstance(response, list) or len(response) != 1 for response in responses):
        raise RuntimeError("invalid trial responses")
    tokens = [[r["output_ids"] for r in response] for response in responses]
    if any(len(ids[0]) != 2 or any(type(t) is not int or not 0 <= t < config["vocab_size"]
                                 for t in ids[0]) for ids in tokens):
        raise RuntimeError("invalid generated token IDs")
    for case, ids in zip(CASES, tokens):
        payloads = [load_payload(e, root, config) for e in groups[case][-2:]]
        if ids[0] != [int(p["logits"].argmax(-1).item()) for p in payloads]:
            raise RuntimeError("response tokens disagree with captured greedy logits")
    baseline_equal &= tokens[0] == tokens[2]
    # Exclude first chunk in all arms: the initial baseline may compile native kernels.
    times = [sum(e["forward_ns"] for e in groups[case][1:] if e["phase"] == "prefill")
             for case in CASES]
    sparse_times = [
        sum(r["inclusive_ns"] for e in groups[case][1:6] for r in e["rows"]
            if r["op"] == "trial.sparse" and r["metadata"]["ratio"] == 4) for case in CASES]
    report = {
        "job": identity["job"], "base_commit": BASE_COMMIT,
        "node": identity["node"], "environment": identity["environment"],
        "complete_sha256": digest(root / "complete.json"),
        "event_sha256": {str(p): digest(p) for p in (root / "worker").glob("events.*.jsonl")},
        "response_sha256": {case: digest(root / f"{case}-responses.json") for case in CASES},
        "capture_identity_sha256": digest(root / "identity.json"),
        "analyzer_sha256": digest(__file__),
        "baseline_repeatability_bytes": baseline_equal,
        "generated_token_ids": tokens, "candidate_tokens_match": tokens[0] == tokens[1],
        "forwards": rows, "prefill_excluding_first_chunk_ns": times,
        "c4_sparse_excluding_first_chunk_ns": sparse_times,
        "observed_prefill_speedup_vs_baseline_a": times[0] / times[1] if baseline_equal else None,
        "observed_prefill_speedup_vs_baseline_b": times[2] / times[1] if baseline_equal else None,
        "baseline_b_over_a": times[2] / times[0],
        "scope": "One baseline/candidate/baseline sandwich, one B1S3072 prompt, G2. "
                 "Five matched prefill chunks after symmetric first-chunk exclusion. Timed "
                 "forward includes identical tracing hooks but no tensor serialization. "
                 "Hidden comparisons sample the last token at all43 layers; logits are full. "
                 "No independent replicas, task accuracy, decode speedup or default promotion.",
        "promotion": "BLOCKED: diagnostic numerical and timing evidence only.",
    }
    write_json(root / "trial-analysis.json", report)
    print(f"C4_TRIAL_ANALYZED baseline_repeatable={baseline_equal} "
          f"tokens_match={report['candidate_tokens_match']} prefill_ns={times}", flush=True)
    if not baseline_equal:
        raise RuntimeError("baseline repeatability failed; timing ratios withheld")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    analyze(parser.parse_args().results.resolve())
