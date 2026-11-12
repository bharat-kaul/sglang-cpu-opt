#!/usr/bin/env python3
"""Actual-input FP32 sparse donor screen; numerical comparisons are not promotion gates."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch

from replay_serving_sparse import same_bytes, validate_sample


CAPTURE_ROOT = Path("/scratch/bkaul/dsv4-serving-screen/job-401206/results/worker/captures")
CAPTURE_HASHES = {
    0: "50a0e3171bbf1a8aad51a08ab8514fe42e43979cb0032cc0763858d4a8bf3b4c",
    2: "aebfb26d370a4790f6cedc46a2e5a136974ad317f902fdf90ba19c426ce8843b",
    3: "9553b3a890dadd979efec584c5498e1f3eed88a38508072e0b911503ca0bcf81",
}
NATIVE = Path("/scratch/bkaul/dsv4-serving-screen/job-401206/extensions/dsv4ck_sa/dsv4ck_sa.so")
NATIVE_HASH = "3a5ffa267cf282289c9daab92b345b10218cf130e5908ee422041998260c428f"
FROZEN = Path("/scratch/bkaul/dsv4-serving-screen/meaningful-20261110-bDqKQY/source")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shared_gemm(q, kv, mask, sink, scale):
    """Matched-operand transcription of _batched_mla_attend's native-branch _oracle."""
    n, h, d = q.shape
    k = kv.shape[0]
    scores = (q.reshape(n * h, d) @ kv.t()).reshape(n, h, k) * scale
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    sinks = sink.reshape(1, h, 1).float().expand(n, h, 1)
    probs = torch.nan_to_num(
        torch.cat([scores, sinks], dim=-1).softmax(dim=-1)[..., :-1], nan=0.0)
    return (probs.reshape(n * h, k) @ kv).reshape(n, h, d)


def shared_chunked(q, kv, mask, sink, scale, chunk=32):
    return torch.cat([shared_gemm(q[i:i + chunk], kv, mask[i:i + chunk], sink, scale)
                      for i in range(0, q.shape[0], chunk)], dim=0)


def gathered_donor(q, kv, mask, sink, scale, donor):
    """Group identical visible lengths; no padding logits or BF16 conversion."""
    counts = mask.sum(dim=1)
    out = torch.zeros_like(q)
    for count in counts.unique(sorted=True).tolist():
        if count == 0:
            continue
        rows = (counts == count).nonzero().flatten()
        # Bound replication scratch to 16 queries, including the long-C4 gather.
        for start in range(0, rows.numel(), 16):
            group = rows[start:start + 16]
            indices = mask.index_select(0, group).nonzero()[:, 1].reshape(group.numel(), count)
            keys = kv[indices]
            result = donor(q.index_select(0, group), keys, sink, scale)
            out.index_copy_(0, group, result)
    return out


def metrics(candidate, reference):
    if (candidate.shape != reference.shape or candidate.dtype != reference.dtype
            or candidate.device != reference.device
            or not torch.isfinite(candidate).all() or not torch.isfinite(reference).all()):
        raise RuntimeError("invalid candidate/reference tensor")
    delta = (candidate.double() - reference.double()).reshape(-1)
    norm = float(torch.linalg.vector_norm(reference.double()))
    return {
        "byte_equal": same_bytes(candidate, reference),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta)) / norm if norm else None,
        "bf16_element_agreement": float((candidate.bfloat16() == reference.bfloat16()).float().mean()),
    }


def measure(sample, native):
    q, kv, mask, sink, captured = (sample[k] for k in ("q", "kv", "mask", "sink", "output"))
    args = (q, kv, mask, sink, sample["scale"])
    arms = {
        "online_native": lambda: native.sparse_attend_masked(*args),
        "shared_fp32_gemm": lambda: shared_gemm(*args),
        "shared_fp32_gemm_chunk32": lambda: shared_chunked(*args),
        "gathered_fp32_donor": lambda: gathered_donor(*args, native.sparse_attend_fp32bmm),
        "gathered_fp32_donor_hx": lambda: gathered_donor(*args, native.sparse_attend_fp32bmm_hx),
    }
    oracle = shared_gemm(*args)
    metrics(oracle, captured)
    samples = {name: [] for name in arms}
    numerical = {}
    first_outputs = {}
    for rep in range(-1, 3):
        names = list(arms) if rep == -1 or rep % 2 == 0 else list(reversed(arms))
        for name in names:
            start = time.perf_counter_ns()
            result = arms[name]()
            elapsed = time.perf_counter_ns() - start
            if not 0 < elapsed < 30_000_000_000:
                raise RuntimeError(f"invalid or over-budget sample: {name}")
            comparison = {"vs_online": metrics(result, captured), "vs_shared_oracle": metrics(result, oracle)}
            if name == "online_native" and not comparison["vs_online"]["byte_equal"]:
                raise RuntimeError("native replay does not reproduce captured output bytes")
            if rep == -1:
                numerical[name] = comparison
                first_outputs[name] = result.clone()
            elif not same_bytes(result, first_outputs[name]):
                raise RuntimeError(f"nondeterministic repeated output: {name}")
            else:
                samples[name].append(elapsed)
    medians = {k: statistics.median(v) for k, v in samples.items()}
    return {
        "layer": sample["layer"], "capture_identity": sample["identity"],
        "q_shape": list(q.shape), "kv_shape": list(kv.shape),
        "visible_per_query": mask.sum(dim=1).tolist(),
        "samples_ns": samples, "median_ns": medians,
        "range_ns": {k: [min(v), max(v)] for k, v in samples.items()},
        "speedup_vs_online": {k: medians["online_native"] / v for k, v in medians.items()},
        "numerical": numerical, "repeated_output_bytes_equal": True,
        "accuracy_disposition": "Non-byte-identical candidates are numerically unapproved, "
                                "regardless of small errors or speed. No serving promotion.",
    }


def run(output):
    if output.exists():
        raise RuntimeError("refusing to overwrite sparse donor screen")
    if torch.get_num_threads() != 32 or sorted(os.sched_getaffinity(0)) != list(range(32)):
        raise RuntimeError("fixed32 CPU0-31 required")
    if sha(NATIVE) != NATIVE_HASH:
        raise RuntimeError("captured native binary changed")
    # Bind the reused binary to its recorded build source, rather than a current source label.
    build = NATIVE.with_name("build.ninja").read_text()
    expected_cpp = str(FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp")
    if expected_cpp not in build:
        raise RuntimeError("native build does not refer to captured frozen source")
    expected_event = {
        "job": "401206", "pid": 1971307, "sequence": 5, "case": "b1-s3072-meaningful",
        "phase": "prefill", "tokens": 512, "seq_lens": [3072],
        "threads_before": 32, "affinity": [0],
    }
    captures = []
    for layer, expected_hash in CAPTURE_HASHES.items():
        path = CAPTURE_ROOT / f"1971307-5-L{layer}.pt"
        if sha(path) != expected_hash:
            raise RuntimeError("capture hash mismatch")
        sample = torch.load(path, weights_only=True, map_location="cpu")
        validate_sample(sample, expected_event, layer)
        captures.append(sample)
    spec = importlib.util.spec_from_file_location("dsv4ck_sa", NATIVE)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    source_paths = [
        Path(__file__), Path(__file__).with_name("replay_serving_sparse.py"),
        Path(expected_cpp), FROZEN / "plugin/kernels/dsa_pilot/avx512_exp.h",
        FROZEN / "plugin/intel_cpu_models/_dsv4_cpu_infra.py",
        NATIVE.with_name("build.ninja"), Path(torch.__file__).parent / "lib/libtorch_cpu.so",
    ]
    sources = {str(p): sha(p) for p in source_paths}
    report = {
        "job": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename, "pid": os.getpid(),
        "threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__, "capture_job": "401206", "capture_sha256": CAPTURE_HASHES,
        "native_path": str(NATIVE), "native_sha256": NATIVE_HASH, "sources": sources,
        "environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")},
        "scope": "Actual final long-prefill chunk, SWA/C4/C128. Three alternating reused-buffer "
                 "trials after one warm call each, one process. Includes selection/gather/masks/softmax "
                 "and output assembly; excludes upstream common FP32 preparation shared by all arms. "
                 "Not a streaming roofline, whole-phase timing, or task accuracy test.",
        "cases": [],
    }
    output.mkdir(parents=True)
    for sample in captures:
        row = measure(sample, native)
        report["cases"].append(row)
        (output / "partial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"SPARSE_DONOR L{sample['layer']} median_ms="
              f"{ {k: round(v/1e6, 4) for k, v in row['median_ns'].items()} } "
              f"speedup={row['speedup_vs_online']}", flush=True)
    if any(sha(p) != value for p, value in sources.items()) or sha(NATIVE) != NATIVE_HASH:
        raise RuntimeError("source or binary changed during evaluation")
    (output / "complete.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("LONG_PREFILL_SPARSE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
