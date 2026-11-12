#!/usr/bin/env python3
"""Shape-faithful synthetic feasibility screen; not an inference benchmark."""

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import types

import torch

from compression_batch_diagnostic import accumulate_batched, accumulate_scalar
from test_compression_batch_diagnostic import common_inputs, equal_tree


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_case(batch, ratio, history, completion, seed):
    common = common_inputs(ratio, seed)
    layer, _, d, *_ = common
    cw = common[5].shape[1]
    g = torch.Generator().manual_seed(seed + 317)
    position = history + (ratio - 1 if completion else 1)
    next_block = position // ratio
    carry, persist, requests = {}, {}, []
    for pool in range(batch):
        begin = max(0, next_block - 1) * ratio
        n = position - begin
        carry[(layer, pool)] = {
            "kv": torch.randn(n, cw, generator=g),
            "score": torch.randn(n, cw, generator=g) * 3,
            "pos": torch.arange(begin, position), "next_block": next_block,
        }
        persist[(layer, pool)] = (
            torch.randn(next_block, d, generator=g).to(torch.bfloat16),
            torch.arange(next_block) * ratio,
        )
        requests.append((pool, torch.randn(1, cw, generator=g),
                         torch.randn(1, cw, generator=g) * 3,
                         torch.tensor([position]), False))
    return common, (carry, persist), requests


def measure_case(infra, batch, ratio, history, completion, seed):
    common, initial, requests = make_case(batch, ratio, history, completion, seed)
    cache = torch.empty(1)
    backend = types.SimpleNamespace(
        token_to_kv_pool=types.SimpleNamespace(get_extra_key_buffer=lambda _: cache))
    positions = torch.cat([r[3] for r in requests])
    forward_batch = types.SimpleNamespace(extend_seq_lens_cpu=None)
    arms = {"scalar": accumulate_scalar, "batched": accumulate_batched}
    samples = {key: [] for key in arms}
    reference = None
    for rep in range(-1, 3):
        for name in list(arms) if rep == -1 or rep % 2 == 0 else list(reversed(arms)):
            infra._WIN_CARRY, infra._COMPRESSED_PERSIST = copy.deepcopy(initial)
            infra._COMPRESSED_HANDOFF = {}
            start = time.perf_counter_ns()
            result = arms[name](infra, requests, common)
            infra._publish_full_history_handoff(
                backend, forward_batch, common[0], ratio, positions, result, False)
            elapsed = time.perf_counter_ns() - start
            state = (result, infra._WIN_CARRY, infra._COMPRESSED_PERSIST,
                     infra._COMPRESSED_HANDOFF[cache.data_ptr()])
            if reference is None:
                # The first warm arm is scalar; both arms are compared on every timed trial.
                if name != "scalar":
                    raise RuntimeError("baseline must establish reference first")
                reference = copy.deepcopy(state)
            elif not equal_tree(reference, state):
                raise RuntimeError(f"output/carry/history/handoff byte mismatch in {name}")
            if elapsed <= 0 or elapsed > 30_000_000_000:
                raise RuntimeError("invalid or over-budget compression sample")
            if rep >= 0:
                samples[name].append(elapsed)
    medians = {k: statistics.median(v) for k, v in samples.items()}
    return {
        "batch": batch, "ratio": ratio, "history": history, "completion": completion,
        "position": int(positions[0]), "seed": seed,
        "samples_ns": samples, "median_ns": medians,
        "range_ns": {k: [min(v), max(v)] for k, v in samples.items()},
        "saved_ns": medians["scalar"] - medians["batched"],
        "speedup": medians["scalar"] / medians["batched"],
        "output_carry_history_handoff_bytes_match": True,
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    if torch.get_num_threads() != 32 or sorted(os.sched_getaffinity(0)) != list(range(32)):
        raise RuntimeError("screen requires fixed32 CPU0-31")
    from intel_cpu_models import _dsv4_cpu_infra as infra
    import sglang.kernels.ops.attention.dsv4.index_buf_accessor as accessor
    paths = [Path(__file__), Path(inspect.getfile(accumulate_batched)),
             Path(inspect.getfile(common_inputs)), Path(inspect.getfile(infra)),
             Path(inspect.getfile(accessor)),
             Path(torch.__file__).parent / "lib/libtorch_cpu.so"]
    hashes = {str(p): sha(p) for p in paths}
    report = {
        "job": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename, "pid": os.getpid(),
        "threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__, "source_and_library_sha256": hashes,
        "environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")},
        "input_scope": "Deterministic signed random projection rows and shape-faithful synthetic "
                       "carry/history, not captured neural activations. FP32 pool, BF16 norm/RoPE "
                       "boundaries and FP8 roundtrip use the unchanged production producer.",
        "timing_scope": "One process, three alternating reused-buffer trials per case after one "
                        "warm call per arm. State restore/checks excluded; preparation, pool, norm, "
                        "RoPE, FP8, history update and production handoff included. Projection excluded.",
        "cases": [],
    }
    for batch in (1, 32, 64):
        for history in (128, 3072):
            for ratio in (4, 128):
                for completion in (False, True):
                    row = measure_case(infra, batch, ratio, history, completion, seed=17)
                    report["cases"].append(row)
                    (output / "partial.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(f"COMP_BATCH B{batch} S{history} C{ratio} complete={completion} "
                          f"scalar_ms={row['median_ns']['scalar']/1e6:.4f} "
                          f"batched_ms={row['median_ns']['batched']/1e6:.4f} "
                          f"speedup={row['speedup']:.3f}", flush=True)
    if any(sha(p) != value for p, value in hashes.items()):
        raise RuntimeError("screen source/library changed during run")
    report["conditional_cycle_estimates"] = []
    for batch in (1, 32, 64):
        for history in (128, 3072):
            saved = 0
            for ratio, layers in ((4, 21), (128, 20)):
                rows = {r["completion"]: r for r in report["cases"]
                        if (r["batch"], r["history"], r["ratio"]) == (batch, history, ratio)}
                saved += layers * ((ratio - 1) * rows[False]["saved_ns"] + rows[True]["saved_ns"]) / ratio
            report["conditional_cycle_estimates"].append({
                "batch": batch, "history": history, "saved_ms_per_decode_step": saved / 1e6,
                "status": "Synthetic extrapolation only: 21*C4 + 20*C128 at exact cadence. "
                          "No real-layer variation, serving interference or E2E speedup measured.",
            })
    (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPRESSION_BATCH_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.resolve())
