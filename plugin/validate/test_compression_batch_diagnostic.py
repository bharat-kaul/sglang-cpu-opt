import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import torch

from compression_batch_diagnostic import accumulate_batched, accumulate_scalar


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def common_inputs(ratio, seed=0, fp8=True):
    g = torch.Generator().manual_seed(seed)
    d, rd = 512, 64
    angles = torch.outer(torch.arange(4096).float(), 1 / (160000 ** (torch.arange(0, rd, 2).float() / rd)))
    return (2 if ratio == 4 else 3, ratio, d, rd, ratio == 4,
            torch.randn(ratio, (2 if ratio == 4 else 1) * d, generator=g) * 0.1,
            torch.rand(d, generator=g) + 0.5, 1e-6,
            torch.polar(torch.ones_like(angles), angles), fp8, torch.bfloat16)


def evaluate(infra, fn, initial, requests, common):
    infra._WIN_CARRY, infra._COMPRESSED_PERSIST = copy.deepcopy(initial)
    result = fn(infra, requests, common)
    return copy.deepcopy((result, infra._WIN_CARRY, infra._COMPRESSED_PERSIST))


class CompressionBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from intel_cpu_models import _dsv4_cpu_infra
        cls.infra = _dsv4_cpu_infra

    def test_request_boundaries_history_and_rounding(self):
        for ratio in (4, 128):
            for fp8 in (False, True):
                for seed in (0, 7):
                    common = common_inputs(ratio, seed, fp8)
                    cw = common[5].shape[1]
                    g = torch.Generator().manual_seed(seed + 13)
                    # Heterogeneous request lengths cover first/partial/multiple windows.
                    lengths = (0, ratio - 1, ratio, 2 * ratio + 1)
                    requests = [(i, torch.randn(n, cw, generator=g),
                                 torch.randn(n, cw, generator=g) * 3, torch.arange(n), True)
                                for i, n in enumerate(lengths)]
                    old = evaluate(self.infra, accumulate_scalar, ({}, {}), requests, common)
                    new = evaluate(self.infra, accumulate_batched, ({}, {}), requests, common)
                    self.assertTrue(equal_tree(old, new), (ratio, fp8, seed, "prefill"))
                    # Advance asynchronously; some requests finish a window while others do not.
                    for step in range(1, 6):
                        requests = [(i, torch.randn(1, cw, generator=g),
                                     torch.randn(1, cw, generator=g) * 3,
                                     torch.tensor([n + step - 1]), False)
                                    for i, n in enumerate(lengths)]
                        baseline = (old[1], old[2])
                        old = evaluate(self.infra, accumulate_scalar, baseline, requests, common)
                        new = evaluate(self.infra, accumulate_batched, baseline, requests, common)
                        self.assertTrue(equal_tree(old, new), (ratio, fp8, seed, step))
                    requests = [(1, torch.zeros(ratio, cw), torch.zeros(ratio, cw),
                                 torch.arange(ratio), True)]
                    baseline = (old[1], old[2])
                    old = evaluate(self.infra, accumulate_scalar, baseline, requests, common)
                    new = evaluate(self.infra, accumulate_batched, baseline, requests, common)
                    self.assertTrue(equal_tree(old, new), "slot reuse/reset")
                    mutated = copy.deepcopy(new)
                    mutated[1][(common[0], 1)]["next_block"] += 1
                    self.assertFalse(equal_tree(old, mutated))

    def test_duplicate_slots_rejected_before_mutation(self):
        common = common_inputs(4)
        request = (0, torch.zeros(4, 1024), torch.zeros(4, 1024), torch.arange(4), True)
        self.infra._WIN_CARRY, self.infra._COMPRESSED_PERSIST = {}, {}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            accumulate_batched(self.infra, [request, request], common)
        self.assertEqual(self.infra._WIN_CARRY, {})

    def test_measured_entry_checks_state_and_rejects_bad_candidate(self):
        import bench_compression_batch as bench
        for ratio in (4, 128):
            for completion in (False, True):
                result = bench.measure_case(self.infra, 2, ratio, 128, completion, seed=3)
                self.assertEqual(len(result["samples_ns"]["scalar"]), 3)
                self.assertTrue(result["output_carry_history_handoff_bytes_match"])

        original = bench.accumulate_batched

        def broken(infra, requests, common):
            out = original(infra, requests, common)
            infra._WIN_CARRY[(common[0], requests[0][0])]["next_block"] += 1
            return out

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(bench, "accumulate_batched", broken), \
                patch.object(torch, "get_num_threads", return_value=32), \
                patch.object(bench.os, "sched_getaffinity", return_value=set(range(32))), \
                patch.dict(os.environ, {"SLURM_JOB_ID": "test", "OMP_NUM_THREADS": "32",
                                       "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "1000000"}), \
                patch.object(bench, "sha", return_value="test"):
            out = Path(tmp) / "results"
            with self.assertRaisesRegex(RuntimeError, "byte mismatch"):
                bench.run(out)
            self.assertFalse((out / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
