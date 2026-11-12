import unittest
from contextlib import ExitStack
from pathlib import Path
import tempfile
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import screen_c4_crossover as screen
from screen_c4_crossover import geometry, timed


class CrossoverTests(unittest.TestCase):
    def test_summary_rejects_bad_replicas_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "job-test"
            results.mkdir()
            (results / "exit_code").write_text("0\n")
            (results / "hostname.txt").write_text("testnode\n")
            sources = [root / "source/plugin/validate" / name for name in (
                "screen_c4_crossover.py", "bench_long_prefill_sparse.py",
                "check_c4_numerics.py", "replay_serving_sparse.py")]
            sources += [root / "source/plugin/dsv4_sparse_trial.py", screen.donor.NATIVE,
                        screen.donor.NATIVE.with_name("build.ninja"),
                        screen.donor.FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp",
                        Path(torch.__file__).parent / "lib/libtorch_cpu.so"]
            original = {}
            for replica in range(3):
                folder = results / f"replicate{replica}"
                folder.mkdir()
                identity = {"job": "test", "pid": replica + 100, "replicate": replica,
                            "threads": 32, "affinity": list(range(32)), "node": "testnode",
                            "torch": torch.__version__, "llc_size": "4K",
                            "flush_bytes": 2 * 4096 + 32 * 2**20, "scope": "test fixture",
                            "environment": {"OMP_NUM_THREADS": "32", "OMP_WAIT_POLICY": "active",
                                            "GOMP_SPINCOUNT": "1000000"},
                            "capture_sha256": screen.donor.CAPTURE_HASHES[2],
                            "native_sha256": screen.donor.NATIVE_HASH,
                            "sources": {str(p): "digest" for p in sources}}
                timing = {"samples_ns": {"online": [9, 10, 11], "python_gemm": [4, 5, 6]},
                          "median_ns": {"online": 10, "python_gemm": 5}, "speedup": 2}
                packet = {"identity": identity, "n": 1, "k": 16, "divisor": 1, "pattern": "block",
                          "h": 64, "d": 512, "visible_min": 16, "visible_max": 16,
                          "visible_mean": 16, "density": 1, "numerical_sampled_rows": [0],
                          "numerical": {arm: {"relative_l2": 1e-7} for arm in ("online", "python_gemm")},
                          "timings": {mode: timing for mode in ("warm", "llc_evicted")}}
                capture = copy.deepcopy(packet)
                capture["numerical_sampled_rows"] = [0, 256, 511]
                for name, value in {
                    "identity.json": identity, "complete.json": {"identity": identity, "cases": 1},
                    "calibration.json": {"identity": identity, "gemm_flops": 2 * 4096**3,
                                         "streaming_rmw_logical_bytes": 2 * identity["flush_bytes"],
                                         "samples_ns": {"fp32_gemm": [1, 2, 3], "streaming_rmw": [1, 2, 3]}},
                    "captured.json": capture,
                    "n1-k16-v1-block.json": packet,
                }.items():
                    path = folder / name
                    original[path] = json.dumps(value)
                    path.write_text(original[path])
            with patch.object(screen, "coordinates", return_value=[(1, 16, 1, "block")]), \
                    patch.object(screen.donor, "sha", return_value="digest"):
                result = screen.summarize(root, "test", "digest")
                self.assertEqual(result["rows"]["captured.json"]["warm"]["median_speedup"], 2)
                for key, value in (("n", 64), ("n", True), ("density", 0.5), ("numerical_sampled_rows", [1])):
                    path = results / "replicate0/n1-k16-v1-block.json"
                    changed = json.loads(original[path])
                    changed[key] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(RuntimeError):
                        screen.summarize(root, "test", "digest")
                    path.write_text(original[path])
                path = results / "replicate1/identity.json"
                changed = json.loads(original[path])
                changed["pid"] = 100
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(RuntimeError, "Duplicate process"):
                    screen.summarize(root, "test", "digest")
                path.write_text(original[path])
                path = results / "replicate2/n1-k16-v1-block.json"
                path.unlink()
                with self.assertRaisesRegex(RuntimeError, "coordinates"):
                    screen.summarize(root, "test", "digest")

    def test_run_rejects_coordinate_drift_without_completion_record(self):
        sample = {"q": torch.randn(4, 2, 8), "kv": torch.randn(16, 8),
                  "mask": torch.ones(4, 16, dtype=torch.bool),
                  "sink": torch.randn(2), "scale": 8**-0.5}
        sample["output"] = screen.shared_chunked(*(sample[k] for k in ("q", "kv", "mask", "sink", "scale")))
        native = SimpleNamespace(sparse_attend_masked=screen.shared_chunked)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda unused: None))
        cpp = screen.donor.FROZEN / "plugin/kernels/dsa_pilot/sparse_attend.cpp"
        def fake_read(path, *args, **kwargs):
            return "4K" if path.name == "size" else str(cpp)
        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for name, value in (("QUERIES", (1,)), ("KEYS", (16,)),
                                ("DIVISORS", (1,)), ("PATTERNS", ("block",))):
                stack.enter_context(patch.object(screen, name, value))
            stack.enter_context(patch.object(screen, "load_capture", return_value=sample))
            stack.enter_context(patch.object(screen, "calibration", return_value={"test": True}))
            stack.enter_context(patch.object(screen.donor, "sha", return_value=screen.donor.NATIVE_HASH))
            stack.enter_context(patch.object(screen.importlib.util, "spec_from_file_location", return_value=spec))
            stack.enter_context(patch.object(screen.importlib.util, "module_from_spec", return_value=native))
            stack.enter_context(patch.object(screen.torch, "get_num_threads", return_value=32))
            stack.enter_context(patch.object(screen.os, "sched_getaffinity", return_value=set(range(32))))
            stack.enter_context(patch.object(Path, "read_text", fake_read))
            stack.enter_context(patch.dict(screen.os.environ, {
                "SLURM_JOB_ID": "test", "OMP_NUM_THREADS": "32",
                "OMP_WAIT_POLICY": "active", "GOMP_SPINCOUNT": "1000000",
            }))
            good = Path(directory) / "good"
            screen.run(good, 0)
            self.assertTrue((good / "complete.json").is_file())
            actual_geometry = screen.geometry
            def drift(sample, n, k, divisor, pattern):
                return actual_geometry(sample, n + 1, k, divisor, pattern)
            bad = Path(directory) / "bad"
            with patch.object(screen, "geometry", side_effect=drift):
                with self.assertRaisesRegex(RuntimeError, "declared coordinate"):
                    screen.run(bad, 0)
            self.assertFalse((bad / "complete.json").exists())

    def test_timing_and_numerical_screen_reject_corruption(self):
        generator = torch.Generator().manual_seed(7)
        args = (torch.randn(3, 2, 8, generator=generator),
                torch.randn(16, 8, generator=generator), torch.ones(3, 16, dtype=torch.bool),
                torch.randn(2, generator=generator), 8**-0.5)
        native = SimpleNamespace(sparse_attend_masked=screen.shared_chunked)
        result = timed(args, native, torch.zeros(256, dtype=torch.uint8))
        for mode in ("warm", "llc_evicted"):
            for samples in result["timings"][mode]["samples_ns"].values():
                self.assertEqual(len(samples), 3)
                self.assertTrue(all(v > 0 for v in samples))
        for bad in (torch.full_like(args[0], float("nan")), torch.ones_like(args[0]),
                    torch.zeros(1), args[0].double()):
            native.sparse_attend_masked = lambda *unused: bad
            with self.assertRaises(RuntimeError):
                timed(args, native, torch.zeros(256, dtype=torch.uint8))
        native.sparse_attend_masked = screen.shared_chunked
        with patch.object(screen.donor, "same_bytes", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Repeated output"):
                timed(args, native, torch.zeros(256, dtype=torch.uint8))
        with patch.object(screen, "gathered_fp64", return_value=torch.full_like(args[0], float("inf"))):
            with self.assertRaisesRegex(RuntimeError, "invalid numerical evidence"):
                timed(args, native, torch.zeros(256, dtype=torch.uint8))

    def test_geometry_controls_work_independently(self):
        sample = {"q": torch.randn(4, 2, 8), "kv": torch.randn(16, 8),
                  "sink": torch.randn(2), "scale": 8**-0.5}
        for n in (1, 2, 8):
            for k in (16, 32):
                for divisor in (1, 2, 4, 8, 16):
                    for pattern in ("block", "strided"):
                        q, kv, mask, sink, scale = geometry(sample, n, k, divisor, pattern)
                        self.assertEqual(q.shape, (n, 2, 8))
                        self.assertEqual(kv.shape, (k, 8))
                        self.assertEqual(mask.shape, (n, k))
                        self.assertTrue(torch.equal(mask.sum(1), torch.full((n,), k // divisor)))
                        self.assertTrue(q.is_contiguous() and kv.is_contiguous() and mask.is_contiguous())
        with self.assertRaises(ValueError):
            geometry(sample, 2, 16, 2, "invalid")


if __name__ == "__main__":
    unittest.main()
