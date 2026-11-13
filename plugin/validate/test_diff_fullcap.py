import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

import diff_fullcap


class FullCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a = {"pf.L0.r0": torch.tensor([[1., -2.]]), "logits.0.r0": torch.zeros(1, 2)}
        self.b = {"pf.L0.r1": self.a["pf.L0.r0"].clone(), "logits.0.r1": torch.zeros(1, 2)}
        self.expected = {
            "schema_version": 1,
            "tensors": {key: {"shape": [1, 2], "dtype": "float32", "layout": "replicated"}
                        for key in ("pf.L0", "logits.0")},
        }
        self.run_metadata = None

    def run_compare(self, *extra):
        torch.save(self.a, self.root / "a.pt")
        torch.save(self.b, self.root / "b.pt")
        if self.run_metadata is not None:
            for side in ("a", "b"):
                record = copy.deepcopy(self.run_metadata[side])
                record["capture_sha256"] = hashlib.sha256((self.root / f"{side}.pt").read_bytes()).hexdigest()
                (self.root / f"{side}.pt.oracle.json").write_text(json.dumps(record))
        inventory = self.root / "expected.json"
        inventory.write_text(json.dumps(self.expected))
        return subprocess.run([
            sys.executable, "-B", diff_fullcap.__file__, str(self.root / "a.pt"), str(self.root / "b.pt"),
            "--expected", str(inventory),
            "--expected-sha256", hashlib.sha256(inventory.read_bytes()).hexdigest(),
            "--atol", "0", "--rtol", "0", *extra,
        ], capture_output=True, text=True, timeout=30)

    def test_valid_replicated_signals_and_zero_tensors_pass(self):
        self.a["pf.L0.r0"] *= 1e-30
        self.b["pf.L0.r1"] *= 1e-30
        result = self.run_compare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TENSOR PARITY PASSED", result.stdout)
        aliased = list(result.args)
        aliased[4] = aliased[3]
        result = subprocess.run(aliased, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("alias the same capture", result.stderr)

    def test_missing_on_one_or_both_sides_and_rank_collisions_reject(self):
        original_a, original_b = copy.deepcopy(self.a), copy.deepcopy(self.b)
        for mutation in ("one-side", "both-sides", "duplicate", "empty"):
            with self.subTest(mutation=mutation):
                self.a, self.b = copy.deepcopy(original_a), copy.deepcopy(original_b)
                if mutation == "one-side":
                    del self.b["pf.L0.r1"]
                elif mutation == "both-sides":
                    del self.b["pf.L0.r1"]
                    del self.a["pf.L0.r0"]
                elif mutation == "duplicate":
                    self.b["pf.L0.r2"] = self.b["pf.L0.r1"].clone()
                else:
                    self.a, self.b = {}, {}
                result = self.run_compare()
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertNotIn("TENSOR PARITY PASSED", result.stdout)

    def test_invalid_reference_candidate_shapes_and_values_reject(self):
        original_a, original_b = copy.deepcopy(self.a), copy.deepcopy(self.b)
        for side in ("a", "b"):
            for value in (torch.tensor([[float("nan"), 2.]]), torch.tensor([[float("inf"), 2.]]),
                          torch.ones(2, 1), torch.ones(1, 2, dtype=torch.float64)):
                with self.subTest(side=side, value=value):
                    self.a, self.b = copy.deepcopy(original_a), copy.deepcopy(original_b)
                    getattr(self, side)["pf.L0.r0" if side == "a" else "pf.L0.r1"] = value
                    result = self.run_compare()
                    self.assertEqual(result.returncode, 2, result.stdout)

    def test_cosine_alone_cannot_hide_magnitude_regression_or_invalid_tolerance(self):
        self.b["pf.L0.r1"] *= 2
        result = self.run_compare()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("DIVERGE", result.stdout)
        result = self.run_compare("--atol", "nan")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.expected["tensors"]["pf.L0"]["layout"] = "sharded"
        result = self.run_compare()
        self.assertEqual(result.returncode, 2, result.stdout)

    def test_run_metadata_is_bound_to_expected_inputs_weights_and_rank(self):
        self.b = copy.deepcopy(self.a)
        common = {
            "checkpoint_sha256": "a" * 64, "runner_sha256": "b" * 64,
            "input_ids": [1, 2], "max_new_tokens": 1, "dtype": "float32",
            "context_length": 128, "chunked_prefill_size": 64, "disable_radix_cache": True,
        }
        self.expected["run"] = common
        self.run_metadata = {
            side: {**common, "schema_version": 1, "device": device, "tp_size": tp, "capture_rank": 0}
            for side, device, tp in (("a", "cuda", 4), ("b", "cpu", 1))
        }
        result = self.run_compare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("matched checkpoint/input/run metadata", result.stdout)
        torch.save({"pf.L0.r0": torch.zeros(1, 2)}, self.root / "b.pt")
        tampered = subprocess.run(result.args, capture_output=True, text=True)
        self.assertEqual(tampered.returncode, 2)
        self.assertIn("capture digest mismatch", tampered.stderr)
        for field, value in (("input_ids", [1, 3]), ("input_ids", [True, 2]),
                             ("checkpoint_sha256", "c" * 64), ("dtype", "float16"),
                             ("max_new_tokens", True), ("disable_radix_cache", 1)):
            with self.subTest(field=field, value=value):
                original = self.run_metadata["b"][field]
                self.run_metadata["b"][field] = value
                result = self.run_compare()
                self.assertEqual(result.returncode, 2, result.stdout)
                self.run_metadata["b"][field] = original


if __name__ == "__main__":
    unittest.main()
