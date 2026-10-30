import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from report_matched_allocations import HERE, report, summarize


class MatchedAllocationReportTests(unittest.TestCase):
    def setUp(self):
        self.runs = [
            json.loads((HERE / "results/matched_allocations" / f"replicate-{i}.json").read_text())
            for i in range(1, 4)
        ]

    def test_actual_evidence(self):
        s = summarize(self.runs)
        self.assertEqual(s["cold"]["fp32"]["median_us"], 710.949)
        self.assertEqual(s["cold"]["bf16"]["median_us"], 330.163)
        self.assertIn("no preallocated output", report(s))
        self.assertEqual(s["allocation_audit"]["fp32"]["dtype_conversion_copies"], 0)
        self.assertEqual(s["allocation_audit"]["bf16"]["dtype_conversion_copies"], 0)

    def test_invalid_evidence_rejected(self):
        mutations = (
            lambda r: r[1].update(pid=r[0]["pid"]),
            lambda r: r[1].update(threads=128),
            lambda r: r[1].update(node="different"),
            lambda r: r[1]["allocation_audit"]["bf16"].update(dtype_conversion_copies=1),
            lambda r: r[1]["allocation_audit"]["bf16"].update(positive_self_allocations=[]),
            lambda r: r[1]["cold"]["timings"]["bf16"].update(median_us=1),
            lambda r: r[1]["cold"]["timings"]["bf16"]["samples_us"].__setitem__(0, float("nan")),
            lambda r: r[1]["accuracy"].pop(),
            lambda r: r[1]["build"]["sources"].update(**{"sparse_attn.cpp": "wrong"}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                runs = copy.deepcopy(self.runs)
                mutation(runs)
                with self.assertRaises(ValueError):
                    summarize(runs)

    def test_cli_rejects_without_writes(self):
        with tempfile.TemporaryDirectory(prefix="matched-allocation-test-") as tmp:
            directory = Path(tmp)
            self.runs[1]["allocation_audit"]["bf16"]["dtype_conversion_copies"] = 1
            for i, data in enumerate(self.runs, 1):
                (directory / f"replicate-{i}.json").write_text(json.dumps(data))
            result = subprocess.run(
                [sys.executable, str(HERE / "report_matched_allocations.py"), "--results", tmp],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allocation contract failed", result.stderr)
            self.assertFalse((directory / "summary.json").exists())
            self.assertFalse((directory / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
